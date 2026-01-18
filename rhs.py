#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVART Timing Protocol Decoder + Benchmark GUI
----------------------------------------------
Integrates:
  - Live decoding GUI (scores, penalties, period, CRC)
  - Time integrity benchmark (dt_wall, dt_game, error), series CSV + optional histograms

Run:
  python rt_gui.py

License: MIT
"""

import sys
import os
import threading
import time
import queue
import re
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union

# Optional serial support
try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
    HAS_SERIAL = True
except Exception:
    HAS_SERIAL = False

import tkinter as tk
from tkinter import ttk, messagebox

STX = 0x02
ETX = 0x03
LF  = 0x0A

MESSAGE_LEN = 55  # bytes / ASCII chars

# Period / mode mapping based on the documentation
PERIOD_MAP: Dict[str, str] = {
    '1': '1st Period',
    '2': '2nd Period',
    '3': '3rd Period',
    '4': 'Overtime',
    '5': 'Intermission',
    '6': 'Shootout',
    '7': 'Wall Clock',
    '8': 'Pre-game Countdown',
}

# -------------------- Paths (from bm.py) --------------------

def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = get_app_dir()
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------- Protocol helpers --------------------

@dataclass
class Penalty:
    player_number: Optional[int]  # None if slot unused
    minutes: Optional[int]
    seconds: Optional[int]

    def is_empty(self) -> bool:
        return self.player_number is None

@dataclass
class RevartMessage:
    raw: str
    clock: Tuple[int, int, int]   # (MM, SS, DD) or (HH,MM,SS) when mode 7/8
    clock_text: str              # e.g. "19:44:18"
    running: bool                # T flag
    period_code: str             # single char '1'..'8'
    period_text: str             # mapped text
    home_score: int
    away_score: int
    home_penalties: List[Penalty]  # length 3
    away_penalties: List[Penalty]  # length 3
    crc_hex: str                 # two ASCII HEX chars
    crc_ok: bool
    receive_ts: Optional[float] = None  # perf_counter timestamp at frame boundary

def xor_crc_ascii_hex(segment_bytes: bytes) -> str:
    """Compute 8-bit XOR over given bytes and return two-digit uppercase hex string."""
    acc = 0
    for b in segment_bytes:
        acc ^= b
    return f"{acc:02X}"

def parse_two_int(s: str) -> Optional[int]:
    if not s or any(c not in "0123456789" for c in s):
        return None
    try:
        return int(s)
    except ValueError:
        return None

def parse_penalty_block(block: str) -> Penalty:
    """block is 6 chars: ZZMMSS or spaces if unused."""
    if len(block) != 6:
        raise ValueError("Penalty block must be 6 chars")
    if block.strip() == "":
        return Penalty(None, None, None)
    zz = block[0:2]
    mm = block[2:4]
    ss = block[4:6]
    pnum = parse_two_int(zz)
    pmin = parse_two_int(mm)
    psec = parse_two_int(ss)
    return Penalty(pnum, pmin, psec)

def split_penalties(segment: str) -> List[Penalty]:
    if len(segment) != 18:
        raise ValueError("Penalties segment must be 18 chars (3x6)")
    return [parse_penalty_block(segment[i:i+6]) for i in range(0, 18, 6)]

def decode_message(msg: str) -> RevartMessage:
    """
    Decode a single 55-char message.
    Format:
    <STX>MM:SS:DDT P XXYY HHHHH... (18) AAA... (18) <ETX><CRC><LF>
    """
    if len(msg) != MESSAGE_LEN:
        raise ValueError(f"Message length must be {MESSAGE_LEN}, got {len(msg)}")
    if msg[0] != chr(STX):
        raise ValueError("Missing STX")
    if msg[-1] != chr(LF):
        raise ValueError("Missing LF")

    clock_text = msg[1:9]
    t_flag = msg[9]
    p_code = msg[10]
    score_pair = msg[11:15]
    home_pen = msg[15:33]
    away_pen = msg[33:51]
    etx_char = msg[51]
    crc_hex = msg[52:54]

    if etx_char != chr(ETX):
        raise ValueError("Missing ETX")

    # compute CRC over bytes STX..ETX inclusive
    crc_segment = msg[:52].encode('ascii', errors='strict')
    computed_crc = xor_crc_ascii_hex(crc_segment)
    crc_ok = (computed_crc.upper() == crc_hex.upper())

    if not re.match(r"^\d{2}:\d{2}:\d{2}$", clock_text):
        raise ValueError("Clock field must be NN:NN:NN")

    a, b, c = clock_text.split(":")
    MM = int(a); SS = int(b); DD = int(c)

    if t_flag not in ('0', '1'):
        raise ValueError("T flag must be '0' or '1'")
    running = (t_flag == '1')

    period_text = PERIOD_MAP.get(p_code, f"Unknown({p_code})")

    if len(score_pair) != 4 or not score_pair.isdigit():
        raise ValueError("Score field must be 4 digits (XXYY)")
    home_score = int(score_pair[0:2])
    away_score = int(score_pair[2:4])

    home_penalties = split_penalties(home_pen)
    away_penalties = split_penalties(away_pen)

    return RevartMessage(
        raw=msg,
        clock=(MM, SS, DD),
        clock_text=clock_text,
        running=running,
        period_code=p_code,
        period_text=period_text,
        home_score=home_score,
        away_score=away_score,
        home_penalties=home_penalties,
        away_penalties=away_penalties,
        crc_hex=crc_hex,
        crc_ok=crc_ok
    )

# -------------------- Benchmark (from bm.py) --------------------

@dataclass
class Parsed:
    raw: str
    recv_ts: float
    valid_crc: bool
    running: bool
    mode: str
    clock_text: str
    mm: int
    ss: int
    dd: int
    total_hundredths: Optional[int]
    home_score: int
    away_score: int

def parse_message(line: str, recv_ts: Optional[float] = None) -> Parsed:
    if len(line) != MESSAGE_LEN:
        raise ValueError(f"Message length must be {MESSAGE_LEN}, got {len(line)}")
    if line[0] != chr(STX) or line[51] != chr(ETX) or line[-1] != chr(LF):
        raise ValueError("Bad framing (STX/ETX/LF)")
    body = line[:52].encode("ascii", errors="strict")
    crc_rx = line[52:54]
    crc_ok = (xor_crc_ascii_hex(body).upper() == crc_rx.upper())

    clock_text = line[1:9]
    if not re.match(r"^\d{2}:\d{2}:\d{2}$", clock_text):
        raise ValueError("Clock field malformed")

    T = line[9]
    P = line[10]
    running = (T == "1")

    mm, ss, dd = clock_text.split(":")
    MM = int(mm); SS = int(ss); DD = int(dd)

    sc = line[11:15]
    if not sc.isdigit():
        raise ValueError("Score field malformed")
    home = int(sc[:2]); away = int(sc[2:])

    if P in {"1", "2", "3", "4", "5", "6"}:
        total_hundredths = (MM * 60 + SS) * 100 + DD
    else:
        total_hundredths = None

    return Parsed(
        raw=line,
        recv_ts=time.perf_counter() if recv_ts is None else recv_ts,
        valid_crc=crc_ok,
        running=running,
        mode=P,
        clock_text=clock_text,
        mm=MM, ss=SS, dd=DD,
        total_hundredths=total_hundredths,
        home_score=home, away_score=away
    )

def normalize_frame(line: str) -> Optional[str]:
    """
    Try to recover a full 55-char frame.

    Accepts:
      - exact 55-char frames (preferred)
      - lines >= 55 (takes last 55 chars)
      - 54-char frames missing LF (adds LF)
    Returns None if cannot normalize.
    """
    if not line:
        return None

    if len(line) == MESSAGE_LEN:
        return line

    if len(line) == MESSAGE_LEN - 1:
        if line and line[0] == chr(STX) and len(line) > 51 and line[51] == chr(ETX):
            return line + chr(LF)
        return None

    if len(line) > MESSAGE_LEN:
        cand = line[-MESSAGE_LEN:]
        if cand and cand[0] == chr(STX) and cand[51] == chr(ETX) and cand[-1] == chr(LF):
            return cand
        if len(cand) == MESSAGE_LEN and cand[0] == chr(STX) and cand[51] == chr(ETX):
            if cand[-1] != chr(LF):
                cand = cand[:-1] + chr(LF)
                if cand[-1] == chr(LF):
                    return cand
        return None

    return None

class SeriesWriter:
    """
    Stream samples into a temporary CSV file with auto-flush.
    On finalize(), assemble final CSV with a stats header at the beginning.
    """
    def __init__(self, final_path: str, flush_every: int = 1):
        self.final_path = final_path
        self.tmp_path = final_path + ".tmp"
        self.flush_every = max(1, int(flush_every))
        self._f = open(self.tmp_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._count = 0

    def write_row(self, dt_wall_ms: float, dt_game_ms: float, err_ms: float):
        self._w.writerow([f"{dt_wall_ms:.3f}", f"{dt_game_ms:.3f}", f"{err_ms:.3f}"])
        self._count += 1
        if (self._count % self.flush_every) == 0:
            self._f.flush()

    def close(self):
        try:
            self._f.flush()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass

    def finalize(self, stats_lines: List[str]):
        self.close()
        with open(self.final_path, "w", newline="") as out:
            for line in stats_lines:
                out.write(f"# {line}\n")
            out.write("#\n")
            out.write("dt_wall_ms,dt_game_ms,error_ms\n")
            with open(self.tmp_path, "r", newline="") as inp:
                for chunk in inp:
                    out.write(chunk)
        try:
            os.remove(self.tmp_path)
        except Exception:
            pass

class Benchmark:
    def __init__(self, ignore_bad_crc: bool = True, series_writer: Optional[SeriesWriter] = None):
        self.ignore_bad_crc = ignore_bad_crc
        self.prev: Optional[Parsed] = None
        self.prev_any: Optional[Parsed] = None

        self.dt_wall_ms: List[float] = []
        self.dt_wall_all_ms: List[float] = []
        self.dt_game_ms: List[float] = []
        self.err_ms: List[float] = []

        self.total_frames = 0
        self.bad_crc = 0
        self.skipped = 0

        self.series_writer = series_writer

    def feed(self, frame: str, recv_ts: Optional[float] = None):
        self.total_frames += 1
        try:
            p = parse_message(frame, recv_ts=recv_ts)
        except Exception:
            self.skipped += 1
            return

        if not p.valid_crc:
            self.bad_crc += 1
            if self.ignore_bad_crc:
                return

        if self.prev_any is not None:
            dt_wall_all = (p.recv_ts - self.prev_any.recv_ts) * 1000.0
            self.dt_wall_all_ms.append(dt_wall_all)
        self.prev_any = p

        # Ignore non-game-time modes
        if (p.total_hundredths is None):
            self.prev = p
            return

        if self.prev is not None and self.prev.running and self.prev.total_hundredths is not None:
            dt_wall = (p.recv_ts - self.prev.recv_ts) * 1000.0

            diff = p.total_hundredths - self.prev.total_hundredths
            if diff == 0:
                dt_game = 0.0
            elif diff < 0:
                dt_game = (-diff) * 10.0
            else:
                dt_game = diff * 10.0

            err = dt_game - dt_wall

            self.dt_wall_ms.append(dt_wall)
            self.dt_game_ms.append(dt_game)
            self.err_ms.append(err)

            if self.series_writer is not None:
                self.series_writer.write_row(dt_wall, dt_game, err)

        self.prev = p

    def stats(self) -> Optional[Dict[str, float]]:
        n_game = len(self.err_ms)
        if n_game == 0:
            return None

        errors = self.err_ms
        abs_err = [abs(e) for e in errors]

        mean_err = sum(errors) / n_game
        rms = (sum(e * e for e in errors) / n_game) ** 0.5
        p95_abs = sorted(abs_err)[max(0, int(0.95 * n_game) - 1)]

        wall = self.dt_wall_all_ms
        game = self.dt_game_ms

        wall_n = len(wall)
        wall_mean = (sum(wall) / wall_n) if wall_n else 0.0
        wall_rms = ((sum(x * x for x in wall) / wall_n) ** 0.5) if wall_n else 0.0
        wall_p95 = sorted(wall)[max(0, int(0.95 * wall_n) - 1)] if wall_n else 0.0

        game_mean = sum(game) / n_game
        game_rms = (sum(x * x for x in game) / n_game) ** 0.5
        game_p95 = sorted(game)[max(0, int(0.95 * n_game) - 1)]

        zeros_wall = sum(1 for x in wall if x == 0.0)
        zeros_game = sum(1 for x in game if x == 0.0)
        zeros_err  = sum(1 for x in errors if x == 0.0)

        return {
            "total_frames": float(self.total_frames),
            "wall_samples": float(wall_n),
            "game_samples": float(n_game),
            "used_samples": float(n_game),
            "bad_crc": float(self.bad_crc),
            "skipped": float(self.skipped),
            "avg_period_ms": float(wall_mean),
            "mean_error_ms": float(mean_err),
            "rms_error_ms": float(rms),
            "p95_abs_error_ms": float(p95_abs),
            "wall_mean": float(wall_mean),
            "game_mean": float(game_mean),
            "wall_rms": float(wall_rms),
            "game_rms": float(game_rms),
            "wall_p95": float(wall_p95),
            "game_p95": float(game_p95),
            "zeros_wall": float(zeros_wall),
            "zeros_game": float(zeros_game),
            "zeros_err": float(zeros_err),
        }

    def stats_lines_for_file(self) -> List[str]:
        s = self.stats()
        if not s:
            w = self.wall_stats()
            if not w:
                return ["No samples collected."]
            return [
                "REVART wall-clock statistics (no running samples)",
                f"frames_total={int(w['total_frames'])}",
                f"wall_samples={int(w['wall_samples'])}",
                f"bad_crc={int(w['bad_crc'])} ignored_bad_crc={self.ignore_bad_crc}",
                f"skipped_parse={int(w['skipped'])}",
                f"wall_mean_ms={w['wall_mean']:.3f}",
                f"wall_rms_ms={w['wall_rms']:.3f}",
                f"wall_p95_ms={w['wall_p95']:.3f}",
                f"zeros_wall={int(w['zeros_wall'])}",
                f"generated_at_epoch={time.time():.3f}",
            ]
        return [
            "REVART time integrity benchmark results",
            f"frames_total={int(s['total_frames'])}",
            f"wall_samples={int(s['wall_samples'])}",
            f"game_samples={int(s['game_samples'])}",
            f"bad_crc={int(s['bad_crc'])} ignored_bad_crc={self.ignore_bad_crc}",
            f"skipped_parse={int(s['skipped'])}",
            f"avg_wall_period_ms={s['avg_period_ms']:.3f}",
            f"mean_error_ms={s['mean_error_ms']:.3f}",
            f"rms_error_ms={s['rms_error_ms']:.3f}",
            f"p95_abs_error_ms={s['p95_abs_error_ms']:.3f}",
            f"wall_mean_ms={s['wall_mean']:.3f} wall_rms_ms={s['wall_rms']:.3f} wall_p95_ms={s['wall_p95']:.3f}",
            f"game_mean_ms={s['game_mean']:.3f} game_rms_ms={s['game_rms']:.3f} game_p95_ms={s['game_p95']:.3f}",
            f"zeros_wall={int(s['zeros_wall'])} zeros_game={int(s['zeros_game'])} zeros_err={int(s['zeros_err'])}",
            f"generated_at_epoch={time.time():.3f}",
        ]

    def wall_stats(self) -> Optional[Dict[str, float]]:
        n = len(self.dt_wall_all_ms)
        if n == 0:
            return None

        wall = self.dt_wall_all_ms
        wall_mean = sum(wall) / n
        wall_rms = (sum(x * x for x in wall) / n) ** 0.5
        wall_p95 = sorted(wall)[max(0, int(0.95 * n) - 1)]
        zeros_wall = sum(1 for x in wall if x == 0.0)

        return {
            "total_frames": float(self.total_frames),
            "wall_samples": float(n),
            "bad_crc": float(self.bad_crc),
            "skipped": float(self.skipped),
            "wall_mean": float(wall_mean),
            "wall_rms": float(wall_rms),
            "wall_p95": float(wall_p95),
            "zeros_wall": float(zeros_wall),
        }

    def _fixed_edges_0_to_max(self, values: List[float], bins: int = 100) -> List[float]:
        maxv = max(values) if values else 0.0
        if maxv <= 0.0:
            return [0.0, 1.0]
        step = maxv / bins
        return [i * step for i in range(bins + 1)]

    def plot_wall_hist(self, bins: int = 100, save_path: Optional[str] = None):
        wall = [x for x in self.dt_wall_all_ms if x >= 0.0]
        if not wall:
            raise RuntimeError("No wall dt samples to plot.")
        import matplotlib.pyplot as plt  # optional dependency

        edges = self._fixed_edges_0_to_max(wall, bins=bins)
        plt.figure()
        plt.hist(wall, bins=edges)
        plt.xlabel("Wall interval dt [ms]")
        plt.ylabel("Count")
        plt.title(f"Histogram of wall-clock intervals (N={len(wall)})")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    def plot_game_hist(self, bins: int = 100, save_path: Optional[str] = None):
        game = [x for x in self.dt_game_ms if x >= 0.0]
        if not game:
            raise RuntimeError("No game dt samples to plot.")
        import matplotlib.pyplot as plt  # optional dependency

        edges = self._fixed_edges_0_to_max(game, bins=bins)
        plt.figure()
        plt.hist(game, bins=edges)
        plt.xlabel("Game interval dt [ms]")
        plt.ylabel("Count")
        plt.title(f"Histogram of game-clock intervals (N={len(game)})")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

# -------------------- Capture settings --------------------

@dataclass
class Cfg:
    # CLI params (bm.py)
    port: str = ""
    baud: int = 38400                # protocol default (bm.py used 115200 by default)
    series: str = "output.txt"                 # csv filename or path
    hist_wall: str = "wall_clock_hist"              # png filename or path
    hist_game: str = "game_clock_hist"              # png filename or path
    flush_every: int = 5
    heartbeat_s: float = 5

def _is_abs_path(p: str) -> bool:
    try:
        return os.path.isabs(p)
    except Exception:
        return False

def _as_out_path(user_path: str) -> str:
    """
    bm.py writes outputs into ./data by default. In GUI we follow the same rule:
      - if user_path is absolute -> use as-is
      - else -> place inside DATA_DIR
    """
    if not user_path:
        return ""
    return user_path if _is_abs_path(user_path) else os.path.join(DATA_DIR, user_path)

# -------------------- Readers (raw frames) --------------------

@dataclass
class FrameItem:
    raw: str
    ts: float

class BaseReader(threading.Thread):
    def __init__(self, out_queue: "queue.Queue[FrameItem]"):
        super().__init__(daemon=True)
        self._out = out_queue
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

class SerialFrameReader(BaseReader):
    def __init__(self, port: str, baud: int, out_queue: "queue.Queue[FrameItem]"):
        if not HAS_SERIAL:
            raise RuntimeError("pyserial is required for serial capture.")
        super().__init__(out_queue)
        self._port = port
        self._baud = int(baud)
        self._ser = serial.Serial(self._port, baudrate=self._baud, timeout=0)  # non-blocking

    def run(self):
        buf = bytearray()
        try:
            while not self._stop.is_set():
                n = self._ser.in_waiting
                if n:
                    buf += self._ser.read(n)
                    while True:
                        idx = buf.find(b"\n")
                        if idx < 0:
                            break
                        frame = bytes(buf[:idx+1])
                        del buf[:idx+1]
                        ts = time.perf_counter()
                        try:
                            s = frame.decode("ascii", errors="strict")
                        except Exception:
                            continue
                        self._out.put(FrameItem(raw=s, ts=ts))
                else:
                    time.sleep(0.001)
        finally:
            try:
                self._ser.close()
            except Exception:
                pass

# -------------------- Settings window --------------------

class SettingsWin(tk.Toplevel):
    def __init__(self, parent: "App", cfg: Cfg):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.configure(bg="#1E1B22")
        self.parent = parent
        self.cfg = cfg

        pad = {"padx": 10, "pady": 6}

        frm = tk.Frame(self, bg="#1E1B22")
        frm.pack(fill="both", expand=True, padx=12, pady=12)

        def row(label: str, widget: tk.Widget, r: int):
            tk.Label(frm, text=label, fg="#DCE0E6", bg="#1E1B22").grid(row=r, column=0, sticky="w", **pad)
            widget.grid(row=r, column=1, sticky="ew", **pad)

        frm.grid_columnconfigure(1, weight=1)

        self.baud_var = tk.StringVar(value=str(cfg.baud))
        self.baud_ent = ttk.Entry(frm, textvariable=self.baud_var, width=30)
        row("Baud rate", self.baud_ent, 0)

        self.series_var = tk.StringVar(value=cfg.series)
        self.series_ent = ttk.Entry(frm, textvariable=self.series_var, width=30)
        row("Statistics file name", self.series_ent, 1)        

        self.hw_var = tk.StringVar(value=cfg.hist_wall)
        self.hw_ent = ttk.Entry(frm, textvariable=self.hw_var, width=30)
        row("Hist wall PNG file name", self.hw_ent, 2)

        self.hg_var = tk.StringVar(value=cfg.hist_game)
        self.hg_ent = ttk.Entry(frm, textvariable=self.hg_var, width=30)
        row("Hist game PNG file name", self.hg_ent, 3)

        self.flush_var = tk.StringVar(value=str(cfg.flush_every))
        self.flush_ent = ttk.Entry(frm, textvariable=self.flush_var, width=30)
        row("Flush every (seconds)", self.flush_ent, 4)

        self.hb_var = tk.StringVar(value=str(cfg.heartbeat_s))
        self.hb_ent = ttk.Entry(frm, textvariable=self.hb_var, width=30)
        row("Heartbeat (seconds)", self.hb_ent, 5)

        btns = tk.Frame(frm, bg="#1E1B22")
        btns.grid(row=6, column=0, columnspan=2, sticky="e", pady=(12, 0))

        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="right")

    def _apply(self):
        try:
            baud = int(self.baud_var.get().strip())
            if baud <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid baud", "Baud must be a positive integer.")
            return

        try:
            flush = int(self.flush_var.get().strip() or "1")
            if flush <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid flush-every", "Flush-every must be a positive integer.")
            return

        try:
            hb = float(self.hb_var.get().strip())
            if hb < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid heartbeat", "Heartbeat must be a non-negative number (0 = disable).")
            return

        self.cfg.baud = baud
        self.cfg.series = self.series_var.get().strip()
        self.cfg.hist_wall = self.hw_var.get().strip()
        self.cfg.hist_game = self.hg_var.get().strip()
        self.cfg.flush_every = flush
        self.cfg.heartbeat_s = hb

        self.parent._log_bm("Settings applied.")
        self.destroy()

# ---------------- GUI -----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("REVART Timing Protocol")
        self.geometry("860x700")
        self.configure(bg="#1E1B22")  # dark background

        self.cfg = Cfg()

        self.q: "queue.Queue[FrameItem]" = queue.Queue()
        self.reader: Optional[BaseReader] = None
        self.connected = False
        self._connecting = False

        self.last_received_ts: Optional[float] = None

        self.bench: Optional[Benchmark] = None
        self.series_writer: Optional[SeriesWriter] = None
        self.capture_active = False
        self._starting_capture = False
        self._cap_start_ts: Optional[float] = None
        self._last_hb_ts: Optional[float] = None

        self._build_top_bar()
        self._build_center()
        self._build_sides()
        self._build_bottom()

        self.after(20, self._poll_queue)
        self.after(20, self._update_since_label)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------- Layout parts -------------

    def _build_top_bar(self):
        top = tk.Frame(self, bg="#1E1B22")
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(8, 6))

        self.period_var = tk.StringVar(value="1st Period")
        self.period_label = tk.Label(top, textvariable=self.period_var, fg="#DCE0E6",
                                     bg="#2B2731", font=("Segoe UI", 14, "bold"),
                                     padx=16, pady=8)
        self.period_label.pack(side=tk.LEFT)

        tk.Label(top, text="", bg="#1E1B22").pack(side=tk.LEFT, expand=True)

        right = tk.Frame(top, bg="#1E1B22")
        right.pack(side=tk.RIGHT)

        self.src_status = tk.Label(right, text="Port: Disconnected", fg="#A0AEC0", bg="#1E1B22")
        self.src_status.grid(row=0, column=0, padx=6)

        self.capture_status_var = tk.StringVar(value="Capture: Ready")
        self.capture_status = tk.Label(right, textvariable=self.capture_status_var,
                                       fg="#A0AEC0", bg="#1E1B22")
        self.capture_status.grid(row=1, column=0, padx=6, pady=(4, 0), sticky="w")        

        since_box = tk.Frame(right, bg="#1E1B22")
        since_box.grid(row=0, column=1, padx=(6, 6))

        tk.Label(since_box, text="Since last:", fg="#A0AEC0", bg="#1E1B22",
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        self.since_val_var = tk.StringVar(value="-- ms")
        self.since_val_label = tk.Label(since_box, textvariable=self.since_val_var,
                                        fg="#A0AEC0", bg="#1E1B22",
                                        font=("Consolas", 10, "bold"),
                                        width=9, anchor="e")
        self.since_val_label.pack(side="left")

        # Port selector (serial source)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(right, width=15, state="readonly", textvariable=self.port_var)
        self._refresh_ports()
        self.port_combo.grid(row=0, column=2, padx=6)

        self.connect_btn = ttk.Button(right, text="CONNECT", command=self._connect, width=11)
        self.connect_btn.grid(row=0, column=3, padx=6)

        self.disconnect_btn = ttk.Button(right, text="DISCONNECT", command=self._disconnect, state="disabled", width=13)
        self.disconnect_btn.grid(row=0, column=4, padx=6)

        self.settings_btn = ttk.Button(right, text="Settings", command=self._open_settings, width=11)
        self.settings_btn.grid(row=0, column=5, padx=6)

        self.capture_btn = ttk.Button(right, text="CAPTURE", command=self._start_capture, state="disabled", width=11)
        self.capture_btn.grid(row=1, column=3, padx=6, pady=(4, 0))

        self.stop_btn = ttk.Button(right, text="STOP", command=self._stop_capture, state="disabled", width=11)
        self.stop_btn.grid(row=1, column=4, padx=6, pady=(4, 0))

    def _build_center(self):
        center = tk.Frame(self, bg="#1E1B22")
        center.pack(side=tk.TOP, fill=tk.X, padx=12, pady=6)

        self.clock_var = tk.StringVar(value="20:00.00")
        self.clock_label = tk.Label(center, textvariable=self.clock_var,
                                    font=("Segoe UI", 48, "bold"),
                                    fg="#F2F4F8", bg="#1E1B22")
        self.clock_label.pack(side=tk.TOP, pady=(6, 2))

        self.run_var = tk.StringVar(value="Stopped")
        self.run_label = tk.Label(center, textvariable=self.run_var, fg="#F2F4F8",
                                  bg="#1E1B22", font=("Segoe UI", 12, "bold"))
        self.run_label.pack(side=tk.TOP)

    def _build_sides(self):
        body = tk.Frame(self, bg="#1E1B22")
        body.pack(side=tk.TOP, fill=tk.X, padx=12, pady=6)

        self.left = self._team_panel(body, "Home team", "")
        self.left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.right = self._team_panel(body, "Away team", "")
        self.right.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

    def _team_panel(self, parent, team_name, label_above):
        frame = tk.Frame(parent, bg="#2B2731", bd=0, highlightthickness=0)
        header = tk.Frame(frame, bg="#8B1E2D")
        header.pack(side=tk.TOP, fill=tk.X)

        name_lbl = tk.Label(header, text=team_name, fg="white", bg="#8B1E2D", font=("Segoe UI", 16, "bold"))
        name_lbl.pack(side=tk.LEFT, padx=10, pady=8)

        role_lbl = tk.Label(header, text=label_above, fg="#F5B7B1", bg="#8B1E2D", font=("Segoe UI", 10, "bold"))
        role_lbl.pack(side=tk.RIGHT, padx=10)

        score_frame = tk.Frame(frame, bg="#2B2731")
        score_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 2))

        score_title = tk.Label(score_frame, text="SCORE", fg="#A0AEC0", bg="#2B2731")
        score_title.pack(side=tk.LEFT, padx=10)
        score_var = tk.StringVar(value="0")
        score_label = tk.Label(score_frame, textvariable=score_var, fg="#FFFFFF", bg="#2B2731", font=("Segoe UI", 26, "bold"))
        score_label.pack(side=tk.RIGHT, padx=10)

        pens_frame = tk.Frame(frame, bg="#2B2731")
        pens_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 10))

        rows = []
        for _ in range(3):
            row = tk.Frame(pens_frame, bg="#3A3342")
            row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

            pnum = tk.StringVar(value="--")
            tk.Label(row, textvariable=pnum, width=4, fg="#F2F2F2", bg="#3A3342",
                     font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=6, pady=6)

            ptime = tk.StringVar(value="--:--")
            tk.Label(row, textvariable=ptime, width=6, fg="#F2F2F2", bg="#3A3342",
                     font=("Segoe UI", 12, "bold")).pack(side=tk.RIGHT, padx=6, pady=6)

            rows.append((pnum, ptime))

        frame.score_var = score_var          # type: ignore[attr-defined]
        frame.penalty_rows = rows            # type: ignore[attr-defined]
        return frame

    def _build_bottom(self):
        bottom = tk.Frame(self, bg="#1E1B22")
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=12, pady=8)

        # Last message section
        last_frame = tk.Frame(bottom, bg="#1E1B22")
        last_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(last_frame, text="Last message", fg="#A0AEC0", bg="#1E1B22",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        self.last_msg = tk.Text(last_frame, height=1, bg="#0F0D11", fg="#E2E8F0", insertbackground="#E2E8F0")
        self.last_msg.pack(fill=tk.BOTH, expand=True)
        self.last_msg.configure(state="disabled")

        #ttk.Separator(bottom, orient="horizontal").pack(fill=tk.X, pady=8)

        # Benchmark section
        bm_frame = tk.Frame(bottom, bg="#1E1B22")
        bm_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(bm_frame, text="Benchmark", fg="#A0AEC0", bg="#1E1B22",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        self.bm_log = tk.Text(bm_frame, height=6, bg="#0F0D11", fg="#E2E8F0", insertbackground="#E2E8F0")
        self.bm_log.pack(fill=tk.BOTH, expand=True)
        self.bm_log.configure(state="disabled")

    # ------------- Helpers -------------

    def _refresh_ports(self):
        ports: List[str] = []
        if HAS_SERIAL:
            try:
                ports = [p.device for p in serial.tools.list_ports.comports()]
            except Exception:
                ports = []
        self.port_combo["values"] = ports
        if ports:
            preferred = self.cfg.port or self.port_var.get()
            if preferred in ports:
                self.port_combo.current(ports.index(preferred))
            else:
                self.port_combo.current(0)
        else:
            self.port_var.set("")

    def _log_last(self, s: str):
        self.last_msg.configure(state="normal")
        self.last_msg.delete("1.0", "end")
        self.last_msg.insert("end", s)
        self.last_msg.configure(state="disabled")

    def _log_bm(self, s: str):
        self.bm_log.configure(state="normal")
        self.bm_log.insert("end", s + "\n")
        self.bm_log.see("end")
        self.bm_log.configure(state="disabled")

    # ------------- Capture control -------------

    def _open_settings(self):
        SettingsWin(self, self.cfg)

    def _connect(self):
        if self.connected or getattr(self, "_connecting", False):
            return

        self._connecting = True
        self.src_status.config(text="Port: Opening...")
        self.connect_btn.config(state="disabled")
        self.disconnect_btn.config(state="disabled")
        self.capture_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.settings_btn.config(state="disabled")
        self.port_combo.config(state="disabled")

        def worker():
            try:
                port = self.port_var.get().strip()
                if not port:
                    raise RuntimeError("Select a serial port.")
                self.cfg.port = port
                reader = SerialFrameReader(port, self.cfg.baud, self.q)
                status_text = f"Port: {port} @ {self.cfg.baud}"

                def ui_start():
                    self.reader = reader
                    self.connected = True
                    self.src_status.config(text=status_text)
                    self.connect_btn.config(state="disabled")
                    self.disconnect_btn.config(state="normal")
                    self.capture_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.settings_btn.config(state="disabled")
                    self.port_combo.config(state="disabled")
                    self.capture_status_var.set("Capture: Ready")
                    self._connecting = False
                    self.reader.start()

                self.after(0, ui_start)

            except Exception as e:
                err_msg = str(e)
                def ui_fail():
                    self._connecting = False
                    self.connected = False
                    self.reader = None
                    self.src_status.config(text="Port: Disconnected")
                    self.connect_btn.config(state="normal")
                    self.disconnect_btn.config(state="disabled")
                    self.capture_btn.config(state="disabled")
                    self.stop_btn.config(state="disabled")
                    self.settings_btn.config(state="normal")
                    self.port_combo.config(state="readonly")
                    self.capture_status_var.set("Capture: Ready")
                    messagebox.showerror("Connect error", err_msg)
                self.after(0, ui_fail)

        threading.Thread(target=worker, daemon=True).start()

    def _disconnect(self):
        if not self.connected and not getattr(self, "_connecting", False):
            return

        if self.capture_active:
            self._stop_capture()

        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception:
                pass
        self.reader = None

        self.connected = False
        self._connecting = False

        self.src_status.config(text="Port: Disconnected")
        self.connect_btn.config(state="normal")
        self.disconnect_btn.config(state="disabled")
        self.capture_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.settings_btn.config(state="normal")
        self.port_combo.config(state="readonly")
        self.capture_status_var.set("Capture: Ready")
        self.capture_status.config(fg="#A0AEC0")

    def _start_capture(self):
        if self.capture_active or getattr(self, "_starting_capture", False):
            return

        if not self.connected:
            messagebox.showerror("Not connected", "Connect to a source before starting capture.")
            return

        self._starting_capture = True

        try:
            os.makedirs(DATA_DIR, exist_ok=True)
        except Exception as e:
            self._starting_capture = False
            messagebox.showerror("Capture error", f"Failed to create data directory: {e}")
            return

        self.bm_log.configure(state="normal")
        self.bm_log.delete("1.0", "end")
        self.bm_log.configure(state="disabled")

        series_path = _as_out_path(self.cfg.series)
        self.series_writer = None
        try:
            if series_path:
                self.series_writer = SeriesWriter(series_path, flush_every=self.cfg.flush_every)
        except Exception as e:
            self._starting_capture = False
            messagebox.showerror("Capture error", str(e))
            return

        self.bench = Benchmark(
            ignore_bad_crc=True,
            series_writer=self.series_writer,
        )

        self.capture_active = True
        self._cap_start_ts = time.perf_counter()
        self._last_hb_ts = self._cap_start_ts
        self.capture_status_var.set("Capture: Runnig")
        self.capture_status.config(fg="#3ED38A")

        if series_path:
            self._log_bm(f"Series CSV (temp): {series_path}.tmp")
        self._log_bm(f"Capture started. Outputs in: {DATA_DIR}")

        self.stop_btn.config(state="normal")
        self.capture_btn.config(state="disabled")
        self.settings_btn.config(state="disabled")
        self._starting_capture = False

    def _stop_capture(self):
        if not self.capture_active:
            return

        self.capture_status_var.set("Capture: Saving")
        self.capture_status.config(fg="#D3973E")
        self.update_idletasks()

        # Finalize series
        if self.series_writer is not None and self.bench is not None:
            try:
                stats_lines = self.bench.stats_lines_for_file()
                self.series_writer.finalize(stats_lines)
                self._log_bm(f"Series CSV written: {_as_out_path(self.cfg.series)}")
            except Exception as e:
                self._log_bm(f"[ERR] Series finalize failed: {e}")
        self.series_writer = None

        # Histograms
        if self.bench is not None:
            try:
                s = self.bench.stats()
                w = None if s is not None else self.bench.wall_stats()

                if self.cfg.hist_wall:
                    p = _as_out_path(self.cfg.hist_wall)
                    self.bench.plot_wall_hist(bins=100, save_path=p)
                    self._log_bm(f"Wall histogram written: {p}.png")

                if s is not None and self.cfg.hist_game:
                    p = _as_out_path(self.cfg.hist_game)
                    self.bench.plot_game_hist(bins=100, save_path=p)
                    self._log_bm(f"Game histogram written: {p}.png")
            except ImportError:
                self._log_bm("[ERR] matplotlib is required to write histograms.")
            except Exception as e:
                self._log_bm(f"[ERR] Histogram error: {e}")

            # Summary
            if s is None:
                w = self.bench.wall_stats()
                if w is None:
                    self._log_bm("No samples collected.")
                else:
                    self._log_bm("")
                    self._log_bm("Summary (wall only):")
                    self._log_bm(f"Frames total: {int(w['total_frames'])}")
                    self._log_bm(f"Wall samples: {int(w['wall_samples'])}")
                    self._log_bm(f"CRC bad: {int(w['bad_crc'])}  Skipped parse: {int(w['skipped'])}")
                    self._log_bm(f"Wall mean: {w['wall_mean']:.2f} ms")
                    self._log_bm(f"Wall RMS: {w['wall_rms']:.2f} ms")
                    self._log_bm(f"Wall 95th %: {w['wall_p95']:.2f} ms")
            else:
                self._log_bm("")
                self._log_bm("Summary:")
                self._log_bm(f"Frames total: {int(s['total_frames'])}")
                self._log_bm(f"Wall samples: {int(s['wall_samples'])}")
                self._log_bm(f"Game samples: {int(s['game_samples'])}")
                self._log_bm(f"CRC bad: {int(s['bad_crc'])}  Skipped parse: {int(s['skipped'])}")
                self._log_bm(f"Avg wall period: {s['avg_period_ms']:.2f} ms")
                self._log_bm(f"Mean error (game - wall): {s['mean_error_ms']:.2f} ms")
                self._log_bm(f"RMS error: {s['rms_error_ms']:.2f} ms")
                self._log_bm(f"95th % abs error: {s['p95_abs_error_ms']:.2f} ms")
        self.bench = None

        self.capture_active = False
        self._cap_start_ts = None
        self._last_hb_ts = None

        self.capture_btn.config(state="normal" if self.connected else "disabled")
        self.stop_btn.config(state="disabled")
        self.settings_btn.config(state="disabled" if self.connected else "normal")
        self.port_combo.config(state="disabled" if self.connected else "readonly")
        self.capture_status_var.set("Capture: Ready")
        self.capture_status.config(fg="#A0AEC0")

    def _on_close(self):
        try:
            self._disconnect()
        except Exception:
            pass
        self.destroy()

    # ------------- Update UI -------------

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                self.last_received_ts = item.ts

                # Heartbeat (bm.py behavior)
                if self.capture_active and self.cfg.heartbeat_s > 0 and self._last_hb_ts is not None and self.bench is not None:
                    now = time.perf_counter()
                    if (now - self._last_hb_ts) >= self.cfg.heartbeat_s:
                        elapsed_h = 0
                        elapsed_m = 0
                        if self._cap_start_ts is not None:
                            elapsed_s = int(now - self._cap_start_ts)
                            elapsed_h = elapsed_s // 3600
                            elapsed_m = (elapsed_s % 3600) // 60
                        self._log_bm(
                            f"[HB] {elapsed_h:d}h {elapsed_m:02d}m | messages={self.bench.total_frames} runnnig samples={len(self.bench.err_ms)}"
                        )
                        self._last_hb_ts = now

                nf = normalize_frame(item.raw)
                if nf is None:
                    continue

                # Benchmark feed
                if self.capture_active and self.bench is not None:
                    self.bench.feed(nf, recv_ts=item.ts)

                # Decoder view (penalties, etc.)
                try:
                    rm = decode_message(nf)
                    rm.receive_ts = item.ts
                    self._apply_message(rm)
                except Exception as e:
                    # Still show raw frame
                    self._log_last(f"{nf}\nDecode error: {e}\n")
        except queue.Empty:
            pass
        self.after(20, self._poll_queue)

    def _update_since_label(self):
        if self.last_received_ts is None:
            new_text = "-- ms"
            new_color = "#A0AEC0"
        else:
            delta_ms = int((time.perf_counter() - self.last_received_ts) * 1000)
            new_text = f"{delta_ms} ms"
            new_color = "#3ED38A" if delta_ms <= 21 else "#E53E3E"
        self.since_val_var.set(new_text)
        self.since_val_label.config(fg=new_color)
        self.after(20, self._update_since_label)

    def _apply_message(self, rm: RevartMessage):
        self.period_var.set(PERIOD_MAP.get(rm.period_code, "Unknown"))

        if rm.period_code in ('7', '8'):
            display = rm.clock_text
        else:
            display = f"{rm.clock_text[0:5]}.{rm.clock_text[6:8]}"
        self.clock_var.set(display)

        self.run_var.set("Running" if rm.running else "Stopped")
        self.clock_label.config(fg="#3ED38A" if rm.running else "#FFFFFF")
        self.run_label.config(fg="#3ED38A" if rm.running else "#FFFFFF")

        self.left.score_var.set(str(rm.home_score))
        self.right.score_var.set(str(rm.away_score))

        def set_rows(frame, pens: List[Penalty]):
            for i, (pvar, tvar) in enumerate(frame.penalty_rows):
                if i < len(pens) and not pens[i].is_empty():
                    p = pens[i]
                    pvar.set(f"{p.player_number:02d}" if p.player_number is not None else "--")
                    if p.minutes is not None and p.seconds is not None:
                        tvar.set(f"{p.minutes:01d}:{p.seconds:02d}")
                    else:
                        tvar.set("--:--")
                else:
                    pvar.set("--")
                    tvar.set("--:--")

        set_rows(self.left, rm.home_penalties)
        set_rows(self.right, rm.away_penalties)

        crc_status = "OK" if rm.crc_ok else "BAD"
        self._log_last(f"{rm.raw.strip("\n")}")
        #    CRC={rm.crc_hex} ({crc_status})\n

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
