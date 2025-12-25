#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVART Timing Protocol Decoder + Simple GUI
-------------------------------------------
- Decodes messages based on REVART TIMING Protocol.
- Run:
    python revart_timing_decoder_gui.py

Author: Martin Ťavoda, Revart s.r.o. (with ChatGPT 5.1)
License: MIT
"""

import sys
import threading
import time
import queue
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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
PERIOD_MAP = {
    '1': '1st Period',
    '2': '2nd Period',
    '3': '3rd Period',
    '4': 'Overtime',
    '5': 'Intermission',
    '6': 'Shootout',
    '7': 'Wall Clock',
    '8': 'Pre-game Countdown',
}

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
    clock: Tuple[int,int,int]  # (MM, SS, DD) or (HH,MM,SS) when mode 7/8
    clock_text: str            # Display string as sent, e.g. "19:44:18"
    running: bool              # T flag
    period_code: str           # single char '1'..'8'
    period_text: str           # mapped text
    home_score: int
    away_score: int
    home_penalties: List[Penalty]  # length 3
    away_penalties: List[Penalty]  # length 3
    crc_hex: str               # two ASCII HEX chars
    crc_ok: bool
    # high-resolution receive timestamp (perf_counter), set by reader thread
    receive_ts: Optional[float] = None

def xor_crc_ascii_hex(segment_bytes: bytes) -> str:
    """
    Compute 8-bit XOR over given bytes and return two-digit uppercase hex string.
    """
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
    """
    block is 6 chars: ZZMMSS or spaces if unused.
    """
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
    <STX>MM:SS:DDT P XXYY HHHHHHHHHHHHHHHHH AAA... <ETX><CRC><LF>
    """
    if len(msg) != MESSAGE_LEN:
        raise ValueError(f"Message length must be {MESSAGE_LEN}, got {len(msg)}")

    if msg[0] != chr(STX):
        raise ValueError("Missing STX")
    if msg[-1] != chr(LF):
        raise ValueError("Missing LF")

    # positions
    # [0] STX
    # [1:9] clock "MM:SS:DD" (8 chars)
    # [9] T
    # [10] P
    # [11:15] XXYY (4 chars)
    # [15:33] home penalties (18)
    # [33:51] away penalties (18)
    # [51] ETX
    # [52:54] CRC (2)
    # [54] LF

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

    # parse clock
    if not re.match(r"^\d{2}:\d{2}:\d{2}$", clock_text):
        raise ValueError("Clock field must be MM:SS:DD (or HH:MM:SS in modes 7/8)")

    a, b, c = clock_text.split(":")
    MM = int(a)
    SS = int(b)
    DD = int(c)

    # running flag
    if t_flag not in ('0', '1'):
        raise ValueError("T flag must be '0' or '1'")
    running = (t_flag == '1')

    # period/mode
    period_text = PERIOD_MAP.get(p_code, f"Unknown({p_code})")

    # score
    if len(score_pair) != 4 or not score_pair.isdigit():
        raise ValueError("Score field must be 4 digits (XXYY)")
    home_score = int(score_pair[0:2])
    away_score = int(score_pair[2:4])

    # penalties
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

# ---------------- GUI -----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("REVART Timing Protocol - Decoder GUI")
        self.geometry("1050x640")
        self.configure(bg="#1E1B22")  # dark background

        self.queue: "queue.Queue[RevartMessage]" = queue.Queue()

        # last received high-resolution timestamp (perf_counter)
        self.last_received_ts: Optional[float] = None

        self._build_top_bar()
        self._build_center()
        self._build_sides()
        self._build_bottom_log()

        # Reader
        self.reader = None  # type: ignore
        # start queue poll and since-label updater (20 ms for responsive display)
        self.after(20, self._poll_queue)
        self.after(20, self._update_since_label)

    # ------------- Layout parts -------------

    def _build_top_bar(self):
        top = tk.Frame(self, bg="#1E1B22")
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(8, 6))

        # Period/Mode display
        self.period_var = tk.StringVar(value="1st Period")  # localized example
        self.period_label = tk.Label(top, textvariable=self.period_var, fg="#DCE0E6",
                                     bg="#2B2731", font=("Segoe UI", 14, "bold"),
                                     padx=16, pady=8)
        self.period_label.pack(side=tk.LEFT)

        # Spacer
        tk.Label(top, text="", bg="#1E1B22").pack(side=tk.LEFT, expand=True)

        # Serial widgets
        right = tk.Frame(top, bg="#1E1B22")
        right.pack(side=tk.RIGHT)

        self.serial_status = tk.Label(right, text="Mode: Disconnected", fg="#A0AEC0", bg="#1E1B22")
        self.serial_status.grid(row=0, column=0, padx=6)

        # --- inside _build_top_bar() ---
        since_box = tk.Frame(right, bg="#1E1B22")
        since_box.grid(row=0, column=1, padx=(6,6))

        tk.Label(
            since_box,
            text="Since last:",
            fg="#A0AEC0",
            bg="#1E1B22",
            font=("Segoe UI", 10, "bold")
        ).pack(side="left")

        self.since_val_var = tk.StringVar(value="-- ms")
        self.since_val_label = tk.Label(
            since_box,
            textvariable=self.since_val_var,
            fg="#A0AEC0",
            bg="#1E1B22",
            font=("Consolas", 10, "bold"),
            width=9,          # constant width
            anchor="e"
        )
        self.since_val_label.pack(side="left")


        self.port_combo = ttk.Combobox(right, width=24, state="readonly")
        if HAS_SERIAL:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        else:
            ports = []
        self.port_combo["values"] = ports
        if ports:
            self.port_combo.current(0)
        self.port_combo.grid(row=0, column=2, padx=6)

        self.connect_btn = ttk.Button(right, text="Connect", command=self._toggle_connect)
        self.connect_btn.grid(row=0, column=3, padx=6)

    def _build_center(self):
        center = tk.Frame(self, bg="#1E1B22")
        center.pack(side=tk.TOP, fill=tk.X, padx=12, pady=6)

        # Big clock
        self.clock_var = tk.StringVar(value="20:00.00")
        self.clock_label = tk.Label(center, textvariable=self.clock_var,
                                    font=("Segoe UI", 48, "bold"),
                                    fg="#F2F4F8", bg="#1E1B22")
        self.clock_label.pack(side=tk.TOP, pady=(6,2))

        # Start/Stop indicator
        self.run_var = tk.StringVar(value="Stopped")
        self.run_label = tk.Label(center, textvariable=self.run_var, fg="#F2F4F8",
                                  bg="#1E1B22", font=("Segoe UI", 12, "bold"))
        self.run_label.pack(side=tk.TOP)

    def _build_sides(self):
        body = tk.Frame(self, bg="#1E1B22")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=6)

        self.left = self._team_panel(body, "Home team", "")
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.right = self._team_panel(body, "Away team", "")
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

    def _team_panel(self, parent, team_name, label_above):
        frame = tk.Frame(parent, bg="#2B2731", bd=0, highlightthickness=0)
        header = tk.Frame(frame, bg="#8B1E2D")
        header.pack(side=tk.TOP, fill=tk.X)

        name_lbl = tk.Label(header, text=team_name, fg="white", bg="#8B1E2D", font=("Segoe UI", 16, "bold"))
        name_lbl.pack(side=tk.LEFT, padx=10, pady=8)

        role_lbl = tk.Label(header, text=label_above, fg="#F5B7B1", bg="#8B1E2D", font=("Segoe UI", 10, "bold"))
        role_lbl.pack(side=tk.RIGHT, padx=10)

        score_frame = tk.Frame(frame, bg="#2B2731")
        score_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 4))

        score_title = tk.Label(score_frame, text="SCORE", fg="#A0AEC0", bg="#2B2731")
        score_title.pack(side=tk.LEFT, padx=10)
        score_var = tk.StringVar(value="0")
        score_label = tk.Label(score_frame, textvariable=score_var, fg="#FFFFFF", bg="#2B2731", font=("Segoe UI", 26, "bold"))
        score_label.pack(side=tk.RIGHT, padx=10)

        # Penalty list
        pens_frame = tk.Frame(frame, bg="#2B2731")
        pens_frame.pack(side=tk.TOP, fill=tk.X, pady=(6, 10))

        rows = []
        for i in range(3):
            row = tk.Frame(pens_frame, bg="#3A3342")
            row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

            pnum = tk.StringVar(value="--")
            pnum_lbl = tk.Label(row, textvariable=pnum, width=4, fg="#F2F2F2", bg="#3A3342", font=("Segoe UI", 12, "bold"))
            pnum_lbl.pack(side=tk.LEFT, padx=6, pady=6)

            ptime = tk.StringVar(value="--:--")
            ptime_lbl = tk.Label(row, textvariable=ptime, width=6, fg="#F2F2F2", bg="#3A3342", font=("Segoe UI", 12, "bold"))
            ptime_lbl.pack(side=tk.RIGHT, padx=6, pady=6)

            rows.append((pnum, ptime))

        # Attach references
        frame.score_var = score_var          # type: ignore
        frame.penalty_rows = rows            # type: ignore
        return frame

    def _build_bottom_log(self):
        bottom = tk.Frame(self, bg="#1E1B22")
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=12, pady=8)

        lbl = tk.Label(bottom, text="Last message", fg="#A0AEC0", bg="#1E1B22")
        lbl.pack(anchor="w")

        self.last_msg = tk.Text(bottom, height=4, bg="#0F0D11", fg="#E2E8F0", insertbackground="#E2E8F0")
        self.last_msg.pack(fill=tk.BOTH, expand=True)
        self.last_msg.configure(state="disabled")

    # ------------- Reader management -------------

    def _toggle_connect(self):
        # If pyserial is missing, stop right here
        if not HAS_SERIAL:
            messagebox.showerror(
                "Serial unavailable",
                "pyserial is required."
            )
            return

        # If already connected, request stop
        if self.reader and self.reader.is_alive():
            self.reader.stop()
            self.reader = None
            self.connect_btn.config(text="Connect")
            self.serial_status.config(text="Mode: Disconnected")
            return

        # Try to connect
        port = self.port_combo.get()
        if not port:
            messagebox.showwarning("No port", "Select a serial port first.")
            return

        try:
            # Opening the port happens in __init__ now (main thread),
            # so we can catch any error and show it nicely.
            self.reader = SerialReader(port, self.queue)
        except Exception as e:
            messagebox.showerror("Serial error", f"Could not open {port}:\n{e}")
            self.reader = None
            self.connect_btn.config(text="Connect")
            self.serial_status.config(text="Mode: Disconnected")
            return

        # If we got here, the port opened successfully
        self.reader.start()
        self.connect_btn.config(text="Disconnect")
        self.serial_status.config(text=f"Mode: Serial {port}")


    # ------------- Update UI -------------

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                # prefer high-resolution timestamp from reader if present
                if getattr(msg, 'receive_ts', None) is not None:
                    self.last_received_ts = msg.receive_ts
                else:
                    self.last_received_ts = time.perf_counter()
                self._apply_message(msg)
        except queue.Empty:
            pass
        self.after(20, self._poll_queue)

    def _update_since_label(self):
        """
        Update ms since last message. Color green if <= 20 ms else red.
        Only update UI when text or color changes to reduce redraws.
        """
        now = time.perf_counter()
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
        # top
        period_text = PERIOD_MAP.get(rm.period_code, "Unknown")
        self.period_var.set(period_text)

        # clock
        # Display: MM:SS.DD for modes 1-6; HH:MM:SS for modes 7/8
        if rm.period_code in ('7', '8'):
            display = rm.clock_text
        else:
            display = f"{rm.clock_text[0:5]}.{rm.clock_text[6:8]}"
        self.clock_var.set(display)
        self.run_var.set("Running" if rm.running else "Stopped")
        
        self.clock_label.config(fg="#3ED38A" if rm.running else "#FFFFFF")
        self.run_label.config(fg="#3ED38A" if rm.running else "#FFFFFF")

        # scores
        self.left.score_var.set(str(rm.home_score))
        self.right.score_var.set(str(rm.away_score))

        # penalties
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

        # log
        self.last_msg.configure(state="normal")
        self.last_msg.delete("1.0", "end")
        crc_status = "OK" if rm.crc_ok else "BAD"
        self.last_msg.insert("end", f"{rm.raw}\nCRC={rm.crc_hex} ({crc_status})\n")
        self.last_msg.configure(state="disabled")


# --------------- Readers -----------------

class BaseReader(threading.Thread):
    def __init__(self, out_queue: "queue.Queue[RevartMessage]"):
        super().__init__(daemon=True)
        self._out = out_queue
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

class SerialReader(BaseReader):
    def __init__(self, port: str, out_queue: "queue.Queue[RevartMessage]"):
        super().__init__(out_queue)
        self._port = port

        # Non-blocking: timeout=0
        self._ser = serial.Serial(self._port, baudrate=38400, timeout=0)

    def run(self):
        buf = bytearray()
        try:
            while not self._stop.is_set():
                n = self._ser.in_waiting
                if n:
                    buf += self._ser.read(n)

                    # split by LF
                    while True:
                        idx = buf.find(b"\n")
                        if idx < 0:
                            break
                        frame = bytes(buf[:idx+1])
                        del buf[:idx+1]

                        # Timestamp as close as possible to frame boundary
                        ts = time.perf_counter()

                        try:
                            s = frame.decode("ascii", errors="strict")
                            if len(s) == MESSAGE_LEN:
                                rm = decode_message(s)
                                rm.receive_ts = ts
                                self._out.put(rm)
                        except Exception:
                            pass
                else:
                    # Yield CPU but keep latency low
                    time.sleep(0.001)
        finally:
            try:
                self._ser.close()
            except Exception:
                pass


def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
