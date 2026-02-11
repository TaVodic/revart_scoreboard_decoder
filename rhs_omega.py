import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


STX = 0x02
ETX = 0x03


def xor_crc(frame_without_crc: bytes) -> int:
    crc = 0
    for b in frame_without_crc:
        crc ^= b
    return crc


def parse_digit_char(b: int) -> int | None:
    if 48 <= b <= 57:
        return b - 48
    return None


def parse_numeric_ascii(raw: bytes) -> int | None:
    s = raw.decode("ascii", errors="ignore").strip()
    if not s or not s.isdigit():
        return None
    return int(s)


def parse_mmss(raw: bytes) -> str:
    s = raw.decode("ascii", errors="replace")
    if len(s) == 4 and s[:2].isdigit() and s[2:].isdigit():
        return f"{s[:2]}:{s[2:]}"
    return s


@dataclass
class ParsedFrame:
    msg_type: str
    parsed: dict[str, Any]
    raw: bytes
    crc_ok: bool


class ProtocolParser:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[ParsedFrame]:
        self._buffer.extend(data)
        frames: list[ParsedFrame] = []

        while True:
            stx_pos = self._buffer.find(bytes([STX]))
            if stx_pos == -1:
                self._buffer.clear()
                break
            if stx_pos > 0:
                del self._buffer[:stx_pos]

            etx_pos = self._buffer.find(bytes([ETX]), 1)
            if etx_pos == -1:
                break
            if etx_pos + 1 >= len(self._buffer):
                break

            raw = bytes(self._buffer[: etx_pos + 2])
            del self._buffer[: etx_pos + 2]

            body = raw[:-1]
            crc_byte = raw[-1]
            crc_ok = xor_crc(body) == crc_byte

            msg_type = self._message_type(body)
            parsed = self._parse_message(body) if crc_ok else {"error": "CRC mismatch"}
            frames.append(ParsedFrame(msg_type=msg_type, parsed=parsed, raw=raw, crc_ok=crc_ok))

        return frames

    def _message_type(self, body: bytes) -> str:
        if len(body) < 2:
            return "?"
        first = body[1]
        if first == ord("F") and len(body) >= 3:
            return body[1:3].decode("ascii", errors="replace")
        return chr(first) if 32 <= first <= 126 else f"0x{first:02X}"

    def _parse_message(self, body: bytes) -> dict[str, Any]:
        msg_type = self._message_type(body)
        info = body[2:] if msg_type.startswith("F") else body[2:]
        if msg_type.startswith("F"):
            info = body[3:-1]
        else:
            info = body[2:-1]

        if msg_type == "D":
            return self._parse_d(info)
        if msg_type in ("F1", "F2"):
            return self._parse_f_team(info)
        if msg_type in ("F3", "F4"):
            return self._parse_f_points(info)
        if msg_type == "C":
            return self._parse_c(info)
        if msg_type == "O":
            return self._parse_o(info)
        if msg_type == "T":
            return self._parse_t(info)
        if msg_type == "N":
            return self._parse_n(info)
        return {"raw_info_ascii": info.decode("latin1", errors="replace")}

    def _parse_d(self, info: bytes) -> dict[str, Any]:
        if len(info) < 23:
            return {"error": f"unexpected D info length {len(info)}"}
        return {
            "clock": info[0:5].decode("ascii", errors="replace"),
            "home_score": parse_numeric_ascii(info[5:8]),
            "away_score": parse_numeric_ascii(info[8:11]),
            "home_faults": parse_numeric_ascii(info[11:12]),
            "away_faults": parse_numeric_ascii(info[12:13]),
            "home_timeouts": parse_numeric_ascii(info[13:14]),
            "away_timeouts": parse_numeric_ascii(info[14:15]),
            "period": parse_numeric_ascii(info[15:16]),
            "service": parse_numeric_ascii(info[16:17]),
            "start_stop": parse_numeric_ascii(info[17:18]),
            "horn": parse_numeric_ascii(info[18:19]),
            "timeout_active": info[19:21].decode("ascii", errors="replace").strip(),
            "possession_active": info[21:23].decode("ascii", errors="replace").strip(),
        }

    def _parse_f_team(self, info: bytes) -> dict[str, Any]:
        players = []
        expected = 16 * 3
        if len(info) < expected:
            return {"error": f"unexpected F1/F2 info length {len(info)}"}

        for i in range(16):
            dm = info[i * 3]
            um = info[i * 3 + 1]
            p = info[i * 3 + 2]

            on_field = (dm & 0x40) != 0
            tens = parse_digit_char(dm & 0x3F)
            units = parse_digit_char(um)
            faults = parse_digit_char(p)
            if tens is None and units is None:
                shirt = None
            elif tens is None and units is not None:
                shirt = units
            elif tens is not None and units is not None:
                shirt = tens * 10 + units
            else:
                shirt = None

            players.append(
                {
                    "idx": i + 1,
                    "shirt": shirt,
                    "on_field": on_field,
                    "faults": faults,
                }
            )
        return {"players": players}

    def _parse_f_points(self, info: bytes) -> dict[str, Any]:
        expected = 16 * 2
        if len(info) < expected:
            return {"error": f"unexpected F3/F4 info length {len(info)}"}
        players = []
        for i in range(16):
            tens = parse_digit_char(info[i * 2])
            units = parse_digit_char(info[i * 2 + 1])
            if tens is None and units is None:
                points = None
            elif tens is None and units is not None:
                points = units
            elif tens is not None and units is not None:
                points = tens * 10 + units
            else:
                points = None
            players.append({"idx": i + 1, "points": points})
        return {"players": players}

    def _parse_c(self, info: bytes) -> dict[str, Any]:
        if len(info) < 26:
            return {"error": f"unexpected C info length {len(info)}"}
        timers = []
        for i in range(6):
            start = i * 4
            timers.append(parse_mmss(info[start : start + 4]))
        return {
            "timers": timers,
            "penalty_local": parse_digit_char(info[24]),
            "penalty_away": parse_digit_char(info[25]),
        }

    def _parse_o(self, info: bytes) -> dict[str, Any]:
        if len(info) < 44:
            return {"error": f"unexpected O info length {len(info)}"}
        timers = []
        for i in range(6):
            start = i * 7
            player = info[start : start + 2].decode("ascii", errors="replace").strip()
            mm = info[start + 2 : start + 4]
            sep = chr(info[start + 4])
            ss = info[start + 5 : start + 7]
            timers.append(
                {
                    "player": player,
                    "time": f"{mm.decode('ascii', errors='replace')}{sep}{ss.decode('ascii', errors='replace')}",
                }
            )
        return {
            "timers": timers,
            "penalty_local": parse_digit_char(info[42]),
            "penalty_away": parse_digit_char(info[43]),
        }

    def _parse_t(self, info: bytes) -> dict[str, Any]:
        if len(info) < 24:
            return {"error": f"unexpected T info length {len(info)}"}
        date = info[0:8].decode("ascii", errors="replace")
        day_time = info[8:16].decode("ascii", errors="replace")
        return {"date": date, "time": day_time}

    def _parse_n(self, info: bytes) -> dict[str, Any]:
        def decode_12(b: bytes) -> str:
            return b.decode("latin1", errors="replace").rstrip()

        if len(info) < 24:
            return {"error": f"unexpected N info length {len(info)}"}
        home_team = decode_12(info[0:12])
        away_team = decode_12(info[12:24])
        remainder = info[24:]
        chunks = [decode_12(remainder[i : i + 12]) for i in range(0, len(remainder), 12) if remainder[i : i + 12]]
        home_players = chunks[:16]
        away_players = chunks[16:32]
        return {"home_team": home_team, "away_team": away_team, "home_players": home_players, "away_players": away_players}


class UartReader(threading.Thread):
    def __init__(self, port: str, baud: int, out_queue: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_queue = out_queue
        self.stop_event = threading.Event()

    def run(self) -> None:
        parser = ProtocolParser()
        try:
            with serial.Serial(self.port, self.baud, timeout=0.2) as ser:
                self.out_queue.put(("status", f"Connected to {self.port} @ {self.baud}"))
                while not self.stop_event.is_set():
                    data = ser.read(512)
                    if not data:
                        continue
                    for frame in parser.feed(data):
                        self.out_queue.put(("frame", frame))
        except Exception as exc:
            self.out_queue.put(("status", f"Connection error: {exc}"))
        finally:
            self.out_queue.put(("status", "Disconnected"))

    def stop(self) -> None:
        self.stop_event.set()


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Omega UART Monitor")
        self.root.geometry("1150x740")

        self.queue: queue.Queue = queue.Queue()
        self.reader: UartReader | None = None

        self.home_roster: dict[int, dict[str, Any]] = {i: {"shirt": None, "faults": None, "on_field": False, "points": None} for i in range(1, 17)}
        self.away_roster: dict[int, dict[str, Any]] = {i: {"shirt": None, "faults": None, "on_field": False, "points": None} for i in range(1, 17)}

        self._build_ui()
        self.refresh_ports()
        self.root.after(50, self.process_queue)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="COM Port").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(top, textvariable=self.port_var, width=15, state="readonly")
        self.port_box.pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(top, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT)

        ttk.Label(top, text="Baud").pack(side=tk.LEFT, padx=(16, 4))
        self.baud_var = tk.StringVar(value="9600")
        self.baud_entry = ttk.Entry(top, textvariable=self.baud_var, width=8)
        self.baud_entry.pack(side=tk.LEFT)

        self.connect_btn = ttk.Button(top, text="Connect", command=self.connect)
        self.connect_btn.pack(side=tk.LEFT, padx=(16, 6))
        self.disconnect_btn = ttk.Button(top, text="Disconnect", command=self.disconnect, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        content = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        upper = ttk.Panedwindow(content, orient=tk.HORIZONTAL)
        content.add(upper, weight=4)

        lower = ttk.Frame(content)
        content.add(lower, weight=2)

        self._build_score_panel(upper)
        self._build_players_panel(upper)
        self._build_log_panel(lower)

    def _build_score_panel(self, parent: ttk.Panedwindow) -> None:
        frame = ttk.LabelFrame(parent, text="Main data", padding=8)
        parent.add(frame, weight=1)

        self.vars: dict[str, tk.StringVar] = {}
        keys = [
            ("clock", "Clock (D)"),
            ("home_score", "Home score (D)"),
            ("away_score", "Away score (D)"),
            ("home_faults", "Home faults (D)"),
            ("away_faults", "Away faults (D)"),
            ("home_timeouts", "Home timeouts (D)"),
            ("away_timeouts", "Away timeouts (D)"),
            ("period", "Period (D)"),
            ("service", "Service (D)"),
            ("start_stop", "Start/Stop (D)"),
            ("horn", "Horn (D)"),
            ("timeout_active", "Timeout active (D)"),
            ("possession_active", "Possession active (D)"),
            ("date", "Date (T)"),
            ("time", "Time (T)"),
            ("home_team", "Home team (N)"),
            ("away_team", "Away team (N)"),
        ]
        highlighted_keys = {"clock", "home_score", "away_score", "period", "start_stop", "time"}

        for row, (k, label) in enumerate(keys):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
            v = tk.StringVar(value="-")
            self.vars[k] = v
            if k in highlighted_keys:
                tk.Label(frame, textvariable=v, bg="#FFFF00").grid(row=row, column=1, sticky="w", pady=1)
            else:
                ttk.Label(frame, textvariable=v).grid(row=row, column=1, sticky="w", pady=1)

        self.penalties_var = tk.StringVar(value="-")
        ttk.Label(frame, text="Penalty timers (C)").grid(row=len(keys), column=0, sticky="w", padx=(0, 10), pady=(8, 1))
        tk.Label(frame, textvariable=self.penalties_var, bg="#FFFF00", wraplength=350, justify=tk.LEFT).grid(
            row=len(keys), column=1, sticky="w", pady=(8, 1)
        )

    def _build_players_panel(self, parent: ttk.Panedwindow) -> None:
        wrapper = ttk.LabelFrame(parent, text="Players (F)", padding=8)
        parent.add(wrapper, weight=2)

        cols = ("idx", "shirt", "on_field", "faults", "points")
        self.home_tree = ttk.Treeview(wrapper, columns=cols, show="headings", height=16)
        self.away_tree = ttk.Treeview(wrapper, columns=cols, show="headings", height=16)

        for tree, team in ((self.home_tree, "Home"), (self.away_tree, "Away")):
            for c, h in zip(cols, ("Player#", "Shirt (F1/F2)", "On field (F1/F2)", "Faults (F1/F2)", "Points (F3/F4)")):
                tree.heading(c, text=h)
                tree.column(c, width=70, anchor=tk.CENTER)
            tree.tag_configure("top3", background="#FFFF00")
            tree.column("on_field", width=80)
            tree.insert("", tk.END, values=(f"{team} team", "", "", "", ""))
            for i in range(1, 17):
                tags = ("top3",) if i <= 3 else ()
                tree.insert("", tk.END, iid=f"{team[0]}{i}", values=(i, "", "", "", ""), tags=tags)

        self.home_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.away_tree.grid(row=0, column=1, sticky="nsew")
        wrapper.columnconfigure(0, weight=1)
        wrapper.columnconfigure(1, weight=1)
        wrapper.rowconfigure(0, weight=1)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Raw Frames / Events", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        self.log = ScrolledText(frame, height=12, font=("Consolas", 10))
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state=tk.DISABLED)

    def refresh_ports(self) -> None:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def connect(self) -> None:
        if self.reader is not None:
            return
        port = self.port_var.get().strip()
        if not port:
            self.set_status("Select a COM port")
            return
        try:
            baud = int(self.baud_var.get().strip())
        except ValueError:
            self.set_status("Invalid baud value")
            return

        self.reader = UartReader(port=port, baud=baud, out_queue=self.queue)
        self.reader.start()
        self.connect_btn.configure(state=tk.DISABLED)
        self.disconnect_btn.configure(state=tk.NORMAL)
        self.set_status("Connecting...")

    def disconnect(self) -> None:
        if self.reader is None:
            return
        self.reader.stop()
        self.reader = None
        self.connect_btn.configure(state=tk.NORMAL)
        self.disconnect_btn.configure(state=tk.DISABLED)

    def process_queue(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self.set_status(payload)
            elif kind == "frame":
                self.handle_frame(payload)

        self.root.after(50, self.process_queue)

    def handle_frame(self, frame: ParsedFrame) -> None:
        timestamp = time.strftime("%H:%M:%S")
        raw_hex = " ".join(f"{b:02X}" for b in frame.raw)
        status = "OK" if frame.crc_ok else "BAD_CRC"
        self.append_log(f"[{timestamp}] {frame.msg_type:<2} {status}  {raw_hex}")

        if not frame.crc_ok:
            return

        msg_type = frame.msg_type
        parsed = frame.parsed

        if msg_type == "D":
            for k, v in parsed.items():
                if k in self.vars:
                    self.vars[k].set("-" if v is None else str(v))
        elif msg_type == "T":
            self.vars["date"].set(parsed.get("date", "-"))
            self.vars["time"].set(parsed.get("time", "-"))
        elif msg_type == "N":
            self.vars["home_team"].set(parsed.get("home_team", "-"))
            self.vars["away_team"].set(parsed.get("away_team", "-"))
        elif msg_type == "F1":
            self.apply_team_message(self.home_roster, parsed)
            self.refresh_team_tree(self.home_tree, self.home_roster, "H")
        elif msg_type == "F2":
            self.apply_team_message(self.away_roster, parsed)
            self.refresh_team_tree(self.away_tree, self.away_roster, "A")
        elif msg_type == "F3":
            self.apply_points_message(self.home_roster, parsed)
            self.refresh_team_tree(self.home_tree, self.home_roster, "H")
        elif msg_type == "F4":
            self.apply_points_message(self.away_roster, parsed)
            self.refresh_team_tree(self.away_tree, self.away_roster, "A")
        elif msg_type == "C":
            timers = parsed.get("timers", [])
            p_local = parsed.get("penalty_local")
            p_away = parsed.get("penalty_away")
            self.penalties_var.set(f"Timers: {timers} | 10-min local={p_local} away={p_away}")
        elif msg_type == "O":
            timers = parsed.get("timers", [])
            p_local = parsed.get("penalty_local")
            p_away = parsed.get("penalty_away")
            self.penalties_var.set(f"Olympic: {timers} | 10-min local={p_local} away={p_away}")

    def apply_team_message(self, roster: dict[int, dict[str, Any]], parsed: dict[str, Any]) -> None:
        players = parsed.get("players", [])
        for p in players:
            idx = p["idx"]
            roster[idx]["shirt"] = p["shirt"]
            roster[idx]["faults"] = p["faults"]
            roster[idx]["on_field"] = p["on_field"]

    def apply_points_message(self, roster: dict[int, dict[str, Any]], parsed: dict[str, Any]) -> None:
        players = parsed.get("players", [])
        for p in players:
            idx = p["idx"]
            roster[idx]["points"] = p["points"]

    def refresh_team_tree(self, tree: ttk.Treeview, roster: dict[int, dict[str, Any]], prefix: str) -> None:
        for i in range(1, 17):
            row = roster[i]
            tree.item(
                f"{prefix}{i}",
                values=(
                    i,
                    "" if row["shirt"] is None else row["shirt"],
                    "Y" if row["on_field"] else "",
                    "" if row["faults"] is None else row["faults"],
                    "" if row["points"] is None else row["points"],
                ),
            )

    def append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 600:
            self.log.delete("1.0", "100.0")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.append_log(text)


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.disconnect(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
