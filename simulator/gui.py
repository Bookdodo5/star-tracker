"""
PC-side control GUI for the simulator (tkinter, stdlib).

A pure remote HTTP client — it launches nothing locally; it talks to the Pi-side simulator's
control API (see simulator/stream_server.py). Separate from the repo-root gui.py (which builds
the DLL / runs the tracker). Run on a machine with a display:

    python -m simulator.gui                    # connect to 127.0.0.1:8090
    python -m simulator.gui --host 10.0.0.5:8090

Panels: live commands, render config sliders, tracker Start/Stop + delay calibration, and a
readout that polls GET /status. Quick network calls run inline; slow ones (status poll,
calibrate) run on worker threads so the UI never blocks.
"""
from __future__ import annotations

import argparse
import threading
import tkinter as tk
import webbrowser
from tkinter import ttk

from . import control

CONFIG_SLIDERS = [  # (key, from, to, resolution)
    ("gain", 0.1, 5.0, 0.1), ("gamma", 1.0, 8.0, 0.1), ("saturation_cap", 50, 255, 5),
    ("noise_sigma", 0, 50, 1), ("blur_sigma", 0, 5, 0.1),
    ("streak_len", 0, 30, 1), ("streak_angle", 0, 180, 5),
]


class SimulatorGUI:
    """Builds the control window against a root; all actions hit the simulator HTTP API."""

    def __init__(self, root: tk.Misc, host: str = control.DEFAULT_HOST):
        self.root = root
        self.host_var = tk.StringVar(value=host)
        self.status_var = tk.StringVar(value="ready")
        self._build()
        self._poll_status()

    def host(self) -> str:
        return self.host_var.get().strip()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.grid(sticky="nsew")
        ttk.Label(top, text="Simulator host:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.host_var, width=22).grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="Open viewer", command=lambda: webbrowser.open(f"http://{self.host()}/")) \
            .grid(row=0, column=2, padx=4)

        # --- command panel ---
        cmd = ttk.LabelFrame(top, text="Commands", padding=6)
        cmd.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
        self.ra_var, self.dec_var, self.roll_var = tk.StringVar(value="83.8"), tk.StringVar(value="-5.4"), tk.StringVar(value="0")
        for i, (lbl, var) in enumerate([("RA", self.ra_var), ("DEC", self.dec_var), ("Roll", self.roll_var)]):
            ttk.Label(cmd, text=lbl).grid(row=0, column=2 * i, sticky="e")
            ttk.Entry(cmd, textvariable=var, width=7).grid(row=0, column=2 * i + 1)
        ttk.Button(cmd, text="Point at", command=self._point).grid(row=0, column=6, padx=4)

        self.slew_axis = tk.StringVar(value="ra")
        self.slew_delta, self.slew_rate = tk.StringVar(value="10"), tk.StringVar(value="5")
        ttk.OptionMenu(cmd, self.slew_axis, "ra", "ra", "dec", "roll").grid(row=1, column=0)
        ttk.Entry(cmd, textvariable=self.slew_delta, width=6).grid(row=1, column=1)
        ttk.Label(cmd, text="deg @").grid(row=1, column=2)
        ttk.Entry(cmd, textvariable=self.slew_rate, width=6).grid(row=1, column=3)
        ttk.Label(cmd, text="deg/s").grid(row=1, column=4)
        ttk.Button(cmd, text="Slew", command=self._slew).grid(row=1, column=6, padx=4)

        self.roll_rate = tk.StringVar(value="3")
        ttk.Label(cmd, text="Roll").grid(row=2, column=0, sticky="e")
        ttk.Entry(cmd, textvariable=self.roll_rate, width=6).grid(row=2, column=1)
        ttk.Label(cmd, text="deg/s").grid(row=2, column=2)
        ttk.Button(cmd, text="Roll forever", command=self._roll).grid(row=2, column=6, padx=4)

        self.free_var = tk.StringVar()
        ttk.Entry(cmd, textvariable=self.free_var, width=28).grid(row=3, column=0, columnspan=5, pady=2, sticky="w")
        ttk.Button(cmd, text="Send", command=self._free).grid(row=3, column=6, padx=4)

        # --- config sliders ---
        cfg = ttk.LabelFrame(top, text="Render config (release to apply)", padding=6)
        cfg.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        self.cfg_vars = {}
        for r, (key, lo, hi, res) in enumerate(CONFIG_SLIDERS):
            ttk.Label(cfg, text=key).grid(row=r, column=0, sticky="w")
            var = tk.DoubleVar()
            self.cfg_vars[key] = var
            s = tk.Scale(cfg, variable=var, from_=lo, to=hi, resolution=res,
                         orient="horizontal", length=220, showvalue=True)
            s.grid(row=r, column=1, sticky="ew")
            s.bind("<ButtonRelease-1>", lambda e: self._push_config())

        # --- tracker panel ---
        trk = ttk.LabelFrame(top, text="Tracker", padding=6)
        trk.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(trk, text="Start", command=lambda: self._async(control.tracker, self.host(), "start")).grid(row=0, column=0)
        ttk.Button(trk, text="Stop", command=lambda: self._async(control.tracker, self.host(), "stop")).grid(row=0, column=1)
        ttk.Button(trk, text="Calibrate delay", command=lambda: self._async(control.calibrate, self.host())).grid(row=0, column=2, padx=4)
        ttk.Button(trk, text="Flash check", command=lambda: self._async(control.flash_check, self.host())).grid(row=0, column=3, padx=4)

        # --- readout ---
        out = ttk.LabelFrame(top, text="Live", padding=6)
        out.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        self.readout = tk.StringVar(value="—")
        ttk.Label(out, textvariable=self.readout, font=("Consolas", 10), justify="left").grid(sticky="w")

        # --- tracker log ---
        log = ttk.LabelFrame(top, text="Tracker output", padding=6)
        log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=4)
        top.rowconfigure(5, weight=1)
        self.tracker_log = tk.Text(log, height=10, width=80, font=("Consolas", 9),
                                   bg="#111", fg="#ddd", state="disabled", wrap="none")
        sb = ttk.Scrollbar(log, command=self.tracker_log.yview)
        self.tracker_log.configure(yscrollcommand=sb.set)
        self.tracker_log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        log.rowconfigure(0, weight=1); log.columnconfigure(0, weight=1)

        ttk.Label(top, textvariable=self.status_var, foreground="#555").grid(row=6, column=0, columnspan=3, sticky="w")

    # --- command actions (quick POSTs, run inline) ---
    def _send(self, line: str) -> None:
        try:
            control.send_command(self.host(), line)
            self.status_var.set(f"sent: {line}")
        except Exception as exc:  # network/HTTP errors shouldn't kill the UI
            self.status_var.set(f"error: {exc}")

    def _point(self):
        self._send(f"point_at {self.ra_var.get()} {self.dec_var.get()} {self.roll_var.get()}")

    def _slew(self):
        self._send(f"slew {self.slew_axis.get()} {self.slew_delta.get()} {self.slew_rate.get()}")

    def _roll(self):
        self._send(f"roll {self.roll_rate.get()} forever")

    def _free(self):
        line = self.free_var.get().strip()
        if line:
            self._send(line)

    def _push_config(self):
        config = {k: v.get() for k, v in self.cfg_vars.items()}
        self._async(control.set_config, self.host(), config)

    # --- threaded actions (slow: don't block the UI) ---
    def _async(self, fn, *args) -> None:
        def run():
            try:
                result = fn(*args)
                self.root.after(0, lambda r=result: self.status_var.set(str(r)))
            except Exception as exc:  # bind exc as a default — Python clears it after the except block
                self.root.after(0, lambda e=exc: self.status_var.set(f"error: {e}"))
        threading.Thread(target=run, daemon=True).start()

    def _poll_status(self) -> None:
        def run():
            try:
                status = control.get_status(self.host())
                self.root.after(0, lambda: self._show_status(status))
            except Exception:
                self.root.after(0, lambda: self.readout.set("(no connection)"))
        threading.Thread(target=run, daemon=True).start()
        self.root.after(500, self._poll_status)

    def _show_status(self, status: dict) -> None:
        m = status.get("metrics", {})
        self.readout.set(
            f"pointing_err : {m.get('pointing_err_deg')}°\n"
            f"roll_err     : {m.get('roll_err_deg')}°\n"
            f"delay        : {m.get('delay')} s\n"
            f"sync_ok      : {m.get('sync_ok')}\n"
            f"fps          : {m.get('fps')}\n"
            f"tracker      : {'running' if m.get('tracker_running') else 'stopped'}\n"
            f"truth        : {m.get('truth')}\n"
            f"estimate     : {m.get('est')}"
        )
        lines = status.get("tracker_lines")
        if lines is not None:
            text = "\n".join(lines)
            self.tracker_log.configure(state="normal")
            self.tracker_log.delete("1.0", "end")
            self.tracker_log.insert("end", text)
            self.tracker_log.see("end")
            self.tracker_log.configure(state="disabled")


def main() -> None:
    p = argparse.ArgumentParser(description="Simulator control GUI")
    p.add_argument("--host", default=control.DEFAULT_HOST, help="host:port (default 127.0.0.1:8090)")
    args = p.parse_args()
    root = tk.Tk()
    root.title("Star Simulator Control")
    SimulatorGUI(root, args.host)
    root.mainloop()


def _selftest() -> None:
    """Self-check: a command entry builds the right line and calls send_command (no network)."""
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as exc:
        print(f"gui.py self-check skipped (no display): {exc}")
        return
    captured = {}
    orig = control.send_command
    control.send_command = lambda host, line: captured.update(host=host, line=line) or {"ok": True}
    try:
        app = SimulatorGUI(root, host="testhost:9999")
        app.ra_var.set("83.8"); app.dec_var.set("-5.4"); app.roll_var.set("0")
        app._point()
        assert captured["line"] == "point_at 83.8 -5.4 0", captured
        assert captured["host"] == "testhost:9999", captured
        app.roll_rate.set("3"); app._roll()
        assert captured["line"] == "roll 3 forever", captured
        # _async error path: the deferred status lambda must not NameError on 'exc'.
        app.root.after = lambda ms, func=None: func() if func else None   # run callbacks inline
        orig_thread = threading.Thread
        threading.Thread = lambda target=None, daemon=None: type("T", (), {"start": staticmethod(target)})()
        try:
            app._async(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            assert "error: boom" in app.status_var.get(), app.status_var.get()
            app._async(lambda: {"ok": True})
            assert app.status_var.get() == "{'ok': True}", app.status_var.get()
        finally:
            threading.Thread = orig_thread
    finally:
        control.send_command = orig
        root.destroy()
    print("gui.py self-check passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        main()
