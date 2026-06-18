"""
Tiny GUI facade over run.ps1 + the batch eval script. Stdlib only (tkinter).

Buttons just shell out to the existing commands and stream their output into the
log pane -- no command logic lives here, so anything run.ps1 can do, this can do.

Run:  python gui.py
"""
import subprocess, sys, threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
RUN_PS1 = ROOT / "run.ps1"
EVAL_PY = ROOT / "scripts" / "eval_orbit_frames.py"


def stream(cmd, log):
    """Run cmd, pipe stdout+stderr line by line into the log widget."""
    log.insert("end", f"\n$ {' '.join(str(c) for c in cmd)}\n"); log.see("end")
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.insert("end", line); log.see("end")
        proc.wait()
        log.insert("end", f"[exit {proc.returncode}]\n"); log.see("end")
    except Exception as exc:  # noqa: BLE001 - surface any launch failure in the log
        log.insert("end", f"[error] {exc}\n"); log.see("end")


def run_async(cmd, log):
    threading.Thread(target=stream, args=(cmd, log), daemon=True).start()


def ps(*args):
    """Build a `powershell -File run.ps1 ...` invocation."""
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(RUN_PS1), *map(str, args)]


def main():
    win = tk.Tk()
    win.title("Star Tracker")
    win.geometry("900x600")

    top = ttk.Frame(win, padding=8); top.pack(fill="x")
    ttk.Label(top, text="FOV (deg):").pack(side="left")
    fov = ttk.Entry(top, width=8); fov.insert(0, "17.75"); fov.pack(side="left", padx=(2, 12))

    log = tk.Text(win, bg="#111", fg="#ddd", insertbackground="#ddd", wrap="none")
    log.pack(fill="both", expand=True, padx=8, pady=8)

    def build():   run_async(ps("build"), log)
    def test():    run_async(ps("test"), log)
    def identify():
        f = filedialog.askopenfilename(title="Pick image",
                                       filetypes=[("Images", "*.png *.ppm"), ("All", "*.*")])
        if f: run_async(ps("identify", f, fov.get()), log)
    def batch():
        d = filedialog.askdirectory(title="Pick frame folder (must contain output.txt)")
        if d: run_async([sys.executable, str(EVAL_PY), d, fov.get()], log)

    for label, fn in (("Build", build), ("Synthetic test", test),
                      ("Identify image…", identify), ("Batch eval folder…", batch)):
        ttk.Button(top, text=label, command=fn).pack(side="left", padx=3)
    ttk.Button(top, text="Clear", command=lambda: log.delete("1.0", "end")).pack(side="right")

    win.mainloop()


if __name__ == "__main__":
    main()
