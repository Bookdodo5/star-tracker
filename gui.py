"""
Star Tracker GUI. Stdlib only (tkinter).

Two functions, the whole product:
  Build  -- compile the in-process identification library (live/libstar_live.dll),
            regenerating the baked star database first if it is missing.
  Live   -- capture frames (webcam / video file / screen) and stream attitude
            (RA / DEC / roll), with optional FOV self-calibration.

Everything shells out to the same commands you'd type by hand.

Run:  python gui.py
"""
import subprocess, sys, threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
LIVE_BUILD = ROOT / "live" / "build-mingw"
DLL = LIVE_BUILD / "libstar_live.dll"
GENERATED = ROOT / "identifier" / "generated" / "catalog_db_generated.c"

_current = {"proc": None}  # last-launched process, for Stop


def build_commands():
    """Commands to (regenerate the DB if missing and) build the live DLL."""
    cmds = []
    if not GENERATED.exists():  # baked DB absent on a clean checkout -> regenerate
        cmds.append([sys.executable, str(ROOT / "identifier" / "tools" / "export_catalog_db.py")])
        cmds.append([sys.executable, str(ROOT / "identifier" / "tools" / "export_tetra_db.py")])
    cmds.append(["cmake", "-S", str(ROOT / "live"), "-B", str(LIVE_BUILD),
                 "-G", "MinGW Makefiles", "-DCMAKE_BUILD_TYPE=Release"])
    cmds.append(["cmake", "--build", str(LIVE_BUILD)])
    return cmds


def identify_command(mode, path, fov, fov_search, morph, glob_pat):
    """`python identify.py <mode> <path> ...` from the identify-section values."""
    sub = {"image": "image", "folder": "batch", "video": "video"}[mode]
    cmd = [sys.executable, str(ROOT / "identify.py"), sub, path, "--fov", fov, "--morph", morph]
    if fov_search:
        cmd.append("--fov-search")
    if mode == "folder" and glob_pat.strip():
        cmd += ["--glob", glob_pat.strip()]
    return cmd


def live_command(v):
    """`python live_identify.py ...` from the live-tab field values."""
    cmd = [sys.executable, str(ROOT / "live_identify.py"),
           "--source", v["source"], "--fov", v["fov"], "--morph", v["morph"], "--scale", v["scale"]]
    for flag, key in (("--fov-search", "fov_search"), ("--show", "show"), ("--list-monitors", "list_monitors")):
        if v[key]:
            cmd.append(flag)
    for flag, key in (("--save", "save"), ("--monitor", "monitor"), ("--region", "region"),
                      ("--cam-width", "cam_width"), ("--cam-height", "cam_height")):
        if str(v[key]).strip():
            cmd += [flag, str(v[key]).strip()]
    return cmd


def stream(cmds, log):
    """Run one or more commands in sequence, piping output into the log; stop on failure."""
    for cmd in cmds:
        log.insert("end", f"\n$ {' '.join(cmd)}\n"); log.see("end")
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            _current["proc"] = proc
            for line in proc.stdout:
                log.insert("end", line); log.see("end")
            proc.wait()
            log.insert("end", f"[exit {proc.returncode}]\n"); log.see("end")
            if proc.returncode != 0:
                break
        except Exception as exc:  # noqa: BLE001 - surface any launch failure in the log
            log.insert("end", f"[error] {exc}\n"); log.see("end")
            break


def run_async(cmds, log):
    threading.Thread(target=stream, args=(cmds, log), daemon=True).start()


def stop():
    proc = _current.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()


# Live-tab fields: (key, label, kind, default). kind: text | int | float | check
LIVE_FIELDS = [
    ("source", "Source (0=webcam, video path, or 'screen')", "text", "0"),
    ("fov", "FOV seed (deg)", "float", "17.75"),
    ("fov_search", "Self-calibrate FOV (recover + lock)", "check", False),
    ("morph", "Morph passes (0 = keep faint stars)", "int", "1"),
    ("scale", "Downscale factor", "float", "1.0"),
    ("show", "Show video window", "check", False),
    ("save", "Save annotated video to (path)", "text", ""),
    ("monitor", "Screen source: monitor #", "text", ""),
    ("region", "Screen source: region x,y,w,h", "text", ""),
    ("cam_width", "Webcam width", "text", ""),
    ("cam_height", "Webcam height", "text", ""),
    ("list_monitors", "List monitors and exit", "check", False),
]


def main():
    win = tk.Tk()
    win.title("Star Tracker")
    win.geometry("960x720")
    win.minsize(820, 560)

    log = tk.Text(win, bg="#111", fg="#ddd", insertbackground="#ddd", wrap="none", height=12)

    # --- bottom first, so the log always keeps real space regardless of the forms above ---
    bar = ttk.Frame(win, padding=(10, 6)); bar.pack(side="bottom", fill="x")
    ttk.Button(bar, text="Stop", command=stop).pack(side="left")
    ttk.Button(bar, text="Clear log", command=lambda: log.delete("1.0", "end")).pack(side="left", padx=4)
    log.pack(side="bottom", fill="both", expand=True, padx=10, pady=(4, 6))

    # Build
    top = ttk.Frame(win, padding=(10, 8)); top.pack(side="top", fill="x")
    ttk.Button(top, text="Build", command=lambda: run_async(build_commands(), log)).pack(side="left")
    ttk.Label(top, text="compile the identification library (run once)").pack(side="left", padx=8)

    # Live and Identify side by side so neither pushes the log off-screen
    cols = ttk.Frame(win); cols.pack(side="top", fill="x", padx=10, pady=(4, 0))

    # Live identify
    form = ttk.LabelFrame(cols, text="Live identify", padding=10); form.pack(side="left", fill="both", expand=True)
    vars_ = {}
    for row, (key, label, kind, default) in enumerate(LIVE_FIELDS):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        if kind == "check":
            var = tk.BooleanVar(value=bool(default))
            ttk.Checkbutton(form, variable=var).grid(row=row, column=1, sticky="w")
        else:
            var = tk.StringVar(value=str(default))
            ttk.Entry(form, textvariable=var, width=36).grid(row=row, column=1, sticky="we")
            if key == "source":
                def pick(v=var):
                    p = filedialog.askopenfilename(title="Pick a video file")
                    if p:
                        v.set(p)
                ttk.Button(form, text="…", width=3, command=pick).grid(row=row, column=2, padx=(4, 0))
        vars_[key] = var
    form.columnconfigure(1, weight=1)

    def run_live():
        if not DLL.exists():
            log.insert("end", "\n[gui] library not built yet -- click Build first.\n"); log.see("end")
            return
        values = {k: (var.get()) for k, var in vars_.items()}
        run_async([live_command(values)], log)
    ttk.Button(form, text="Run live", command=run_live).grid(row=len(LIVE_FIELDS), column=0,
                                                             columnspan=3, sticky="w", pady=(8, 0))

    # Identify (still image / folder / video)
    idf = ttk.LabelFrame(cols, text="Identify", padding=10)
    idf.pack(side="left", fill="both", expand=True, padx=(10, 0))
    mode = tk.StringVar(value="image")
    ipath = tk.StringVar(value="")
    ifov = tk.StringVar(value="10")
    isearch = tk.BooleanVar(value=False)
    imorph = tk.StringVar(value="0")
    iglob = tk.StringVar(value="*.ppm")
    ttk.Label(idf, text="Mode").grid(row=0, column=0, sticky="w", padx=(0, 8))
    ttk.Combobox(idf, textvariable=mode, values=["image", "folder", "video"], width=8,
                 state="readonly").grid(row=0, column=1, sticky="w")
    ttk.Label(idf, text="Path").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
    ttk.Entry(idf, textvariable=ipath, width=40).grid(row=1, column=1, sticky="we")

    def ipick():
        p = filedialog.askdirectory() if mode.get() == "folder" else filedialog.askopenfilename()
        if p:
            ipath.set(p)
    ttk.Button(idf, text="…", width=3, command=ipick).grid(row=1, column=2, padx=(4, 0))
    ttk.Label(idf, text="FOV seed (deg)").grid(row=2, column=0, sticky="w", padx=(0, 8))
    ttk.Entry(idf, textvariable=ifov, width=10).grid(row=2, column=1, sticky="w")
    ttk.Checkbutton(idf, text="Self-calibrate FOV", variable=isearch).grid(row=3, column=1, sticky="w")
    ttk.Label(idf, text="Morph passes").grid(row=4, column=0, sticky="w", padx=(0, 8))
    ttk.Entry(idf, textvariable=imorph, width=10).grid(row=4, column=1, sticky="w")
    ttk.Label(idf, text="Folder glob").grid(row=5, column=0, sticky="w", padx=(0, 8))
    ttk.Entry(idf, textvariable=iglob, width=14).grid(row=5, column=1, sticky="w")
    idf.columnconfigure(1, weight=1)

    def run_identify():
        if not DLL.exists():
            log.insert("end", "\n[gui] library not built yet -- click Build first.\n"); log.see("end")
            return
        if not ipath.get().strip():
            log.insert("end", "\n[gui] pick a path to identify.\n"); log.see("end")
            return
        run_async([identify_command(mode.get(), ipath.get().strip(), ifov.get(),
                                    isearch.get(), imorph.get(), iglob.get())], log)
    ttk.Button(idf, text="Run identify", command=run_identify).grid(row=6, column=0, columnspan=3,
                                                                    sticky="w", pady=(8, 0))

    win.mainloop()


if __name__ == "__main__":
    main()
