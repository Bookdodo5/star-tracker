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
import json, math, subprocess, sys, threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
LIVE_BUILD = ROOT / "live" / "build-mingw"
DLL = LIVE_BUILD / "libstar_live.dll"
GENERATED = ROOT / "identifier" / "generated" / "catalog_db_generated.c"
DB_META = ROOT / "identifier" / "generated" / "db_meta.json"
DB_CENTER = ROOT / "db_center.py"
PROVEN_RADIUS_DEG, PROVEN_DIAG_DEG = 3.5, math.hypot(7.0, 4.0)

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


def live_command(v):
    """`python live_identify.py ...` from the live-tab field values."""
    cmd = [sys.executable, str(ROOT / "live_identify.py"),
           "--source", v["source"], "--fov", v["fov"], "--morph", v["morph"], "--scale", v["scale"]]
    for flag, key in (("--fov-search", "fov_search"), ("--show", "show"), ("--list-monitors", "list_monitors"),
                      ("--quiet", "quiet"), ("--timing", "timing")):
        if v[key]:
            cmd.append(flag)
    for flag, key in (("--save", "save"), ("--monitor", "monitor"), ("--region", "region"),
                      ("--cam-width", "cam_width"), ("--cam-height", "cam_height")):
        if str(v[key]).strip():
            cmd += [flag, str(v[key]).strip()]
    return cmd


def stream(cmds, log, on_done=None):
    """Run one or more commands in sequence, piping output into the log; stop on failure.
    on_done() is called once when all commands finish (e.g. to refresh the DB info panel)."""
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
    if on_done is not None:
        try:
            on_done()
        except Exception:  # noqa: BLE001 - a UI refresh must never crash the worker thread
            pass


def run_async(cmds, log, on_done=None):
    threading.Thread(target=stream, args=(cmds, log, on_done), daemon=True).start()


def stop():
    proc = _current.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()


# --- Database tab -----------------------------------------------------------

# (key, label, default). radius uses "" default = auto (proven ratio for the FOV).
DB_FIELDS = [
    ("fov_w", "FOV width (deg)", "7.0"),
    ("fov_h", "FOV height (deg)", "4.0"),
    ("mag", "Member mag limit L (fainter = denser)", "7.5"),
    ("bmc", "Anchor mag limit BMC", "7.0"),
    ("radius", "Gather radius (deg, blank = auto)", ""),
    ("k", "K companions per anchor (higher = redundancy)", "9"),
    ("density_thresh", "Density threshold (size knob; higher/off = bigger)", "15"),
    ("k_low", "K for dense anchors", "6"),
    ("fields", "Eval/coverage: random fields", "2000"),
    ("tol", "Eval: correct tolerance (deg)", "0.5"),
]


def _auto_radius(fov_w, fov_h):
    """The proven radius/diagonal ratio applied to this FOV (matches db_center's default)."""
    return PROVEN_RADIUS_DEG * math.hypot(fov_w, fov_h) / PROVEN_DIAG_DEG


def db_command(v, action):
    """Builds a `python db_center.py <action> ...` command from the DB-tab field values.
    action is 'build', 'coverage', or 'eval'."""
    cmd = [sys.executable, str(DB_CENTER), action]
    if action in ("build", "coverage"):
        cmd += ["--fov-w", v["fov_w"], "--fov-h", v["fov_h"],
                "--mag", v["mag"], "--bmc", v["bmc"], "--k", v["k"],
                "--density-thresh", v["density_thresh"], "--k-low", v["k_low"]]
        if str(v["radius"]).strip():
            cmd += ["--radius", v["radius"]]
    if action in ("coverage", "eval"):
        cmd += ["--fields", v["fields"]]
    if action == "coverage":
        cmd.append("--robust")
    if action == "eval":
        cmd += ["--tol", v["tol"], "--robust"]
    return cmd


def db_info_text():
    """One multi-line summary of the current DB from db_meta.json, for the info panel."""
    if not DB_META.exists():
        return "No DB built yet. Set a FOV and click Build DB + DLL."
    try:
        m = json.loads(DB_META.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "db_meta.json unreadable."
    stale = "STALE - rebuild" if _dll_stale() else "up to date"
    lines = [
        f"FOV: {m.get('fov_w')} x {m.get('fov_h')} deg   (max edge {m.get('max_edge_deg', 0):.2f})",
        f"members V<={m.get('mag')}  anchors V<={m.get('bmc')}  radius {m.get('radius_deg', 0):.2f} deg",
        f"K={m.get('k')}  tetrads {m.get('tetrads'):,}   size {m.get('db_bytes', 0)/1e6:.1f} MB",
        f"DLL: {stale}   built {m.get('generated_utc')}",
    ]
    evals = m.get("evals", [])
    if evals:
        e = evals[-1]
        robust = f"  robust {e['robust_rate']*100:.1f}%" if "robust_rate" in e else ""
        lines.append(f"last eval: solve {e.get('solve_rate', 0)*100:.1f}%  "
                     f"acc {e.get('accuracy', 0)*100:.1f}%{robust}")
    else:
        lines.append("last eval: none (click Evaluate)")
    return "\n".join(lines)


def _dll_stale():
    if not DLL.exists():
        return True
    t = DLL.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > t for p in
               (GENERATED, ROOT / "identifier" / "generated" / "tetra_db_generated.c"))


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
    ("quiet", "Quiet (suppress NULL / no-solve lines)", "check", False),
    ("timing", "Show centroid timing", "check", False),
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

    notebook = ttk.Notebook(win); notebook.pack(side="top", fill="both", expand=False, padx=10, pady=(6, 0))
    live_tab = ttk.Frame(notebook); notebook.add(live_tab, text="Live")
    db_tab = ttk.Frame(notebook); notebook.add(db_tab, text="Database")

    # --- Live tab ---
    # Build
    top = ttk.Frame(live_tab, padding=(10, 8)); top.pack(side="top", fill="x")
    ttk.Button(top, text="Build", command=lambda: run_async(build_commands(), log)).pack(side="left")
    ttk.Label(top, text="compile the identification library (run once)").pack(side="left", padx=8)

    cols = ttk.Frame(live_tab); cols.pack(side="top", fill="x", padx=10, pady=(4, 0))

    # Live identify (a still image or video file also works as --source: OpenCV reads both)
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

    # --- Database tab ---
    dbf = ttk.Frame(db_tab, padding=10); dbf.pack(side="top", fill="x")
    info = ttk.LabelFrame(dbf, text="Current database", padding=10); info.pack(side="top", fill="x")
    info_label = ttk.Label(info, text=db_info_text(), justify="left", font=("TkFixedFont", 9))
    info_label.pack(side="left", anchor="w")

    def refresh_info():
        info_label.config(text=db_info_text())
    ttk.Button(info, text="Refresh", command=refresh_info).pack(side="right", anchor="ne")

    knobs = ttk.LabelFrame(dbf, text="Build / measure a DB for your camera FOV", padding=10)
    knobs.pack(side="top", fill="x", pady=(8, 0))
    dbvars = {}
    meta = {}
    if DB_META.exists():
        try:
            meta = json.loads(DB_META.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    meta_defaults = {"fov_w": "fov_w", "fov_h": "fov_h", "mag": "mag", "bmc": "bmc",
                     "k": "k", "density_thresh": "density_thresh", "k_low": "k_low"}
    for row, (key, label, default) in enumerate(DB_FIELDS):
        ttk.Label(knobs, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        prefill = str(meta.get(meta_defaults[key], default)) if key in meta_defaults and meta else default
        var = tk.StringVar(value=prefill)
        ttk.Entry(knobs, textvariable=var, width=16).grid(row=row, column=1, sticky="w")
        dbvars[key] = var
    knobs.columnconfigure(1, weight=1)

    def db_run(action):
        values = {k: var.get() for k, var in dbvars.items()}
        try:  # show the auto radius that will be used, if left blank
            if action in ("build", "coverage") and not values["radius"].strip():
                r = _auto_radius(float(values["fov_w"]), float(values["fov_h"]))
                log.insert("end", f"\n[gui] auto radius = {r:.2f} deg for FOV "
                                  f"{values['fov_w']}x{values['fov_h']}\n"); log.see("end")
        except ValueError:
            pass
        run_async([db_command(values, action)], log, on_done=refresh_info)

    btns = ttk.Frame(knobs); btns.grid(row=len(DB_FIELDS), column=0, columnspan=2, sticky="w", pady=(10, 0))
    ttk.Button(btns, text="Estimate coverage (fast)", command=lambda: db_run("coverage")).pack(side="left")
    ttk.Button(btns, text="Build DB + DLL", command=lambda: db_run("build")).pack(side="left", padx=6)
    ttk.Button(btns, text="Evaluate", command=lambda: db_run("eval")).pack(side="left")

    # Sweep: vary one knob across values and print a coverage table (the "what number?" helper).
    swp = ttk.Frame(knobs); swp.grid(row=len(DB_FIELDS) + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
    ttk.Label(swp, text="Sweep").pack(side="left")
    sweep_knob = tk.StringVar(value="k")
    ttk.Combobox(swp, textvariable=sweep_knob, width=14, state="readonly",
                 values=["k", "density-thresh", "radius", "mag", "bmc", "k-low"]).pack(side="left", padx=4)
    ttk.Label(swp, text="values").pack(side="left")
    sweep_vals = tk.StringVar(value="6,9,12,16")
    ttk.Entry(swp, textvariable=sweep_vals, width=22).pack(side="left", padx=4)

    # box key -> db_center flag, so the sweep reuses every knob from the form except the swept one.
    KNOB_FLAGS = {"mag": "--mag", "bmc": "--bmc", "radius": "--radius", "k": "--k",
                  "density_thresh": "--density-thresh", "k_low": "--k-low"}

    def db_sweep():
        v = {k: var.get() for k, var in dbvars.items()}
        swept = sweep_knob.get()  # flag name with dashes, e.g. "density-thresh"
        cmd = [sys.executable, str(DB_CENTER), "sweep", "--fov-w", v["fov_w"], "--fov-h", v["fov_h"],
               "--knob", swept, "--values", sweep_vals.get(), "--fields", v["fields"], "--robust"]
        for key, flag in KNOB_FLAGS.items():
            if flag.lstrip("-") == swept:      # skip the swept knob: --values drives it
                continue
            if str(v[key]).strip():            # blank radius stays auto
                cmd += [flag, v[key]]
        run_async([cmd], log)
    ttk.Button(swp, text="Run sweep", command=db_sweep).pack(side="left", padx=4)

    win.mainloop()


if __name__ == "__main__":
    main()
