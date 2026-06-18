"""
Evaluate a folder of orbit frames against ground-truth RA/DEC (output.txt).

Runs TETRA per frame and compares each recovered boresight to the truth in
<folder>/output.txt (lines: "index<TAB>ra_deg,dec_deg").

Pipeline per frame: PNG --PIL--> PPM --centroid_extract--> csv --demo_centroid_compare--> RA/DEC.
We try a few FOVs and keep the lowest-residual *internal* solve (the solver never
sees the truth), then score that solve's angular error vs truth.

Usage: python scripts/eval_orbit_frames.py [folder] [nominal_fov]
       defaults: cache/KnacksatOrbit_frame 17.7
"""
import re, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CENTROID_EXE = ROOT / "centroid" / "build-mingw" / "centroid_extract.exe"
DEMO_EXE = ROOT / "identifier" / "build-generated-release" / "demo_centroid_compare.exe"

SUCCESS_RE = re.compile(r"^TETRA success=(true|false) matches=(\d+) mean_residual_arcsec=(\d+) max_residual_arcsec=(\d+) score=(-?\d+) time_us=(\d+)")
ATT_RE = re.compile(r"^TETRA attitude_ra_deg=([-\d.]+) attitude_dec_deg=([-\d.]+) attitude_roll_deg=([-\d.]+)")


def angular_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle angle between two sky points, degrees."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def load_truth(folder):
    truth = {}
    idx = 0
    for line in (folder / "output.txt").read_text().splitlines():
        if not line.strip():
            continue
        idx += 1
        ra, dec = line.split(",")
        truth[idx] = (float(ra), float(dec))
    return truth


def run_demo(csv, w, h, fov):
    """Return dict(success,residual,time_us,ra,dec,roll) for TETRA."""
    out = subprocess.run([str(DEMO_EXE), str(csv), str(w), str(h), str(fov)],
                         check=True, capture_output=True, text=True).stdout
    res = {"success": False, "residual": None, "time_us": None, "ra": None, "dec": None, "roll": None}
    for line in out.splitlines():
        m = SUCCESS_RE.match(line)
        if m:
            res["success"] = m.group(1) == "true"
            res["residual"] = int(m.group(3))
            res["time_us"] = int(m.group(6))
        m = ATT_RE.match(line)
        if m:
            res.update(ra=float(m.group(1)), dec=float(m.group(2)), roll=float(m.group(3)))
    return res


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache" / "KnacksatOrbit_frame"
    if not folder.is_absolute():
        folder = ROOT / folder
    nominal = float(sys.argv[2]) if len(sys.argv) > 2 else 17.7
    fovs = [nominal, nominal-0.05, nominal+0.05, nominal-0.1, nominal+0.1, nominal-0.15, nominal+0.15]

    for exe in (CENTROID_EXE, DEMO_EXE):
        if not exe.exists():
            sys.exit(f"Missing {exe} -- run `.\\run.ps1 build`")
    truth = load_truth(folder)
    images = sorted(folder.glob("*.png"))
    out_dir = ROOT / "outputs"; out_dir.mkdir(exist_ok=True)
    tmp_ppm, tmp_csv = out_dir / "_eval.ppm", out_dir / "_eval.csv"

    rows = []
    t0 = time.time()
    print(f"[eval] {len(images)} frames, FOV candidates {fovs}")
    for i, png in enumerate(images, 1):
        idx = int(re.search(r"(\d+)", png.stem).group(1))
        tru = truth.get(idx)
        img = Image.open(png).convert("RGB"); img.save(tmp_ppm); w, h = img.size
        subprocess.run([str(CENTROID_EXE), str(tmp_ppm), str(tmp_csv)], check=True, capture_output=True)
        best = None
        for fov in fovs:
            r = run_demo(tmp_csv, w, h, fov)
            if r["success"] and (best is None or r["residual"] < best["residual"]):
                best = {**r, "fov": fov}
        err = angular_sep_deg(best["ra"], best["dec"], tru[0], tru[1]) if (best and tru) else None
        rows.append({"frame": idx, "name": png.name, "truth": tru, "TETRA": {**(best or {}), "err_deg": err}})
        if i % 10 == 0 or i == len(images):
            rate = i / max(time.time()-t0, 1e-6)
            print(f"  {i:3d}/{len(images)}  TETRA_ok={sum(1 for r in rows if r['TETRA'].get('success'))}"
                  f"  ({rate:.2f} fr/s, ETA {(len(images)-i)/max(rate,1e-6):.0f}s)")

    # write CSV
    csv_path = out_dir / f"{folder.name}_eval.csv"
    with open(csv_path, "w") as f:
        f.write("frame,truth_ra,truth_dec,"
                "tetra_success,tetra_ra,tetra_dec,tetra_err_deg,tetra_resid_arcsec,tetra_time_us\n")
        for r in rows:
            t, tr = r["TETRA"], r["truth"]
            def cell(d, k): return "" if d.get(k) is None else (f"{d[k]:.4f}" if isinstance(d.get(k), float) else d[k])
            f.write(f"{r['frame']},{tr[0]:.4f},{tr[1]:.4f},"
                    f"{bool(t.get('success'))},{cell(t,'ra')},{cell(t,'dec')},{cell(t,'err_deg')},{cell(t,'residual')},{cell(t,'time_us')}\n")
    print(f"[csv] wrote {csv_path}")

    # summary
    n = len(rows)
    print(f"\n{'='*52}\nSUMMARY  ({n} frames, {folder.name})\n{'='*52}")
    print(f"{'metric':<34}{'TETRA':>18}")
    line = lambda name, tv: print(f"{name:<34}{tv:>18}")

    def stat(tol):
        solved = [r["TETRA"] for r in rows if r["TETRA"].get("success")]
        correct = [s for s in solved if s["err_deg"] is not None and s["err_deg"] <= tol]
        return len(solved), len(correct)

    tsolved, _ = stat(1e9)
    line("solved (success=true)", f"{tsolved}/{n} ({100*tsolved/n:.1f}%)")
    for tol in (0.25, 0.5, 1.0):
        _, tc = stat(tol)
        line(f"correct <= {tol} deg of truth", f"{tc}/{n} ({100*tc/n:.1f}%)")
    te = [r["TETRA"]["err_deg"] for r in rows if r["TETRA"].get("success") and r["TETRA"]["err_deg"] is not None]
    line("angular err median (deg)", f"{np.median(te):.3f}" if te else "-")
    line("angular err mean (deg)", f"{np.mean(te):.3f}" if te else "-")
    line("angular err max (deg)", f"{np.max(te):.3f}" if te else "-")
    tt = [r["TETRA"]["time_us"]/1000 for r in rows if r["TETRA"].get("success") and r["TETRA"].get("time_us") is not None]
    line("identify time mean (ms)", f"{np.mean(tt):.1f}" if tt else "-")
    line("identify time max (ms)", f"{np.max(tt):.1f}" if tt else "-")
    print(f"{'='*52}\ntotal wall time: {time.time()-t0:.0f}s for {n} frames "
          f"({n/(time.time()-t0):.2f} frames/s incl. {len(fovs)}-FOV sweep)")


if __name__ == "__main__":
    main()
