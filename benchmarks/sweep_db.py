"""
Find the smallest TETRA DB (by baked bytes) that solves >=99% of "valid" 7x4 deg fields.

Two modes:
  --eval            Load the CURRENT DLL/DB and print one accuracy JSON line (run as a
                    subprocess by the sweep so each DB is measured in a fresh process).
  (default) sweep   For each magnitude limit L: regenerate both DBs, rebuild the DLL,
                    then subprocess --eval. Reports the size<->accuracy curve and the
                    smallest DB meeting the gate.

Gate (see .omc/specs/deep-interview-tycho2-db-sweep.md):
  - Reference sky R = Tycho-2 V<=VALID_MAG (7.5); a field is VALID if it contains
    >=MIN_BRIGHT (6) R-stars in the 7x4 footprint.
  - A solve is CORRECT if the boresight is within TOL (0.5 deg) of truth.
  - accuracy = correct / valid; PASS if >=0.99. Objective: minimize baked DB bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GEN = ROOT / "identifier" / "generated"
CATALOG_C = GEN / "catalog_db_generated.c"
TETRA_C = GEN / "tetra_db_generated.c"

FOV_W, FOV_H = 7.0, 4.0
VALID_MAG = 7.5
MIN_BRIGHT = 6
TOL_DEG = 0.5
N_FIELDS = 2000
SEED = 12345


def _rotation(ra_deg, dec_deg, roll_deg):
    """catalog->camera rotation with rows (east', north', boresight); roll rotates in-plane."""
    ra, dec, roll = np.radians([ra_deg, dec_deg, roll_deg])
    e = np.array([-np.sin(ra), np.cos(ra), 0.0])
    n = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    b = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    x = e * np.cos(roll) + n * np.sin(roll)
    y = -e * np.sin(roll) + n * np.cos(roll)
    return np.vstack([x, y, b])  # row 2 = boresight, matches attitude_to_radecroll


def _sep_deg(ra1, dec1, ra2, dec2):
    a = np.radians([ra1, dec1]); b = np.radians([ra2, dec2])
    d = (np.sin(a[1]) * np.sin(b[1]) +
         np.cos(a[1]) * np.cos(b[1]) * np.cos(a[0] - b[0]))
    return np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))


def evaluate():
    """Runs the synthetic gate on the current DLL/DB; returns a metrics dict."""
    import live_identify as L
    from src.star_tracker_core import load_db_catalog

    lib = L.load_lib()
    if not hasattr(lib, "identify_vectors"):
        raise SystemExit("DLL lacks identify_vectors -- rebuild live/ after adding the entry.")

    R = load_db_catalog(VALID_MAG)
    ra = R["RA_deg"].to_numpy(); dec = R["DEC_deg"].to_numpy()
    cat = np.column_stack([
        np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
        np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
        np.sin(np.radians(dec)),
    ])  # R already sorted brightest-first by load_db_catalog

    tan_w, tan_h = np.tan(np.radians(FOV_W / 2)), np.tan(np.radians(FOV_H / 2))
    rng = np.random.default_rng(SEED)
    t_ra = rng.uniform(0, 360, N_FIELDS)
    t_dec = np.degrees(np.arcsin(rng.uniform(-1, 1, N_FIELDS)))
    t_roll = rng.uniform(0, 360, N_FIELDS)

    valid = solved = correct = 0
    for i in range(N_FIELDS):
        M = _rotation(t_ra[i], t_dec[i], t_roll[i])
        obs = cat @ M.T  # rows -> camera-frame vectors (already brightest-first)
        z = obs[:, 2]
        infov = (z > 0) & (np.abs(obs[:, 0] / z) <= tan_w) & (np.abs(obs[:, 1] / z) <= tan_h)
        field = obs[infov]  # preserves brightness order
        if len(field) < MIN_BRIGHT:
            continue  # not a valid field (empty/sparse sky) -> excluded
        valid += 1
        att = L.solve_vectors(lib, field[:20])  # sensor sees brightest ~20
        if att is None:
            continue
        solved += 1
        if _sep_deg(t_ra[i], t_dec[i], att[0], att[1]) <= TOL_DEG:
            correct += 1

    db_bytes = CATALOG_C.stat().st_size + TETRA_C.stat().st_size
    return {
        "mag": float(os.environ.get("STAR_DB_MAG", "?") or "?") if os.environ.get("STAR_DB_MAG") else None,
        "valid": valid, "solved": solved, "correct": correct,
        "accuracy": (correct / valid) if valid else 0.0,
        "solve_rate": (solved / valid) if valid else 0.0,
        "db_bytes": db_bytes,
    }


def _regen_and_build(mag):
    """Regenerates both DBs at magnitude limit `mag` and rebuilds the DLL. Returns True on success."""
    env = dict(os.environ, STAR_DB_MAG=str(mag))
    for tool in ("export_catalog_db.py", "export_tetra_db.py"):
        print(f"  [{mag}] {tool} ...", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "identifier" / "tools" / tool)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-800:]); print(r.stderr[-800:]); return False
    print(f"  [{mag}] building DLL ...", flush=True)
    r = subprocess.run(["cmake", "--build", str(ROOT / "live" / "build-mingw")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-800:]); print(r.stderr[-800:]); return False
    return True


def main():
    p = argparse.ArgumentParser(description="DB size<->coverage sweep")
    p.add_argument("--eval", action="store_true", help="evaluate the current DB only (prints JSON)")
    p.add_argument("--mags", default="6.0,6.25,6.5,6.75,7.0,7.25,7.5,7.75,8.0")
    args = p.parse_args()

    if args.eval:
        print("RESULT " + json.dumps(evaluate()))
        return

    mags = [float(x) for x in args.mags.split(",")]
    rows = []
    for mag in mags:
        if not _regen_and_build(mag):
            print(f"  [{mag}] regen/build FAILED, skipping")
            continue
        r = subprocess.run([sys.executable, __file__, "--eval"], env=dict(os.environ, STAR_DB_MAG=str(mag)),
                           capture_output=True, text=True)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if not line:
            print(r.stdout[-800:]); print(r.stderr[-800:]); print(f"  [{mag}] eval FAILED"); continue
        m = json.loads(line[len("RESULT "):]); m["mag"] = mag; rows.append(m)
        print(f"  L={mag}: valid={m['valid']} solved={m['solved']} correct={m['correct']} "
              f"acc={m['accuracy']*100:.2f}% solve%={m['solve_rate']*100:.1f} db={m['db_bytes']/1e6:.2f}MB", flush=True)

    print("\n=== size<->accuracy curve ===")
    print(f"{'L':>5} {'valid':>6} {'acc%':>7} {'solve%':>7} {'DB_MB':>7}")
    for m in rows:
        print(f"{m['mag']:>5} {m['valid']:>6} {m['accuracy']*100:>7.2f} {m['solve_rate']*100:>7.1f} {m['db_bytes']/1e6:>7.2f}")
    passing = [m for m in rows if m["accuracy"] >= 0.99]
    if passing:
        best = min(passing, key=lambda m: m["db_bytes"])
        print(f"\nSMALLEST DB >=99%: L={best['mag']}  acc={best['accuracy']*100:.2f}%  "
              f"DB={best['db_bytes']/1e6:.2f}MB  valid={best['valid']}")
    else:
        print("\nNo DB reached 99% accuracy-over-valid in the swept range.")


if __name__ == "__main__":
    main()
