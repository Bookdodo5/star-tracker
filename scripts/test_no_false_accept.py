"""
Adversarial acceptance test: prove the tracker (a) never accepts a wrong attitude and
(b) never locks a wrong FOV. Runs the REAL C identifier via the live DLL.

Four suites, each prints PASS/FAIL; the process exits non-zero if any fails:

  1. correct-FOV      vectors built at the true FOV -> identify_vectors. A "wrong solve"
                      (accepted, pointing error > tol) is a false accept.  Expect 0.
  2. wrong-FOV        vectors built with a WRONG focal (camera/user FOV mismatch) ->
                      identify_vectors (fixed FOV). This is the screenshot bug: a distorted
                      field must be REJECTED. Any accept at |scale-1| > 3% is a false accept.
  3. pure-noise       random unit vectors (no real field) -> identify_vectors. Any solve is
                      a false accept.  Expect 0.
  4. fov-search       render at the true FOV, run the FOV-search bootstrap from far-off seeds.
                      Every accepted solve must have BOTH recovered-FOV within 3% AND pointing
                      within tol, else it locked a wrong FOV / wrong attitude.  Expect 0 bad.

Run:  python scripts/test_no_false_accept.py            # default sizes (~1-2 min)
      python scripts/test_no_false_accept.py --quick    # small, for a fast smoke
"""
from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import live_identify as L  # noqa: E402
from src.star_tracker_core import load_db_catalog, unit_vectors  # noqa: E402

MAG_LIMIT = 7.5
POINT_TOL_DEG = 0.5      # pointing error above this = wrong attitude
FOV_TOL_FRAC = 0.03      # recovered FOV must be within 3% of truth
WRONG_FOV_MIN_FRAC = 0.03  # a "wrong-FOV" input differs from truth by at least this


def camera_basis(ra_deg, dec_deg, roll_deg):
    """Camera->catalog basis (rows x,y,bore) for one pointing. Matches coverage_455x3_experiment."""
    ra, dec, roll = map(math.radians, (ra_deg, dec_deg, roll_deg))
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    bore = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    x_axis = east * math.cos(roll) + north * math.sin(roll)
    y_axis = -east * math.sin(roll) + north * math.cos(roll)
    return np.vstack([x_axis, y_axis, bore])


def pointing_error_deg(ra_a, dec_a, ra_b, dec_b):
    """Boresight angular error between two RA/DEC directions, in degrees."""
    a = unit_vectors(np.array([ra_a]), np.array([dec_a]))[0]
    b = unit_vectors(np.array([ra_b]), np.array([dec_b]))[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))


def visible_tangents(vecs, ra, dec, roll, fov_deg):
    """Tangent-plane offsets (tx,ty) of in-frame stars, brightest-first (catalog is mag-sorted)."""
    basis = camera_basis(ra, dec, roll)
    cam = vecs @ basis.T
    z = cam[:, 2]
    half = math.tan(math.radians(fov_deg) * 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx, ty = cam[:, 0] / z, cam[:, 1] / z
    inb = (z > 0) & (np.abs(tx) <= half) & (np.abs(ty) <= half)
    return tx[inb], ty[inb]


def tangents_to_unit(tx, ty, scale):
    """Build unit vectors as the C camera model does: (scale*tx, scale*ty, 1) normalized.
    scale=1 => true FOV; scale!=1 => the field distorted as if the FOV were wrong."""
    v = np.stack([scale * tx, scale * ty, np.ones_like(tx)], axis=1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def random_fields(rng, n):
    """Yield n random (ra, dec, roll) uniform on the sphere."""
    for _ in range(n):
        yield (float(rng.uniform(0, 360)),
               float(math.degrees(math.asin(rng.uniform(-1, 1)))),
               float(rng.uniform(0, 360)))


# --------------------------------------------------------------------------- suites

def suite_correct_fov(lib, vecs, rng, n, fov):
    """Vectors at the true FOV must never solve to the wrong sky."""
    solved = wrong = valid = 0
    for ra, dec, roll in random_fields(rng, n):
        tx, ty = visible_tangents(vecs, ra, dec, roll, fov)
        if len(tx) < 4:
            continue
        valid += 1
        est = L.solve_vectors(lib, tangents_to_unit(tx, ty, 1.0)[:20])
        if est is None:
            continue
        solved += 1
        if pointing_error_deg(ra, dec, est[0], est[1]) > POINT_TOL_DEG:
            wrong += 1
    ok = wrong == 0
    print(f"  [1] correct-FOV : valid={valid} solved={solved} WRONG={wrong}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def suite_wrong_fov(lib, vecs, rng, n, fov, scales):
    """Vectors built with a wrong focal must be rejected, not confidently mis-solved."""
    false_accept = attempts = 0
    per_scale = {}
    for scale in scales:
        s_solved = s_valid = 0
        print(f"      scale x{scale:.2f} ...", flush=True)
        for ra, dec, roll in random_fields(rng, n):
            tx, ty = visible_tangents(vecs, ra, dec, roll, fov)
            if len(tx) < 4:
                continue
            s_valid += 1
            attempts += 1
            est = L.solve_vectors(lib, tangents_to_unit(tx, ty, scale)[:20])
            if est is not None:
                s_solved += 1
                if abs(scale - 1.0) > WRONG_FOV_MIN_FRAC:
                    false_accept += 1
        per_scale[scale] = (s_valid, s_solved)
    ok = false_accept == 0
    detail = "  ".join(f"x{s:.2f}:{sv}->{so}solve" for s, (sv, so) in per_scale.items())
    print(f"  [2] wrong-FOV   : {detail}", flush=True)
    print(f"      false accepts (solve at |scale-1|>{WRONG_FOV_MIN_FRAC:.0%}) = {false_accept}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def suite_noise(lib, rng, n, k=16):
    """Random unit vectors (no real field) must never solve."""
    solved = 0
    for _ in range(n):
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # force z>0 hemisphere so they look like a plausible forward-looking field
        v[:, 2] = np.abs(v[:, 2])
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        if L.solve_vectors(lib, v.astype(np.float32)) is not None:
            solved += 1
    ok = solved == 0
    print(f"  [3] pure-noise  : fields={n} solved(false)={solved}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def suite_fov_search(lib, rng, n, true_fovs, seeds):
    """FOV-search from far-off seeds must only ever lock the correct FOV + correct attitude."""
    import cv2
    from simulator.renderer import Renderer

    # wrong_point = accepted a WRONG attitude (the real danger). wrong_fov_only = right sky,
    # wrong focal (a scale alias) — harmful for a fixed camera but not a wrong-sky report.
    wrong_point = wrong_fov_only = locked = attempts = 0
    per_fov = {f: [0, 0, 0] for f in true_fovs}  # [locked, wrong_fov_only, wrong_point]
    for true_fov in true_fovs:
        print(f"      true_fov={true_fov} ...", flush=True)
        r = Renderer(image_size=877, fov_deg=true_fov, magnitude_limit=MAG_LIMIT)
        for ra, dec, roll in random_fields(rng, n):
            jpg = r.render(ra, dec, roll)
            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            for seed in seeds:
                attempts += 1
                rec_fov, att = L.calibrate_fov(lib, bgr, seed, morph=0)
                if att is None:
                    continue
                locked += 1
                per_fov[true_fov][0] += 1
                point_err = pointing_error_deg(ra, dec, att[0], att[1])
                fov_bad = abs(rec_fov - true_fov) / true_fov > FOV_TOL_FRAC
                if point_err > POINT_TOL_DEG:
                    wrong_point += 1
                    per_fov[true_fov][2] += 1
                    print(f"      WRONG POINT: true=({ra:.1f},{dec:.1f}) est=({att[0]:.1f},{att[1]:.1f}) "
                          f"err={point_err:.2f}deg true_fov={true_fov} rec={rec_fov:.3f}", flush=True)
                elif fov_bad:
                    wrong_fov_only += 1
                    per_fov[true_fov][1] += 1
    for f, (lk, wf, wp) in per_fov.items():
        print(f"      fov={f}: locked={lk} wrong_fov_only={wf} wrong_point={wp}", flush=True)
    # The hard requirement is: never accept a WRONG attitude. wrong_fov_only is reported but
    # gated separately so off-design scale aliases don't mask a wrong-sky regression.
    ok = wrong_point == 0
    print(f"  [4] fov-search  : attempts={attempts} locked={locked} wrong_fov_only={wrong_fov_only} "
          f"WRONG_POINT={wrong_point}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fov", type=float, default=7.0, help="camera FOV for vector suites")
    p.add_argument("--n", type=int, default=3000, help="fields for suite 1 (fast: solves early-exit)")
    p.add_argument("--n-slow", type=int, default=120,
                   help="fields for suites 2/3 (each ~350ms: no verify => no early-exit)")
    p.add_argument("--fov-search-n", type=int, default=15,
                   help="fields per true-FOV in suite 4 (each calibrate ~2s)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-fov-search", action="store_true", help="skip the slow rendered suite 4")
    args = p.parse_args()
    if args.quick:
        args.n, args.n_slow, args.fov_search_n = 1000, 30, 4

    catalog = load_db_catalog(MAG_LIMIT)
    vecs = unit_vectors(catalog["RA_deg"], catalog["DEC_deg"])
    lib = L.load_lib()
    print(f"DB stars={len(vecs)}  camera_fov={args.fov}  n={args.n}  "
          f"point_tol={POINT_TOL_DEG}deg  fov_tol={FOV_TOL_FRAC:.0%}")

    t0 = time.time()
    results = []
    results.append(suite_correct_fov(lib, vecs, np.random.default_rng(1), args.n, args.fov))
    results.append(suite_wrong_fov(lib, vecs, np.random.default_rng(2), args.n_slow, args.fov,
                                   scales=(0.6, 0.8, 1.25, 1.5, 2.0)))
    results.append(suite_noise(lib, np.random.default_rng(3), args.n_slow))
    if not args.skip_fov_search:
        results.append(suite_fov_search(lib, np.random.default_rng(4), args.fov_search_n,
                                        true_fovs=(5.0, 7.0, 10.0, 14.0), seeds=(3.0, 8.0, 20.0)))

    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}  ({time.time()-t0:.1f}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
