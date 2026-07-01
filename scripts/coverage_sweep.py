"""
Fast pure-Python coverage sweep — find the smallest DB config that solves >=99% of valid
7x4 deg fields, WITHOUT compiling a 400 MB DLL per candidate.

A field solves iff >=1 four-subset of its brightest-10 footprint stars is a DB tetrad
(the C solver verifies at 100% correctness once a tetrad matches, so coverage == solve-rate).
We replicate DB tetrad generation (brightest-K within field_radius of each catalog star) as a
set of integer keys, then test the same 2000 seeded synthetic fields the C harness uses.

    python scripts/coverage_sweep.py --bmc 6.0,6.5,7.0 --fieldstars 8,10,12,16

Mirrors the ANCHORED generator in export_tetra_db.py: each tetra is owned by its brightest
star (anchor V<=BMC), members are V<=L (=7.5, the validity reference), neighbours are gathered
within the FOV DIAGONAL and tetras are kept if max edge <= diagonal. Reports coverage and
tetrad count (proxy for DB bytes) per (BMC, K). No compile needed.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.star_tracker_core import load_db_catalog, anchored_allcombos_tetrads  # noqa: E402

FOV_W, FOV_H = 7.0, 4.0
VALID_MAG = 7.5           # L: member limit AND validity reference (kept equal so ids align)
MIN_BRIGHT = 6
N_FIELDS = 2000
SEED = 12345
QUERY_STARS = 16          # matches TETRA_MAX_QUERY_STARS in identify_tetra.c
FOV_DIAG = math.hypot(FOV_W, FOV_H)
GATHER_RAD = math.radians(FOV_DIAG)   # gather companions out to the full diagonal
MAX_EDGE_RAD = GATHER_RAD             # max pairwise edge: 4 stars must fit one frame


def _vecs(df):
    ra = np.radians(df["RA_deg"].to_numpy()); dec = np.radians(df["DEC_deg"].to_numpy())
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _key(ids):
    a, b, c, d = sorted(ids)
    return (a << 48) | (b << 32) | (c << 16) | d


def build_tetrad_keys(vecs, n_anchors, radius_deg, max_nb,
                      density_thresh=None, k_low=None):
    """Set of integer keys for anchored all-combos tetrads at the given gather radius and K."""
    gather = math.radians(radius_deg)
    tetrads = anchored_allcombos_tetrads(vecs, n_anchors, gather, MAX_EDGE_RAD, max_nb,
                                          density_thresh=density_thresh, k_low=k_low)
    return {_key(t) for t in tetrads}, len(tetrads)


def coverage(keys, ref_vecs):
    """Fraction of valid fields whose brightest-10 contains >=1 tetrad present in `keys`."""
    tan_w, tan_h = math.tan(math.radians(FOV_W / 2)), math.tan(math.radians(FOV_H / 2))
    rng = np.random.default_rng(SEED)
    t_ra = rng.uniform(0, 360, N_FIELDS)
    t_dec = np.degrees(np.arcsin(rng.uniform(-1, 1, N_FIELDS)))
    t_roll = rng.uniform(0, 360, N_FIELDS)
    valid = covered = 0
    for i in range(N_FIELDS):
        ra, dec, roll = map(math.radians, (t_ra[i], t_dec[i], t_roll[i]))
        e = np.array([-math.sin(ra), math.cos(ra), 0.0])
        n = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
        b = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
        x = e * math.cos(roll) + n * math.sin(roll)
        y = -e * math.sin(roll) + n * math.cos(roll)
        M = np.vstack([x, y, b])
        obs = ref_vecs @ M.T
        z = obs[:, 2]
        infov = np.where((z > 0) & (np.abs(obs[:, 0] / z) <= tan_w) & (np.abs(obs[:, 1] / z) <= tan_h))[0]
        if len(infov) < MIN_BRIGHT:
            continue
        valid += 1
        bright = infov[:QUERY_STARS]  # ref_vecs is brightest-first, infov preserves order
        if any(_key(c) in keys for c in itertools.combinations(bright.tolist(), 4)):
            covered += 1
    return covered, valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bmc", default="7.0", help="anchor magnitude limit")
    p.add_argument("--radii", default="3.5", help="gather radii (deg) to try")
    p.add_argument("--maxnb", default="9", help="K_high: max neighbours for sparse anchors")
    p.add_argument("--adaptive", action="store_true",
                   help="sweep adaptive-K: vary density_thresh and k_low")
    p.add_argument("--density-thresh", default="6,8,10,12",
                   help="neighbour counts above which k_low is used")
    p.add_argument("--k-low", default="5,6,7",
                   help="K to use for dense anchors")
    args = p.parse_args()

    ref = load_db_catalog(VALID_MAG)
    ref_vecs = _vecs(ref)
    vmag = ref["Vmag"].to_numpy()

    results = {}

    if args.adaptive:
        print(f"=== Adaptive-K sweep (BMC={args.bmc} R={args.radii} K_high={args.maxnb}) ===")
        print(f"{'thresh':>7} {'k_low':>6} {'tetrads':>10} {'cover%':>8}  vs K=9 baseline")
        bmc = float(args.bmc); radius = float(args.radii); k_high = int(args.maxnb)
        n_anchors = int((vmag <= bmc).sum())
        # baseline
        keys9, nt9 = build_tetrad_keys(ref_vecs, n_anchors, radius, k_high)
        cov9, valid9 = coverage(keys9, ref_vecs)
        print(f"  baseline K={k_high}: {nt9:>10,} tetrads  {100*cov9/valid9:.2f}%")
        for dt in (int(x) for x in args.density_thresh.split(",")):
            for kl in (int(x) for x in args.k_low.split(",")):
                if kl >= k_high:
                    continue
                keys, nt = build_tetrad_keys(ref_vecs, n_anchors, radius, k_high,
                                              density_thresh=dt, k_low=kl)
                cov, valid = coverage(keys, ref_vecs)
                pct = 100 * cov / valid
                flag = "✓" if pct >= 99.0 else " "
                reduction = 100 * (1 - nt / nt9)
                results[(dt, kl)] = (nt, cov, valid)
                print(f"  thresh>{dt:2d} k_low={kl}: {nt:>10,} tetrads  {pct:.2f}%  "
                      f"({reduction:.1f}% smaller)  {flag}", flush=True)
        print("\n>=99% configs, fewest tetrads first:")
        ok = [(k, v) for k, v in results.items() if v[1]/v[2] >= 0.99]
        for (dt, kl), (nt, cov, valid) in sorted(ok, key=lambda x: x[1][0]):
            print(f"  thresh>{dt} k_low={kl}: {nt:,} tetrads  {100*cov/valid:.2f}%")
        if not ok:
            print("  none reached 99%")
    else:
        print(f"{'BMC':>5} {'rad':>5} {'K':>4} {'anchors':>8} {'tetrads':>10} {'valid':>6} {'cover%':>7}")
        for bmc in (float(x) for x in args.bmc.split(",")):
            n_anchors = int((vmag <= bmc).sum())
            for radius in (float(x) for x in args.radii.split(",")):
                for k in (int(x) for x in args.maxnb.split(",")):
                    keys, nt = build_tetrad_keys(ref_vecs, n_anchors, radius, k)
                    cov, valid = coverage(keys, ref_vecs)
                    results[(bmc, radius, k)] = (n_anchors, nt, cov, valid)
                    print(f"{bmc:>5} {radius:>5} {k:>4} {n_anchors:>8} {nt:>10} {valid:>6} "
                          f"{100*cov/valid:>7.2f}", flush=True)
        print("\nConfigs >=99% coverage, fewest tetrads first:")
        ok = [(k, v) for k, v in results.items() if v[2] / v[3] >= 0.99]
        for (bmc, radius, knb), (na, nt, cov, valid) in sorted(ok, key=lambda kv: kv[1][1]):
            print(f"  BMC={bmc} radius={radius} K={knb}: anchors={na} tetrads={nt} cover={100*cov/valid:.2f}%")
        if not ok:
            print("  none reached 99% in this grid")


if __name__ == "__main__":
    main()
