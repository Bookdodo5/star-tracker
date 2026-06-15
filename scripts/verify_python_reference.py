"""
Reproducible accuracy test for Python TETRA and Pyramid references.
Fixed seed (numpy 42), 20 synthetic fields. Saves results to outputs/python_reference_baseline.txt.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.star_tracker_core import (
    load_catalog, build_tetra_database, TetraMatcher,
    build_pair_database, PyramidMatcher,
    filter_stars_by_fov,
)

SEED = 42
N_FIELDS = 20
TETRA_FOV = 15.0   # matches DB build FOV; 10 deg gives ~44% due to cap=40 truncation
PYRAMID_FOV = 8.0
MAG_LIMIT = 6.5
TETRA_MAX_STARS = 12
PYRAMID_MAX_STARS = 10


def random_ra_dec(rng: np.random.Generator) -> tuple[float, float]:
    ra = rng.uniform(0.0, 360.0)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))
    return float(ra), float(dec)


def run_batch(algo: str, matcher, catalog, fov: float, max_stars: int, n: int, seed: int) -> tuple[float, list]:
    rng = np.random.default_rng(seed)
    correct = 0
    results = []
    for i in range(n):
        ra, dec = random_ra_dec(rng)
        visible = filter_stars_by_fov(catalog, ra, dec, fov, MAG_LIMIT)
        if len(visible) < 4:
            results.append({"field": i, "ra": ra, "dec": dec, "skipped": True})
            continue
        result = matcher.identify(ra, dec, fov, MAG_LIMIT, max_stars_query=max_stars)
        ok = result.get("correct", False)
        if ok:
            correct += 1
        results.append({"field": i, "ra": ra, "dec": dec, "correct": ok, "score": result.get("score")})
    accuracy = correct / n * 100.0
    return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["tetra", "pyramid", "both"], default="both")
    parser.add_argument("--n", type=int, default=N_FIELDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    print("Loading catalog...")
    catalog = load_catalog()

    # Get git SHA of Tetra/Tetra.ipynb (authoritative reference marker)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:Tetra/Tetra.ipynb"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        sha = "unknown"

    lines = [
        f"Python Reference Baseline - seed={args.seed}, n={args.n}",
        f"Tetra/Tetra.ipynb SHA: {sha}",
        "",
    ]

    if args.algo in ("tetra", "both"):
        print(f"Building TETRA database (fov={TETRA_FOV}, mag<={MAG_LIMIT})...")
        db = build_tetra_database(catalog, fov_deg=15.0, mag_limit=MAG_LIMIT, max_tetras_per_anchor=40)
        matcher = TetraMatcher(catalog, db)
        acc, _ = run_batch("tetra", matcher, catalog, TETRA_FOV, TETRA_MAX_STARS, args.n, args.seed)
        line = f"TETRA accuracy: {acc:.1f}% on {args.n} fields (FOV={TETRA_FOV}°, max_stars={TETRA_MAX_STARS})"
        print(line)
        lines.append(line)

    if args.algo in ("pyramid", "both"):
        print(f"Building Pyramid database (fov<=20°, mag<={MAG_LIMIT})...")
        pair_db, db_stars = build_pair_database(catalog, algorithm_name="pyramid",
                                                max_fov_deg=20.0, mag_limit=MAG_LIMIT)
        matcher = PyramidMatcher(catalog, pair_db, db_stars)
        acc, _ = run_batch("pyramid", matcher, catalog, PYRAMID_FOV, PYRAMID_MAX_STARS, args.n, args.seed)
        line = f"Pyramid accuracy: {acc:.1f}% on {args.n} fields (FOV={PYRAMID_FOV}°, max_stars={PYRAMID_MAX_STARS})"
        print(line)
        lines.append(line)

    baseline_path = outputs / "python_reference_baseline.txt"
    baseline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {baseline_path}")


if __name__ == "__main__":
    main()
