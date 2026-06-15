"""
Pyramid parity harness: three encoding-matched checkpoints that diagnose why C Pyramid fails.

Checkpoint 1 - DB Coverage: Does the C Pyramid DB contain a seed pair from the top-10 visible stars?
Checkpoint 2 - Tolerance Match: Is the best pair's sep_code error within PYRAMID_SEED_TOL_CODE=328?
Checkpoint 3 - Grow success: Can the seed pair be grown to >=4 matched stars within tolerance?

The C DB uses MAX_FOV_DEG=15 and has full pairs (no diversity pruning).
The Python DB uses max_fov_deg=20 with cap/bin diversity pruning.

Run: python scripts/parity_harness_pyramid.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

# Force UTF-8 stdout so degree signs / em-dashes print on cp932 (JP) Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.star_tracker_core import (
    load_catalog, build_pair_database,
    filter_stars_by_fov, unit_vectors, angular_distance_pair,
)

# Test field: Orion region
RA = 83.8
DEC = -5.4
FOV = 10.0
MAG_LIMIT = 6.5

# C Pyramid constants (from identify_pyramid.c)
C_MAX_FOV_DEG = 15.0
C_SEED_TOL_CODE = 328    # PYRAMID_SEED_TOL_CODE
C_GROW_TOL_CODE = 590    # PYRAMID_GROW_TOL_CODE
C_MAX_QUERY_STARS = 10   # PYRAMID_MAX_QUERY_STARS
MAX_SEP_RAD = math.radians(C_MAX_FOV_DEG)

# Python Pyramid constants (from identify in star_tracker_core.py)
PY_SEED_TOL_DEG = 0.10
PY_GROW_TOL_DEG = 0.18
PY_MAX_STARS = 10

CHECKPOINT2_TOL_CODE = C_SEED_TOL_CODE   # same as C
CHECKPOINT3_MIN_GROW = 4                  # must grow to at least 4 stars


def sep_to_code(sep_rad: float, max_sep_rad: float = MAX_SEP_RAD) -> int:
    """Mirrors C's angle_to_code() for Pyramid separations."""
    scaled = sep_rad / max_sep_rad * 65535.0
    scaled = max(0.0, min(65535.0, scaled))
    return int(scaled + 0.5)


def code_to_deg(code: int, max_sep_rad: float = MAX_SEP_RAD) -> float:
    return math.degrees(code / 65535.0 * max_sep_rad)


def c_style_sep_code(sep_rad: float) -> int:
    return sep_to_code(sep_rad, MAX_SEP_RAD)


def build_c_style_pair_index(catalog_rows: list[tuple[int, float]],
                              vecs: np.ndarray) -> dict[frozenset, int]:
    """
    Builds a {frozenset(hr_a, hr_b) -> sep_code} dict equivalent to the C DB.
    Only includes pairs with sep <= MAX_FOV_DEG and both stars in the filtered catalog.
    """
    n = len(catalog_rows)
    pairs: dict[frozenset, int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            sep = angular_distance_pair(vecs[i], vecs[j])
            if sep <= MAX_SEP_RAD:
                hr_i, _ = catalog_rows[i]
                hr_j, _ = catalog_rows[j]
                pairs[frozenset([hr_i, hr_j])] = c_style_sep_code(sep)
    return pairs


def simulate_grow(
    seed_hrs: tuple[int, int],
    all_visible_hrs: list[int],
    all_visible_vecs: np.ndarray,
    catalog_rows: list[tuple[int, float]],
    catalog_vecs: np.ndarray,
    c_pairs: dict[frozenset, int],
    grow_tol_code: int,
    max_sep_rad: float = MAX_SEP_RAD,
) -> list[int]:
    """
    Simulates the Pyramid grow step: given a seed pair (HR IDs), checks how many
    additional visible stars can be assigned a catalog ID within grow tolerance.
    Returns the list of matched HR IDs (seed pair + grown matches).
    """
    matched = list(seed_hrs)
    seed_vecs = [catalog_vecs[next(idx for idx, (hr, _) in enumerate(catalog_rows) if hr == h)]
                 for h in seed_hrs]

    for obs_idx, (obs_hr, obs_vec) in enumerate(zip(all_visible_hrs, all_visible_vecs)):
        if obs_hr in seed_hrs:
            continue
        # Find catalog candidates reachable from both seed stars within grow tolerance
        candidates: set[int] = set()
        for seed_hr, seed_vec in zip(seed_hrs, seed_vecs):
            obs_sep_rad = angular_distance_pair(obs_vec, seed_vec)
            obs_code = c_style_sep_code(obs_sep_rad)
            lo = max(0, obs_code - grow_tol_code)
            hi = min(65535, obs_code + grow_tol_code)
            # Scan C DB for catalog stars near seed_hr within this tolerance
            for (hr_a, hr_b), sep in c_pairs.items():
                if seed_hr in (hr_a, hr_b) and lo <= sep <= hi:
                    other = hr_b if hr_a == seed_hr else hr_a
                    candidates.add(other)
        # Candidates must be consistent with ALL already-matched stars
        for match_hr in matched:
            match_vec_idx = next((idx for idx, (hr, _) in enumerate(catalog_rows) if hr == match_hr), None)
            if match_vec_idx is None:
                continue
            match_vec = catalog_vecs[match_vec_idx]
            obs_sep_rad = angular_distance_pair(obs_vec, match_vec)
            obs_code = c_style_sep_code(obs_sep_rad)
            lo = max(0, obs_code - grow_tol_code)
            hi = min(65535, obs_code + grow_tol_code)
            candidates = {c for c in candidates
                          if abs(c_pairs.get(frozenset([match_hr, c]), -99999) - obs_code) <= grow_tol_code
                             or frozenset([match_hr, c]) not in c_pairs}
        if len(candidates) == 1:
            matched.append(candidates.pop())
        elif len(candidates) > 1:
            # Multiple candidates: not uniquely constrained
            pass
    return matched


def main() -> None:
    print(f"=== Pyramid Parity Harness: RA={RA}, DEC={DEC}, FOV={FOV}° ===\n")

    catalog = load_catalog()
    print("Loading Python Pyramid DB (fov<=20, diversity-pruned)...")
    py_pair_db, db_stars = build_pair_database(
        catalog, algorithm_name="pyramid", max_fov_deg=20.0, mag_limit=MAG_LIMIT
    )
    print(f"  Python DB: {len(py_pair_db)} pairs\n")

    visible = filter_stars_by_fov(catalog, RA, DEC, FOV, MAG_LIMIT)
    visible = visible.sort_values("Vmag").head(C_MAX_QUERY_STARS).reset_index(drop=True)
    print(f"Top-{C_MAX_QUERY_STARS} stars (by brightness, Vmag<={MAG_LIMIT}) in field:")
    for rank, row in visible.iterrows():
        print(f"  #{rank+1:2d}  HR={int(row['HR_clean']):5d}  Vmag={row['Vmag']:.2f}  "
              f"RA={row['RA_deg']:.2f}  DEC={row['DEC_deg']:.2f}")

    obs_hrs = [int(r["HR_clean"]) for _, r in visible.iterrows()]
    obs_vecs = unit_vectors(visible["RA_deg"].to_numpy(), visible["DEC_deg"].to_numpy())

    # Build a subset catalog of all bright stars for C-style pair DB
    all_bright = catalog[catalog["Vmag"] <= MAG_LIMIT].sort_values("Vmag").reset_index(drop=True)
    cat_rows = [(int(r["HR_clean"]), r["Vmag"]) for _, r in all_bright.iterrows()]
    cat_vecs = unit_vectors(all_bright["RA_deg"].to_numpy(), all_bright["DEC_deg"].to_numpy())

    print(f"\nBuilding C-style pair index (fov<={C_MAX_FOV_DEG}°, {len(cat_rows)} catalog stars) ...")
    print("  (This may take 30-60 seconds for the full bright catalog)")

    # Only build pairs among stars close to the test field to keep it fast
    field_visible_all = filter_stars_by_fov(catalog, RA, DEC, C_MAX_FOV_DEG * 2, MAG_LIMIT)
    local_rows = [(int(r["HR_clean"]), r["Vmag"]) for _, r in field_visible_all.iterrows()]
    local_vecs = unit_vectors(field_visible_all["RA_deg"].to_numpy(), field_visible_all["DEC_deg"].to_numpy())
    c_pairs = build_c_style_pair_index(local_rows, local_vecs)
    print(f"  Local C-style pairs (within {C_MAX_FOV_DEG*2}° of field center): {len(c_pairs)}\n")

    # ---- Checkpoint 1: DB Coverage ----
    print("--- Checkpoint 1: DB Coverage (C-style DB) ---")
    cp1_pass = False
    best_seed_error = 999999
    best_seed_pair: tuple[int, int] | None = None
    best_seed_code = 0

    for i in range(len(obs_hrs)):
        for j in range(i + 1, len(obs_hrs)):
            hr_a, hr_b = obs_hrs[i], obs_hrs[j]
            obs_sep_rad = angular_distance_pair(obs_vecs[i], obs_vecs[j])
            obs_code = c_style_sep_code(obs_sep_rad)
            pair_key = frozenset([hr_a, hr_b])
            if pair_key in c_pairs:
                db_code = c_pairs[pair_key]
                err = abs(db_code - obs_code)
                if err < best_seed_error:
                    best_seed_error = err
                    best_seed_pair = (hr_a, hr_b)
                    best_seed_code = db_code

    if best_seed_pair is not None:
        cp1_pass = True
        sep_deg = code_to_deg(best_seed_code)
        print(f"  Best seed pair: HR {best_seed_pair[0]} - HR {best_seed_pair[1]}")
        print(f"  DB sep_code={best_seed_code} ({sep_deg:.2f}°), sep_code error={best_seed_error}")
        print(f"  CHECKPOINT 1: PASS (seed pair found in C DB)")
    else:
        print(f"  No pair from top-{C_MAX_QUERY_STARS} visible stars found in C DB!")
        print(f"  CHECKPOINT 1: FAIL (DB coverage gap)")

    # ---- Checkpoint 2: Tolerance Match ----
    print("\n--- Checkpoint 2: Tolerance Match (PYRAMID_SEED_TOL_CODE={}) ---".format(C_SEED_TOL_CODE))
    cp2_pass = False
    if best_seed_pair is not None:
        cp2_pass = best_seed_error <= CHECKPOINT2_TOL_CODE
        tol_deg = code_to_deg(CHECKPOINT2_TOL_CODE)
        print(f"  Best seed pair sep_code error: {best_seed_error} (tolerance: {C_SEED_TOL_CODE} = {tol_deg:.3f}°)")
        py_seed_tol_code = sep_to_code(math.radians(PY_SEED_TOL_DEG))
        print(f"  Python seed tolerance: {PY_SEED_TOL_DEG}° = code {py_seed_tol_code}  "
              f"(C is {'tighter' if C_SEED_TOL_CODE < py_seed_tol_code else 'looser'})")
        print(f"  CHECKPOINT 2: {'PASS' if cp2_pass else 'FAIL'} "
              f"({'within' if cp2_pass else 'OUTSIDE'} C seed tolerance)")
    else:
        print("  No seed pair — skipping.")

    # ---- Checkpoint 3: Grow success ----
    print("\n--- Checkpoint 3: Grow success (target >= 4 matched stars) ---")
    cp3_pass = False
    grow_result_count = 0
    if best_seed_pair is not None and cp2_pass:
        grow_tol_code = C_GROW_TOL_CODE
        matched = simulate_grow(
            best_seed_pair, obs_hrs, obs_vecs,
            local_rows, local_vecs, c_pairs,
            grow_tol_code,
        )
        grow_result_count = len(matched)
        cp3_pass = grow_result_count >= CHECKPOINT3_MIN_GROW
        print(f"  Simulated grow result: {grow_result_count} matched stars (need >= {CHECKPOINT3_MIN_GROW})")
        print(f"  Matched HRs: {matched}")
        print(f"  CHECKPOINT 3: {'PASS' if cp3_pass else 'FAIL'}")
    else:
        print("  Seed pair not within tolerance — skipping grow.")

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"  CP1 DB Coverage:    {'PASS' if cp1_pass else 'FAIL'}")
    print(f"  CP2 Tolerance:      {'PASS' if cp2_pass else 'FAIL'}")
    print(f"  CP3 Grow Success:   {'PASS' if cp3_pass else 'FAIL'}")

    root_cause = []
    if not cp1_pass:
        root_cause.append("C DB has no pair for any two of the top-10 visible stars (DB coverage gap)")
    elif not cp2_pass:
        root_cause.append(
            f"Seed pair exists in DB but sep_code error={best_seed_error} > PYRAMID_SEED_TOL_CODE={C_SEED_TOL_CODE}; "
            f"increase tolerance or check pair encoding"
        )
    elif not cp3_pass:
        root_cause.append(
            f"Seed pair found and in tolerance, but grow only reached {grow_result_count} stars "
            f"(need {CHECKPOINT3_MIN_GROW}); grow tolerance or branch ordering may be wrong"
        )

    report_lines = [
        "\nPyramid Parity Harness Report",
        f"  Field: RA={RA} DEC={DEC} FOV={FOV}",
        f"  CP1 DB Coverage: {'PASS' if cp1_pass else 'FAIL'}",
        f"  CP2 Tolerance: {'PASS' if cp2_pass else 'FAIL'} (error={best_seed_error}, tol={C_SEED_TOL_CODE})",
        f"  CP3 Grow: {'PASS' if cp3_pass else 'FAIL'} (grew to {grow_result_count} stars)",
        f"  Overall: {'PASS' if (cp1_pass and cp2_pass and cp3_pass) else 'FAIL'}",
    ]
    if root_cause:
        report_lines.append("  Root cause(s):")
        for rc in root_cause:
            report_lines.append(f"    - {rc}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "divergence_report.txt"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    sep_line = "\n" + "=" * 60 + "\n"
    report_path.write_text(
        "\n".join(report_lines) + sep_line + existing,
        encoding="utf-8",
    )
    print(f"\nReport appended to {report_path}")

    if root_cause:
        print("\nROOT CAUSE(S):")
        for rc in root_cause:
            print(f"  * {rc}")
    else:
        print("\nAll checkpoints pass.")


if __name__ == "__main__":
    main()
