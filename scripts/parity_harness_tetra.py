"""
TETRA parity harness: three encoding-matched checkpoints that diagnose why C TETRA fails.

Checkpoint 1 - DB Coverage: Does the DB contain >=1 tetrad with >=3 of the top-12 visible stars?
Checkpoint 2 - Feature Error: Is the best DB match within a 200-unit L1 window when using C-style encoding?
Checkpoint 3 - Star Cap: Does the matching tetrad fall within the top-8 stars (C's TETRA_MAX_QUERY_STARS)?

Run: python scripts/parity_harness_tetra.py
"""
from __future__ import annotations
import itertools
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
    load_catalog, build_tetra_database,
    filter_stars_by_fov, unit_vectors, angular_distance_pair,
    sorted_edge_feature,
)

# Test field: Orion region
RA = 83.8
DEC = -5.4
FOV = 10.0      # test FOV in degrees
DB_FOV = 15.0   # DB was built at this FOV
MAG_LIMIT = 6.5
MAX_STARS_C = 8   # TETRA_MAX_QUERY_STARS in C
MAX_STARS_PY = 12

CHECKPOINT1_JACCARD_THRESHOLD = 0.5   # >=2/4 stars must match (Jaccard >=0.5)
CHECKPOINT2_L1_THRESHOLD = 200        # L1 error threshold for "reachable by KD-tree"
CHECKPOINT3_LABEL = f"top-{MAX_STARS_C} (C cap)"

PI = math.pi


def angle_to_code(angle_rad: float, max_angle_rad: float = PI) -> int:
    """Mirrors C's angle_to_code() — quantizes angle to uint16 via max_angle."""
    scaled = angle_rad / max_angle_rad * 65535.0
    scaled = max(0.0, min(65535.0, scaled))
    return int(scaled + 0.5)


def c_style_feature(vectors4: np.ndarray) -> np.ndarray | None:
    """
    Computes the TETRA feature exactly as the C code does:
    1. Compute 6 angular distances (float)
    2. Quantize each via angle_to_code (divide by pi, not by max)
    3. Sort
    4. Normalize 5 shortest by longest using integer division
    """
    edges_rad = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = angular_distance_pair(vectors4[i], vectors4[j])
            edges_rad.append(d)
    edge_codes = sorted([angle_to_code(d) for d in edges_rad])
    if edge_codes[5] == 0:
        return None
    feature = np.array(
        [int(edge_codes[k] * 65535) // edge_codes[5] for k in range(5)],
        dtype=np.uint16,
    )
    return feature


def python_style_feature(vectors4: np.ndarray) -> np.ndarray | None:
    """
    Computes the TETRA feature as stored in the DB (Python style):
    1. Compute ratio arc_len_i / arc_len_max (float)
    2. Multiply by 65535 and round to uint16
    """
    feat_float = sorted_edge_feature(vectors4)
    if feat_float is None:
        return None
    return np.round(feat_float * 65535.0).astype(np.uint16)


def l1_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).sum())


def db_feature_array(db: pd.DataFrame) -> np.ndarray:
    """Returns DB features as a (N, 5) uint16 array."""
    return np.round(db[["f1", "f2", "f3", "f4", "f5"]].to_numpy(np.float32) * 65535).astype(np.uint16)


def hr_set(row: pd.Series) -> frozenset:
    return frozenset([int(row["hr_a"]), int(row["hr_b"]), int(row["hr_c"]), int(row["hr_d"])])


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b)


def main() -> None:
    print(f"=== TETRA Parity Harness: RA={RA}, DEC={DEC}, FOV={FOV}° ===\n")

    catalog = load_catalog()
    print(f"Loading TETRA DB (fov={DB_FOV}, cap=40)...")
    db = build_tetra_database(catalog, fov_deg=DB_FOV, mag_limit=MAG_LIMIT, max_tetras_per_anchor=40)
    print(f"  DB size: {len(db)} tetrads\n")

    visible = filter_stars_by_fov(catalog, RA, DEC, FOV, MAG_LIMIT)
    visible = visible.sort_values("Vmag").head(MAX_STARS_PY).reset_index(drop=True)
    print(f"Top-{MAX_STARS_PY} stars (by brightness, Vmag<={MAG_LIMIT}) in field:")
    for rank, row in visible.iterrows():
        print(f"  #{rank+1:2d}  HR={int(row['HR_clean']):5d}  Vmag={row['Vmag']:.2f}  "
              f"RA={row['RA_deg']:.2f}  DEC={row['DEC_deg']:.2f}")

    visible_hrs = set(int(r["HR_clean"]) for _, r in visible.iterrows())
    top8_hrs = set(int(r["HR_clean"]) for _, r in visible.head(MAX_STARS_C).iterrows())
    vecs = unit_vectors(visible["RA_deg"].to_numpy(), visible["DEC_deg"].to_numpy())

    db_features = db_feature_array(db)
    db_hr_sets = [hr_set(row) for _, row in db.iterrows()]

    # ---- Checkpoint 1: DB Coverage ----
    # Check whether ANY DB tetrad has all 4 stars fully contained in visible_top12.
    # Jaccard similarity against a 12-star set would max out at 4/12=0.33 even for a
    # perfect tetrad, so containment ("dbs <= visible_hrs") is the right test.
    print("\n--- Checkpoint 1: DB Coverage ---")
    best_overlap = 0
    best_db_idx = -1
    best_db_hrs: frozenset = frozenset()
    for db_idx, dbs in enumerate(db_hr_sets):
        overlap = len(dbs & visible_hrs)
        if overlap > best_overlap:
            best_overlap = overlap
            best_db_idx = db_idx
            best_db_hrs = dbs
    in_top12 = best_db_hrs <= visible_hrs
    in_top8 = best_db_hrs <= top8_hrs
    print(f"  Best DB entry HRs: {sorted(best_db_hrs)}")
    print(f"  Stars matched to visible_top12: {best_overlap}/4")
    print(f"  All 4 stars in top-12? {in_top12}   All 4 in top-8? {in_top8}")
    cp1_pass = in_top12  # full containment means DB covers this field
    print(f"  CHECKPOINT 1: {'PASS' if cp1_pass else 'FAIL'} "
          f"({'fully covered' if cp1_pass else f'only {best_overlap}/4 overlap'})")

    # ---- Checkpoint 2: Feature Encoding Error ----
    print("\n--- Checkpoint 2: Feature Encoding Error ---")
    cp2_pass = False
    best_encoding_error = 999999
    best_combo_rank = None
    if best_db_idx >= 0:
        db_feat = db_features[best_db_idx]
        db_row = db.iloc[best_db_idx]
        match_hrs = frozenset([int(db_row["hr_a"]), int(db_row["hr_b"]), int(db_row["hr_c"]), int(db_row["hr_d"])])
        # Find those stars among visible
        match_ranks = [r for r, row in visible.iterrows() if int(row["HR_clean"]) in match_hrs]
        if len(match_ranks) >= 4:
            match_vecs = vecs[match_ranks[:4]]
            py_feat = python_style_feature(match_vecs)
            c_feat = c_style_feature(match_vecs)
            py_err = l1_distance(py_feat, db_feat) if py_feat is not None else -1
            c_err = l1_distance(c_feat, db_feat) if c_feat is not None else -1
            print(f"  Matching tetrad star ranks: {[r+1 for r in match_ranks[:4]]} (1=brightest)")
            print(f"  DB feature:            {db_feat.tolist()}")
            if py_feat is not None:
                print(f"  Python-style query:    {py_feat.tolist()}  (L1 error vs DB: {py_err})")
            if c_feat is not None:
                print(f"  C-style query:         {c_feat.tolist()}  (L1 error vs DB: {c_err})")
            best_encoding_error = min(v for v in [py_err, c_err] if v >= 0)
            best_combo_rank = [r+1 for r in match_ranks[:4]]
            cp2_pass = best_encoding_error <= CHECKPOINT2_L1_THRESHOLD
        else:
            print(f"  Matching tetrad has only {len(match_ranks)}/4 stars in visible top-12 — cannot compute feature.")
    else:
        print("  No DB entry found — cannot compute feature.")
    print(f"  CHECKPOINT 2: {'PASS' if cp2_pass else 'FAIL'} (best L1={best_encoding_error} vs threshold {CHECKPOINT2_L1_THRESHOLD})")

    # ---- Checkpoint 3: Star Cap ----
    print("\n--- Checkpoint 3: Star Cap (C uses top-8, Python uses top-12) ---")
    cp3_pass = False
    if best_combo_rank is not None:
        in_cap = all(r <= MAX_STARS_C for r in best_combo_rank)
        cp3_pass = in_cap
        print(f"  Matching tetrad star ranks: {best_combo_rank}")
        print(f"  All ranks <= {MAX_STARS_C}? {in_cap}  => C {'CAN' if in_cap else 'CANNOT'} see this tetrad")
    else:
        print("  No matching tetrad found in top-12.")
    print(f"  CHECKPOINT 3: {'PASS' if cp3_pass else 'FAIL'} (needs all ranks <={MAX_STARS_C} for C to try)")

    # ---- Scan: best tetrad reachable within top-8 ----
    print("\n--- Scan: best DB match reachable within C star cap ---")
    best_c_jaccard = 0.0
    best_c_l1 = 999999
    for i, j, k, l in itertools.combinations(range(min(MAX_STARS_C, len(visible))), 4):
        query_vecs = vecs[[i, j, k, l]]
        query_hrs = frozenset([int(visible.iloc[r]["HR_clean"]) for r in [i, j, k, l]])
        c_feat = c_style_feature(query_vecs)
        if c_feat is None:
            continue
        # Find the closest DB entry
        l1_errors = np.abs(db_features.astype(np.int32) - c_feat.astype(np.int32)).sum(axis=1)
        min_idx = int(np.argmin(l1_errors))
        min_l1 = int(l1_errors[min_idx])
        jac = jaccard(query_hrs, db_hr_sets[min_idx])
        if jac > best_c_jaccard or (jac == best_c_jaccard and min_l1 < best_c_l1):
            best_c_jaccard = jac
            best_c_l1 = min_l1
    print(f"  Best tetrad reachable by C (within top-{MAX_STARS_C}): Jaccard={best_c_jaccard:.3f}, min L1={best_c_l1}")

    # ---- Summary ----
    print("\n=== Summary ===")
    print(f"  CP1 DB Coverage:      {'PASS' if cp1_pass else 'FAIL'}")
    print(f"  CP2 Feature Encoding: {'PASS' if cp2_pass else 'FAIL'}")
    print(f"  CP3 Star Cap:         {'PASS' if cp3_pass else 'FAIL'}")

    all_pass = cp1_pass and cp2_pass and cp3_pass
    root_cause = []
    if not cp1_pass:
        root_cause.append("DB has no tetrad covering enough of the visible field stars (DB coverage gap)")
    if cp1_pass and not cp2_pass:
        root_cause.append(f"Feature encoding mismatch exceeds L1={CHECKPOINT2_L1_THRESHOLD} (C double-quantization error)")
    if cp1_pass and cp2_pass and not cp3_pass:
        root_cause.append(f"Correct tetrad needs stars ranked >{MAX_STARS_C} (C star cap too small: increase TETRA_MAX_QUERY_STARS)")

    report_lines = [
        "TETRA Parity Harness Report",
        f"  Field: RA={RA} DEC={DEC} FOV={FOV}",
        f"  CP1 DB Coverage: {'PASS' if cp1_pass else 'FAIL'} (overlap={best_overlap}/4, best reachable Jaccard={best_c_jaccard:.3f})",
        f"  CP2 Feature Encoding: {'PASS' if cp2_pass else 'FAIL'} (L1={best_encoding_error})",
        f"  CP3 Star Cap: {'PASS' if cp3_pass else 'FAIL'} (ranks={best_combo_rank})",
        f"  Overall: {'PASS' if all_pass else 'FAIL'}",
    ]
    if root_cause:
        report_lines.append("  Root cause(s):")
        for rc in root_cause:
            report_lines.append(f"    - {rc}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "divergence_report.txt"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    sep = "\n" + "=" * 60 + "\n"
    report_path.write_text(
        "TETRA Parity Harness\n" + "\n".join(report_lines) + sep + existing,
        encoding="utf-8",
    )
    print(f"\nReport written to {report_path}")

    if root_cause:
        print("\nROOT CAUSE(S):")
        for rc in root_cause:
            print(f"  * {rc}")
    else:
        print("\nAll checkpoints pass - failure is downstream of parity harness scope.")


if __name__ == "__main__":
    main()
