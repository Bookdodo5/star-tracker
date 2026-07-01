"""
Generates the TETRA KD-tree C array using field-coverage enumeration.

For every star in the full catalog (including dim ones used as field centres
by batch_synthetic_compare), this script simulates the same visible-star
collection that the C identifier uses: top MAX_FIELD_STARS brightest mag<=6.5
stars within FIELD_RADIUS_RAD (= FOV/2 = 7.5 deg).  Every 4-combination from
that field is added to the database.

Why this beats the old anchor+cap=100 scheme:
  anchor+cap cuts off valid combinations at iteration position ~21 000 in
  combinations(98 neighbours, 3) while cap stops at 100, so most observable
  tetrads were never generated.
"""
from __future__ import annotations

import itertools
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GENERATED_TETRA_SOURCE = ROOT / "identifier" / "generated" / "tetra_db_generated.c"

# Anchored all-combinations generation (each tetra is owned by its BRIGHTEST star = anchor):
#   - anchors: stars brighter than BMC (so even sparse fields, whose brightest is faint, get one).
#   - members: stars brighter than L (matches the 7.5 validity reference).
#   - gather radius R ~= FOV HALF-DIAGONAL so the companion pool is in-frame (avoids the
#     out-of-frame "annulus" that makes brightest-K select the wrong stars); every star is an
#     anchor, so edge-anchor fields are covered via a more central anchor's compact subset.
#   - max tetra edge <= FOV DIAGONAL so the 4 stars fit one frame.
# Swept knobs are env-overridable so the size sweep rebuilds without editing source.
FOV_W = float(os.environ.get("STAR_DB_FOV_W", "7.0"))
FOV_H = float(os.environ.get("STAR_DB_FOV_H", "4.0"))
FOV_DIAG_DEG = math.hypot(FOV_W, FOV_H)
MAG_LIMIT = float(os.environ.get("STAR_DB_MAG", "7.5"))         # L: faintest tetra member
BMC = float(os.environ.get("STAR_DB_BMC", "7.0"))              # faintest anchor (brightest star)
MAX_FIELD_STARS = int(os.environ.get("STAR_DB_FIELDSTARS", "9"))   # K_high: dimmer neighbours (sparse anchors)
GATHER_RAD = math.radians(float(os.environ.get("STAR_DB_RADIUS", "3.5")))  # ~half-diagonal (7x4 winner)
# Adaptive-K: anchors with >DENSITY_THRESH dimmer neighbours in gather radius use K_LOW instead of K_HIGH.
# Dense sky regions are covered by many overlapping anchors; sparser combos per anchor still maintain >=99%.
DENSITY_THRESH = int(os.environ.get("STAR_DB_DENSITY_THRESH", "15"))   # neighbour count threshold
K_LOW = int(os.environ.get("STAR_DB_K_LOW", "6"))                       # K for dense anchors
MAX_SEP_RAD = math.radians(FOV_DIAG_DEG)   # max pairwise edge: 4 stars must fit one frame


def _ang(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    dot = ax * bx + ay * by + az * bz
    return math.acos(max(-1.0, min(1.0, dot)))


def _sorted_edge_feature(vecs: list[tuple[float, float, float]]) -> list[int] | None:
    """Five shortest-to-longest ratios of 6 pairwise edges, quantised to uint16.
    Returns None for degenerate or too-wide tetrads (max edge > MAX_SEP_RAD)."""
    edges = sorted(
        _ang(vecs[i][0], vecs[i][1], vecs[i][2], vecs[j][0], vecs[j][1], vecs[j][2])
        for i in range(4)
        for j in range(i + 1, 4)
    )
    if edges[5] == 0.0 or edges[5] > MAX_SEP_RAD:
        return None
    return [min(65535, int(e / edges[5] * 65535 + 0.5)) for e in edges[:5]]


def _build_kd(tetra_rows: list[dict], kd_nodes: list[dict]) -> int:
    if not tetra_rows:
        return -1
    split_axis = max(
        range(5),
        key=lambda k: max(r["feature"][k] for r in tetra_rows) - min(r["feature"][k] for r in tetra_rows),
    )
    tetra_rows.sort(key=lambda r: r["feature"][split_axis])
    median_index = len(tetra_rows) // 2
    node = tetra_rows[median_index]
    node_index = len(kd_nodes)
    node["axis"] = split_axis
    node["left"] = -1
    node["right"] = -1
    kd_nodes.append(node)
    node["left"] = _build_kd(tetra_rows[:median_index], kd_nodes)
    node["right"] = _build_kd(tetra_rows[median_index + 1 :], kd_nodes)
    return node_index


def main() -> None:
    from src.star_tracker_core import load_db_catalog, unit_vectors, anchored_allcombos_tetrads

    print(f"Loading Tycho-2 catalog (members V <= {MAG_LIMIT})...")
    df_all = load_db_catalog(MAG_LIMIT)  # sorted brightest-first; index = brightness rank

    member_vecs_np = unit_vectors(df_all["RA_deg"], df_all["DEC_deg"])
    member_hrs = df_all["HR_clean"].astype(int).tolist()
    member_vecs = [(float(member_vecs_np[i, 0]), float(member_vecs_np[i, 1]), float(member_vecs_np[i, 2]))
                   for i in range(len(df_all))]
    n_anchors = int((df_all["Vmag"] <= BMC).sum())  # anchors = prefix brighter than BMC
    print(f"  {len(member_vecs)} members (V<={MAG_LIMIT}); {n_anchors} anchors (V<={BMC}); "
          f"R_gather={math.degrees(GATHER_RAD):.2f}° max_edge={FOV_DIAG_DEG:.2f}° K={MAX_FIELD_STARS}")

    print(f"Generating anchored all-combos tetrads (adaptive: thresh>{DENSITY_THRESH} → K={K_LOW})...")
    combos = anchored_allcombos_tetrads(member_vecs_np, n_anchors, GATHER_RAD, MAX_SEP_RAD, MAX_FIELD_STARS,
                                         density_thresh=DENSITY_THRESH, k_low=K_LOW)
    tetra_rows: list[dict] = []
    for combo in combos:
        feature = _sorted_edge_feature([member_vecs[i] for i in combo])
        if feature is None:
            continue
        tetra_rows.append({"hr_ids": [member_hrs[i] for i in combo], "feature": feature})
    print(f"  Tetrads: {len(tetra_rows)}")

    print("Building KD-tree...")
    kd_nodes: list[dict] = []
    _build_kd(tetra_rows, kd_nodes)
    print(f"  KD-tree nodes: {len(kd_nodes)}")

    GENERATED_TETRA_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {GENERATED_TETRA_SOURCE} ...")
    with GENERATED_TETRA_SOURCE.open("w", newline="\n") as f:
        f.write("/**\n")
        f.write(" * Generated TETRA array KD-tree database (anchored generation).\n")
        f.write(f" * anchors V<={BMC}, members V<={MAG_LIMIT}, K={MAX_FIELD_STARS},\n")
        f.write(f" * gather_radius={FOV_DIAG_DEG:.2f}deg, max_edge={FOV_DIAG_DEG:.2f}deg"
                f" (FOV {FOV_W}x{FOV_H}).\n")
        f.write(" * Do not edit by hand; rerun export_tetra_db.py instead.\n")
        f.write(" */\n")
        f.write('#include "tetra_db.h"\n\n')
        f.write(f"const uint32_t tetra_kd_node_count = {len(kd_nodes)};\n\n")
        f.write("const TetraKdNode tetra_kd_nodes[] = {\n")
        for node in kd_nodes:
            feat = ", ".join(str(v) for v in node["feature"])
            hrs = ", ".join(str(v) for v in node["hr_ids"])
            f.write(f"    {{{{{feat}}}, {{{hrs}}}, {node['left']}, {node['right']}, {node['axis']}}},\n")
        f.write("};\n")

    print(GENERATED_TETRA_SOURCE)


if __name__ == "__main__":
    main()
