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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GENERATED_TETRA_SOURCE = ROOT / "identifier" / "generated" / "tetra_db_generated.c"
FOV_DEG = 10.0
MAX_SEP_RAD = math.radians(FOV_DEG)       # max pairwise edge in a valid tetrad
# Single pass at 5° radius covers all tetrads observable in a 10° FOV.
FIELD_RADII_RAD = [math.radians(5.0)]
MAG_LIMIT_Q100 = 650                      # faintest star allowed as a field member
MAX_FIELD_STARS = 8                       # top-N brightest per field


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


def _build_tetrads(
    bright_vecs: list[tuple[float, float, float]],
    bright_hrs: list[int],
    all_vecs: list[tuple[float, float, float]],
    field_radius_rad: float,
    seen: set[frozenset[int]],
    rows: list[dict],
) -> None:
    """
    Iterates all_vecs as field centres with the given field_radius_rad.
    Appends newly seen 4-combinations into rows, deduplicating via seen.
    Called once per supported FOV radius so both get coverage.
    """
    n_bright = len(bright_vecs)
    n_all = len(all_vecs)
    cos_field = math.cos(field_radius_rad)
    t0 = time.time()

    for centre_idx in range(n_all):
        cx, cy, cz = all_vecs[centre_idx]

        field: list[int] = []
        for i in range(n_bright):
            if len(field) == MAX_FIELD_STARS:
                break
            x, y, z = bright_vecs[i]
            if cx * x + cy * y + cz * z >= cos_field:
                field.append(i)

        if len(field) < 4:
            continue

        for combo in itertools.combinations(field, 4):
            key = frozenset(combo)
            if key in seen:
                continue
            feature = _sorted_edge_feature([bright_vecs[i] for i in combo])
            if feature is None:
                continue
            seen.add(key)
            rows.append({"hr_ids": [bright_hrs[i] for i in combo], "feature": feature})

        elapsed = time.time() - t0
        if (centre_idx + 1) % 1000 == 0 or centre_idx == n_all - 1:
            frac = (centre_idx + 1) / n_all
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0.0
            print(f"    {centre_idx + 1}/{n_all} centres | {len(rows)} tetrads"
                  f" | elapsed {elapsed:.1f}s | eta {eta:.1f}s")


def main() -> None:
    from src.star_tracker_core import load_catalog, unit_vectors

    print("Loading catalog...")
    df_all = load_catalog()
    df_all = df_all.sort_values("Vmag").reset_index(drop=True)

    # Bright stars used as field members (mag <= 6.5)
    df_bright = df_all[df_all["Vmag"] <= MAG_LIMIT_Q100 / 100.0].reset_index(drop=True)
    bright_vecs_np = unit_vectors(df_bright["RA_deg"], df_bright["DEC_deg"])
    bright_hrs = df_bright["HR_clean"].astype(int).tolist()
    bright_vecs = [(float(bright_vecs_np[i, 0]), float(bright_vecs_np[i, 1]), float(bright_vecs_np[i, 2]))
                   for i in range(len(df_bright))]
    print(f"  {len(bright_vecs)} bright stars (mag <= {MAG_LIMIT_Q100 / 100:.1f})")

    # All catalog stars used as field centres (including dim ones, matching batch harness)
    all_vecs_np = unit_vectors(df_all["RA_deg"], df_all["DEC_deg"])
    all_vecs = [(float(all_vecs_np[i, 0]), float(all_vecs_np[i, 1]), float(all_vecs_np[i, 2]))
                for i in range(len(df_all))]
    print(f"  {len(all_vecs)} total catalog stars (field centres)")

    radii_deg = [math.degrees(r) for r in FIELD_RADII_RAD]
    print(f"Building field-coverage TETRA DB (fov={FOV_DEG}°,"
          f" field_radii={radii_deg}°, max_field_stars={MAX_FIELD_STARS})...")
    seen: set[frozenset[int]] = set()
    tetra_rows: list[dict] = []
    for radius_rad in FIELD_RADII_RAD:
        print(f"  Pass field_radius={math.degrees(radius_rad):.1f}°...")
        _build_tetrads(bright_vecs, bright_hrs, all_vecs, radius_rad, seen, tetra_rows)
    print(f"  Unique tetrads after all passes: {len(tetra_rows)}")

    print("Building KD-tree...")
    kd_nodes: list[dict] = []
    _build_kd(tetra_rows, kd_nodes)
    print(f"  KD-tree nodes: {len(kd_nodes)}")

    GENERATED_TETRA_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {GENERATED_TETRA_SOURCE} ...")
    with GENERATED_TETRA_SOURCE.open("w", newline="\n") as f:
        f.write("/**\n")
        f.write(" * Generated TETRA array KD-tree database.\n")
        f.write(f" * Field-coverage: top-{MAX_FIELD_STARS} stars per centre,\n")
        f.write(f" * field_radii={radii_deg}deg,"
                f" max_edge={FOV_DEG}deg, mag<={MAG_LIMIT_Q100 / 100:.1f}.\n")
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
