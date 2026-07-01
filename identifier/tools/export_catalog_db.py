from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GENERATED_CATALOG_SOURCE = ROOT / "identifier" / "generated" / "catalog_db_generated.c"
MAG_LIMIT = float(os.environ.get("STAR_DB_MAG", "7.5"))  # shared with export_tetra_db.py


def q15(value: float) -> int:
    """
    Quantizes one unit-vector component to signed Q15.
    """

    return max(-32767, min(32767, round(value * 32767)))


def _build_catalog_kd(
    entries: list[tuple[float, float, float, int]],
    kd_nodes: list[dict],
) -> int:
    """
    Recursively builds a flat KD-tree from (x, y, z, hr) catalog entries.
    Returns the index of the root node in kd_nodes, or -1 for an empty list.
    Each node is added to kd_nodes in pre-order so the root lands at index 0.
    """
    if not entries:
        return -1

    axis = max(
        range(3),
        key=lambda d: max(e[d] for e in entries) - min(e[d] for e in entries),
    )
    entries.sort(key=lambda e: e[axis])
    mid = len(entries) // 2
    node_index = len(kd_nodes)
    node: dict = {
        "x": entries[mid][0],
        "y": entries[mid][1],
        "z": entries[mid][2],
        "hr": entries[mid][3],
        "axis": axis,
        "left": -1,
        "right": -1,
    }
    kd_nodes.append(node)
    node["left"] = _build_catalog_kd(entries[:mid], kd_nodes)
    node["right"] = _build_catalog_kd(entries[mid + 1 :], kd_nodes)
    return node_index


def main() -> None:
    """
    Generates the C catalog array, HR lookup table, and KD-tree for fast nearest-star lookup.
    """

    from src.star_tracker_core import load_db_catalog

    print(f"Loading Tycho-2 catalog (V <= {MAG_LIMIT})...")
    df = load_db_catalog(MAG_LIMIT)
    catalog_rows: list[tuple[int, int, int, int, int]] = []
    kd_entries: list[tuple[float, float, float, int]] = []
    for hr_id, ra_degrees, dec_degrees, visual_magnitude in zip(
        df["HR_clean"].astype(int), df["RA_deg"], df["DEC_deg"], df["Vmag"]
    ):
        ra_radians = math.radians(ra_degrees)
        dec_radians = math.radians(dec_degrees)
        unit_x = math.cos(dec_radians) * math.cos(ra_radians)
        unit_y = math.cos(dec_radians) * math.sin(ra_radians)
        unit_z = math.sin(dec_radians)
        catalog_rows.append((hr_id, q15(unit_x), q15(unit_y), q15(unit_z), round(visual_magnitude * 100)))
        kd_entries.append((unit_x, unit_y, unit_z, hr_id))

    hr_lookup_count = max(hr_id for hr_id, *_ in catalog_rows) + 1
    hr_to_catalog_index = [0xFFFF] * hr_lookup_count
    for catalog_index, (hr_id, *_rest) in enumerate(catalog_rows):
        hr_to_catalog_index[hr_id] = catalog_index

    print(f"Building KD-tree over {len(kd_entries)} catalog stars...")
    kd_nodes: list[dict] = []
    _build_catalog_kd(kd_entries, kd_nodes)
    print(f"  KD-tree nodes: {len(kd_nodes)}")

    GENERATED_CATALOG_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_CATALOG_SOURCE.open("w", newline="\n") as file:
        file.write("/**\n")
        file.write(" * Generated catalog database with KD-tree for nearest-star lookup.\n")
        file.write(f" * Source: data/tycho2.csv (Tycho-2, V <= {MAG_LIMIT})\n")
        file.write(" * Do not edit by hand; rerun export_catalog_db.py instead.\n")
        file.write(" */\n")
        file.write('#include "catalog_db.h"\n\n')
        file.write(f"const uint16_t catalog_star_count = {len(catalog_rows)};\n")
        file.write(f"const uint16_t hr_to_catalog_index_count = {hr_lookup_count};\n\n")
        file.write("const CatalogStar catalog_stars[] = {\n")
        for hr_id, unit_x_q15, unit_y_q15, unit_z_q15, magnitude_q100 in catalog_rows:
            file.write(f"    {{{hr_id}, {unit_x_q15}, {unit_y_q15}, {unit_z_q15}, {magnitude_q100}}},\n")
        file.write("};\n\n")
        file.write("const uint16_t hr_to_catalog_index[] = {\n")
        for chunk_start in range(0, len(hr_to_catalog_index), 12):
            file.write("    " + ", ".join(str(index) for index in hr_to_catalog_index[chunk_start : chunk_start + 12]) + ",\n")
        file.write("};\n\n")
        file.write(f"const uint32_t catalog_kd_node_count = {len(kd_nodes)};\n\n")
        file.write("const CatalogKdNode catalog_kd_nodes[] = {\n")
        for node in kd_nodes:
            file.write(
                f"    {{{node['x']:.9f}f, {node['y']:.9f}f, {node['z']:.9f}f, 0.0f,"
                f" {node['left']}, {node['right']}, {node['hr']}, {node['axis']}, 0}},\n"
            )
        file.write("};\n")

    print(GENERATED_CATALOG_SOURCE)


if __name__ == "__main__":
    main()
