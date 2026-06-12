from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "cache" / "tetra_fov15_mag6.5_cap40.csv"
GENERATED_TETRA_SOURCE = ROOT / "C" / "generated" / "tetra_db_generated.c"


def feature_code(value: str) -> int:
    """
    Quantizes one normalized TETRA feature to uint16.
    """

    return max(0, min(65535, round(float(value) * 65535)))


def build_kd(tetra_rows: list[dict[str, object]], kd_nodes: list[dict[str, object]]) -> int:
    """
    Builds an array KD-tree and returns the root node index.
    """

    if not tetra_rows:
        return -1
    # Use the widest feature dimension as the split axis to keep the tree balanced.
    split_axis = max(
        range(5),
        key=lambda feature_index: max(row["feature"][feature_index] for row in tetra_rows)
        - min(row["feature"][feature_index] for row in tetra_rows),
    )
    tetra_rows.sort(key=lambda row: row["feature"][split_axis])
    median_index = len(tetra_rows) // 2
    kd_node = tetra_rows[median_index]
    node_index = len(kd_nodes)
    kd_node["axis"] = split_axis
    kd_node["left"] = -1
    kd_node["right"] = -1
    kd_nodes.append(kd_node)
    kd_node["left"] = build_kd(tetra_rows[:median_index], kd_nodes)
    kd_node["right"] = build_kd(tetra_rows[median_index + 1 :], kd_nodes)
    return node_index


def main() -> None:
    """
    Generates the TETRA KD-tree C array from the cached TETRA CSV.
    """

    tetra_rows: list[dict[str, object]] = []
    with INPUT.open(newline="") as file:
        for row in csv.DictReader(file):
            tetra_rows.append(
                {
                    "hr_ids": [int(row["hr_a"]), int(row["hr_b"]), int(row["hr_c"]), int(row["hr_d"])],
                    "feature": [feature_code(row[f"f{feature_number}"]) for feature_number in range(1, 6)],
                }
            )

    kd_nodes: list[dict[str, object]] = []
    build_kd(tetra_rows, kd_nodes)
    GENERATED_TETRA_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_TETRA_SOURCE.open("w", newline="\n") as file:
        file.write("/**\n")
        file.write(" * Generated TETRA array KD-tree database.\n")
        file.write(" * Source: cache/tetra_fov15_mag6.5_cap40.csv\n")
        file.write(" * Do not edit by hand; rerun export_tetra_db.py instead.\n")
        file.write(" */\n")
        file.write('#include "tetra_db.h"\n\n')
        file.write(f"const uint32_t tetra_kd_node_count = {len(kd_nodes)};\n\n")
        file.write("const TetraKdNode tetra_kd_nodes[] = {\n")
        for kd_node in kd_nodes:
            feature_values = ", ".join(str(value) for value in kd_node["feature"])
            hr_ids = ", ".join(str(value) for value in kd_node["hr_ids"])
            file.write(
                f"    {{{{{feature_values}}}, {{{hr_ids}}}, "
                f"{kd_node['left']}, {kd_node['right']}, {kd_node['axis']}}},\n"
            )
        file.write("};\n")

    print(GENERATED_TETRA_SOURCE)


if __name__ == "__main__":
    main()
