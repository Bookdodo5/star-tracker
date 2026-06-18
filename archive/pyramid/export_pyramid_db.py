from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "catalog.bin"
GENERATED_PYRAMID_SOURCE = ROOT / "identifier" / "generated" / "pyramid_db_generated.c"
MAX_FOV_DEG = 10.0
MAX_SEP_RAD = math.radians(MAX_FOV_DEG)
MAX_MAG_Q100 = 650


def sep_code(sep_rad: float) -> int:
    """
    Quantizes a pair separation to the Pyramid uint16 code scale.
    """

    return max(0, min(65535, round(sep_rad / MAX_SEP_RAD * 65535)))


def parse_ra(text: str) -> float | None:
    """
    Converts fixed-width catalog RA text to degrees.
    """

    text = text.strip()
    if len(text) < 6:
        return None
    try:
        return (float(text[:2]) + float(text[2:4]) / 60.0 + float(text[4:]) / 3600.0) * 15.0
    except ValueError:
        return None


def parse_dec(text: str) -> float | None:
    """
    Converts fixed-width catalog DEC text to degrees.
    """

    text = text.strip()
    if len(text) < 6:
        return None
    try:
        sign = -1.0 if text[0] == "-" else 1.0
        body = text[1:] if text[0] in "+-" else text
        return sign * (float(body[:2]) + float(body[2:4]) / 60.0 + float(body[4:]) / 3600.0)
    except ValueError:
        return None


def load_bright_catalog() -> list[tuple[int, float, float, float]]:
    """
    Loads catalog stars used by the Pyramid database as HR and unit-vector rows.
    """

    rows: list[tuple[int, float, float, float]] = []
    for line in CATALOG.read_text(errors="ignore").splitlines():
        try:
            hr_id = int(line[0:4])
            ra_degrees = parse_ra(line[75:83])
            dec_degrees = parse_dec(line[83:90])
            magnitude_q100 = round(float(line[102:107]) * 100)
        except ValueError:
            continue
        if ra_degrees is None or dec_degrees is None or magnitude_q100 > MAX_MAG_Q100:
            continue
        ra_radians = math.radians(ra_degrees)
        dec_radians = math.radians(dec_degrees)
        rows.append(
            (
                hr_id,
                math.cos(dec_radians) * math.cos(ra_radians),
                math.cos(dec_radians) * math.sin(ra_radians),
                math.sin(dec_radians),
            )
        )
    return rows


def build_full_pairs(catalog_rows: list[tuple[int, float, float, float]]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], list[int]]:
    """
    Builds full seed pairs and full symmetric HR adjacency rows for Pyramid lookup.
    """

    max_hr_id = max(hr_id for hr_id, *_vector in catalog_rows)
    pair_rows: list[tuple[int, int, int]] = []
    neighbors_by_hr: list[list[tuple[int, int]]] = [[] for _ in range(max_hr_id + 1)]
    min_dot = math.cos(MAX_SEP_RAD)
    for first_index, (first_hr_id, first_x, first_y, first_z) in enumerate(catalog_rows):
        for second_hr_id, second_x, second_y, second_z in catalog_rows[first_index + 1 :]:
            dot_product = first_x * second_x + first_y * second_y + first_z * second_z
            if dot_product < min_dot:
                continue
            dot_product = max(-1.0, min(1.0, dot_product))
            separation_code = sep_code(math.acos(dot_product))
            pair_rows.append((first_hr_id, second_hr_id, separation_code))
            neighbors_by_hr[first_hr_id].append((second_hr_id, separation_code))
            neighbors_by_hr[second_hr_id].append((first_hr_id, separation_code))

    flat_neighbors: list[tuple[int, int]] = []
    neighbor_starts: list[int] = []
    for hr_neighbors in neighbors_by_hr:
        neighbor_starts.append(len(flat_neighbors))
        flat_neighbors.extend(sorted(hr_neighbors, key=lambda item: item[1]))
    neighbor_starts.append(len(flat_neighbors))
    pair_rows.sort(key=lambda item: item[2])
    return pair_rows, flat_neighbors, neighbor_starts


def main() -> None:
    """
    Generates the sorted Pyramid pair database C array.
    """

    pair_rows, neighbor_rows, neighbor_starts = build_full_pairs(load_bright_catalog())

    GENERATED_PYRAMID_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_PYRAMID_SOURCE.open("w", newline="\n") as file:
        file.write("/**\n")
        file.write(" * Generated Pyramid pair database sorted by sep_code.\n")
        file.write(f" * Source: data/catalog.bin full pairs filtered to {MAX_FOV_DEG:g} deg.\n")
        file.write(" * Do not edit by hand; rerun export_pyramid_db.py instead.\n")
        file.write(" */\n")
        file.write('#include "pyramid_db.h"\n\n')
        file.write(f"const float pyramid_max_sep_rad = {MAX_SEP_RAD:.9f}f;\n")
        file.write(f"const uint32_t pyramid_pair_count = {len(pair_rows)};\n\n")
        file.write("const PairRow pyramid_pairs_by_sep[] = {\n")
        for first_hr_id, second_hr_id, separation_code in pair_rows:
            file.write(f"    {{{first_hr_id}, {second_hr_id}, {separation_code}}},\n")
        file.write("};\n")
        file.write(f"\nconst uint32_t pyramid_neighbor_count = {len(neighbor_rows)};\n\n")
        file.write("const PairNeighbor pyramid_neighbors_by_hr[] = {\n")
        for neighbor_hr_id, separation_code in neighbor_rows:
            file.write(f"    {{{neighbor_hr_id}, {separation_code}}},\n")
        file.write("};\n\n")
        file.write(f"const uint32_t pyramid_neighbor_start_count = {len(neighbor_starts)};\n\n")
        file.write("const uint32_t pyramid_neighbor_starts[] = {\n")
        for chunk_start in range(0, len(neighbor_starts), 12):
            values = ", ".join(str(value) for value in neighbor_starts[chunk_start : chunk_start + 12])
            file.write(f"    {values},\n")
        file.write("};\n")

    print(GENERATED_PYRAMID_SOURCE)


if __name__ == "__main__":
    main()
