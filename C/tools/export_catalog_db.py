from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "catalog.bin"
GENERATED_CATALOG_SOURCE = ROOT / "C" / "generated" / "catalog_db_generated.c"


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


def q15(value: float) -> int:
    """
    Quantizes one unit-vector component to signed Q15.
    """

    return max(-32767, min(32767, round(value * 32767)))


def main() -> None:
    """
    Generates the C catalog array and HR lookup table.
    """

    catalog_rows: list[tuple[int, int, int, int, int]] = []
    for line in CATALOG.read_text(errors="ignore").splitlines():
        try:
            hr_id = int(line[0:4])
            ra_degrees = parse_ra(line[75:83])
            dec_degrees = parse_dec(line[83:90])
            visual_magnitude = float(line[102:107])
        except ValueError:
            continue
        if ra_degrees is None or dec_degrees is None:
            continue
        ra_radians = math.radians(ra_degrees)
        dec_radians = math.radians(dec_degrees)
        unit_x = math.cos(dec_radians) * math.cos(ra_radians)
        unit_y = math.cos(dec_radians) * math.sin(ra_radians)
        unit_z = math.sin(dec_radians)
        catalog_rows.append((hr_id, q15(unit_x), q15(unit_y), q15(unit_z), round(visual_magnitude * 100)))

    hr_lookup_count = max(hr_id for hr_id, *_ in catalog_rows) + 1
    hr_to_catalog_index = [0xFFFF] * hr_lookup_count
    for catalog_index, (hr_id, *_rest) in enumerate(catalog_rows):
        hr_to_catalog_index[hr_id] = catalog_index

    GENERATED_CATALOG_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_CATALOG_SOURCE.open("w", newline="\n") as file:
        file.write("/**\n")
        file.write(" * Generated catalog database.\n")
        file.write(" * Source: data/catalog.bin\n")
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
        file.write("};\n")

    print(GENERATED_CATALOG_SOURCE)


if __name__ == "__main__":
    main()
