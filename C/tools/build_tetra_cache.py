from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "catalog.bin"
OUTPUT = ROOT / "cache" / "tetra_fov15_mag6.5_near60_cap80.csv"
MAX_FOV_DEG = 15.0
MAX_MAG = 6.5
MAX_LOCAL_NEIGHBORS = 60
MAX_TETRAS_PER_ANCHOR = 80


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


def load_catalog() -> list[tuple[int, float, tuple[float, float, float]]]:
    """
    Loads bright catalog rows as HR, magnitude, and unit-vector values.
    """

    rows: list[tuple[int, float, tuple[float, float, float]]] = []
    for line in CATALOG.read_text(errors="ignore").splitlines():
        try:
            hr_id = int(line[0:4])
            ra_degrees = parse_ra(line[75:83])
            dec_degrees = parse_dec(line[83:90])
            magnitude = float(line[102:107])
        except ValueError:
            continue
        if ra_degrees is None or dec_degrees is None or magnitude > MAX_MAG:
            continue
        ra_radians = math.radians(ra_degrees)
        dec_radians = math.radians(dec_degrees)
        rows.append(
            (
                hr_id,
                magnitude,
                (
                    math.cos(dec_radians) * math.cos(ra_radians),
                    math.cos(dec_radians) * math.sin(ra_radians),
                    math.sin(dec_radians),
                ),
            )
        )
    return sorted(rows, key=lambda row: row[1])


def angular_distance(first_vector: tuple[float, float, float], second_vector: tuple[float, float, float]) -> float:
    """
    Computes angular distance between two unit vectors.
    """

    dot_product = sum(first_vector[axis] * second_vector[axis] for axis in range(3))
    return math.acos(max(-1.0, min(1.0, dot_product)))


def tetra_feature(vectors: list[tuple[float, float, float]]) -> list[float] | None:
    """
    Computes the normalized five-edge TETRA feature.
    """

    edges = sorted(angular_distance(vectors[first_index], vectors[second_index]) for first_index, second_index in itertools.combinations(range(4), 2))
    if edges[-1] <= 0.0:
        return None
    return [edge / edges[-1] for edge in edges[:5]]


def main() -> None:
    """
    Builds the 15-degree TETRA cache CSV used by the C exporter.
    """

    stars = load_catalog()
    max_separation = math.radians(MAX_FOV_DEG)
    rows: list[dict[str, float | int]] = []
    for anchor_index, (anchor_hr_id, _anchor_mag, anchor_vector) in enumerate(stars):
        neighbor_distances = [
            (star_index, angular_distance(anchor_vector, stars[star_index][2]))
            for star_index in range(anchor_index + 1, len(stars))
        ]
        neighbor_indices = [
            star_index
            for star_index, distance in sorted(neighbor_distances, key=lambda item: (item[1], stars[item[0]][1]))
            if distance <= max_separation
        ][:MAX_LOCAL_NEIGHBORS]
        added_count = 0
        for combo in itertools.combinations(neighbor_indices, 3):
            if added_count >= MAX_TETRAS_PER_ANCHOR:
                break
            indices = (anchor_index, *combo)
            vectors = [stars[index][2] for index in indices]
            all_edges = [angular_distance(vectors[first], vectors[second]) for first, second in itertools.combinations(range(4), 2)]
            if max(all_edges) > max_separation:
                continue
            feature = tetra_feature(vectors)
            if feature is None:
                continue
            rows.append(
                {
                    "hr_a": anchor_hr_id,
                    "hr_b": stars[combo[0]][0],
                    "hr_c": stars[combo[1]][0],
                    "hr_d": stars[combo[2]][0],
                    "f1": feature[0],
                    "f2": feature[1],
                    "f3": feature[2],
                    "f4": feature[3],
                    "f5": feature[4],
                }
            )
            added_count += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["hr_a", "hr_b", "hr_c", "hr_d", "f1", "f2", "f3", "f4", "f5"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{OUTPUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
