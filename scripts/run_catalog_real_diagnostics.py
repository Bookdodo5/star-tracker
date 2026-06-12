from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.bin"
REAL_IMAGE_PATH = PROJECT_ROOT / "Centroid" / "test-image" / "10h16m56s-59-51-22.png"
CENTROID_CSV_PATH = PROJECT_ROOT / "Centroid" / "test-image" / "stars.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "catalog_real_diagnostics"

IMAGE_SIZE = 877
CENTER_RA_DEG = (10 + 16 / 60 + 56 / 3600) * 15
CENTER_DEC_ABS_DEG = 59 + 51 / 60 + 22 / 3600
MAGNITUDE_LIMIT = 8.0


@dataclass(frozen=True)
class CatalogStar:
    """
    Stores one parsed catalog star.
    """

    hr_id: int
    ra_deg: float
    dec_deg: float
    magnitude: float


@dataclass(frozen=True)
class CentroidPoint:
    """
    Stores one centroid measured from the image.
    """

    obs_id: int
    x: float
    y: float
    brightness: int


@dataclass(frozen=True)
class PlateMatch:
    """
    Stores one catalog-to-centroid match after plate fitting.
    """

    hr_id: int
    obs_id: int
    error_px: float
    magnitude: float
    brightness: int


@dataclass(frozen=True)
class SimilarityTransform:
    """
    Maps simple catalog plate coordinates to image pixels.
    """

    scale: float
    cos_angle: float
    sin_angle: float
    tx: float
    ty: float


def parse_ra(text: str) -> float | None:
    """
    Converts fixed-width Yale catalog RA text to degrees.
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
    Converts fixed-width Yale catalog DEC text to degrees.
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


def load_catalog() -> list[CatalogStar]:
    """
    Loads bright stars from catalog.bin.
    """

    catalog_stars: list[CatalogStar] = []
    for line in CATALOG_PATH.read_text(errors="ignore").splitlines():
        try:
            hr_id = int(line[:4])
            ra_deg = parse_ra(line[75:83])
            dec_deg = parse_dec(line[83:90])
            magnitude = float(line[102:107])
        except ValueError:
            continue
        if ra_deg is None or dec_deg is None or magnitude > MAGNITUDE_LIMIT:
            continue
        catalog_stars.append(CatalogStar(hr_id, ra_deg, dec_deg, magnitude))
    return sorted(catalog_stars, key=lambda star: star.magnitude)


def load_centroids() -> list[CentroidPoint]:
    """
    Loads the centroid CSV created from the real image.
    """

    centroids: list[CentroidPoint] = []
    with CENTROID_CSV_PATH.open(newline="") as file:
        for row in csv.DictReader(file):
            centroids.append(CentroidPoint(int(row["index"]) - 1, float(row["x"]), float(row["y"]), int(row["brightness"])))
    return centroids


def simple_plate_points(catalog_stars: list[CatalogStar], center_dec_deg: float, fov_deg: float) -> list[dict[str, float | int]]:
    """
    Projects catalog stars into a simple RA/DEC plate for fitting.
    """

    pixels_per_degree = IMAGE_SIZE / fov_deg
    points: list[dict[str, float | int]] = []
    for star in catalog_stars:
        delta_ra_deg = ((star.ra_deg - CENTER_RA_DEG + 180.0) % 360.0) - 180.0
        plate_x = IMAGE_SIZE * 0.5 + delta_ra_deg * math.cos(math.radians(center_dec_deg)) * pixels_per_degree
        plate_y = IMAGE_SIZE * 0.5 - (star.dec_deg - center_dec_deg) * pixels_per_degree
        if -IMAGE_SIZE <= plate_x <= IMAGE_SIZE * 2 and -IMAGE_SIZE <= plate_y <= IMAGE_SIZE * 2:
            points.append({"hr_id": star.hr_id, "x": plate_x, "y": plate_y, "magnitude": star.magnitude})
    return points


def transform_from_pairs(
    first_catalog: dict[str, float | int],
    second_catalog: dict[str, float | int],
    first_centroid: CentroidPoint,
    second_centroid: CentroidPoint,
) -> SimilarityTransform | None:
    """
    Estimates a similarity transform from two catalog-to-centroid assignments.
    """

    catalog_dx = float(second_catalog["x"]) - float(first_catalog["x"])
    catalog_dy = float(second_catalog["y"]) - float(first_catalog["y"])
    centroid_dx = second_centroid.x - first_centroid.x
    centroid_dy = second_centroid.y - first_centroid.y
    catalog_distance = math.hypot(catalog_dx, catalog_dy)
    centroid_distance = math.hypot(centroid_dx, centroid_dy)
    if catalog_distance < 1.0 or centroid_distance < 1.0:
        return None
    scale = centroid_distance / catalog_distance
    cos_angle = (catalog_dx * centroid_dx + catalog_dy * centroid_dy) / (catalog_distance * centroid_distance)
    sin_angle = (catalog_dx * centroid_dy - catalog_dy * centroid_dx) / (catalog_distance * centroid_distance)
    rotated_x = scale * (cos_angle * float(first_catalog["x"]) - sin_angle * float(first_catalog["y"]))
    rotated_y = scale * (sin_angle * float(first_catalog["x"]) + cos_angle * float(first_catalog["y"]))
    return SimilarityTransform(scale, cos_angle, sin_angle, first_centroid.x - rotated_x, first_centroid.y - rotated_y)


def transform_point(transform: SimilarityTransform, x_value: float, y_value: float) -> tuple[float, float]:
    """
    Applies one similarity transform to a catalog plate point.
    """

    return (
        transform.scale * (transform.cos_angle * x_value - transform.sin_angle * y_value) + transform.tx,
        transform.scale * (transform.sin_angle * x_value + transform.cos_angle * y_value) + transform.ty,
    )


def score_transform(
    transform: SimilarityTransform,
    catalog_points: list[dict[str, float | int]],
    centroids: list[CentroidPoint],
    tolerance_px: float,
) -> tuple[int, float, float, list[PlateMatch]]:
    """
    Scores one fitted plate transform by nearest-neighbor matches and brightness consistency.
    """

    matches: list[PlateMatch] = []
    used_obs_ids: set[int] = set()
    for catalog_point in sorted(catalog_points, key=lambda row: float(row["magnitude"])):
        projected_x, projected_y = transform_point(transform, float(catalog_point["x"]), float(catalog_point["y"]))
        nearest = min(centroids, key=lambda point: (point.x - projected_x) ** 2 + (point.y - projected_y) ** 2)
        error_px = math.hypot(nearest.x - projected_x, nearest.y - projected_y)
        if error_px <= tolerance_px and nearest.obs_id not in used_obs_ids:
            used_obs_ids.add(nearest.obs_id)
            matches.append(PlateMatch(int(catalog_point["hr_id"]), nearest.obs_id, error_px, float(catalog_point["magnitude"]), nearest.brightness))
    mean_error = float(np.mean([match.error_px for match in matches])) if matches else float("nan")
    rank_corr = float(np.corrcoef([match.brightness for match in matches], [-match.magnitude for match in matches])[0, 1]) if len(matches) >= 3 else float("nan")
    return len(matches), mean_error, rank_corr, matches


def find_best_fit(
    catalog_points: list[dict[str, float | int]],
    centroids: list[CentroidPoint],
) -> tuple[SimilarityTransform | None, list[PlateMatch]]:
    """
    Finds the best plate fit with scale constrained near the assumed FOV.
    """

    best_transform: SimilarityTransform | None = None
    best_matches: list[PlateMatch] = []
    best_key = (-1, -999.0)
    bright_catalog = sorted(catalog_points, key=lambda row: float(row["magnitude"]))[:15]
    for first_index, first_catalog in enumerate(bright_catalog):
        for second_catalog in bright_catalog[first_index + 1 :]:
            for first_centroid in centroids:
                for second_centroid in centroids:
                    if first_centroid.obs_id == second_centroid.obs_id:
                        continue
                    transform = transform_from_pairs(first_catalog, second_catalog, first_centroid, second_centroid)
                    if transform is None or not 0.8 <= transform.scale <= 1.2:
                        continue
                    count, mean_error, rank_corr, matches = score_transform(transform, catalog_points, centroids, 35.0)
                    key = (count, (rank_corr if not math.isnan(rank_corr) else -2.0) - mean_error * 0.01)
                    if key > best_key:
                        best_key = key
                        best_transform = transform
                        best_matches = matches
    return best_transform, best_matches


def draw_overlay(
    output_path: Path,
    title: str,
    transform: SimilarityTransform | None,
    catalog_points: list[dict[str, float | int]],
    centroids: list[CentroidPoint],
    matches: list[PlateMatch],
) -> None:
    """
    Draws real-image overlay with centroids, projected catalog points, and fitted matches.
    """

    image = Image.open(REAL_IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), title, fill=(255, 255, 0))
    for centroid in centroids:
        draw.ellipse((centroid.x - 8, centroid.y - 8, centroid.x + 8, centroid.y + 8), outline=(0, 255, 255), width=2)
        draw.text((centroid.x + 10, centroid.y + 4), str(centroid.obs_id), fill=(0, 255, 255))
    if transform is not None:
        for catalog_point in sorted(catalog_points, key=lambda row: float(row["magnitude"]))[:60]:
            x_value, y_value = transform_point(transform, float(catalog_point["x"]), float(catalog_point["y"]))
            if 0 <= x_value < IMAGE_SIZE and 0 <= y_value < IMAGE_SIZE:
                draw.rectangle((x_value - 4, y_value - 4, x_value + 4, y_value + 4), outline=(255, 0, 0), width=1)
        for match in matches:
            centroid = next(point for point in centroids if point.obs_id == match.obs_id)
            draw.ellipse((centroid.x - 14, centroid.y - 14, centroid.x + 14, centroid.y + 14), outline=(0, 255, 0), width=3)
            draw.text((centroid.x + 16, centroid.y - 12), f"HR{match.hr_id} {match.error_px:.1f}px", fill=(0, 255, 0))
    image.save(output_path)


def write_summary(rows: list[dict[str, float | int | str]]) -> None:
    """
    Writes plate sweep summary CSV.
    """

    summary_path = OUTPUT_DIR / "plate_sweep_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dec_sign", "fov_deg", "matches", "mean_error_px", "rank_corr", "scale", "rotation_deg"])
        writer.writeheader()
        writer.writerows(rows)
    print(summary_path)


def main() -> None:
    """
    Runs catalog-vs-real diagnostics and writes visual outputs.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog_stars = load_catalog()
    centroids = load_centroids()
    rows: list[dict[str, float | int | str]] = []
    best_by_label: dict[str, tuple[SimilarityTransform | None, list[PlateMatch], list[dict[str, float | int]]]] = {}
    for dec_sign, center_dec_deg in (("positive", CENTER_DEC_ABS_DEG), ("negative", -CENTER_DEC_ABS_DEG)):
        for fov_deg in (10, 15, 20):
            catalog_points = simple_plate_points(catalog_stars, center_dec_deg, fov_deg)
            transform, matches = find_best_fit(catalog_points, centroids)
            mean_error = float(np.mean([match.error_px for match in matches])) if matches else float("nan")
            rank_corr = float(np.corrcoef([match.brightness for match in matches], [-match.magnitude for match in matches])[0, 1]) if len(matches) >= 3 else float("nan")
            rows.append({
                "dec_sign": dec_sign,
                "fov_deg": fov_deg,
                "matches": len(matches),
                "mean_error_px": round(mean_error, 3) if not math.isnan(mean_error) else "",
                "rank_corr": round(rank_corr, 3) if not math.isnan(rank_corr) else "",
                "scale": round(transform.scale, 4) if transform is not None else "",
                "rotation_deg": round(math.degrees(math.atan2(transform.sin_angle, transform.cos_angle)), 3) if transform is not None else "",
            })
            best_by_label[f"{dec_sign}_{fov_deg}"] = (transform, matches, catalog_points)
    write_summary(rows)
    for key in ("positive_10", "negative_10", "negative_20"):
        transform, matches, catalog_points = best_by_label[key]
        draw_overlay(OUTPUT_DIR / f"overlay_{key}.png", key, transform, catalog_points, centroids, matches)
        print(f"{key}: matches={len(matches)}")
    best_row = max(rows, key=lambda row: (int(row["matches"]), float(row["rank_corr"]) if row["rank_corr"] != "" else -2.0))
    print("best", best_row)


if __name__ == "__main__":
    main()
