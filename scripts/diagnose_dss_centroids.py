"""
Diagnoses why the DSS real-image solve fails by checking whether the detected
centroids correspond to catalog stars at the KNOWN truth attitude.

It projects all mag<=limit catalog stars onto the image plane using the same
pinhole model the renderer/C identifier use, then matches each detected centroid
to its nearest projected catalog star. It does this for four orientations of the
detected centroids (identity, horizontal mirror, vertical mirror, 180 rotation)
to expose any flip/chirality mismatch between real DSS images and the synthetic
render convention.

Usage:
    python scripts/diagnose_dss_centroids.py \
        --stars outputs/test_dss_stars.csv --ra 83.8 --dec -5.4 --fov 10 --size 877
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from render_catalog_test_image import load_catalog, project_stars


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_centroids(path: Path) -> list[tuple[float, float]]:
    """Reads the detected centroid CSV (index,x,y,brightness)."""
    points: list[tuple[float, float]] = []
    with path.open() as file:
        reader = csv.DictReader(file)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    return points


def transform(point: tuple[float, float], size: int, mode: str) -> tuple[float, float]:
    """Applies one of the four orientation transforms to a pixel coordinate."""
    x, y = point
    if mode == "identity":
        return x, y
    if mode == "mirror_x":
        return (size - 1) - x, y
    if mode == "mirror_y":
        return x, (size - 1) - y
    if mode == "rotate180":
        return (size - 1) - x, (size - 1) - y
    raise ValueError(mode)


def nearest(point: tuple[float, float], projected) -> tuple[float, int]:
    """Returns (distance_px, hr_id) of the nearest projected star to a point."""
    best_dist = float("inf")
    best_hr = -1
    for star in projected:
        dist = math.hypot(point[0] - star.x, point[1] - star.y)
        if dist < best_dist:
            best_dist = dist
            best_hr = star.hr_id
    return best_dist, best_hr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stars", type=Path, default=PROJECT_ROOT / "outputs" / "test_dss_stars.csv")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "catalog.bin")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--fov", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=877)
    parser.add_argument("--magnitude-limit", type=float, default=6.5)
    parser.add_argument("--match-threshold-px", type=float, default=12.0)
    args = parser.parse_args()

    stars = load_catalog(args.catalog, args.magnitude_limit)
    projected = project_stars(stars, args.ra, args.dec, args.size, args.fov)
    centroids = read_centroids(args.stars)

    print(f"catalog mag<={args.magnitude_limit}: {len(stars)} stars")
    print(f"projected into {args.size}px FOV={args.fov} at RA={args.ra} DEC={args.dec}: {len(projected)} stars")
    print(f"detected centroids: {len(centroids)}")
    print()

    for mode in ("identity", "mirror_x", "mirror_y", "rotate180"):
        matches = 0
        total_dist = 0.0
        details = []
        for point in centroids:
            tp = transform(point, args.size, mode)
            dist, hr = nearest(tp, projected)
            if dist <= args.match_threshold_px:
                matches += 1
                total_dist += dist
            details.append((tp, dist, hr))
        mean = (total_dist / matches) if matches else float("nan")
        print(f"[{mode:9s}] matches within {args.match_threshold_px}px: {matches}/{len(centroids)}"
              f"  mean_dist={mean:.2f}px")
        for (tp, dist, hr) in details:
            flag = "OK " if dist <= args.match_threshold_px else "   "
            print(f"    {flag}({tp[0]:6.1f},{tp[1]:6.1f}) -> HR{hr:<5d} dist={dist:7.1f}px")
        print()


if __name__ == "__main__":
    main()
