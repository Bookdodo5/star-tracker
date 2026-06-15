"""
Real-image batch benchmark: fetches DSS2 Red images for many sky fields, runs the
full pipeline (centroid extraction -> TETRA/Pyramid identification), and reports the
fraction of fields each algorithm solves to within an angular tolerance of truth.

Unlike batch_synthetic_compare (which builds observed vectors directly from the
catalog), this exercises the real image path end to end, including centroiding and
the camera model.

Fetched images are cached under cache/real_images/ so re-runs do not re-download.

Usage:
    python scripts/batch_real_image_compare.py --count 12 --fov 10 --tolerance-deg 0.5
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.star_tracker_core import load_catalog, unit_vectors
from fetch_dss_image import fetch_image, image_to_ppm

DEMO_EXE = ROOT / "identifier" / "build-generated-release" / "demo_centroid_compare.exe"
CENTROID_EXE = ROOT / "centroid" / "build-mingw" / "centroid_extract.exe"
CACHE_DIR = ROOT / "cache" / "real_images"
SCRATCH_DIR = ROOT / "cache" / "real_batch_scratch"

ATTITUDE_RE = re.compile(
    r"(TETRA|Pyramid) attitude_ra_deg=(-?\d+\.?\d*) attitude_dec_deg=(-?\d+\.?\d*)"
)
SUCCESS_RE = re.compile(r"(TETRA|Pyramid) success=(true|false)")


def angular_error_deg(ra_a: float, dec_a: float, ra_b: float, dec_b: float) -> float:
    """Returns the angular separation in degrees between two RA/DEC directions."""
    va = unit_vectors(np.array([ra_a]), np.array([dec_a]))[0]
    vb = unit_vectors(np.array([ra_b]), np.array([dec_b]))[0]
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def select_field_centers(count: int, fov_deg: float, min_field_stars: int, seed: int):
    """
    Selects field centers (RA, DEC) from catalog stars whose surrounding field holds
    at least min_field_stars bright (mag<=6.5) stars, so identification is feasible.
    """
    df = load_catalog()
    bright = df[df["Vmag"] <= 6.5].reset_index(drop=True)
    bright_vecs = unit_vectors(bright["RA_deg"].to_numpy(), bright["DEC_deg"].to_numpy())
    cos_radius = math.cos(math.radians(fov_deg * 0.5))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bright))
    centers = []
    for idx in order:
        if len(centers) == count:
            break
        center_vec = bright_vecs[idx]
        dots = bright_vecs @ center_vec
        field_count = int(np.count_nonzero(dots >= cos_radius))
        if field_count >= min_field_stars:
            centers.append((float(bright["RA_deg"].iloc[idx]), float(bright["DEC_deg"].iloc[idx])))
    return centers


def fetch_cached_ppm(ra: float, dec: float, fov_deg: float, pixels: int) -> Path:
    """Fetches (or reuses a cached) DSS PPM for one field and returns its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"ra{ra:+09.4f}_dec{dec:+09.4f}_fov{fov_deg:g}_px{pixels}".replace("+", "p").replace("-", "m")
    ppm_path = CACHE_DIR / f"{tag}.ppm"
    if ppm_path.exists() and ppm_path.stat().st_size > 0:
        return ppm_path
    image = fetch_image(ra, dec, fov_deg, pixels)
    ppm_path.write_bytes(image_to_ppm(image))
    return ppm_path


def run_pipeline(ppm_path: Path, size: int, fov_deg: float):
    """
    Runs centroid extraction + identification on one PPM and returns a dict with
    detected star count and per-algorithm (success, ra, dec), or None on failure.
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    stars_csv = SCRATCH_DIR / (ppm_path.stem + "_stars.csv")
    centroid = subprocess.run(
        [str(CENTROID_EXE), str(ppm_path), str(stars_csv)],
        cwd=str(SCRATCH_DIR), capture_output=True, text=True,
    )
    if centroid.returncode != 0 or not stars_csv.exists():
        return None

    demo = subprocess.run(
        [str(DEMO_EXE), str(stars_csv), str(size), str(size), str(fov_deg)],
        cwd=str(SCRATCH_DIR), capture_output=True, text=True,
    )
    out = demo.stdout
    detected = 0
    match = re.search(r"Detected stars: (\d+)", out)
    if match:
        detected = int(match.group(1))

    result = {"detected": detected,
              "TETRA": {"success": False, "ra": None, "dec": None},
              "Pyramid": {"success": False, "ra": None, "dec": None}}
    for algo, flag in SUCCESS_RE.findall(out):
        result[algo]["success"] = (flag == "true")
    for algo, ra_s, dec_s in ATTITUDE_RE.findall(out):
        result[algo]["ra"] = float(ra_s)
        result[algo]["dec"] = float(dec_s)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-image batch benchmark over DSS fields")
    parser.add_argument("--count", type=int, default=12, help="number of fields to test")
    parser.add_argument("--fov", type=float, default=10.0, help="field of view in degrees")
    parser.add_argument("--size", type=int, default=877, help="image side length in pixels")
    parser.add_argument("--tolerance-deg", type=float, default=0.5, help="max attitude error to count as correct")
    parser.add_argument("--min-field-stars", type=int, default=8, help="min bright stars required in a field")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "real_batch_latest.csv")
    args = parser.parse_args()

    if not DEMO_EXE.exists() or not CENTROID_EXE.exists():
        raise SystemExit("Build the C and Centroid targets first (run.ps1 build).")

    print(f"Selecting {args.count} field centers (FOV={args.fov}, >= {args.min_field_stars} bright stars)...")
    centers = select_field_centers(args.count, args.fov, args.min_field_stars, args.seed)
    print(f"  selected {len(centers)} fields")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_lines = ["index,ra,dec,detected,tetra_success,tetra_err_deg,tetra_correct,"
                 "pyramid_success,pyramid_err_deg,pyramid_correct"]

    valid = 0
    tetra_correct = 0
    pyramid_correct = 0
    fetch_failures = 0
    start = time.time()

    for i, (ra, dec) in enumerate(centers):
        try:
            ppm_path = fetch_cached_ppm(ra, dec, args.fov, args.size)
        except Exception as exc:  # network / SkyView errors should not abort the batch
            print(f"  field {i + 1}/{len(centers)} RA={ra:.2f} DEC={dec:.2f}: FETCH FAILED ({exc})")
            fetch_failures += 1
            continue

        result = run_pipeline(ppm_path, args.size, args.fov)
        if result is None or result["detected"] < 4:
            print(f"  field {i + 1}/{len(centers)} RA={ra:.2f} DEC={dec:.2f}: too few centroids, skipped")
            continue
        valid += 1

        row = {}
        for algo in ("TETRA", "Pyramid"):
            info = result[algo]
            correct = False
            err = float("nan")
            if info["success"] and info["ra"] is not None:
                err = angular_error_deg(ra, dec, info["ra"], info["dec"])
                correct = err <= args.tolerance_deg
            row[algo] = (info["success"], err, correct)
        if row["TETRA"][2]:
            tetra_correct += 1
        if row["Pyramid"][2]:
            pyramid_correct += 1

        csv_lines.append(
            f"{i},{ra:.4f},{dec:.4f},{result['detected']},"
            f"{row['TETRA'][0]},{row['TETRA'][1]:.4f},{row['TETRA'][2]},"
            f"{row['Pyramid'][0]},{row['Pyramid'][1]:.4f},{row['Pyramid'][2]}"
        )

        elapsed = time.time() - start
        done = i + 1
        eta = elapsed / done * (len(centers) - done)
        print(f"  field {done}/{len(centers)} RA={ra:7.2f} DEC={dec:7.2f} detected={result['detected']:2d} "
              f"| TETRA {'OK ' if row['TETRA'][2] else 'no '}({row['TETRA'][1]:.3f}deg) "
              f"Pyramid {'OK ' if row['Pyramid'][2] else 'no '}({row['Pyramid'][1]:.3f}deg) "
              f"| elapsed {elapsed:.0f}s eta {eta:.0f}s")

    args.output.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print("\n=== Real-image batch summary ===")
    print(f"  fields requested: {len(centers)}   fetch failures: {fetch_failures}   valid (>=4 centroids): {valid}")
    if valid > 0:
        print(f"  TETRA   accuracy: {tetra_correct}/{valid} = {tetra_correct * 100.0 / valid:.1f}% "
              f"(within {args.tolerance_deg} deg)")
        print(f"  Pyramid accuracy: {pyramid_correct}/{valid} = {pyramid_correct * 100.0 / valid:.1f}% "
              f"(within {args.tolerance_deg} deg)")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
