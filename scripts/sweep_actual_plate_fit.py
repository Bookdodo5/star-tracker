from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_actual_image.py"


def run_fit(stars_csv_path: Path, center_dec: float, fov: float) -> tuple[int, float, float, float]:
    """
    Runs the actual-image evaluator and extracts match count, error, scale, and rotation.
    """

    command = [
        "python",
        str(EVALUATOR),
        "--stars",
        str(stars_csv_path),
        "--image-size",
        "877",
        "--fov",
        str(fov),
        "--center-dec",
        str(center_dec),
        "--tolerance-px",
        "35",
        "--min-scale",
        "0.8",
        "--max-scale",
        "1.2",
    ]
    stdout = subprocess.run(command, cwd=PROJECT_ROOT, check=True, text=True, capture_output=True).stdout
    matched_count = 0
    mean_error_px = 1e9
    scale = 0.0
    rotation_deg = 0.0
    for line in stdout.splitlines():
        if line.startswith("matched_count="):
            parts = dict(part.split("=") for part in line.split())
            matched_count = int(parts["matched_count"])
            mean_error_px = float(parts["mean_error_px"])
            scale = float(parts["scale"])
            rotation_deg = float(parts["rotation_deg"])
    return matched_count, mean_error_px, scale, rotation_deg


def main() -> None:
    """
    Sweeps plausible FOV and DEC sign values to test whether the real image matches catalog.bin.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--stars", required=True, type=Path)
    args = parser.parse_args()

    base_dec = 59 + 51 / 60 + 22 / 3600
    print("dec_sign,fov_deg,matched_count,mean_error_px,scale,rotation_deg")
    for center_dec in (base_dec, -base_dec):
        for fov in (8, 9, 10, 11, 12, 14, 15, 18, 20):
            matched_count, mean_error_px, scale, rotation_deg = run_fit(args.stars, center_dec, fov)
            sign = "+" if center_dec > 0 else "-"
            print(f"{sign},{fov},{matched_count},{mean_error_px:.2f},{scale:.4f},{rotation_deg:.2f}")


if __name__ == "__main__":
    main()
