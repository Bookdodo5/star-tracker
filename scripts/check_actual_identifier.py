from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = PROJECT_ROOT / "C" / "build-generated-release" / "demo_centroid_compare.exe"


def load_truth(path: Path) -> set[int]:
    """
    Loads expected HR IDs produced by the actual-image plate evaluator.
    """

    with path.open(newline="") as file:
        return {int(row["hr_id"]) for row in csv.DictReader(file)}


def run_identifier(executable_path: Path, stars_csv_path: Path, width: int, height: int, fov: float) -> str:
    """
    Runs the C identifier and returns stdout.
    """

    command = [str(executable_path), str(stars_csv_path), str(width), str(height), str(fov)]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=True, text=True, capture_output=True).stdout


def parse_algorithm_hrs(identifier_stdout: str) -> dict[str, set[int]]:
    """
    Parses matched HR IDs from C identifier stdout.
    """

    algorithm_hrs: dict[str, set[int]] = {"TETRA": set(), "Pyramid": set()}
    pattern = re.compile(r"^(TETRA|Pyramid) match,\d+,(\d+),\d+$")
    for line in identifier_stdout.splitlines():
        parsed = pattern.match(line.strip())
        if parsed is not None:
            algorithm_hrs[parsed.group(1)].add(int(parsed.group(2)))
    return algorithm_hrs


def main() -> None:
    """
    Compares C identifier output against actual-image HR ground truth.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--stars", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fov", required=True, type=float)
    parser.add_argument("--exe", default=DEFAULT_EXE, type=Path)
    args = parser.parse_args()

    expected_hrs = load_truth(args.truth)
    identifier_stdout = run_identifier(args.exe, args.stars, args.width, args.height, args.fov)
    algorithm_hrs = parse_algorithm_hrs(identifier_stdout)
    print(identifier_stdout)
    print(f"expected_hrs={sorted(expected_hrs)}")
    for algorithm_name, matched_hrs in algorithm_hrs.items():
        hits = matched_hrs & expected_hrs
        accuracy = len(hits) / len(matched_hrs) * 100.0 if matched_hrs else 0.0
        print(f"{algorithm_name} actual_hits={len(hits)} matched={len(matched_hrs)} precision_pct={accuracy:.1f} hits={sorted(hits)}")


if __name__ == "__main__":
    main()
