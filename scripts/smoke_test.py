from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import (
    BatchConfig,
    build_tetra_database,
    load_catalog,
    run_batch,
    summarize_results,
    TetraMatcher,
)


def main() -> None:
    catalog = load_catalog()
    tetra_db = build_tetra_database(catalog, fov_deg=8.0, max_tetras_per_anchor=20)

    algorithms = {
        "tetra": TetraMatcher(catalog, tetra_db).identify,
    }

    for name, identify in algorithms.items():
        single = identify(120.0, 15.0, 12.0, 6.5, 10)
        batch = run_batch(identify, BatchConfig(samples=8, fov_deg=12.0, mag_limit=6.5, max_stars_query=10))
        print(name, single["outcome"], summarize_results(batch))


if __name__ == "__main__":
    main()
