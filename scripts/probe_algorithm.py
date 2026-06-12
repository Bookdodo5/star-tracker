from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import (
    BatchConfig,
    PyramidMatcher,
    TetraMatcher,
    build_pair_database,
    build_tetra_database,
    load_catalog,
    run_batch,
    summarize_results,
)


def main(name: str) -> None:
    started = time.time()
    catalog = load_catalog()
    if name == "tetra":
        matcher = TetraMatcher(catalog, build_tetra_database(catalog, fov_deg=8.0, max_tetras_per_anchor=20))
        top = 10
    elif name == "pyramid":
        pair_db, stars = build_pair_database(catalog, "pyramid")
        matcher = PyramidMatcher(catalog, pair_db, stars)
        top = 8
    else:
        raise ValueError(name)

    print(name, "setup_seconds", round(time.time() - started, 2), flush=True)
    single = matcher.identify(120.0, 15.0, 12.0, 6.5, top)
    print(name, "single", single["outcome"], single["n_stars"], single["matched_ids"], flush=True)
    batch = run_batch(matcher.identify, BatchConfig(samples=8, fov_deg=12.0, mag_limit=6.5, max_stars_query=top))
    print(batch.to_string(index=False), flush=True)
    print(name, summarize_results(batch), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
