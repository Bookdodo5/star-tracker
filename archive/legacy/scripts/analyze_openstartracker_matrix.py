from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import (
    BatchConfig,
    OpenStarTrackerMatcher,
    build_pair_database,
    load_catalog,
    run_batch,
    summarize_results,
)


catalog = load_catalog()
pair_db, stars = build_pair_database(catalog, "openstartracker")
matcher = OpenStarTrackerMatcher(catalog, pair_db, stars)

for fov in [8.0, 12.0, 18.0]:
    for mag_limit in [5.0, 6.0, 6.5, None]:
        label = "ALL" if mag_limit is None else f"<={mag_limit:g}"
        results = run_batch(
            matcher.identify,
            BatchConfig(samples=8, fov_deg=fov, mag_limit=mag_limit, max_stars_query=12),
            show_progress=False,
        )
        summary = summarize_results(results)
        print(f"FOV={fov:g} mag={label} summary={summary}")
        failures = results[results["outcome"] == "failure"]
        if len(failures):
            print(failures[["ra", "dec", "n_stars", "score", "mean_residual_deg"]].to_string(index=False))
