from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import (
    PyramidMatcher,
    build_pair_database,
    load_catalog,
    run_confusion_matrix,
)


catalog = load_catalog()
pair_db, db_stars = build_pair_database(catalog, "pyramid")
matcher = PyramidMatcher(catalog, pair_db, db_stars)

started = time.perf_counter()
confusion_df, summary_df = run_confusion_matrix(
    matcher.identify,
    fov_values=[8.0, 12.0, 18.0],
    mag_limits=[5.0, 6.0, 6.5, None],
    samples=8,
    max_stars_query=10,
)
elapsed = time.perf_counter() - started

print(confusion_df.to_string(index=False))
print(summary_df[["FOV", "Magnitude", "seconds", "accuracy_pct"]].to_string(index=False))
print(f"total_seconds={elapsed:.2f}")
