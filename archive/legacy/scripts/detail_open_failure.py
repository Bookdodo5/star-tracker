from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import (
    OpenStarTrackerMatcher,
    build_pair_database,
    filter_stars_by_fov,
    load_catalog,
)


catalog = load_catalog()
pair_db, stars = build_pair_database(catalog, "openstartracker")
matcher = OpenStarTrackerMatcher(catalog, pair_db, stars)

ra = 231.791443
dec = 45.186626
result = matcher.identify(ra, dec, 12.0, 6.5, 10)
visible = filter_stars_by_fov(catalog, ra, dec, 12.0, 6.5)

print(result["outcome"], result["n_stars"], result["matched_ids"], result["score"], result["mean_residual_deg"])
print(visible[["HR_clean", "Vmag"]].head(20).to_string(index=False))
