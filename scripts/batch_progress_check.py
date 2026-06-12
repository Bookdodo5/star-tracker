from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import BatchConfig, TetraMatcher, build_tetra_database, load_catalog, run_batch


catalog = load_catalog()
database = build_tetra_database(catalog)
matcher = TetraMatcher(catalog, database)
run_batch(matcher.identify, BatchConfig(samples=5, fov_deg=10.0, mag_limit=6.5, max_stars_query=8))
