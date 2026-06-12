from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import build_tetra_database, load_catalog, run_confusion_matrix, TetraMatcher


catalog = load_catalog()
database = build_tetra_database(catalog)
matcher = TetraMatcher(catalog, database)
run_confusion_matrix(matcher.identify, [8.0], [6.5, None], samples=2, max_stars_query=8)
