from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import CACHE_DIR, build_pair_database, build_tetra_database, load_catalog


catalog = load_catalog()
tetra = build_tetra_database(catalog)
pyramid_pairs, _ = build_pair_database(catalog, "pyramid", force=True)

files = [
    CACHE_DIR / "tetra_fov8_mag6.5_cap20.csv",
    CACHE_DIR / "pyramid_pairs_fov20_mag6.5_min0.25_starsall_cap15_bin2.csv",
]

print(f"tetra rows={len(tetra):,}")
print(f"pyramid rows={len(pyramid_pairs):,}")
for path in files:
    print(f"{path.name} {path.stat().st_size / 1024**2:.2f} MB")
