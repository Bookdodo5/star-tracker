"""
Retired Python Pyramid reference implementation.
Wraps archive/pyramid/pyramid_python_reference.py PyramidMatcher.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import load_catalog
from archive.pyramid.pyramid_python_reference import build_pair_database, PyramidMatcher


def build_matcher(max_fov_deg: float = 20.0, mag_limit: float = 6.5) -> tuple:
    """Loads catalog and builds (or loads cached) Pyramid pair database, returns (catalog, matcher)."""
    catalog = load_catalog()
    pair_db, db_stars = build_pair_database(catalog, algorithm_name="pyramid",
                                            max_fov_deg=max_fov_deg, mag_limit=mag_limit)
    return catalog, PyramidMatcher(catalog, pair_db, db_stars)


def match(matcher_tuple: tuple, ra_deg: float, dec_deg: float, fov_deg: float,
          mag_limit: float = 6.5, max_stars: int = 10) -> dict:
    """Run Pyramid identification for a pointing. Returns the result dict from star_tracker_core."""
    catalog, matcher = matcher_tuple
    return matcher.identify(ra_deg, dec_deg, fov_deg, mag_limit, max_stars_query=max_stars)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pyramid reference identifier")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--fov", type=float, default=10.0)
    parser.add_argument("--mag", type=float, default=6.5)
    args = parser.parse_args()

    print("Loading catalog and Pyramid database...")
    mt = build_matcher()
    result = match(mt, args.ra, args.dec, args.fov, args.mag)
    print(f"Correct: {result.get('correct', False)}")
    print(f"Matched HRs: {result.get('matched_ids', [])}")
    print(f"Branches: {result.get('branches', 'N/A')}")
