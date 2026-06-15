"""
Canonical Python TETRA reference implementation.
Wraps src/star_tracker_core.py TetraMatcher — the authoritative algorithm source.
All algorithm logic lives in star_tracker_core; this module provides a stable entry point.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import load_catalog, build_tetra_database, TetraMatcher


def build_matcher(fov_deg: float = 15.0, mag_limit: float = 6.5, max_tetras_per_anchor: int = 40) -> tuple:
    """Loads catalog and builds (or loads cached) TETRA database, returns (catalog, matcher)."""
    catalog = load_catalog()
    db = build_tetra_database(catalog, fov_deg=fov_deg, mag_limit=mag_limit,
                              max_tetras_per_anchor=max_tetras_per_anchor)
    return catalog, TetraMatcher(catalog, db)


def match(matcher_tuple: tuple, ra_deg: float, dec_deg: float, fov_deg: float,
          mag_limit: float = 6.5, max_stars: int = 12) -> dict:
    """Run TETRA identification for a pointing. Returns the result dict from star_tracker_core."""
    catalog, matcher = matcher_tuple
    return matcher.identify(ra_deg, dec_deg, fov_deg, mag_limit, max_stars_query=max_stars)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TETRA reference identifier")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--fov", type=float, default=10.0)
    parser.add_argument("--mag", type=float, default=6.5)
    args = parser.parse_args()

    print("Loading catalog and TETRA database...")
    mt = build_matcher()
    result = match(mt, args.ra, args.dec, args.fov, args.mag)
    print(f"Success: {result.get('correct', False)}")
    print(f"Matched HRs: {result.get('matched_ids', [])}")
    print(f"Score: {result.get('score', 'N/A')}")
