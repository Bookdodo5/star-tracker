"""
Tracker feed: parse the star tracker's estimated attitude out of its stdout lines.

The tracker is left completely unchanged — we just read the lines it already prints and
pull an ``(ra, dec, roll)`` out of them. Two formats are recognised, covering every
existing driver in this repo:

    ``... attitude_ra_deg=123.4 attitude_dec_deg=-5.6 attitude_roll_deg=7.8 ...``  (demo_centroid_compare)
    ``... RA=123.4  DEC=-5.6  ROLL=7.8 ...``                                       (pi_identify / identify)

Usage: pipe the tracker into the simulator, e.g.
    python pi_identify.py --fov 10 | python -m simulator.main --compare-stdin ...
"""
from __future__ import annotations

import re
from typing import Optional

_KV = re.compile(r"attitude_ra_deg=(-?\d+\.?\d*)\s+attitude_dec_deg=(-?\d+\.?\d*)\s+attitude_roll_deg=(-?\d+\.?\d*)")
_RA = re.compile(r"\bRA=\s*(-?\d+\.?\d*)\s+DEC=\s*(-?\d+\.?\d*)\s+ROLL=\s*(-?\d+\.?\d*)")


def parse_line(line: str) -> Optional[tuple[float, float, float]]:
    """Returns ``(ra_deg, dec_deg, roll_deg)`` from a tracker stdout line, or None."""
    m = _KV.search(line) or _RA.search(line)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    except ValueError:
        return None


def _demo() -> None:
    """Self-check: both stdout formats parse; NULL and junk lines return None."""
    assert parse_line("t attitude_ra_deg=100.5 attitude_dec_deg=-5.4 attitude_roll_deg=12.0 s") \
        == (100.5, -5.4, 12.0)
    assert parse_line("frame 12 | RA= 100.5  DEC=  -5.4  ROLL=  12.0  (5 fps)") == (100.5, -5.4, 12.0)
    assert parse_line("frame 13 | NULL") is None
    assert parse_line("random noise") is None
    print("feed.py self-check passed")


if __name__ == "__main__":
    _demo()
