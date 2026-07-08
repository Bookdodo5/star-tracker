"""
Attitude representation and error metrics for the star simulator.

An attitude here is a 3-tuple ``(ra_deg, dec_deg, roll_deg)`` describing where the
simulated camera boresight points on the sky and how it is rolled about that boresight.
That is exactly the form the star tracker reports, so truth and estimate compare directly.

We deliberately compare boresight direction and roll *separately* (pointing error in
degrees, roll error in degrees) rather than composing a single quaternion. Quaternion
composition depends on the east/north/boresight chirality convention documented in
CLAUDE.md, and getting it subtly wrong silently inflates the error. Boresight + roll is
convention-free and physically meaningful, so it is the primary metric.
"""
from __future__ import annotations

import math


def radec_to_vec(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    """Converts an RA/DEC pointing to a unit boresight vector in catalog coordinates."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))


def vec_to_radec(vec: tuple[float, float, float]) -> tuple[float, float]:
    """Converts a unit boresight vector back to (ra_deg, dec_deg) in [0,360) / [-90,90]."""
    x, y, z = vec
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return ra, dec


def angular_sep_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle angle in degrees between two (ra_deg, dec_deg) pointings."""
    va = radec_to_vec(*a)
    vb = radec_to_vec(*b)
    dot = max(-1.0, min(1.0, va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]))
    return math.degrees(math.acos(dot))


def wrap_deg(angle: float) -> float:
    """Wraps an angle to (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


def roll_diff_deg(truth_roll: float, est_roll: float) -> float:
    """Signed shortest roll difference in degrees, wrapped to (-180, 180]."""
    return wrap_deg(truth_roll - est_roll)


def attitude_error(truth: tuple[float, float, float],
                   est: tuple[float, float, float]) -> tuple[float, float]:
    """
    Returns ``(pointing_err_deg, roll_err_deg)`` between a truth and estimated attitude.
    ``pointing_err_deg`` is the great-circle boresight error; ``roll_err_deg`` is the
    signed wrapped roll difference. Both are convention-free.
    """
    pointing = angular_sep_deg((truth[0], truth[1]), (est[0], est[1]))
    roll = abs(roll_diff_deg(truth[2], est[2]))
    return pointing, roll


def _demo() -> None:
    """Self-check: identical attitudes ~0 error; a known 5 deg RA offset ~5 deg pointing error."""
    p, r = attitude_error((10.0, 20.0, 30.0), (10.0, 20.0, 30.0))
    assert p < 1e-6 and r < 1e-6, (p, r)
    # 5 deg of RA at dec=0 is 5 deg on the sky.
    p, r = attitude_error((100.0, 0.0, 0.0), (105.0, 0.0, 0.0))
    assert abs(p - 5.0) < 1e-6, p
    # roll wrap: 359 vs 1 is 2 deg apart, not 358.
    assert abs(roll_diff_deg(359.0, 1.0)) - 2.0 < 1e-9
    # vec round-trip.
    ra, dec = vec_to_radec(radec_to_vec(123.4, -45.6))
    assert abs(ra - 123.4) < 1e-6 and abs(dec + 45.6) < 1e-6, (ra, dec)
    print("attitude.py self-check passed")


if __name__ == "__main__":
    _demo()
