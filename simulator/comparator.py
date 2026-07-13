"""
Comparator: truth vs. estimate, with the pipeline delay handled gracefully.

THE DELAY PROBLEM
-----------------
When you command the simulator to show attitude A, that attitude does not reach the
tracker's output instantly. It travels:

    command -> render -> stream to phone -> phone displays -> camera exposes ->
    centroid -> TETRA -> estimated attitude

That whole chain is a lag of a few hundred milliseconds. If the commanded attitude is
*moving* (roll/slew), comparing the estimate that arrives at time ``t`` against the truth
at time ``t`` is wrong by exactly the motion during the lag. Naively you'd see a big fake
error and chase a bug that isn't there.

HOW IT IS HANDLED (so you don't have to think about it)
-------------------------------------------------------
1. The simulator records a timestamped *truth timeline* — every rendered attitude with the
   monotonic time it was shown, and whether it was moving at that instant.
2. Each estimate is stamped with the time it was *received*. We look up truth at
   ``t_received - pipeline_delay`` (delay-compensated matching), not at ``t_received``.
3. By default, accuracy stats are computed only over samples where the matched truth was
   **static** (a ``hold``/settled ``point_at`` window). During a hold, the truth is the
   same for a whole second regardless of the exact delay, so the error number is immune to
   any delay-estimate error. This is the "no headache" default: park the attitude, measure.
4. ``--pipeline-delay`` sets the compensation (moving samples are still logged to the CSV
   and can be summarised with ``summary(static_only=False)``). If you don't know the delay,
   run ``estimate_delay`` on a run that contains a step: it finds the lag between a
   commanded jump and when the estimate follows (the /calibrate-delay route does this).

Every sample (static and moving) is always logged to CSV; only the *summary* filters.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .attitude import attitude_error


@dataclass
class TruthSample:
    """One point on the truth timeline: when it was shown, what, and whether it was moving."""
    t: float
    ra: float
    dec: float
    roll: float
    moving: bool


class TruthTimeline:
    """A bounded, time-ordered history of shown truth attitudes, queried by timestamp."""

    def __init__(self, maxlen: int = 4096):
        self._samples: deque[TruthSample] = deque(maxlen=maxlen)

    def record(self, t: float, ra: float, dec: float, roll: float, moving: bool) -> None:
        """Appends a shown truth attitude."""
        self._samples.append(TruthSample(t, ra, dec, roll, moving))

    def at(self, t: float, tolerance: float = 0.5) -> Optional[TruthSample]:
        """Returns the truth sample nearest ``t``, or None if none is within ``tolerance`` s."""
        best, best_dt = None, tolerance
        for s in self._samples:
            dt = abs(s.t - t)
            if dt <= best_dt:
                best, best_dt = s, dt
        return best


@dataclass
class Comparison:
    """One matched (truth, estimate) result."""
    t_recv: float
    pointing_err_deg: float
    roll_err_deg: float
    truth_moving: bool
    truth: tuple[float, float, float]
    est: tuple[float, float, float]


class Comparator:
    """
    Matches estimates to delay-compensated truth, logs every comparison, and summarises
    accuracy over static samples by default.
    """

    def __init__(self, timeline: TruthTimeline, pipeline_delay: float = 0.30,
                 roll_sign: float = 1.0, roll_offset: float = 0.0,
                 csv_path: Optional[Path] = None):
        self._timeline = timeline
        self._delay = pipeline_delay
        self._roll_sign = roll_sign
        self._roll_offset = roll_offset
        self._results: list[Comparison] = []
        self._csv_path = csv_path
        self._writer = None
        self._csv_file = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._writer = csv.writer(self._csv_file)
            self._writer.writerow(["t_recv", "truth_moving", "pointing_err_deg", "roll_err_deg",
                                   "truth_ra", "truth_dec", "truth_roll",
                                   "est_ra", "est_dec", "est_roll"])

    def set_delay(self, delay: float) -> None:
        """Updates the pipeline delay used for matching (e.g. after a calibration)."""
        self._delay = delay

    def add_estimate(self, t_recv: float, est: tuple[float, float, float]) -> Optional[Comparison]:
        """
        Matches an estimate received at ``t_recv`` to truth at ``t_recv - delay`` and records
        the comparison. Returns the :class:`Comparison`, or None if no truth matched.
        """
        truth = self._timeline.at(t_recv - self._delay)
        if truth is None:
            return None
        # Align the estimate's roll to the truth's roll convention before scoring.
        est_aligned = (est[0], est[1], (self._roll_sign * est[2] + self._roll_offset) % 360.0)
        pointing, roll = attitude_error((truth.ra, truth.dec, truth.roll), est_aligned)
        c = Comparison(t_recv, pointing, roll, truth.moving,
                       (truth.ra, truth.dec, truth.roll), est_aligned)
        self._results.append(c)
        if self._writer:
            self._writer.writerow([f"{t_recv:.4f}", int(truth.moving), f"{pointing:.4f}", f"{roll:.4f}",
                                   f"{truth.ra:.4f}", f"{truth.dec:.4f}", f"{truth.roll:.4f}",
                                   f"{est_aligned[0]:.4f}", f"{est_aligned[1]:.4f}", f"{est_aligned[2]:.4f}"])
            self._csv_file.flush()
        return c

    def summary(self, static_only: bool = True, pointing_pass_deg: float = 0.5) -> dict:
        """
        Summary over recorded comparisons. By default only static-truth samples count, so
        the numbers are immune to pipeline-delay uncertainty.
        """
        rows = [c for c in self._results if (not static_only or not c.truth_moving)]
        if not rows:
            return {"samples": 0}
        pts = sorted(c.pointing_err_deg for c in rows)
        n = len(pts)
        within = sum(1 for c in rows if c.pointing_err_deg <= pointing_pass_deg)
        return {
            "samples": n,
            "static_only": static_only,
            "mean_pointing_deg": sum(pts) / n,
            "median_pointing_deg": pts[n // 2],
            "p95_pointing_deg": pts[min(n - 1, int(0.95 * n))],
            "max_pointing_deg": pts[-1],
            "accuracy_pct": 100.0 * within / n,
            "pass_threshold_deg": pointing_pass_deg,
        }

    def close(self) -> None:
        """Closes the CSV file if one was opened."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None


def estimate_delay(estimates: list[tuple[float, tuple[float, float, float]]],
                   step_time: float, new_pointing: tuple[float, float], match_deg: float = 0.5) -> Optional[float]:
    """
    One-shot pipeline-delay measurement from a commanded step.

    Given a known instant ``step_time`` when the truth jumped to ``new_pointing`` and the
    recorded ``(t_recv, (ra,dec,roll))`` estimates, returns the delay = (first time an
    estimate lands within ``match_deg`` of the new pointing) - ``step_time``. Run this once
    on a run that starts static, jumps, then holds; use the result as ``--pipeline-delay``.
    """
    from .attitude import angular_sep_deg
    for t_recv, est in sorted(estimates):
        if t_recv < step_time:
            continue
        if angular_sep_deg(new_pointing, (est[0], est[1])) <= match_deg:
            return t_recv - step_time
    return None


def _demo() -> None:
    """Self-check: delay compensation matches a moving truth; static scoring ignores lag."""
    tl = TruthTimeline()
    # Truth sweeps RA 0->10 over t=0..10 while "moving"; estimate arrives lagged by 0.3s.
    for i in range(101):
        t = i * 0.1
        tl.record(t, ra=t, dec=0.0, roll=0.0, moving=True)
    cmp = Comparator(tl, pipeline_delay=0.3)
    # Estimate received at t=5.3 reflects truth shown at t=5.0 (ra=5.0). Compensation should
    # match it to ra~5.0, giving near-zero error despite the lag.
    c = cmp.add_estimate(5.3, (5.0, 0.0, 0.0))
    assert c is not None and c.pointing_err_deg < 0.05, c
    # Without any static samples, static-only summary reports nothing (nothing to trust).
    assert cmp.summary(static_only=True)["samples"] == 0
    # A static hold: truth parked at (5,0), estimate off by ~0.02 deg -> counted, passes.
    tl2 = TruthTimeline()
    tl2.record(0.0, 5.0, 0.0, 0.0, moving=False)
    cmp2 = Comparator(tl2, pipeline_delay=0.3)
    cmp2.add_estimate(0.3, (5.02, 0.0, 0.0))
    s = cmp2.summary(static_only=True)
    assert s["samples"] == 1 and s["accuracy_pct"] == 100.0, s
    print("comparator.py self-check passed")


if __name__ == "__main__":
    _demo()
