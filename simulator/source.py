"""
Attitude sources: the swappable front of the simulator.

Everything downstream (renderer, stream, comparator) only calls ``attitude(now)`` and
never cares where the attitude came from. That is the seam that makes the requirement
"the attitude can come from live commands or from other sources" true without touching
the render/stream code.

Implementations:
    CommandQueueSource  attitude driven by an operator command script (the default)
    ReplaySource        attitude replayed from a recorded CSV (t,ra,dec,roll)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

from .commands import Command, Resolver


class AttitudeSource(Protocol):
    """Anything that can report an attitude at a monotonic time."""

    def attitude(self, now: float) -> tuple[tuple[float, float, float], bool]:
        """Returns ``((ra_deg, dec_deg, roll_deg), moving)`` at monotonic time ``now``."""
        ...


class CommandQueueSource:
    """Drives attitude from an ordered command queue via :class:`~simulator.commands.Resolver`."""

    def __init__(self, commands: list[Command], start_attitude: tuple[float, float, float]):
        self._resolver = Resolver(commands, start_attitude)

    def attitude(self, now: float):
        return self._resolver.attitude(now)

    def inject(self, command: Command, now: float) -> None:
        """Interrupts the current motion with a live command (see Resolver.inject)."""
        self._resolver.inject(command, now)

    def display_color(self):
        """Full-frame fill colour for the active blank/flash command, else None."""
        return self._resolver.display_color()


class ReplaySource:
    """
    Replays an attitude timeline from a CSV with columns ``t,ra,dec,roll`` (t in seconds
    from the recording start). Linearly interpolates between samples; clamps at the ends.
    Reuses the same recorded-frame idea as the existing orbit-frame datasets, but as a
    live truth feed rather than static images.
    """

    def __init__(self, csv_path: Path):
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((float(row["t"]), float(row["ra"]), float(row["dec"]), float(row["roll"])))
        if not rows:
            raise ValueError(f"{csv_path} has no rows")
        self._rows = sorted(rows, key=lambda r: r[0])
        self._t0 = None

    def attitude(self, now: float):
        if self._t0 is None:
            self._t0 = now
        t = now - self._t0
        rows = self._rows
        if t <= rows[0][0]:
            _, ra, dec, roll = rows[0]
            return (ra, dec, roll), False
        if t >= rows[-1][0]:
            _, ra, dec, roll = rows[-1]
            return (ra, dec, roll), False
        # Find the bracketing pair and interpolate.
        for (t0, ra0, dec0, roll0), (t1, ra1, dec1, roll1) in zip(rows, rows[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return (ra0 + f * (ra1 - ra0), dec0 + f * (dec1 - dec0),
                        roll0 + f * (roll1 - roll0)), True
        _, ra, dec, roll = rows[-1]
        return (ra, dec, roll), False


def _demo() -> None:
    """Self-check: replay interpolates the midpoint between samples."""
    import io
    rows = "t,ra,dec,roll\n0,0,0,0\n2,10,0,0\n"
    r = ReplaySource.__new__(ReplaySource)
    reader = csv.DictReader(io.StringIO(rows))
    r._rows = sorted(((float(x["t"]), float(x["ra"]), float(x["dec"]), float(x["roll"])) for x in reader),
                     key=lambda z: z[0])
    r._t0 = 0.0
    (ra, _, _), moving = r.attitude(1.0)  # halfway between 0 and 10
    assert abs(ra - 5.0) < 1e-9 and moving, ra
    print("source.py self-check passed")


if __name__ == "__main__":
    _demo()
