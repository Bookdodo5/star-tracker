"""
Device-Under-Test abstractions — the UVM idea, minus the SystemC.

One protocol, several implementations, so the same scenario/scoreboard drives any of them:

    OpticalDUT   full hardware-in-the-loop: render → phone → camera → tracker (the impressive
                 path; lives in main.py's TrackerController, not here).
    SoftwareDUT  headless: render attitude → observed unit vectors → solve_vectors (no camera).
                 This is the fast, repeatable regression path and the seam for the paper's
                 positional-noise injection (perturb the star vectors, not the phone screen).

The SoftwareDUT bypasses the optical + centroid stages, so it exercises the identifier +
attitude solve directly — ideal for regression tests and Monte-Carlo sweeps.
"""
from __future__ import annotations

from typing import Optional, Protocol

import live_identify as L

from .renderer import Renderer

Attitude = tuple[float, float, float]


class DUT(Protocol):
    """Anything that turns a true attitude into an estimated attitude (or None if unsolved)."""

    def solve(self, attitude: Attitude, config: Optional[dict] = None) -> Optional[Attitude]:
        ...


class SoftwareDUT:
    """Headless DUT: true attitude → observed vectors → identifier → estimated attitude."""

    def __init__(self, renderer: Renderer):
        self._lib = L.load_lib()
        self._renderer = renderer

    def solve(self, attitude: Attitude, config: Optional[dict] = None) -> Optional[Attitude]:
        ra, dec, roll = attitude
        pixel_noise = float((config or {}).get("pixel_noise", 0.0))
        vecs = self._renderer.observed_vectors(ra, dec, roll, pixel_noise=pixel_noise)
        if len(vecs) < 4:
            return None
        result = L.solve_vectors(self._lib, vecs)
        return (result[0], result[1], result[2]) if result else None


def _demo() -> None:
    """Self-check: the headless DUT recovers a known attitude (verifies the vector convention)."""
    dut = SoftwareDUT(Renderer(image_size=877, fov_deg=10.0, magnitude_limit=7.5))
    truth = (83.8, -5.4, 0.0)
    est = dut.solve(truth)
    assert est is not None, "SoftwareDUT failed to solve a dense field"
    from .attitude import angular_sep_deg
    err = angular_sep_deg((truth[0], truth[1]), (est[0], est[1]))
    assert err < 0.2, f"headless solve off by {err:.3f}° — vector convention wrong? est={est}"
    print(f"dut.py self-check passed (headless solve err={err:.4f}°)")


if __name__ == "__main__":
    _demo()
