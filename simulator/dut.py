"""
Device-Under-Test: star-tracker stand-ins at three levels of realism. All expose the same
``solve(attitude) -> attitude | None`` method, so ``dynamics.run_detumble`` (or any test)
drives any of them unchanged.

SoftwareDUT   "true attitude -> observed unit vectors -> solve_vectors". No image, no
              centroid stage — exercises TETRA + verify + attitude solve only. Fast
              (~10 ms/solve); the seam for positional-noise injection.
ImageDUT      "true attitude -> rendered frame (PSF/brightness) -> C centroid extraction
              -> TETRA -> attitude". The full software pipeline — everything the real rig
              does except physical optics/camera. Slower (renders + full C solve per call).
OpticalDUT    hardware-in-the-loop: HTTP client of a running ``python -m simulator serve``.
              Commands the displayed attitude, waits out the optical pipeline
              (render -> phone -> camera -> tracker), returns the tracker's fresh estimate.
              Seconds per solve; the caller's virtual physics pauses while it waits.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

import live_identify as L

from . import control
from .renderer import Renderer
from .state import DEFAULT_CONFIG

Attitude = tuple[float, float, float]


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


class ImageDUT:
    """Full-pipeline DUT: render the actual frame, then image → C centroids → TETRA → attitude.

    Unlike SoftwareDUT, the solver gets *pixels*: the PSF/Pogson rendering and the C
    centroid extractor are in the loop, so a regression anywhere in the chain fails here.
    ``config`` accepts renderer params (noise_sigma, blur_sigma, gain, ...) for aberration
    injection; the render FOV is pinned to the renderer's construction FOV.
    """

    def __init__(self, renderer: Renderer, morph: int = 0):
        self._lib = L.load_lib()
        self._renderer = renderer
        self._morph = morph  # 0 = keep small/faint rendered stars (point sources)

    def solve(self, attitude: Attitude, config: Optional[dict] = None) -> Optional[Attitude]:
        ra, dec, roll = attitude
        render_config = dict(DEFAULT_CONFIG, **(config or {}))
        render_config["fov_deg"] = self._renderer.fov_deg  # never let a config default re-scale the field
        jpg = self._renderer.render(ra, dec, roll, config=render_config)
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        result = L.solve(self._lib, bgr, self._renderer.fov_deg, self._morph)
        return (result[0], result[1], result[2]) if result else None


class OpticalDUT:
    """
    Hardware-in-the-loop DUT: drives a running ``python -m simulator serve`` over HTTP.

    ``solve(attitude)`` = POST ``point_at`` so the rig displays the attitude, sleep
    ``settle_s`` so the optical pipeline flushes (frames of the *old* attitude drain out of
    phone/camera/solver), then wait for one estimate strictly newer than the flush point
    and return it. Stop-and-stare: the caller's virtual physics is paused during the wait,
    so pipeline delay costs loop rate (~settle_s + one solve period per step), not
    correctness.

    Serve-side requirements: started with ``--tracker`` (or ``--compare-stdin``) so
    estimates flow into the metrics. ``settle_s`` must exceed the measured pipeline delay
    (web "Calibrate delay" button) plus one tracker solve period — too short and you score
    a stale frame.

    ``roll_sign`` / ``roll_offset`` align the tracker's roll convention with the
    commanded one. Symptom of a wrong sign: pointing damps but the roll axis *diverges*
    during detumble (the damping torque becomes positive feedback on roll).
    """

    def __init__(self, host: str = "127.0.0.1:8090", settle_s: float = 1.5,
                 timeout_s: float = 8.0, roll_sign: float = 1.0, roll_offset: float = 0.0):
        self._host = host
        self._settle_s = settle_s
        self._timeout_s = timeout_s
        self._roll_sign = roll_sign
        self._roll_offset = roll_offset

    def _metrics(self) -> dict:
        return control.get_status(self._host)["metrics"]

    def solve(self, attitude: Attitude, config: Optional[dict] = None) -> Optional[Attitude]:
        ra, dec, roll = attitude
        control.send_command(self._host, f"point_at {ra} {dec} {roll}")
        time.sleep(self._settle_s)                      # old-attitude frames drain out
        marker = self._metrics().get("est_t") or -1.0   # estimates at/before this may be stale
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            m = self._metrics()
            est, est_t = m.get("est"), m.get("est_t") or -1.0
            if est is not None and est_t > marker:      # strictly fresher than the flush point
                return (est[0], est[1], (self._roll_sign * est[2] + self._roll_offset) % 360.0)
            time.sleep(0.15)
        return None                                     # tracker never solved the new field


def _demo() -> None:
    """Self-check: both DUTs recover a known attitude (vector convention + full image chain)."""
    from .attitude import angular_sep_deg
    renderer = Renderer(image_size=877, fov_deg=10.0, magnitude_limit=7.5)
    truth = (83.8, -5.4, 0.0)
    for dut_cls in (SoftwareDUT, ImageDUT):
        est = dut_cls(renderer).solve(truth)
        assert est is not None, f"{dut_cls.__name__} failed to solve a dense field"
        err = angular_sep_deg((truth[0], truth[1]), (est[0], est[1]))
        assert err < 0.2, f"{dut_cls.__name__} off by {err:.3f}° — est={est}"
        print(f"dut.py: {dut_cls.__name__} solve err={err:.4f}°")
    print("dut.py self-check passed")


if __name__ == "__main__":
    _demo()
