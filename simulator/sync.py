"""
Flash visual check — a liveness / optical-round-trip sanity read.

The simulator fills the whole frame with R, then G, then B; a watcher on the tracker's
camera-preview MJPEG (`:8080`) detects each colour appearing and reports the round-trip
time. This confirms the loop is live and roughly how long light takes to come back. It is
NOT used to set ``pipeline_delay`` — the preview publishes *after* the solve in
pi_identify.py, so it measures a different latency than the stdout estimate stream (see
comparator.estimate_delay, which measures the quantity that actually needs compensating).
"""
from __future__ import annotations

import statistics
import time
import urllib.request
from typing import Callable, Iterable, Optional

import cv2
import numpy as np


def channel_dominant(mean_bgr, color_rgb, threshold: float = 30.0) -> bool:
    """True if the frame's mean colour is dominated by the flashed colour's channel."""
    r, g, b = color_rgb
    mb, mg, mr = mean_bgr
    if r >= g and r >= b:
        return mr > mg + threshold and mr > mb + threshold
    if g >= r and g >= b:
        return mg > mr + threshold and mg > mb + threshold
    return mb > mr + threshold and mb > mg + threshold


def _detect_one(frames: Iterable[tuple[float, tuple]], color_rgb, t_render: float,
                threshold: float) -> Optional[float]:
    """Returns (t_seen - t_render) for the first frame at/after t_render matching the colour."""
    for t_seen, mean_bgr in frames:
        if t_seen >= t_render and channel_dominant(mean_bgr, color_rgb, threshold):
            return t_seen - t_render
    return None


def preview_mean_frames(preview_url: str, stop_after: float = 3.0):
    """Yields ``(t, mean_bgr)`` from an MJPEG preview stream until ``stop_after`` seconds elapse."""
    deadline = time.monotonic() + stop_after
    with urllib.request.urlopen(preview_url, timeout=stop_after) as stream:
        buf = b""
        while time.monotonic() < deadline:
            buf += stream.read(4096)
            start = buf.find(b"\xff\xd8")      # JPEG SOI
            end = buf.find(b"\xff\xd9", start)  # JPEG EOI
            if start != -1 and end != -1:
                jpg = buf[start:end + 2]
                buf = buf[end + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield time.monotonic(), tuple(img.reshape(-1, 3).mean(axis=0))


def detect_flash(flash_setter: Callable, preview_url: str,
                 colors=((255, 0, 0), (0, 255, 0), (0, 0, 255)),
                 threshold: float = 30.0) -> Optional[float]:
    """
    Flashes each colour, watches the preview for it, and returns the median round-trip time.

    ``flash_setter(color_or_None)`` overrides / clears the displayed frame. Returns None if no
    colour was detected (loop not live / camera not pointed at the screen).
    """
    times = []
    for color in colors:
        flash_setter(color)
        t_render = time.monotonic()
        dt = _detect_one(preview_mean_frames(preview_url), color, t_render, threshold)
        flash_setter(None)  # resume star rendering before the next colour
        if dt is not None:
            times.append(dt)
    return statistics.median(times) if times else None


def _demo() -> None:
    """Self-check: flash detection on synthetic frames."""
    # Black frames then a red frame -> detected at the red frame's time.
    black = (8.0, 8.0, 8.0)
    red_bgr = (8.0, 8.0, 240.0)  # BGR: high red channel
    frames = [(0.0, black), (0.1, black), (0.2, red_bgr)]
    dt = _detect_one(frames, (255, 0, 0), t_render=0.0, threshold=30.0)
    assert dt == 0.2, dt
    assert channel_dominant(red_bgr, (255, 0, 0)) and not channel_dominant(black, (255, 0, 0))
    print("sync.py self-check passed")


if __name__ == "__main__":
    _demo()
