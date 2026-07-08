"""
Shared simulator state: the seam between the render loop and the HTTP control API.

One lock guards ``config`` + ``metrics`` + ``flash_color`` (all small, copy-in/copy-out).
Live commands flow through a ``queue.Queue`` mailbox owned by the render thread — HTTP
handlers only ``put`` command lines, the render thread drains and applies them, so the
attitude ``Resolver`` is never mutated from two threads.
"""
from __future__ import annotations

import queue
import threading

# Default render config. Brightness: physics-based Pogson (flux ∝ 10^(-0.4·mag)) put
# through a display transfer ``gamma`` so faint stars stay above the centroid floor.
DEFAULT_CONFIG = {
    "gain": 1.0,             # overall brightness multiplier
    "gamma": 4.0,            # display transfer; higher = gentler falloff, keeps faint stars visible
    "saturation_cap": 255.0, # brightest pixel value
    "noise_sigma": 0.0,      # additive gaussian sensor noise (0 = off)
    "blur_sigma": 0.0,       # gaussian defocus blur (0 = off)
    "streak_len": 0.0,       # directional motion-streak length in px (0 = off)
    "streak_angle": 0.0,     # streak direction in degrees
}


class SimState:
    """Thread-safe holder for render config, live metrics, a flash override, and a command mailbox."""

    def __init__(self, pipeline_delay: float = 0.30):
        self._lock = threading.Lock()
        self._config = dict(DEFAULT_CONFIG)
        self._metrics = {
            "pointing_err_deg": None, "roll_err_deg": None, "delay": pipeline_delay,
            "sync_ok": False, "fps": 0.0, "truth": None, "est": None, "tracker_running": False,
        }
        self._flash_color = None            # None, or an (r,g,b) tuple to fill the whole frame
        self.commands: queue.Queue[str] = queue.Queue()

    # --- config (render params) ---
    def get_config(self) -> dict:
        with self._lock:
            return dict(self._config)

    def update_config(self, updates: dict) -> dict:
        """Merges only known keys (ignores junk); returns the new config."""
        with self._lock:
            for key, value in updates.items():
                if key in self._config:
                    self._config[key] = float(value)
            return dict(self._config)

    # --- metrics (live readout) ---
    def get_metrics(self) -> dict:
        with self._lock:
            return dict(self._metrics)

    def update_metrics(self, updates: dict) -> None:
        with self._lock:
            self._metrics.update(updates)

    # --- flash override (sync visual check) ---
    def get_flash(self):
        with self._lock:
            return self._flash_color

    def set_flash(self, color) -> None:
        with self._lock:
            self._flash_color = color

    # --- command mailbox ---
    def put_command(self, line: str) -> None:
        self.commands.put(line)

    def drain_commands(self) -> list[str]:
        """Returns all pending command lines (empties the queue)."""
        out = []
        try:
            while True:
                out.append(self.commands.get_nowait())
        except queue.Empty:
            pass
        return out


def _demo() -> None:
    """Self-check: config merge ignores junk; mailbox drains all pending; flash round-trips."""
    s = SimState()
    s.update_config({"gain": 2.0, "bogus": 9})
    assert s.get_config()["gain"] == 2.0 and "bogus" not in s.get_config()
    s.put_command("point_at 1 2"); s.put_command("roll 3 forever")
    assert s.drain_commands() == ["point_at 1 2", "roll 3 forever"]
    assert s.drain_commands() == []
    s.set_flash((255, 0, 0)); assert s.get_flash() == (255, 0, 0)
    s.update_metrics({"fps": 9.5}); assert s.get_metrics()["fps"] == 9.5
    print("state.py self-check passed")


if __name__ == "__main__":
    _demo()
