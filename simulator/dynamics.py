"""
Closed-loop detumble: a hardware-in-the-loop ADCS testbed where the *only* sensor is the
real star tracker.

The loop:

    RigidBody (tumbling)  ->  attitude shown on the phone  ->  star tracker  ->  estimated
    attitude  ->  RateController estimates the spin rate and commands a damping torque  ->
    RigidBody integrates the torque  ->  ...

The controller never sees the true state — it estimates the angular rate purely by
differencing successive tracker attitude estimates, exactly as a real spacecraft would.
This is the software replacement for a commercial ADCS air-bearing + star-simulator rig.

Model note: attitude is (ra, dec, roll) and angular velocity is the three rates
(dra/dt, ddec/dt, droll/dt) in deg/s, integrated as decoupled axes with a diagonal inertia.
This is kinematic rate-damping detumble (B-dot-like), not the full ω×Jω Euler equation —
enough to demonstrate the closed loop; add cross-coupling later if the demo needs it.
"""
from __future__ import annotations

import math

from .attitude import wrap_deg


class RigidBody:
    """A tumbling body: attitude (ra,dec,roll) + angular rates, integrated under torque."""

    def __init__(self, attitude=(83.8, -5.4, 0.0), omega=(4.0, 3.0, 5.0), inertia=(1.0, 1.0, 1.0)):
        self.ra, self.dec, self.roll = attitude
        self.omega = list(omega)          # deg/s about (ra, dec, roll)
        self.inertia = inertia

    def step(self, dt: float, torque=(0.0, 0.0, 0.0)) -> None:
        """Advances the state dt seconds: torque changes the rates, rates change the attitude."""
        for i in range(3):
            self.omega[i] += torque[i] / self.inertia[i] * dt
        self.ra = (self.ra + self.omega[0] * dt) % 360.0
        self.dec = max(-90.0, min(90.0, self.dec + self.omega[1] * dt))
        self.roll = (self.roll + self.omega[2] * dt) % 360.0

    def attitude(self):
        return (self.ra, self.dec, self.roll)

    def rate_magnitude(self) -> float:
        """|ω| in deg/s — the detumble target (drive to ~0)."""
        return math.sqrt(sum(w * w for w in self.omega))


class RateController:
    """
    Estimates the angular rate by differencing successive tracker attitudes and commands a
    damping torque τ = -k·ω_est. Uses ONLY the estimated attitude — never the true state.
    """

    def __init__(self, gain: float = 2.0):
        self.gain = gain
        self._prev = None  # (attitude, t) of the last usable estimate

    def torque(self, estimate, t: float):
        """Returns the damping torque for a new tracker estimate at time t (zero until it can diff)."""
        if estimate is None:
            return (0.0, 0.0, 0.0)
        if self._prev is None:
            self._prev = (estimate, t)
            return (0.0, 0.0, 0.0)
        (pra, pdec, proll), pt = self._prev
        dt = t - pt
        if dt <= 0:
            return (0.0, 0.0, 0.0)
        omega_est = (wrap_deg(estimate[0] - pra) / dt,
                     (estimate[1] - pdec) / dt,
                     wrap_deg(estimate[2] - proll) / dt)
        self._prev = (estimate, t)
        return tuple(-self.gain * w for w in omega_est)


def run_detumble(dut, body: RigidBody = None, controller: RateController = None,
                 duration_s: float = 20.0, dt: float = 0.1, on_step=None) -> list:
    """
    Runs the headless closed-loop detumble and returns the |ω| history.

    ``dut`` is any DUT (SoftwareDUT headless, or an OpticalDUT-like wrapper for HIL).
    ``on_step(t, truth, est, rate_mag)`` is an optional callback for live plotting/streaming.
    """
    body = body or RigidBody()
    controller = controller or RateController()
    history = []
    steps = int(duration_s / dt)
    for i in range(steps):
        t = i * dt
        truth = body.attitude()
        est = dut.solve(truth)                       # the star tracker: the only sensor
        torque = controller.torque(est, t)           # damping from estimated rate only
        body.step(dt, torque)
        history.append(body.rate_magnitude())
        if on_step is not None:
            on_step(t, truth, est, history[-1])
    return history


def _demo() -> None:
    """Self-check: a headless closed loop detumbles — |ω| drops from tumbling to near zero."""
    from .dut import SoftwareDUT
    from .renderer import Renderer
    dut = SoftwareDUT(Renderer(image_size=877, fov_deg=10.0, magnitude_limit=7.5))
    body = RigidBody(attitude=(83.8, -5.4, 0.0), omega=(4.0, 3.0, 5.0))
    history = run_detumble(dut, body, RateController(gain=2.0), duration_s=20.0, dt=0.1)
    start, end = history[0], history[-1]
    assert start > 5.0, start
    assert end < 0.2 * start, f"did not detumble: |w| {start:.2f} -> {end:.2f} deg/s"
    print(f"dynamics.py self-check passed (detumble |w|: {start:.2f} -> {end:.3f} deg/s)")


if __name__ == "__main__":
    _demo()
