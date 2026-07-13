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
import time

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
    damping torque τ = -k·ω_filtered. Uses ONLY the estimated attitude — never the true state.

    ``alpha`` low-passes the raw differenced rate (exponential moving average):
    1.0 = raw differencing (noisy: attitude noise / dt amplifies into torque kicks),
    ~0.3 = smooth (a noise spike is diluted 3:1 against history; response lags a few steps).
    Differencing amplifies estimate noise, so any real sensor path wants alpha < 1.
    """

    def __init__(self, gain: float = 2.0, alpha: float = 1.0):
        self.gain = gain
        self.alpha = alpha
        self._prev = None                     # (attitude, t) of the last usable estimate
        self._omega_filt = (0.0, 0.0, 0.0)    # EMA of the differenced rate estimate

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
        omega_raw = (wrap_deg(estimate[0] - pra) / dt,
                     (estimate[1] - pdec) / dt,
                     wrap_deg(estimate[2] - proll) / dt)
        self._omega_filt = tuple((1.0 - self.alpha) * f + self.alpha * r
                                 for f, r in zip(self._omega_filt, omega_raw))
        self._prev = (estimate, t)
        return tuple(-self.gain * w for w in self._omega_filt)


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


def run_hil_detumble(host: str, gain: float = 0.3, alpha: float = 0.5,
                     control_period: float = 2.0, window_s: float = 4.0,
                     duration_s: float = 90.0, omega0=(1.5, 0.3, 2.0),
                     stop_below: float = 0.05, roll_sign: float = 1.0) -> list:
    """
    REAL-TIME hardware-in-the-loop detumble: the displayed field tumbles continuously —
    no stop-and-stare. The plant never pauses; delay and estimate latency act on the loop
    exactly as they would on a spacecraft.

    Plant: the running simulator itself. ``tumble <rates> forever`` makes it integrate the
    attitude continuously at the commanded body rates; this client changes those rates by
    torque (ω ← ω − k·ω̂·T, inertia 1), so "applying force" = re-commanding the new rates.
    The controller measures ω̂ ONLY from the tracker: it collects timestamped estimates
    (``est_t`` is the serve-side clock, so differencing needs no clock sync), unwraps
    ra/roll, and fits the rate over the trailing ``window_s`` endpoints (a longer baseline
    divides estimate noise, the fix for raw-differencing noise ≈ σ_att/dt), EMA-filtered
    by ``alpha``. Phase lag from the pipeline delay is genuinely present — keep
    ``gain·control_period`` ≲ 0.8 or the loop hunts.

    ``roll_sign``: −1 if the tracker's roll convention is mirrored (symptom: roll rate
    grows while ra/dec damp). Star-poor pointings just pause updates (rates hold) until
    estimates return. Real-camera caveat: motion blur — at ω deg/s and exposure T_exp the
    streak is ω·T_exp degrees; keep initial rates low or exposure short.

    Returns the history of |ω_commanded| (= true rate; the sim executes commands exactly).
    """
    import collections

    from . import control as ctl
    from .attitude import wrap_deg

    omega = list(omega0)
    ctl.send_command(host, f"tumble {omega[0]:.4f} {omega[1]:.4f} {omega[2]:.4f} forever")
    estimates: collections.deque = collections.deque()  # (est_t, ra_unwrapped, dec, roll_unwrapped)
    unwrapped = None
    last_est_t = -1.0
    omega_filt = None
    history = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        step_end = time.monotonic() + control_period
        while time.monotonic() < step_end:              # gather fresh estimates as they stream in
            m = ctl.get_status(host)["metrics"]
            est, est_t = m.get("est"), m.get("est_t") or -1.0
            if est is not None and est_t > last_est_t:
                last_est_t = est_t
                ra, dec, roll = est[0], est[1], (roll_sign * est[2]) % 360.0
                if unwrapped is None:
                    unwrapped = [ra, dec, roll]
                else:                                    # accumulate shortest-path deltas
                    unwrapped[0] += wrap_deg(ra - unwrapped[0] % 360.0)
                    unwrapped[1] = dec
                    unwrapped[2] += wrap_deg(roll - unwrapped[2] % 360.0)
                estimates.append((est_t, *unwrapped))
            time.sleep(0.1)
        while estimates and estimates[0][0] < last_est_t - window_s:
            estimates.popleft()
        if len(estimates) >= 2:
            (t0, *a0), (t1, *a1) = estimates[0], estimates[-1]
            if t1 - t0 >= 0.5:                           # need a baseline for a stable rate
                rate = [(v1 - v0) / (t1 - t0) for v0, v1 in zip(a0, a1)]
                omega_filt = rate if omega_filt is None else \
                    [(1.0 - alpha) * f + alpha * r for f, r in zip(omega_filt, rate)]
                torque = [-gain * f for f in omega_filt]
                omega = [w + tq * control_period for w, tq in zip(omega, torque)]
                ctl.send_command(host, f"tumble {omega[0]:.4f} {omega[1]:.4f} {omega[2]:.4f} forever")
        mag_cmd = math.sqrt(sum(w * w for w in omega))
        mag_est = math.sqrt(sum(f * f for f in omega_filt)) if omega_filt else float("nan")
        history.append(mag_cmd)
        print(f"  |w| true={mag_cmd:5.2f} deg/s  estimated={mag_est:5.2f} deg/s  "
              f"(estimates in window: {len(estimates)})", flush=True)
        if omega_filt is not None and mag_cmd < stop_below:
            break
    ctl.send_command(host, f"tumble {omega[0]:.4f} {omega[1]:.4f} {omega[2]:.4f} forever")
    print(f"HIL real-time detumble: |w| {history[0]:.2f} -> {history[-1]:.3f} deg/s "
          f"over {len(history)} control steps")
    return history


def _demo(full_image_pipeline: bool = False) -> None:
    """
    Self-check: a headless closed loop detumbles — |ω| drops from tumbling to near zero.

    ``full_image_pipeline=True`` swaps in the ImageDUT so every step renders a real frame
    and solves it through image → C centroid extraction → TETRA (the full software chain);
    the default SoftwareDUT feeds unit vectors directly (fast, identifier+solve only).
    """
    from .dut import ImageDUT, SoftwareDUT
    from .renderer import Renderer
    renderer = Renderer(image_size=877, fov_deg=10.0, magnitude_limit=7.5)
    dut = ImageDUT(renderer) if full_image_pipeline else SoftwareDUT(renderer)
    label = "image-pipeline" if full_image_pipeline else "vector"

    def progress(t, truth, est, rate_mag):
        if full_image_pipeline and abs(t * 10) % 50 < 1:  # every 5 s of sim time (slow path)
            print(f"  t={t:5.1f}s  |w|={rate_mag:5.2f} deg/s  est={'ok' if est else 'NULL'}", flush=True)

    body = RigidBody(attitude=(83.8, -5.4, 0.0), omega=(4.0, 3.0, 5.0))
    history = run_detumble(dut, body, RateController(gain=2.0), duration_s=20.0, dt=0.1,
                           on_step=progress)
    start, end = history[0], history[-1]
    assert start > 5.0, start
    assert end < 0.2 * start, f"did not detumble: |w| {start:.2f} -> {end:.2f} deg/s"
    print(f"dynamics.py self-check passed ({label} detumble |w|: {start:.2f} -> {end:.3f} deg/s)")


if __name__ == "__main__":
    import sys
    _demo(full_image_pipeline="--image" in sys.argv)
