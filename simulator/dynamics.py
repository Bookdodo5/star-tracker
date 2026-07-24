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
                     stop_below: float = 0.05, roll_sign: float = 1.0,
                     flush_s: float = 1.5, torque_max: float = 1.0,
                     deadband: float = 0.15, target=None, kp: float = 0.08,
                     slew_max: float = 1.5, pointing_tol: float = 0.3) -> list:
    """
    REAL-TIME hardware-in-the-loop detumble: the displayed field tumbles continuously —
    no stop-and-stare. The plant never pauses; delay and estimate latency act on the loop
    exactly as they would on a spacecraft.

    Plant: the running simulator itself. ``gimbal_tumble <rates> forever`` makes it integrate the
    attitude continuously at the commanded body rates; this client changes those rates by
    torque (ω ← ω − k·ω̂·T, inertia 1), so "applying force" = re-commanding the new rates.
    The controller measures ω̂ ONLY from the tracker: it collects timestamped estimates
    (``est_t`` is the serve-side clock, so differencing needs no clock sync), unwraps
    ra/roll, and fits the rate as the least-squares slope over every estimate in the
    trailing ``window_s`` (longer baseline + all-samples regression both divide estimate
    noise — the fix for raw-differencing noise ≈ σ_att/dt), EMA-filtered by ``alpha``. Phase lag from the pipeline delay is genuinely present — keep
    ``gain·control_period`` ≲ 0.8 or the loop hunts.

    Start-up guard: estimates in flight when the loop starts belong to the pre-tumble
    field (a ``point_at`` jump away), so the first ``flush_s`` of estimates is discarded;
    ``torque_max`` clamps the per-axis torque so one corrupted rate window cannot slam
    the commanded rates; ``deadband`` (deg/s, at the base window) zeroes the torque when
    every axis' estimated rate is noise-level. While coasting, the fit window stretches
    (up to 4×``window_s``) and the threshold shrinks proportionally — a longer baseline
    averages noise down, so genuinely slow drift is eventually distinguished and damped,
    while pure noise never triggers actuation.

    Pointing mode (``target=(ra, dec, roll)``): cascaded control. The outer loop turns
    the attitude error (latest tracker estimate − target, shortest path) into a desired
    rate ``ω_des = clamp(−kp·err, ±slew_max)``; the inner loop is the same rate damper,
    now driving ``ω̂ → ω_des`` instead of → 0. ``slew_max`` caps the approach rate so the
    tracker never sees blur-fast motion; ``pointing_tol`` (deg) is the arrival gate for
    the early-stop. Pure detumble is the ``ω_des = 0`` special case.

    Stability: the rate measurement lags ~6–8 s (fit-window midpoint + EMA + control
    period + pipeline), so the outer loop must be slower than that lag: keep
    ``1/kp ≳ 1.5×lag`` (kp ≲ 0.1). kp=0.25 was measured to limit-cycle at ~8–12° with
    ~25 s period on the real rig — the classic saturated-P-with-delay oscillation.

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
    ctl.send_command(host, f"gimbal_tumble {omega[0]:.4f} {omega[1]:.4f} {omega[2]:.4f} forever")
    # Flush: estimates already in flight belong to the PRE-tumble field (possibly a huge
    # point_at jump away). Differencing across that discontinuity fakes a rate of tens of
    # deg/s and the first torque slams the plant. Wait out the pipeline, then discard
    # everything solved so far by starting the freshness cursor at the current est_t.
    time.sleep(flush_s)
    last_est_t = ctl.get_status(host)["metrics"].get("est_t") or -1.0
    estimates: collections.deque = collections.deque()  # (est_t, ra_unwrapped, dec, roll_unwrapped)
    unwrapped = None
    latest = None                                        # freshest wrapped estimate (pointing error)
    win = window_s                                       # current fit baseline (grows while coasting)
    omega_filt = None
    err = None                                           # pointing error (deg), None until an estimate
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
                latest = (ra, dec, roll)                 # freshest wrapped estimate (pointing error)
            time.sleep(0.1)
        while estimates and estimates[0][0] < last_est_t - win:
            estimates.popleft()
        if len(estimates) >= 2:
            t_first, t_last = estimates[0][0], estimates[-1][0]
            if t_last - t_first >= 0.5:                  # need a baseline for a stable rate
                # Least-squares slope over ALL estimates in the window, not just the two
                # endpoints: every sample contributes, so rate noise drops ~sqrt(N/6)x
                # (~2x at N≈25) at the same effective lag (both estimate the rate at the
                # window midpoint), and a single outlier can no longer own an endpoint.
                n = len(estimates)
                t_mean = sum(e[0] for e in estimates) / n
                denom = sum((e[0] - t_mean) ** 2 for e in estimates)
                rate = [sum((e[0] - t_mean) * e[axis] for e in estimates) / denom
                        for axis in (1, 2, 3)]
                omega_filt = rate if omega_filt is None else \
                    [(1.0 - alpha) * f + alpha * r for f, r in zip(omega_filt, rate)]
                # Pointing: attitude error (shortest path) → desired rate that nulls it,
                # capped at slew_max so the tracker never sees blur-fast motion. With no
                # target (pure detumble) the desired rate is simply zero.
                if target is not None and latest is not None:
                    err = [wrap_deg(latest[0] - target[0]),
                           latest[1] - target[1],
                           wrap_deg(latest[2] - target[2])]
                    omega_des = [max(-slew_max, min(slew_max, -kp * e)) for e in err]
                else:
                    omega_des = [0.0, 0.0, 0.0]
                # Deadband with patience: near zero the fitted rate is mostly estimate
                # noise (≈ 2σ_att / baseline). While every axis is below the threshold,
                # coast AND stretch the fit window — noise shrinks ~1/baseline, so the
                # threshold shrinks with it, and a genuinely drifting body (displacement
                # grows linearly with time) eventually crosses it. Any detection snaps
                # the window back to ``window_s`` for fast response.
                threshold = deadband * (window_s / win)
                rate_err = [f - d for f, d in zip(omega_filt, omega_des)]
                parked = (target is not None and err is not None
                          and max(abs(e) for e in err) <= pointing_tol
                          and all(abs(f) <= deadband for f in omega_filt))
                if parked:
                    # Park: inside tolerance with noise-level measured rates. Null the
                    # commanded rates outright — coasting here would freeze a residual
                    # ±deadband drift into the plant and the field would slowly orbit
                    # the target forever instead of stopping on it.
                    omega = [0.0, 0.0, 0.0]
                    torque = [0.0, 0.0, 0.0]
                    win = min(win * 1.5, 4.0 * window_s)
                elif target is None and all(abs(re) <= threshold for re in rate_err):
                    # Rate deadband is a DETUMBLE guard only. With a target, ω_des=kp·err
                    # sits below any deadband for |err| < deadband/kp (~2°), which turns
                    # the deadband into a position dead zone the field wanders in forever
                    # — so in pointing mode the only coast state is the park above.
                    win = min(win * 1.5, 4.0 * window_s)  # ponytail: cap staleness at 4x
                    torque = [0.0, 0.0, 0.0]
                else:
                    win = window_s
                    # Clamp per-axis torque: one corrupted window (blurred/false solve,
                    # discontinuity) must not spin the plant up faster than
                    # torque_max·control_period per step; an honest estimate self-corrects.
                    torque = [max(-torque_max, min(torque_max, -gain * re)) for re in rate_err]
                omega = [w + tq * control_period for w, tq in zip(omega, torque)]
                ctl.send_command(host, f"gimbal_tumble {omega[0]:.4f} {omega[1]:.4f} {omega[2]:.4f} forever")
        mag_cmd = math.sqrt(sum(w * w for w in omega))
        mag_est = math.sqrt(sum(f * f for f in omega_filt)) if omega_filt else float("nan")
        history.append(mag_cmd)
        err_txt = f"  err=({err[0]:+7.2f},{err[1]:+7.2f},{err[2]:+7.2f}) deg" if err else ""
        print(f"  |w| true={mag_cmd:5.2f} deg/s  estimated={mag_est:5.2f} deg/s{err_txt}  "
              f"(estimates in window: {len(estimates)})", flush=True)
        on_target = target is None or (err is not None and max(abs(e) for e in err) <= pointing_tol)
        if omega_filt is not None and mag_cmd < stop_below and on_target:
            break
    ctl.send_command(host, f"gimbal_tumble {omega[0]:.4f} {omega[1]:.4f} {omega[2]:.4f} forever")
    label = f"point+stabilize at ({target[0]}, {target[1]}, {target[2]})" if target else "detumble"
    print(f"HIL real-time {label}: |w| {history[0]:.2f} -> {history[-1]:.3f} deg/s "
          f"over {len(history)} control steps"
          + (f", final err=({err[0]:+.2f},{err[1]:+.2f},{err[2]:+.2f}) deg" if err else ""))
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
