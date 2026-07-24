"""
Torque-free rigid-body attitude dynamics — a *physical* tumble, unlike the decoupled
per-axis angle-rate ``gimbal_tumble``.

A real spacecraft with no external torque (ΣF, Στ = 0) does NOT drift at constant
(dRA/dt, dDEC/dt, dROLL/dt). Its body angular velocity ω obeys Euler's equation
``I·ω̇ = -ω × (I·ω)`` — so for an asymmetric inertia the rate vector nutates in the body
frame and the boresight traces a looping ("polhode") path on the sky. This module models
that: state is a rotation matrix (catalog→body) + body-frame ω, integrated with Poisson's
kinematic equation and Euler's equation.

The attitude matrix is built to reproduce the renderer's exact projection basis *including
roll* (``media/render_catalog_test_image.camera_basis`` + the in-plane roll rotation), so a
matrix ⇄ (ra, dec, roll) round-trip is exact and a tumble displayed on the phone is a
genuine rigid rotation of the real star field.

Also here: ``BodyRateController`` — the honest detumble controller. It estimates body ω by
differencing successive *full attitudes* (relative rotation → log map), not by differencing
Euler angles, so it works on a real tumble. Torque τ = -gain·ω̂ (body frame).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "media"))
import render_catalog_test_image as R  # noqa: E402  (camera_basis — one source of truth)

from .attitude import vec_to_radec  # noqa: E402


def attitude_to_matrix(ra: float, dec: float, roll: float) -> np.ndarray:
    """
    (ra, dec, roll) → 3×3 rotation matrix whose rows are the camera (body) axes in catalog
    coordinates. Row 2 is the boresight; rows 0/1 are the roll-rotated image x/y axes,
    matching exactly how ``renderer._project`` applies roll (in-plane pixel rotation).
    """
    cx0, cy0, cz = (np.array(v) for v in R.camera_basis(ra, dec))
    theta = math.radians(roll)
    ex = math.cos(theta) * cx0 - math.sin(theta) * cy0
    ey = math.sin(theta) * cx0 + math.cos(theta) * cy0
    return np.array([ex, ey, cz])


def matrix_to_attitude(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`attitude_to_matrix` → (ra_deg, dec_deg, roll_deg in [0,360))."""
    ra, dec = vec_to_radec(tuple(rot[2]))
    cx0, cy0, _ = (np.array(v) for v in R.camera_basis(ra, dec))
    ex = rot[0]
    roll = math.degrees(math.atan2(-float(ex @ cy0), float(ex @ cx0))) % 360.0
    return ra, dec, roll


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rodrigues(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Rotation matrix exp(-[ω]× dt) — Poisson's kinematic increment for a catalog→body matrix
    with a body-frame ω.

    The sign is NOT free: it must pair with Euler's equation ``I·ω̇ = -ω × (I·ω)``, because
    only that pair gives ``d/dt(Rᵀ·I·ω) = 0`` — conservation of the *inertial* angular
    momentum vector, which is what makes the motion genuinely torque-free. Flipping this sign
    alone still conserves |L| and energy (both are sign-blind), so a magnitude-only self-check
    will not catch it; the inertial L direction precesses instead.

    Consequence for the roll coordinate: a right-handed +ω_z about the boresight *decreases*
    ``roll``, because ``roll`` is an image-plane rotation that corresponds to a rotation about
    **−boresight** (the ``roll <rate>`` command has the same handedness).
    """
    angle = float(np.linalg.norm(omega)) * dt
    if angle < 1e-15:
        return np.eye(3)
    k = _skew(omega / np.linalg.norm(omega))
    return np.eye(3) - math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def logvec(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (axis·angle). Inverse of ``exp([v]×)``."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1.0) * 0.5)))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return v * (angle / (2.0 * math.sin(angle)))


class FreeRigidBody:
    """
    Torque-free (default) rigid body. Same interface as ``dynamics.RigidBody``
    (``attitude`` / ``step`` / ``rate_magnitude``) so ``run_detumble`` drives it unchanged.

    ``omega_body_deg`` is the initial body angular velocity in deg/s; ``inertia`` are the
    principal moments (ratios matter, not units). Asymmetric inertia ⇒ real nutation.
    """

    def __init__(self, attitude=(83.8, -5.4, 0.0), omega_body_deg=(4.0, 3.0, 5.0),
                 inertia=(1.0, 1.7, 2.5)):
        self.rot = attitude_to_matrix(*attitude)          # catalog→body
        self.omega = np.radians(np.array(omega_body_deg, float))  # rad/s, body frame
        self.inertia = np.array(inertia, float)

    def step(self, dt: float, torque_body=(0.0, 0.0, 0.0)) -> None:
        """Advance dt seconds under a body-frame torque (default 0 = torque-free tumble).

        Euler's equation is integrated with RK4 (near-exact conservation of angular momentum
        and energy on a torque-free body); the attitude is rotated by the mid-step ω, which
        rodrigues advances exactly for a constant rate.
        """
        tau = np.array(torque_body, float)

        def omega_dot(w):
            return (tau - np.cross(w, self.inertia * w)) / self.inertia

        substeps = max(1, int(math.ceil(dt / 0.05)))
        h = dt / substeps
        for _ in range(substeps):
            w = self.omega
            k1 = omega_dot(w)
            k2 = omega_dot(w + 0.5 * h * k1)
            k3 = omega_dot(w + 0.5 * h * k2)
            k4 = omega_dot(w + h * k3)
            new_omega = w + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            self.rot = rodrigues(0.5 * (w + new_omega), h) @ self.rot
            self.omega = new_omega

    def attitude(self) -> tuple[float, float, float]:
        return matrix_to_attitude(self.rot)

    def rate_magnitude(self) -> float:
        """|ω| in deg/s — the detumble target (drive to ~0)."""
        return math.degrees(float(np.linalg.norm(self.omega)))


class BodyRateController:
    """
    Detumble (and optionally point-and-hold) controller that works on a *real* tumble.

    It recovers the body angular velocity from two successive full attitude estimates
    (relative rotation → log map → ω, exactly inverting the plant's Poisson update) and
    commands τ = -gain·(ω̂ - ω_desired) in the body frame. Uses ONLY the estimated attitude —
    never the true state. ``alpha`` EMA-filters the rate estimate.

    With ``target=(ra, dec, roll)`` this becomes a cascaded pointing controller:
    the outer loop turns the attitude error into a desired rate ``ω_des = kp·φ``, where φ is
    the body-frame rotation vector that carries the current attitude onto the target (from the
    log map — NOT three separate Euler-angle errors, which do not form a vector and would
    misbehave near the poles). ``slew_max`` (deg/s) caps |ω_des| so the tracker never sees
    blur-fast motion. Pure detumble is the ``ω_des = 0`` special case.

    Stability: the outer loop must be slower than the inner one — keep ``kp`` well below
    ``gain`` (default 0.4 vs 2.0), or the pointing loop hunts.
    """

    def __init__(self, gain: float = 2.0, alpha: float = 1.0, target=None,
                 kp: float = 0.4, slew_max: float = 8.0):
        self.gain = gain
        self.alpha = alpha
        self.kp = kp
        self.slew_max = math.radians(slew_max) if slew_max else None
        self.target_rot = attitude_to_matrix(*target) if target is not None else None
        self._prev = None                        # (attitude_matrix, t) of last usable estimate
        self._omega_filt = np.zeros(3)

    def _desired_rate(self, rot: np.ndarray) -> np.ndarray:
        """Outer loop: body-frame rate that drives the current attitude onto the target."""
        if self.target_rot is None:
            return np.zeros(3)
        # Plant convention rot_new = exp(-[φ]×)·rot, so the body-frame rotation taking the
        # current attitude to the target is φ = -log(R_target·R_currentᵀ).
        phi = -logvec(self.target_rot @ rot.T)
        omega_des = self.kp * phi
        if self.slew_max is not None:
            speed = float(np.linalg.norm(omega_des))
            if speed > self.slew_max:
                omega_des = omega_des * (self.slew_max / speed)
        return omega_des

    def torque(self, estimate, t: float):
        if estimate is None:
            return (0.0, 0.0, 0.0)
        rot = attitude_to_matrix(*estimate)
        if self._prev is None:
            self._prev = (rot, t)
            return (0.0, 0.0, 0.0)
        rot_prev, t_prev = self._prev
        dt = t - t_prev
        if dt <= 0:
            return (0.0, 0.0, 0.0)
        # Plant update is rot = exp(-[ω]×dt)·rot_prev, so rot·rot_prevᵀ = exp(-[ω]×dt):
        # log map recovers -ω·dt in the same (body) frame the plant applies torque in.
        omega = -logvec(rot @ rot_prev.T) / dt
        self._omega_filt = (1.0 - self.alpha) * self._omega_filt + self.alpha * omega
        self._prev = (rot, t)
        return tuple(-self.gain * (self._omega_filt - self._desired_rate(rot)))


def _demo() -> None:
    """Self-check: matrix round-trip, pure-roll sign, conservation, and a closed detumble."""
    # (1) attitude ⇄ matrix round-trips exactly.
    for att in [(83.8, -5.4, 0.0), (12.0, 60.0, 200.0), (300.0, -80.0, 47.0)]:
        ra, dec, roll = matrix_to_attitude(attitude_to_matrix(*att))
        assert abs(ra - att[0]) < 1e-6 and abs(dec - att[1]) < 1e-6 and abs(roll - att[2]) < 1e-6, att

    # (2) pure boresight spin (+ω_z, right-handed) holds ra/dec and changes roll at the
    #     commanded rate — NEGATIVE, because `roll` is a rotation about -boresight.
    spin = FreeRigidBody(attitude=(100.0, 0.0, 0.0), omega_body_deg=(0.0, 0.0, 10.0),
                         inertia=(1.0, 1.0, 1.0))
    spin.step(2.0)
    ra, dec, roll = spin.attitude()
    assert abs(ra - 100.0) < 1e-4 and abs(dec) < 1e-4, (ra, dec)
    assert abs(((roll + 20.0 + 180) % 360) - 180) < 1e-3, roll

    # (3) torque-free asymmetric body: the INERTIAL angular momentum vector Rᵀ·I·ω must be
    #     constant in all three components — the real test of torque-free motion. |L| and
    #     energy alone are sign-blind and stay conserved even with a wrong kinematic sign,
    #     so they cannot catch a Poisson/Euler mismatch; the L direction precesses instead.
    body = FreeRigidBody(omega_body_deg=(4.0, 3.0, 5.0), inertia=(1.0, 1.7, 2.5))
    def inertial_momentum(b):
        return b.rot.T @ (b.inertia * b.omega)
    def energy(b):
        return float(0.5 * b.omega @ (b.inertia * b.omega))
    l0, e0 = inertial_momentum(body), energy(body)
    rates = []
    for _ in range(400):
        body.step(0.05)
        rates.append(body.rate_magnitude())
        assert np.allclose(inertial_momentum(body), l0, atol=1e-6 * np.linalg.norm(l0) + 1e-9), \
            f"inertial angular momentum drifted: {l0} -> {inertial_momentum(body)}"
    assert abs(energy(body) - e0) < 1e-4 * e0, (e0, energy(body))
    assert max(rates) - min(rates) > 0.2, "asymmetric body should nutate (|w| varies)"

    # (4) closed loop: BodyRateController drives a free tumble to rest using attitude-only
    #     rate estimates (perfect sensor here — the point is the frame math is consistent).
    plant = FreeRigidBody(omega_body_deg=(4.0, 3.0, 5.0), inertia=(1.0, 1.7, 2.5))
    ctrl = BodyRateController(gain=2.0)
    dt = 0.1
    start = plant.rate_magnitude()
    for i in range(300):
        torque = ctrl.torque(plant.attitude(), i * dt)
        plant.step(dt, torque)
    end = plant.rate_magnitude()
    assert end < 0.05 * start, f"free-body detumble failed: |w| {start:.2f} -> {end:.3f} deg/s"

    # (5) point-and-hold: same loop with a target must converge in BOTH boresight and roll,
    #     from a tumbling start, and then stay there.
    from .attitude import attitude_error
    target = (100.0, 10.0, 30.0)
    plant = FreeRigidBody(attitude=(83.8, -5.4, 0.0), omega_body_deg=(4.0, 3.0, 5.0),
                          inertia=(1.0, 1.7, 2.5))
    ctrl = BodyRateController(gain=2.0, target=target, kp=0.4, slew_max=8.0)
    for i in range(600):
        plant.step(dt, ctrl.torque(plant.attitude(), i * dt))
    point_err, roll_err = attitude_error(target, plant.attitude())
    assert point_err < 0.05 and roll_err < 0.05, f"pointing failed: {point_err:.3f} deg, {roll_err:.3f} deg"
    print(f"freebody.py self-check passed (detumble |w|: {start:.2f} -> {end:.3f} deg/s; "
          f"point-and-hold err {point_err:.4f} deg boresight, {roll_err:.4f} deg roll)")


if __name__ == "__main__":
    _demo()
