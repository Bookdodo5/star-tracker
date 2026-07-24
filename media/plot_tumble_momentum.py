"""
Angular-momentum test of the two tumbles — free drift, NO detumble controller running.

Both trajectories start from the same attitude and the same initial body angular velocity, are
assigned the same inertia, and are measured the same way: body ω is recovered from successive
attitudes (ω = -log(R·R_prevᵀ)/dt), then L_inertial = Rᵀ·(I·ω).

For a torque-free rigid body L_inertial must be constant in ALL THREE components. `free_tumble`
is; `gimbal_tumble` is not — so the constant-(RA,DEC,ROLL)-rate motion is not something any body
can do without a motor continuously feeding it torque. The bottom row is that required torque.

Note |L| and rotational energy are NOT sufficient tests — they can stay conserved while the L
direction precesses. The component plot (middle row) is the real test.

Output: outputs/tumble_momentum.png     Run: python media/plot_tumble_momentum.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.commands import Resolver, parse_commands
from simulator.freebody import attitude_to_matrix, logvec

OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

START = (80.0, 35.0, 10.0)
INERTIA = np.array([1.0, 1.7, 2.5])
RATES = (1.0, 1.0, 0.5)      # deg/s, the gimbal_tumble command rates
DURATION = 50.0              # stays below the dec=+90 clamp (dec starts at 35, rises 1 deg/s)
DT = 0.02


def measure(command, duration=DURATION, dt=DT):
    """
    Runs a command open-loop and returns time, body ω, inertial L, and |dL/dt|.

    ω and L are derived only from the attitude stream, so both tumbles are measured by the
    identical estimator — no model is assumed for either.
    """
    resolver = Resolver(parse_commands(f"point_at {START[0]} {START[1]} {START[2]}\n{command}"),
                        (0.0, 0.0, 0.0))
    steps = int(duration / dt) + 1
    mats = [attitude_to_matrix(*resolver.attitude(i * dt)[0]) for i in range(steps)]
    omega = np.array([-logvec(mats[k + 1] @ mats[k].T) / dt for k in range(steps - 1)])
    momentum = np.array([mats[k].T @ (INERTIA * omega[k]) for k in range(steps - 1)])
    t = np.arange(len(omega)) * dt
    torque = np.linalg.norm(np.diff(momentum, axis=0), axis=1) / dt
    return t, omega, momentum, torque


def main():
    # Same start attitude, same initial body rate, same inertia — only the motion law differs.
    t, omega_g, mom_g, torque_g = measure(f"gimbal_tumble {RATES[0]} {RATES[1]} {RATES[2]} forever")
    w0 = np.degrees(omega_g[0])
    free_cmd = (f"free_tumble {w0[0]:.6f} {w0[1]:.6f} {w0[2]:.6f} "
                f"{INERTIA[0]} {INERTIA[1]} {INERTIA[2]} forever")
    _, omega_f, mom_f, torque_f = measure(free_cmd)

    l0 = np.linalg.norm(mom_g[0])
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    columns = [
        (0, omega_g, mom_g, torque_g, "gimbal_tumble — constant RA/DEC/ROLL rates\n(kinematic)"),
        (1, omega_f, mom_f, torque_f, "free_tumble — torque-free rigid body\n(Euler's equation)"),
    ]
    for col, omega, momentum, torque, title in columns:
        top, mid, bot = axes[0][col], axes[1][col], axes[2][col]

        for i, (lbl, color) in enumerate(zip("xyz", ("tab:blue", "tab:orange", "tab:green"))):
            top.plot(t, np.degrees(omega[:, i]), color=color, label=f"ω_{lbl} (body)")
        top.plot(t, np.degrees(np.linalg.norm(omega, axis=1)), "k--", lw=1.2, label="|ω|")
        top.set_title(title, fontweight="bold")
        top.set_ylabel("body rate (°/s)"); top.legend(fontsize=8, ncol=2); top.grid(alpha=0.3)

        for i, (lbl, color) in enumerate(zip("XYZ", ("tab:blue", "tab:orange", "tab:green"))):
            mid.plot(t, momentum[:, i] / l0, color=color, label=f"L_{lbl} (inertial)")
        mid.plot(t, np.linalg.norm(momentum, axis=1) / l0, "k--", lw=1.2, label="|L|")
        spread = (momentum.max(axis=0) - momentum.min(axis=0)) / l0 * 100.0
        mid.set_ylabel("L / |L₀|"); mid.grid(alpha=0.3); mid.legend(fontsize=8, ncol=2)
        mid.text(0.02, 0.04, f"component drift: {spread[0]:.2f}%, {spread[1]:.2f}%, {spread[2]:.2f}%",
                 transform=mid.transAxes, fontsize=9,
                 bbox=dict(fc="mistyrose" if spread.max() > 1 else "honeydew", ec="0.6"))

        bot.semilogy(t[1:], np.maximum(torque / l0, 1e-16), color="tab:red")
        bot.set_ylabel("|dL/dt| / |L₀|  (1/s)"); bot.set_xlabel("time (s)")
        bot.grid(alpha=0.3, which="both")
        note = ("torque required to sustain this motion" if col == 0 else
                "at the finite-difference floor — i.e. zero")
        bot.text(0.02, 0.88, note, transform=bot.transAxes, fontsize=9)

    for row in axes[:2]:                   # shared linear scales so the columns compare honestly
        lo = min(a.get_ylim()[0] for a in row); hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)
    peak = max(torque_g.max(), torque_f.max()) / l0   # log row: a decade of headroom each side
    for a in axes[2]:
        a.set_ylim(min(torque_f.min(), torque_g.min()) / l0 / 10.0, peak * 10.0)

    fig.suptitle("Is the tumble torque-free? Inertial angular momentum, no controller running\n"
                 "identical start attitude, identical initial body rate, identical inertia",
                 fontweight="bold")
    path = os.path.join(OUT, "tumble_momentum.png")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=130); plt.close(fig)

    for name, momentum, torque in [("gimbal_tumble", mom_g, torque_g), ("free_tumble", mom_f, torque_f)]:
        spread = (momentum.max(axis=0) - momentum.min(axis=0)) / l0 * 100.0
        print(f"{name:15s} L drift/|L0| = {spread[0]:6.2f}% {spread[1]:6.2f}% {spread[2]:6.2f}%   "
              f"|L| drift = {(np.ptp(np.linalg.norm(momentum, axis=1))/l0*100):5.2f}%   "
              f"peak |dL/dt|/|L0| = {torque.max()/l0:.3e} 1/s")
    print("wrote", path)


if __name__ == "__main__":
    main()
