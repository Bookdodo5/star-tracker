"""
Run the closed-loop detumble on a *physical* (torque-free rigid-body) tumble and plot it.

Unlike the old constant-rate ``gimbal_tumble``, the plant here is ``FreeRigidBody``: its body
angular velocity obeys Euler's equation, so the boresight nutates on the sky. The only sensor
is the star tracker (``SoftwareDUT``), and ``BodyRateController`` recovers the body rate by
differencing successive full attitudes, then damps it.

Outputs (to outputs/):
  free_detumble_sky.png        boresight path on the sky — free coast vs the controlled run
  free_detumble_attitude.png   RA / DEC / ROLL and |ω| (true + tracker-estimated) vs time

Run: python media/plot_free_detumble.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.dynamics import run_detumble
from simulator.freebody import BodyRateController, FreeRigidBody, logvec, attitude_to_matrix
from simulator.dut import SoftwareDUT
from simulator.renderer import Renderer
from simulator.attitude import attitude_error

OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

ATT0 = (83.8, -5.4, 0.0)          # start pointing (Orion-ish, a dense field the tracker solves)
OMEGA0 = (4.0, 3.0, 5.0)          # initial body rate, deg/s
INERTIA = (1.0, 1.7, 2.5)         # asymmetric principal moments → nutation
TARGET = (100.0, 10.0, 30.0)      # commanded RA / DEC / ROLL to detumble onto and hold
DURATION = 40.0
DT = 0.1


def coast_path(duration, dt):
    """Torque-free tumble (no control): the raw physical motion, for the sky-path reference."""
    body = FreeRigidBody(attitude=ATT0, omega_body_deg=OMEGA0, inertia=INERTIA)
    ras, decs, ts = [], [], []
    for i in range(int(duration / dt)):
        ra, dec, _ = body.attitude()
        ras.append(ra); decs.append(dec); ts.append(i * dt)
        body.step(dt)
    return np.array(ras), np.array(decs), np.array(ts)


def run_controlled():
    """
    Detumble onto a commanded (RA, DEC, ROLL) with the star tracker in the loop; collect
    truth + estimated-rate history. The controller sees only the tracker's estimate.
    """
    renderer = Renderer(image_size=877, fov_deg=10.0, magnitude_limit=7.5)
    dut = SoftwareDUT(renderer)
    body = FreeRigidBody(attitude=ATT0, omega_body_deg=OMEGA0, inertia=INERTIA)
    controller = BodyRateController(gain=2.0, alpha=0.5, target=TARGET, kp=0.4, slew_max=8.0)

    log = {"t": [], "ra": [], "dec": [], "roll": [], "w_true": [], "w_est": [], "solved": []}
    prev = {"est": None, "t": None}

    def on_step(t, truth, est, w_true):
        log["t"].append(t)
        log["ra"].append(truth[0]); log["dec"].append(truth[1]); log["roll"].append(truth[2])
        log["w_true"].append(w_true)
        log["solved"].append(est is not None)
        # Tracker-derived |ω|: difference successive full attitude estimates (same log-map the
        # controller uses). NaN when the field did not solve, so gaps show as breaks.
        w_est = np.nan
        if est is not None and prev["est"] is not None and t > prev["t"]:
            rel = attitude_to_matrix(*est) @ attitude_to_matrix(*prev["est"]).T
            w_est = np.degrees(np.linalg.norm(logvec(rel)) / (t - prev["t"]))
        if est is not None:
            prev["est"] = est; prev["t"] = t
        log["w_est"].append(w_est)

    run_detumble(dut, body, controller, duration_s=DURATION, dt=DT, on_step=on_step)
    return {k: np.array(v, dtype=float) if k != "solved" else np.array(v) for k, v in log.items()}


def plot_sky(coast, controlled):
    """
    Two boresight sky-paths, time-coloured:
      left  — the raw torque-free tumble (no control): nutation sweeps from Euler's equation.
      right — the detumble run (zoomed): the tracker kills the rate in a short arc, then holds.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6))

    cra, cdec, ct = coast
    sc0 = left.scatter(cra, cdec, c=ct, cmap="plasma", s=5)
    left.plot(cra[0], cdec[0], "o", color="lime", ms=9, label="start", zorder=5)
    fig.colorbar(sc0, ax=left, label="time (s)")
    left.set_title("Free tumble — no control\n(torque-free rigid body, boresight nutates)")
    left.set_xlabel("RA (°)"); left.set_ylabel("DEC (°)"); left.grid(alpha=0.3)
    left.legend(loc="best", fontsize=8)

    sc1 = right.scatter(controlled["ra"], controlled["dec"], c=controlled["t"], cmap="viridis", s=14)
    right.plot(controlled["ra"][0], controlled["dec"][0], "o", color="lime", ms=9, label="start", zorder=5)
    right.plot(TARGET[0], TARGET[1], "X", color="red", ms=14, mew=2,
               label=f"target ({TARGET[0]}, {TARGET[1]}, roll {TARGET[2]}°)", zorder=6)
    right.plot(controlled["ra"][-1], controlled["dec"][-1], "*", color="black", ms=13,
               label="final attitude", zorder=7)
    fig.colorbar(sc1, ax=right, label="time (s)")
    right.set_title("Detumble onto a commanded attitude\n(star tracker is the only sensor)")
    right.set_xlabel("RA (°)"); right.set_ylabel("DEC (°)"); right.grid(alpha=0.3)
    right.legend(loc="best", fontsize=8)

    path = os.path.join(OUT, "free_detumble_sky.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_attitude(c):
    """Stacked RA / DEC / ROLL and |ω| (true + tracker-estimated) versus time."""
    def unwrapped(series, target):
        """Continuous angle series for plotting: removes 360° wraps, then picks the branch
        nearest the target so the target line is meaningful (dec never wraps, so it is left)."""
        out = np.degrees(np.unwrap(np.radians(series)))
        return out - 360.0 * round((out[-1] - target) / 360.0)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for ax, key, lbl, tgt in zip(axes[:3], ("ra", "dec", "roll"),
                                 ("RA (°)", "DEC (°)", "ROLL (°)"), TARGET):
        series = c[key] if key == "dec" else unwrapped(c[key], tgt)
        ax.plot(c["t"], series, color="tab:blue", label="truth")
        ax.axhline(tgt, color="red", ls="--", lw=1.2, label=f"target {tgt}°")
        ax.set_ylabel(lbl); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    axes[0].set_title("Attitude and rotation rate — detumble onto a commanded RA/DEC/ROLL")
    w = axes[3]
    w.semilogy(c["t"], np.maximum(c["w_true"], 1e-3), color="tab:blue", label="|ω| true (commanded plant)")
    w.semilogy(c["t"], np.maximum(c["w_est"], 1e-3), ".", ms=4, color="tab:orange",
               label="|ω| from tracker (attitude differencing)")
    w.set_ylabel("|ω| (°/s)"); w.set_xlabel("time (s)")
    w.grid(alpha=0.3, which="both"); w.legend(fontsize=8)
    path = os.path.join(OUT, "free_detumble_attitude.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main():
    print("Running free-tumble detumble (star tracker in the loop)...", flush=True)
    controlled = run_controlled()
    coast = coast_path(150.0, DT)   # long enough for the nutation sweeps to repeat
    p1 = plot_sky(coast, controlled)
    p2 = plot_attitude(controlled)
    solved = controlled["solved"].mean() * 100.0
    final = (controlled["ra"][-1], controlled["dec"][-1], controlled["roll"][-1])
    point_err, roll_err = attitude_error(TARGET, final)
    print(f"|ω|: {controlled['w_true'][0]:.2f} -> {controlled['w_true'][-1]:.3f} deg/s   "
          f"tracker solved {solved:.0f}% of frames")
    print(f"target {TARGET} -> final ({final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f})   "
          f"boresight err {point_err:.4f}°, roll err {roll_err:.4f}°")
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
