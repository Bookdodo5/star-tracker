"""
Presentation figures for the detumble HIL results (lab-professor slides).

Four figures, each titled with its conclusion, big fonts, no inside jargon:
  1_detumble_works.png   - rotation rate driven to ~0 (successful rate-detumbles)
  2_gain_matters.png     - diverged vs converged run, same axes
  3_point_and_stabilize.png - hero figure: pointing error + rate, final target run
  4_sky_path.png         - the flight to the target on the sky

Reuses the parsing/rate machinery from plot_detumble_logs.py.
Usage: python tools/plot_presentation_figs.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_detumble_logs import (ROOT, parse_log, quat_rate,
                                pointing_error_deg, wrap180)

OUT = os.path.join(ROOT, "outputs", "presentation_figs")

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.labelsize": 16, "lines.linewidth": 2.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})


def segment(log, session, t0, t1):
    """Solved frames of one run window, time rebased to 0."""
    data = parse_log(os.path.join(ROOT, log))[session]
    seg = data[(data[:, 0] >= t0) & (data[:, 0] <= t1)].copy()
    seg[:, 0] -= seg[0, 0]
    return seg


def tumble_onset(tm, rate, threshold=2.0):
    """First time the (smoothed) rate exceeds threshold — trims the pre-run
    quiet padding so every curve starts at its own tumble."""
    idx = np.argmax(rate > threshold)
    return tm[idx] if rate[idx] > threshold else tm[0]


def fig1_detumble_works():
    runs = [
        ("Run A", segment("newrun.log", 6, 87, 110)),
        ("Run B", segment("newrun.log", 6, 449, 486)),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for (label, seg), color in zip(runs, ["tab:blue", "tab:green"]):
        tm, rate = quat_rate(seg)
        t0 = tumble_onset(tm, rate)
        keep = tm >= t0
        tm, rate = tm[keep] - t0, rate[keep]
        ax.plot(tm, rate, color=color,
                label=f"{label} (started at {rate[0]:.0f} °/s)")
        ax.annotate(f"{rate[-1]:.1f} °/s", (tm[-1], rate[-1]),
                    xytext=(8, 8), textcoords="offset points",
                    color=color, fontweight="bold")
    ax.set_title("The controller stops the tumble\n(star tracker is the only sensor)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotation rate (°/s)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "1_detumble_works.png"))
    plt.close(fig)


def fig2_gain_matters():
    bad = segment("newrun.log", 6, 362, 430)    # diverged
    good = segment("newrun.log", 6, 449, 486)   # converged
    fig, ax = plt.subplots(figsize=(11, 6))
    for seg, color, label in [(bad, "tab:red", "gain too high → spins UP"),
                              (good, "tab:green", "tuned gain → spins down")]:
        tm, rate = quat_rate(seg)
        t0 = tumble_onset(tm, rate)
        keep = tm >= t0
        ax.semilogy(tm[keep] - t0, np.maximum(rate[keep], 0.1),
                    color=color, label=label)
    ax.annotate("beyond ~40 °/s stars streak,\nthe tracker loses lock",
                xy=(52, 60), color="tab:red", fontsize=13, ha="center")
    ax.set_title("Same controller, wrong gain — feedback of a noisy,\ndelayed sensor can destabilize")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotation rate (°/s, log scale)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "2_gain_matters.png"))
    plt.close(fig)


def target_run(log="newrun4.log", session=0, t0=12, t1=125):
    """Glitch-filtered target-run segment, hold-tail restored.

    The parser collapses repeated identical solves, so once the tracker locks
    (identical frame -> identical solve) the final steady hold produces no rows;
    append one row at the log's true last frame time so plots show the hold."""
    import re
    seg = segment(log, session, t0, t1)
    ra, dec = wrap180(seg[:, 1]), seg[:, 2]
    d_prev = np.hypot(np.diff(ra, prepend=ra[0]), np.diff(dec, prepend=dec[0]))
    d_next = np.hypot(np.diff(ra, append=ra[-1]), np.diff(dec, append=dec[-1]))
    d_prev[0], d_next[-1] = np.inf, np.inf  # endpoints have only one neighbor
    seg = seg[~((d_prev > 15) & (d_next > 15))]
    t_last = 0.0
    with open(os.path.join(ROOT, log), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"t=\s*([\d.]+)s", line)
            if m:
                t_last = max(t_last, float(m.group(1)))
    if t_last - t0 > seg[-1, 0]:
        tail = seg[-1].copy()
        tail[0] = t_last - t0
        seg = np.vstack([seg, tail])
    return seg


def fig3_locked_on_target():
    seg = target_run()
    tt = seg[:, 0]
    perr = pointing_error_deg(seg[:, 1], seg[:, 2], 0.0, 0.0)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(tt, np.maximum(perr, 0.12), color="tab:blue")
    ax.axhspan(0.05, 0.5, color="tab:green", alpha=0.18)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 120)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle to target (°)")
    ax.set_title("From 80° away to locked on target")
    ax.text(tt[-1], float(np.median(perr[-10:])), " 0.3°, holding",
            color="tab:green", fontweight="bold", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "3_locked_on_target.png"))
    plt.close(fig)
    return float(np.median(perr[-10:]))


def fig4_bullseye():
    seg = target_run()
    ra, dec, tt = wrap180(seg[:, 1]), seg[:, 2], seg[:, 0]
    perr = pointing_error_deg(ra, dec, 0.0, 0.0)
    azimuth = np.arctan2(dec, ra * np.cos(np.radians(dec)))
    rings = np.array([0.3, 1, 3, 10, 30, 90])

    def rmap(deg):
        """Log radial mapping: 0.05 deg at the center, each ring ~3x farther."""
        return np.log10(np.maximum(deg, 0.05) / 0.05)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.plot(azimuth, rmap(perr), color="tab:blue", lw=2, alpha=0.9)
    sc = ax.scatter(azimuth, rmap(perr), c=tt, cmap="viridis", s=12, zorder=3)
    ax.scatter([0], [0], marker="*", s=500, color="red", zorder=5)
    ax.set_rgrids(rmap(rings), labels=[f"{r:g}°" for r in rings],
                  angle=90, fontsize=13)
    ax.set_thetagrids([])
    ax.set_rmax(rmap(95))
    ax.grid(True, alpha=0.4)
    ax.set_title("Camera aim, seen from the target\n(rings: angular distance)", pad=20)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "4_bullseye.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig1_detumble_works()
    fig2_gain_matters()
    settled = fig3_locked_on_target()
    fig4_bullseye()
    assert settled < 0.5, f"hero figure claim broken: settled at {settled:.2f} deg"
    print(f"4 figures written to {OUT}  (final pointing error {settled:.2f} deg)")


if __name__ == "__main__":
    main()
