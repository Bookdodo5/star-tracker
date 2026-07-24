"""
One clean |omega| vs time plot per detumble run (all 8 windows), presentation
style, minimal text. Also a combined 2x4 grid for side-by-side comparison.

Usage: python media/plot_all_omega.py
Output: outputs/presentation_figs/omega/
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_detumble_logs import ROOT, RUNS, parse_log, quat_rate, log_path
from plot_presentation_figs import tumble_onset

OUT = os.path.join(ROOT, "outputs", "presentation_figs", "omega")

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.titleweight": "bold",
    "lines.linewidth": 2.5, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})


def run_rate(log, session, t0, t1):
    """(t, |omega|) for one run window, t rebased to tumble onset."""
    data = parse_log(log_path(log))[session]
    seg = data[(data[:, 0] >= t0) & (data[:, 0] <= t1)]
    tm, rate = quat_rate(seg)
    onset = tumble_onset(tm, rate)
    keep = tm >= onset
    return tm[keep] - onset, rate[keep]


def main():
    os.makedirs(OUT, exist_ok=True)
    curves = []
    for log, si, t0, t1, label, _target in RUNS:
        tm, rate = run_rate(log, si, t0, t1)
        curves.append((label, tm, rate))
        diverged = "DIVERGED" in label
        color = "tab:red" if diverged else "tab:blue"
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(tm, rate, color=color)
        ax.annotate(f"{rate[-1]:.1f} °/s", (tm[-1], rate[-1]),
                    xytext=(6, 6), textcoords="offset points",
                    color=color, fontweight="bold")
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rotation rate (°/s)")
        # the initial point_at teleport reads as a fake huge spike — scale to
        # the physical data (everything after the first 3 s)
        steady = rate[tm > 3.0] if np.any(tm > 3.0) else rate
        ax.set_ylim(0, float(np.max(steady)) * 1.15)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"omega_{label}.png"))
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    for ax, (label, tm, rate) in zip(axes.flat, curves):
        diverged = "DIVERGED" in label
        ax.plot(tm, rate, color="tab:red" if diverged else "tab:blue", lw=2)
        ax.set_title(label.replace("_", " "), fontsize=11)
        steady = rate[tm > 3.0] if np.any(tm > 3.0) else rate
        ax.set_ylim(0, float(np.max(steady)) * 1.15)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("°/s")
    fig.suptitle("Rotation rate vs time — all detumble runs", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "omega_all_grid.png"))
    plt.close(fig)
    print(f"{len(curves)} single plots + grid written to {OUT}")


if __name__ == "__main__":
    main()
