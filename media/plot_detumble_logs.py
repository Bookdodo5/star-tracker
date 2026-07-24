"""
Plot detumble runs from Pi tracker stdout logs (newrun*.log).

Log format: `frame N | t=Xs | RA= .. DEC= .. ROLL= .. Q=(..)` with sessions
separated by `[stream]` header lines (t resets per session). NULL frames are
unsolved. The tracker repeats the last solve at display fps while solving at
~3 Hz, so consecutive duplicate attitudes must be collapsed before any
finite differencing.

Outputs (to outputs/detumble_plots/):
  A_<log>_overview.png       - per log: RA/DEC/ROLL + |w| over the whole session,
                               detumble windows shaded
  B_<runlabel>.png           - per detumble window: attitude, |w| (log-y), sky path
  C_<runlabel>_targeterr.png - target-(0,0,0) runs: pointing + roll error (log-y)

Usage: python media/plot_detumble_logs.py
"""
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "detumble_plots")

FRAME_RE = re.compile(
    r"frame\s+\d+\s+\|\s+t=\s*([\d.]+)s\s+\|\s+"
    r"RA=\s*(-?[\d.]+)\s+DEC=\s*(-?[\d.]+)\s+ROLL=\s*(-?[\d.]+)\s+Q=\(([^)]+)\)"
)

# (log, session_idx, t0, t1, label, target-(ra,dec,roll) or None)
# Windows measured by motion detection (boresight rate > 0.25 deg/s), padded ~5 s.
RUNS = [
    ("newrun.log", 6,  87, 110, "run1_omega_1.5_1_3_short",      None),
    ("newrun.log", 6, 120, 204, "run2_omega_1_1_3_DIVERGED",     None),
    ("newrun.log", 6, 238, 266, "run3_stare",                    None),
    ("newrun.log", 6, 362, 430, "run4_omega_4_-1.5_-2_DIVERGED", None),
    ("newrun.log", 6, 449, 486, "run5_omega_2.5_-3.5_-1",        None),
    ("newrun2.log", 0,  56, 215, "run6_target000_first",         (0.0, 0.0, 0.0)),
    ("newrun3.log", 0,  40, 181, "run7_target000_duration180",   (0.0, 0.0, 0.0)),
    ("newrun4.log", 0,  12, 125, "run8_target000_final",         (0.0, 0.0, 0.0)),
]


def log_path(name):
    """Resolve a log file that may live at the repo root or in cache/."""
    for candidate in (os.path.join(ROOT, name), os.path.join(ROOT, "cache", name)):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"{name} not found in repo root or cache/")


def parse_log(path):
    """Return list of sessions; each session is an (N,5) array of
    [t, ra, dec, roll, *quat...] rows -> actually (t, ra, dec, roll, qx,qy,qz,qw),
    duplicates already collapsed (only frames where the quaternion changed)."""
    sessions = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("[stream]"):
                current = []
                sessions.append(current)
                continue
            m = FRAME_RE.search(line)
            if m and current is not None:
                q = tuple(float(v) for v in m.group(5).split(","))
                row = (float(m.group(1)), float(m.group(2)), float(m.group(3)),
                       float(m.group(4))) + q
                # collapse repeated solves: keep only when the quaternion changes
                if current and current[-1][4:] == row[4:]:
                    continue
                current.append(row)
    return [np.array(s) for s in sessions]


def attitude_matrices(data):
    """Rotation matrix per solve built from RA/DEC/ROLL (full log precision;
    the logged quaternion is only 4 dp, which underflows small rotations).
    Convention doesn't matter — only relative angles between solves are used."""
    ra = np.radians(data[:, 1])
    dec = np.radians(data[:, 2])
    roll = np.radians(data[:, 3])
    cr, sr = np.cos(ra), np.sin(ra)
    cd, sd = np.cos(dec), np.sin(dec)
    bore = np.stack([cd * cr, cd * sr, sd], axis=1)
    north = np.stack([-sd * cr, -sd * sr, cd], axis=1)
    east = np.stack([-sr, cr, np.zeros_like(ra)], axis=1)
    cl, sl = np.cos(roll)[:, None], np.sin(roll)[:, None]
    up = cl * north + sl * east
    right = np.cross(up, bore)
    return np.stack([right, up, bore], axis=1)  # rows = camera axes


def quat_rate(data, baseline=0.5):
    # ponytail: relative angle wraps at 180 deg, so rates above 180/baseline
    # deg/s alias; baseline=0.5 keeps the observed peaks (~190 deg/s) readable.
    """|w| in deg/s: relative rotation angle over a ~`baseline`-second span
    (frame-to-frame differencing amplifies solver noise into garbage)."""
    t = data[:, 0]
    R = attitude_matrices(data)
    tm, rate = [], []
    j = 0
    for i in range(len(t)):
        while t[j] < t[i] - baseline:
            j += 1
        if j >= i or t[i] - t[j] < 0.3 * baseline:
            continue
        rel = R[j] @ R[i].T
        ang = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(rel) - 1) / 2))))
        tm.append(0.5 * (t[i] + t[j]))
        rate.append(ang / (t[i] - t[j]))
    tm, rate = np.array(tm), np.array(rate)
    # rolling median (+-1 s): a single wrong solve makes two huge dtheta pairs,
    # which would read as a >100 deg/s spike; sustained tumble survives a median
    smoothed = np.array([
        np.median(rate[(tm >= ti - 1.0) & (tm <= ti + 1.0)]) for ti in tm
    ])
    return tm, smoothed


def prep_angles(data):
    """Unwrapped RA (pole-masked), DEC, ROLL for plotting."""
    ra = np.unwrap(np.radians(data[:, 1]))
    ra = np.degrees(ra)
    dec = data[:, 2]
    roll = data[:, 3]
    ra_masked = ra.copy()
    ra_masked[np.abs(dec) > 85.0] = np.nan  # RA meaningless near the pole
    return ra_masked, dec, roll


def pointing_error_deg(ra_deg, dec_deg, tra, tdec):
    """Great-circle angle between (ra,dec) and target."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    tra, tdec = math.radians(tra), math.radians(tdec)
    c = (np.sin(dec) * math.sin(tdec)
         + np.cos(dec) * math.cos(tdec) * np.cos(ra - tra)).clip(-1, 1)
    return np.degrees(np.arccos(c))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def plot_overview(log, sessions, windows):
    for si, data in enumerate(sessions):
        if len(data) < 50:
            continue
        ra, dec, roll = prep_angles(data)
        tm, rate = quat_rate(data)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax1.plot(data[:, 0], ra, label="RA (unwrapped)", lw=0.8)
        ax1.plot(data[:, 0], dec, label="DEC", lw=0.8)
        ax1.plot(data[:, 0], roll, label="ROLL", lw=0.8)
        ax1.set_ylabel("deg")
        ax1.legend(loc="upper right", fontsize=8)
        ax2.semilogy(tm, np.maximum(rate, 1e-3), lw=0.8, color="tab:red")
        ax2.set_ylabel("|w| (deg/s, log)")
        ax2.set_xlabel("session time (s)")
        for (t0, t1, label) in windows.get(si, []):
            for ax in (ax1, ax2):
                ax.axvspan(t0, t1, alpha=0.12, color="tab:green")
            ax1.text(t0, ax1.get_ylim()[1], label.split("_", 1)[0],
                     fontsize=7, va="top", rotation=90)
        fig.suptitle(f"{log} session {si}: attitude + angular rate")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"A_{log.replace('.log','')}_s{si}_overview.png"),
                    dpi=150)
        plt.close(fig)


def plot_run(data, t0, t1, label, target):
    seg = data[(data[:, 0] >= t0) & (data[:, 0] <= t1)]
    if len(seg) < 10:
        print(f"  WARNING: {label}: only {len(seg)} solves in window, skipped")
        return None
    tt = seg[:, 0] - seg[0, 0]
    ra, dec, roll = prep_angles(seg)
    tm, rate = quat_rate(seg)
    tm = tm - seg[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    ax.plot(tt, ra, label="RA", lw=0.9)
    ax.plot(tt, dec, label="DEC", lw=0.9)
    ax.plot(tt, roll, label="ROLL", lw=0.9)
    ax.set_xlabel("t since window start (s)")
    ax.set_ylabel("deg")
    ax.set_title("attitude")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(tm, np.maximum(rate, 1e-3), color="tab:red", lw=0.9)
    # peak excludes the first 3 s: the initial point_at is an attitude teleport,
    # and differencing across it reads as a fake several-hundred-deg/s spike
    steady = rate[tm > 3.0] if np.any(tm > 3.0) else rate
    if len(steady):
        peak = float(np.max(steady))
        ax.axhline(peak, color="gray", ls="--", lw=0.8,
                   label=f"peak {peak:.2f} deg/s")
        ax.legend(fontsize=8)
    ax.set_xlabel("t since window start (s)")
    ax.set_ylabel("|w| (deg/s, log)")
    ax.set_title("angular rate")

    ax = axes[2]
    ra_sky = wrap180(seg[:, 1]) if target is not None else seg[:, 1]
    sc = ax.scatter(ra_sky, seg[:, 2], c=tt, cmap="viridis", s=4)
    plt.colorbar(sc, ax=ax, label="t (s)")
    if target is not None:
        ax.plot(target[0], target[1], "r*", ms=14, label="target")
        ax.legend(fontsize=8)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("DEC (deg)")
    ax.set_title("sky path")

    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"B_{label}.png"), dpi=150)
    plt.close(fig)

    result = {
        "label": label, "duration": tt[-1],
        "w_peak": float(np.max(steady)) if len(steady) else float("nan"),
        "w_end": float(np.median(rate[-10:])) if len(rate) >= 10 else float("nan"),
        "final_dec": seg[-1, 2],
    }

    if target is not None:
        perr = pointing_error_deg(seg[:, 1], seg[:, 2], target[0], target[1])
        rerr = np.abs(wrap180(roll - target[2]))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.semilogy(tt, np.maximum(perr, 1e-3), color="tab:blue", lw=0.9)
        ax1.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 deg")
        ax1.set_ylabel("pointing err (deg, log)")
        ax1.legend(fontsize=8)
        ax2.semilogy(tt, np.maximum(rerr, 1e-3), color="tab:orange", lw=0.9)
        ax2.set_ylabel("|roll err| (deg, log)")
        ax2.set_xlabel("t since window start (s)")
        fig.suptitle(f"{label}: error to target RA={target[0]} DEC={target[1]} ROLL={target[2]}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"C_{label}_targeterr.png"), dpi=150)
        plt.close(fig)
        result["final_perr"] = perr[-1]
        # final pointing = median of last 10 solves (robust to a stray frame)
        result["settled_perr"] = float(np.median(perr[-10:]))
    return result


def main():
    os.makedirs(OUT, exist_ok=True)
    logs = {}
    for log in sorted({r[0] for r in RUNS}):
        logs[log] = parse_log(log_path(log))

    windows = {}
    for log, si, t0, t1, label, _ in RUNS:
        windows.setdefault(log, {}).setdefault(si, []).append((t0, t1, label))
    for log, sessions in logs.items():
        plot_overview(log, sessions, windows.get(log, {}))

    print(f"{'run':34s} {'dur(s)':>7s} {'w_peak':>7s} {'w_end':>7s} {'final perr':>11s}")
    results = []
    for log, si, t0, t1, label, target in RUNS:
        r = plot_run(logs[log][si], t0, t1, label, target)
        if r:
            results.append(r)
            perr = f"{r.get('settled_perr', float('nan')):10.3f}" if target else "        --"
            print(f"{r['label']:34s} {r['duration']:7.1f} {r['w_peak']:7.2f} {r['w_end']:7.2f} {perr}")

    # sanity checks
    by = {r["label"]: r for r in results}
    assert by["run8_target000_final"]["settled_perr"] < 0.5, "newrun4 did not converge <0.5 deg"
    assert abs(by["run2_omega_1_1_3_DIVERGED"]["final_dec"]) > 85, "run2 expected at pole"
    assert abs(by["run4_omega_4_-1.5_-2_DIVERGED"]["final_dec"]) > 85, "run4 expected at pole"
    for r in results:
        assert np.isfinite(r["w_peak"]) and r["w_peak"] > 0, f"{r['label']}: bad rate"
    print(f"\nAll checks passed. Plots in {OUT}")


if __name__ == "__main__":
    main()
