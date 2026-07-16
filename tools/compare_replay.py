"""
Compare a tracker log (pi_identify stdout) against the replayed 'actual' trajectory CSV
and plot the pointing/roll error over time.

Alignment:
  start  -> detected as the moment the tracker attitude jumps off its initial hold
            (great-circle distance from the first estimate exceeds a threshold).
  end    -> start + (rows-1)*dt   (running time of the replay; dt = 0.2 s/line here).
Actual sample i is at time start + i*dt; each tracker estimate at running time t is
compared to the actual attitude interpolated at (t - start).
"""
import math
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else "curunrun.log"
ACTUAL = sys.argv[2] if len(sys.argv) > 2 else "cache/FVIDEO/real-timer.txt"
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
OUT = sys.argv[4] if len(sys.argv) > 4 else "outputs/replay_error.png"

_LINE = re.compile(r"frame\s+(\d+)\s*\|\s*t=\s*([\d.]+)s.*RA=\s*(-?[\d.]+)\s+DEC=\s*(-?[\d.]+)\s+ROLL=\s*(-?[\d.]+)")


def unit(ra, dec):
    r, d = math.radians(ra), math.radians(dec)
    return (math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d))


def gc_deg(a, b):
    """Great-circle angle (deg) between two (ra,dec) directions."""
    ua, ub = unit(*a), unit(*b)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(ua, ub))))
    return math.degrees(math.acos(dot))


def lerp_angle(a, b, f):
    d = ((b - a + 180.0) % 360.0) - 180.0
    return a + d * f


def roll_err(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


# --- parse into runs, split on frame-counter reset (log is several runs concatenated) ---
runs, cur, last_frame = [], [], None
for line in open(LOG):
    if "NULL" in line:
        continue
    m = _LINE.search(line)
    if not m:
        continue
    frame = int(m.group(1))
    t, ra, dec, roll = (float(x) for x in m.groups()[1:])
    if last_frame is not None and frame < last_frame:  # counter reset -> new run
        runs.append(cur)
        cur = []
    cur.append((t, ra, dec, roll))
    last_frame = frame
if cur:
    runs.append(cur)
if not runs:
    sys.exit("no attitude lines parsed from " + LOG)
# Pick the run with the largest running-time span (the full replay, not the aborted tries).
est = max(runs, key=lambda r: r[-1][0] - r[0][0])
print(f"{len(runs)} run(s); using the longest: {len(est)} frames, "
      f"{est[-1][0] - est[0][0]:.1f}s span")

# --- load actual trajectory (ra,dec,roll per line) ---
actual = []
with open(ACTUAL) as f:
    header = f.readline()  # ra,dec,roll
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            actual.append((float(parts[0]), float(parts[1]), float(parts[2])))
n = len(actual)
duration = (n - 1) * DT

# --- detect start: first frame that reaches the trajectory's start point (attitude change) ---
# The run holds on its initial pointing, then jumps onto the replay. Aligning t=0 of the
# actual to when the tracker first lands within THRESH of actual[0] also absorbs pipeline lag.
THRESH = 3.0
target = (actual[0][0], actual[0][1])
start_t = None
for t, ra, dec, roll in est:
    if gc_deg(target, (ra, dec)) < THRESH:
        start_t = t
        break
if start_t is None:  # fallback: first departure from the run's initial hold
    a0 = (est[0][1], est[0][2])
    for t, ra, dec, roll in est:
        if gc_deg(a0, (ra, dec)) > 5.0:
            start_t = t
            break
if start_t is None:
    sys.exit("could not detect replay start")
end_t = start_t + duration


def actual_at(rel_t):
    """Interpolate the actual attitude at replay-relative time rel_t (clamped)."""
    x = rel_t / DT
    if x <= 0:
        return actual[0]
    if x >= n - 1:
        return actual[-1]
    i = int(x)
    f = x - i
    ra = lerp_angle(actual[i][0], actual[i + 1][0], f)
    dec = actual[i][1] + (actual[i + 1][1] - actual[i][1]) * f
    roll = lerp_angle(actual[i][2], actual[i + 1][2], f)
    return ra, dec, roll


# --- build error series over the replay window ---
times, point_err, r_err = [], [], []
for t, ra, dec, roll in est:
    if start_t <= t <= end_t:
        rel = t - start_t
        ar, ad, arl = actual_at(rel)
        times.append(rel)
        point_err.append(gc_deg((ar, ad), (ra, dec)))
        r_err.append(roll_err(roll, arl))

mean_p = sum(point_err) / len(point_err)
p95 = sorted(point_err)[int(0.95 * len(point_err))]
print(f"replay start t={start_t:.2f}s  end t={end_t:.2f}s  duration={duration:.1f}s")
print(f"samples={len(point_err)}")
# Angular (great-circle) errors: direction-independent, unlike RA/DEC degrees.
print("=== angular error (deg), great-circle ===")
print(f"boresight pointing: mean={mean_p:.4f}  min={min(point_err):.4f}  max={max(point_err):.4f}  p95={p95:.4f}")
print(f"roll:               mean={sum(r_err)/len(r_err):.4f}  min={min(r_err):.4f}  max={max(r_err):.4f}")

# --- correct the constant boresight misalignment: fit the single best-fit rotation that
#     maps tracker boresights onto truth boresights (Kabsch/Wahba), then residual angle. ---
import numpy as np


def _unit(ra, dec):
    r, d = math.radians(ra), math.radians(dec)
    return np.array([math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)])


obs, tru = [], []
for t, ra, dec, roll in est:
    if start_t <= t <= end_t:
        a_ra, a_dec, _ = actual_at(t - start_t)
        obs.append(_unit(ra, dec))     # tracker boresight
        tru.append(_unit(a_ra, a_dec)) # true boresight
obs, tru = np.array(obs), np.array(tru)
H = tru.T @ obs
U, _, Vt = np.linalg.svd(H)
d = np.sign(np.linalg.det(U @ Vt))
Rot = U @ np.diag([1.0, 1.0, d]) @ Vt        # best-fit tracker->truth rotation
corrected = obs @ Rot.T
dots = np.clip(np.einsum("ij,ij->i", corrected, tru), -1.0, 1.0)
res = np.degrees(np.arccos(dots))
axis_angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rot) - 1) / 2))))
print(f"boresight bias removed (best-fit rotation, {axis_angle:.3f} deg):")
print(f"  residual pointing: mean={res.mean():.4f}  min={res.min():.4f}  max={res.max():.4f}  p95={np.percentile(res,95):.4f}")

# --- plot ---
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(times, point_err, color="#c0504d", lw=0.8)
ax[0].axhline(mean_p, color="#4472c4", ls="--", lw=1, label=f"mean {mean_p:.3f} deg")
ax[0].set_ylabel("pointing error (deg)")
ax[0].set_title(f"Tracker vs actual — replay aligned start={start_t:.1f}s, dur={duration:.1f}s")
ax[0].legend(loc="upper right")
ax[0].grid(alpha=0.3)
ax[1].plot(times, r_err, color="#e08214", lw=0.8)
ax[1].set_ylabel("roll error (deg)")
ax[1].set_xlabel("time since replay start (s)")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote " + OUT)

# --- overlay plots: actual vs tracker, RA and DEC separately ---
tr_t, tr_ra, tr_dec = [], [], []
for t, ra, dec, roll in est:
    if start_t <= t <= end_t:
        tr_t.append(t - start_t)
        tr_ra.append(ra)
        tr_dec.append(dec)
act_t = [i * DT for i in range(n)]
act_ra = [a[0] for a in actual]
act_dec = [a[1] for a in actual]

fig2, ax2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax2[0].plot(act_t, act_ra, color="#1f4e9c", lw=1.6, label="actual")
ax2[0].plot(tr_t, tr_ra, color="#e08214", lw=0.9, label="tracker")
ax2[0].set_ylabel("RA (deg)")
ax2[0].set_title(f"Actual vs tracker — aligned start={start_t:.1f}s, dur={duration:.1f}s")
ax2[0].legend(loc="upper left")
ax2[0].grid(alpha=0.3)
ax2[1].plot(act_t, act_dec, color="#1f4e9c", lw=1.6, label="actual")
ax2[1].plot(tr_t, tr_dec, color="#e08214", lw=0.9, label="tracker")
ax2[1].set_ylabel("DEC (deg)")
ax2[1].set_xlabel("time since replay start (s)")
ax2[1].legend(loc="upper left")
ax2[1].grid(alpha=0.3)
fig2.tight_layout()
OUT2 = OUT.replace(".png", "_overlay.png")
fig2.savefig(OUT2, dpi=120)
print("wrote " + OUT2)

# --- correct the constant RA offset (mean actual-tracker over the window) and re-overlay ---
ra_offsets = []
for rt, rra in zip(tr_t, tr_ra):
    a_ra, _, _ = actual_at(rt)
    ra_offsets.append(((a_ra - rra + 180.0) % 360.0) - 180.0)
ra_shift = sum(ra_offsets) / len(ra_offsets)
tr_ra_corr = [(r + ra_shift) % 360.0 for r in tr_ra]
resid = [abs(((actual_at(rt)[0] - rc + 180.0) % 360.0) - 180.0) for rt, rc in zip(tr_t, tr_ra_corr)]
print(f"RA constant shift applied = {ra_shift:+.4f} deg; "
      f"residual RA error after correction: mean={sum(resid)/len(resid):.4f} deg  max={max(resid):.4f}")

fig3, ax3 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax3[0].plot(act_t, act_ra, color="#1f4e9c", lw=1.6, label="actual")
ax3[0].plot(tr_t, tr_ra_corr, color="#e08214", lw=0.9, label=f"tracker (RA {ra_shift:+.3f}°)")
ax3[0].set_ylabel("RA (deg)")
ax3[0].set_title(f"Actual vs tracker, RA offset corrected {ra_shift:+.3f}° "
                 f"(start={start_t:.1f}s, dur={duration:.1f}s)")
ax3[0].legend(loc="upper left")
ax3[0].grid(alpha=0.3)
ax3[1].plot(act_t, act_dec, color="#1f4e9c", lw=1.6, label="actual")
ax3[1].plot(tr_t, tr_dec, color="#e08214", lw=0.9, label="tracker")
ax3[1].set_ylabel("DEC (deg)")
ax3[1].set_xlabel("time since replay start (s)")
ax3[1].legend(loc="upper left")
ax3[1].grid(alpha=0.3)
fig3.tight_layout()
OUT3 = OUT.replace(".png", "_overlay_corrected.png")
fig3.savefig(OUT3, dpi=120)
print("wrote " + OUT3)

# --- proper correction: bias is constant in great-circle angle, so RA correction is
#     scaled by 1/cos(dec). Fit one angular constant C (deg-gc), apply dRA = C/cos(dec). ---
gc_consts = []
for rt, rra, rdec in zip(tr_t, tr_ra, tr_dec):
    a_ra, _, _ = actual_at(rt)
    dra = ((a_ra - rra + 180.0) % 360.0) - 180.0
    gc_consts.append(dra * math.cos(math.radians(rdec)))
C = sum(gc_consts) / len(gc_consts)
tr_ra_gc = [(r + C / math.cos(math.radians(d))) % 360.0 for r, d in zip(tr_ra, tr_dec)]
resid2 = [abs(((actual_at(rt)[0] - rc + 180.0) % 360.0) - 180.0) for rt, rc in zip(tr_t, tr_ra_gc)]
print(f"great-circle RA constant C={C:+.4f} deg; residual after 1/cos(dec) correction: "
      f"mean={sum(resid2)/len(resid2):.4f} deg  max={max(resid2):.4f}")

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(act_t, act_ra, color="#1f4e9c", lw=1.6, label="actual")
ax4.plot(tr_t, tr_ra_gc, color="#e08214", lw=0.9, label=f"tracker (RA {C:+.3f}°/cos·dec)")
ax4.set_ylabel("RA (deg)"); ax4.set_xlabel("time since replay start (s)")
ax4.set_title(f"RA corrected by constant great-circle offset C={C:+.3f}° (no crossing if bias is angular)")
ax4.legend(loc="upper left"); ax4.grid(alpha=0.3)
fig4.tight_layout()
OUT4 = OUT.replace(".png", "_overlay_gc_corrected.png")
fig4.savefig(OUT4, dpi=120)
print("wrote " + OUT4)

# --- per-axis error stats (RA, DEC, ROLL): mean / median / max of |error| ---
def stats(vals):
    s = sorted(vals)
    return sum(s) / len(s), s[len(s) // 2], s[-1]


ra_e, dec_e, roll_e, tr_roll = [], [], [], []
for t, ra, dec, roll in est:
    if start_t <= t <= end_t:
        a_ra, a_dec, a_roll = actual_at(t - start_t)
        ra_e.append(abs(((a_ra - ra + 180.0) % 360.0) - 180.0))
        dec_e.append(abs(a_dec - dec))
        roll_e.append(roll_err(roll, a_roll))
        tr_roll.append(roll)
print("\n=== per-axis |error| (deg) over aligned window, N=%d ===" % len(ra_e))
print("axis   mean    median   max")
for name, e in (("RA ", ra_e), ("DEC", dec_e), ("ROLL", roll_e)):
    m, md, mx = stats(e)
    print(f"{name}  {m:7.4f} {md:7.4f} {mx:7.4f}")
ra_e_corr = [abs(((actual_at(rt)[0] - rc + 180.0) % 360.0) - 180.0) for rt, rc in zip(tr_t, tr_ra_gc)]
m, md, mx = stats(ra_e_corr)
print(f"RA*  {m:7.4f} {md:7.4f} {mx:7.4f}   (* after constant angular-bias removal)")

# --- ROLL overlay graph ---
act_roll = [a[2] for a in actual]
fig5, ax5 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax5[0].plot(act_t, act_roll, color="#1f4e9c", lw=1.6, label="actual")
ax5[0].plot(tr_t, tr_roll, color="#e08214", lw=0.9, label="tracker")
ax5[0].set_ylabel("ROLL (deg)")
ax5[0].set_title(f"Actual vs tracker ROLL (start={start_t:.1f}s, dur={duration:.1f}s)")
ax5[0].legend(loc="upper left"); ax5[0].grid(alpha=0.3)
ax5[1].plot(tr_t, roll_e, color="#c0504d", lw=0.8)
ax5[1].set_ylabel("|roll error| (deg)"); ax5[1].set_xlabel("time since replay start (s)")
ax5[1].grid(alpha=0.3)
fig5.tight_layout()
OUT5 = OUT.replace(".png", "_roll_overlay.png")
fig5.savefig(OUT5, dpi=120)
print("wrote " + OUT5)
