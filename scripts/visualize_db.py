"""
Four DB coverage visualizations for the TETRA anchored all-combos database.

    python scripts/visualize_db.py           # K=10 (current default)
    python scripts/visualize_db.py --k 9     # K=9 (new smaller DB)
    python scripts/visualize_db.py --k 9 10  # compare both

Outputs to outputs/db_viz/.

Viz 1 - All-sky tetrad density:    where are tetrads concentrated?
Viz 2 - Coverage multiplicity map: how many tetrads match per sky cell? (= redundancy)
Viz 3 - Per-star tetrad count:     which stars are bottlenecks vs over-covered?
Viz 4 - Tetrad span distribution:  max-edge angle histogram by sky region
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.star_tracker_core import load_db_catalog, anchored_allcombos_tetrads  # noqa: E402

# --- DB config (mirrors export_tetra_db.py defaults) ---
MAG_LIMIT = 7.5
BMC       = 7.0
GATHER_DEG = 3.5
FOV_W, FOV_H = 7.0, 4.0
FOV_DIAG = math.hypot(FOV_W, FOV_H)

# --- field simulation (mirrors coverage_sweep.py) ---
N_FIELDS  = 2000
SEED      = 12345
MIN_STARS = 6
QUERY_N   = 16


def _unit_vecs(df):
    ra  = np.radians(df["RA_deg"].to_numpy())
    dec = np.radians(df["DEC_deg"].to_numpy())
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def _galactic_lat(ra_deg, dec_deg):
    """Approximate galactic latitude from J2000 RA/DEC (degrees)."""
    # NGP at RA=192.86, Dec=27.13, l_NCP=122.93
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    ngp_ra, ngp_dec = math.radians(192.8595), math.radians(27.1284)
    sin_b = (np.sin(dec)*math.sin(ngp_dec) +
             np.cos(dec)*np.cos(dec)*np.cos(ra - ngp_ra))
    return np.degrees(np.arcsin(np.clip(sin_b, -1, 1)))


def build_tetrads(k: int, cat_vecs: np.ndarray, n_anchors: int):
    """Returns list of (a,b,c,d) index tuples."""
    gather = math.radians(GATHER_DEG)
    max_edge = math.radians(FOV_DIAG)
    return anchored_allcombos_tetrads(cat_vecs, n_anchors, gather, max_edge, k)


def angular_sep_rad(v1, v2):
    return math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)))))


# ── Viz 1: All-sky tetrad density ────────────────────────────────────────────

def viz1_sky_density(tetrads, cat_vecs, cat_df, k, out_dir):
    """Mollweide all-sky map: tetrad density per deg², anchored at anchor star RA/DEC."""
    ra_arr  = cat_df["RA_deg"].to_numpy()
    dec_arr = cat_df["DEC_deg"].to_numpy()

    anchor_ras  = [ra_arr[t[0]]  for t in tetrads]
    anchor_decs = [dec_arr[t[0]] for t in tetrads]

    # Mollweide coords: lon in [-pi,pi], lat in [-pi/2,pi/2]
    lon = np.radians(np.array(anchor_ras))
    lon = ((lon + math.pi) % (2*math.pi)) - math.pi          # wrap to [-π,π]
    lat = np.radians(np.array(anchor_decs))

    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(111, projection="mollweide")
    ax.set_title(f"Viz 1 — All-sky tetrad density  (K={k}, {len(tetrads):,} tetrads)",
                 fontsize=13, pad=12)

    # 2D histogram in Mollweide bins
    h, xb, yb = np.histogram2d(lon, lat, bins=[360, 180],
                                range=[[-math.pi, math.pi], [-math.pi/2, math.pi/2]])
    # convert to tetrads/deg²
    pix_area = (360/360) * (180/180)  # 1 deg² per bin
    h = h / pix_area
    h[h == 0] = np.nan

    xc = 0.5*(xb[:-1]+xb[1:])
    yc = 0.5*(yb[:-1]+yb[1:])
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    im = ax.pcolormesh(XX, YY, h, norm=mcolors.LogNorm(vmin=1, vmax=h[np.isfinite(h)].max()),
                       cmap="plasma", shading="auto")
    fig.colorbar(im, ax=ax, label="tetrads / deg²", shrink=0.7)

    # Galactic plane
    lons_gp = np.linspace(-180, 180, 720)
    # galactic plane in equatorial: rough sine curve (b=0 crosses equator at ~RA 266°)
    dec_gp = np.degrees(np.arcsin(
        np.sin(np.radians(62.87)) * np.sin(np.radians(lons_gp - 266.4 + 33.0))
    ))
    ax.plot(np.radians(((lons_gp+180) % 360)-180), np.radians(dec_gp),
            "w--", lw=0.8, alpha=0.6, label="Galactic plane")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out = out_dir / f"viz1_sky_density_k{k}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Viz 2: Coverage multiplicity map ─────────────────────────────────────────

def viz2_multiplicity(tetrads, cat_vecs, cat_df, k, out_dir):
    """
    Grid the sky into ~10°×10° cells. For each cell center, place a 7×4° field
    and count how many DB tetrads have all 4 stars visible.
    High count = high redundancy. Low count = fragile coverage.
    """
    # Build fast lookup: for each star index, which tetrads contain it?
    from collections import defaultdict
    star_to_tetrads = defaultdict(set)
    for tid, t in enumerate(tetrads):
        for s in t:
            star_to_tetrads[s].add(tid)

    tan_w = math.tan(math.radians(FOV_W / 2))
    tan_h = math.tan(math.radians(FOV_H / 2))

    # 5°×5° grid cells
    grid_ra  = np.arange(0, 360, 5) + 2.5
    grid_dec = np.arange(-87.5, 90, 5)
    counts = np.zeros((len(grid_ra), len(grid_dec)), dtype=np.int32)

    for i, ra_c in enumerate(grid_ra):
        for j, dec_c in enumerate(grid_dec):
            r, d = math.radians(ra_c), math.radians(dec_c)
            bore  = np.array([math.cos(d)*math.cos(r), math.cos(d)*math.sin(r), math.sin(d)])
            east  = np.array([-math.sin(r), math.cos(r), 0.0])
            north = np.cross(bore, east); north /= np.linalg.norm(north)
            obs = cat_vecs @ bore
            bz  = cat_vecs @ bore
            ex  = cat_vecs @ east
            ny  = cat_vecs @ north
            # stars in FOV
            in_fov = np.where(
                (bz > 0) &
                (np.abs(ex / np.maximum(bz, 1e-9)) <= tan_w) &
                (np.abs(ny / np.maximum(bz, 1e-9)) <= tan_h)
            )[0]
            if len(in_fov) < MIN_STARS:
                counts[i, j] = 0
                continue
            bright_set = frozenset(in_fov[:QUERY_N].tolist())
            # count DB tetrads whose all 4 stars are in the query pool
            matched = set()
            for s in bright_set:
                for tid in star_to_tetrads.get(s, ()):
                    if tid not in matched and all(ts in bright_set for ts in tetrads[tid]):
                        matched.add(tid)
            n_match = len(matched)
            counts[i, j] = n_match

    lon = np.radians(((grid_ra + 180) % 360) - 180)
    lat = np.radians(grid_dec)
    XX, YY = np.meshgrid(lon, lat, indexing="ij")
    C = counts.astype(float)
    C[C == 0] = np.nan

    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(111, projection="mollweide")
    ax.set_title(f"Viz 2 — Coverage multiplicity  (K={k}) — matching tetrads per 7×4° field",
                 fontsize=12, pad=12)
    im = ax.pcolormesh(XX, YY, C,
                       norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(C)),
                       cmap="YlOrRd", shading="auto")
    fig.colorbar(im, ax=ax, label="# matching DB tetrads", shrink=0.7)
    ax.grid(True, alpha=0.3)

    out = out_dir / f"viz2_multiplicity_k{k}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Viz 3: Per-star tetrad participation ─────────────────────────────────────

def viz3_star_participation(tetrads, cat_vecs, cat_df, k, out_dir):
    """
    For each catalog star: how many DB tetrads include it?
    Two panels:
      Left  — Histogram of participation counts (log y)
      Right — All-sky dot map colored by participation count
    Bottleneck stars (count < 10) in red; over-covered (> 200) in blue.
    """
    n_stars  = len(cat_df)
    counts   = np.zeros(n_stars, dtype=np.int32)
    for t in tetrads:
        for s in t:
            counts[s] += 1

    ra_arr  = np.radians(cat_df["RA_deg"].to_numpy())
    dec_arr = np.radians(cat_df["DEC_deg"].to_numpy())
    # wrap RA for Mollweide
    lon = ((ra_arr + math.pi) % (2*math.pi)) - math.pi

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"Viz 3 — Per-star tetrad participation  (K={k}, {n_stars:,} stars)", fontsize=13)

    # Left: histogram
    ax1 = fig.add_subplot(1, 2, 1)
    bins = np.logspace(0, math.log10(counts.max()+1), 60)
    ax1.hist(counts[counts > 0], bins=bins, color="steelblue", edgecolor="none")
    ax1.axvline(10,  color="red",  ls="--", lw=1.2, label="fragile < 10")
    ax1.axvline(200, color="navy", ls="--", lw=1.2, label="redundant > 200")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("# tetrads containing this star")
    ax1.set_ylabel("# stars")
    ax1.set_title("Participation count distribution")
    ax1.legend(fontsize=9)
    n_fragile   = int((counts < 10).sum())
    n_redundant = int((counts > 200).sum())
    ax1.text(0.97, 0.97, f"fragile (<10): {n_fragile}\nredundant (>200): {n_redundant}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Right: sky map
    ax2 = fig.add_subplot(1, 2, 2, projection="mollweide")
    ax2.set_title("All-sky (color = log participation)")
    sc = ax2.scatter(lon, dec_arr, c=np.log10(np.maximum(counts, 1)),
                     s=0.8, cmap="RdYlBu", vmin=0, vmax=math.log10(counts.max()+1),
                     linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax2, shrink=0.7)
    cb.set_label("log₁₀(# tetrads)")
    ticks = [0, 1, 2, 3]; cb.set_ticks(ticks)
    cb.set_ticklabels([f"10^{t}" for t in ticks])
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"viz3_star_participation_k{k}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Viz 4: Tetrad span distribution by sky region ────────────────────────────

def viz4_span_distribution(tetrads, cat_vecs, cat_df, k, out_dir):
    """
    Histogram of max pairwise angular separation (max edge) for all tetrads.
    Split into Galactic latitude bands: |b|<20° (dense plane), 20-60°, >60° (sparse poles).
    Shows whether Galactic-plane tetrads cluster at small spans (dense → short edges)
    or whether DB has similar size distribution everywhere.
    """
    ra_arr  = cat_df["RA_deg"].to_numpy()
    dec_arr = cat_df["DEC_deg"].to_numpy()

    print(f"  computing max edges for {len(tetrads):,} tetrads...")
    max_edges_deg = np.zeros(len(tetrads))
    anchor_b      = np.zeros(len(tetrads))   # galactic lat of anchor

    for i, t in enumerate(tetrads):
        vs = cat_vecs[list(t)]      # (4,3)
        # all 6 pairwise dot products → max angular sep
        max_cos = -1.0
        min_cos =  1.0
        for a, b in itertools.combinations(range(4), 2):
            c = float(np.dot(vs[a], vs[b]))
            if c < min_cos:
                min_cos = c
        max_edges_deg[i] = math.degrees(math.acos(max(-1.0, min(1.0, min_cos))))
        anchor_b[i] = _galactic_lat(ra_arr[t[0]], dec_arr[t[0]])

    bands = [
        ("|b| < 20° (Galactic plane)",  np.abs(anchor_b) < 20,  "firebrick"),
        ("20° ≤ |b| < 60° (mid-lat)",   (np.abs(anchor_b) >= 20) & (np.abs(anchor_b) < 60), "darkorange"),
        ("|b| ≥ 60° (poles)",           np.abs(anchor_b) >= 60, "steelblue"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Viz 4 — Tetrad span (max edge) distribution  (K={k})", fontsize=13)

    bins = np.linspace(0, FOV_DIAG + 0.5, 80)
    for label, mask, color in bands:
        sub = max_edges_deg[mask]
        axes[0].hist(sub, bins=bins, alpha=0.6, color=color, label=f"{label}  n={mask.sum():,}")
        axes[1].hist(sub, bins=bins, alpha=0.6, color=color)

    axes[0].set_xlabel("Max pairwise separation (deg)")
    axes[0].set_ylabel("# tetrads")
    axes[0].set_title("Linear scale")
    axes[0].axvline(FOV_DIAG, color="k", ls="--", lw=1, label=f"FOV diagonal {FOV_DIAG:.1f}°")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Max pairwise separation (deg)")
    axes[1].set_ylabel("# tetrads (log)")
    axes[1].set_title("Log scale")
    axes[1].set_yscale("log")
    axes[1].axvline(FOV_DIAG, color="k", ls="--", lw=1)

    fig.tight_layout()
    out = out_dir / f"viz4_span_distribution_k{k}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", nargs="+", type=int, default=[10],
                   help="K values to visualize (default: 10)")
    p.add_argument("--skip", nargs="+", type=int, default=[],
                   help="viz numbers to skip, e.g. --skip 2 (slow)")
    args = p.parse_args()

    out_dir = ROOT / "outputs" / "db_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = load_db_catalog(MAG_LIMIT)
    cat_vecs = _unit_vecs(cat)
    vmag     = cat["Vmag"].to_numpy()
    n_anchors = int((vmag <= BMC).sum())

    for k in args.k:
        print(f"\n=== K={k}: building tetrad list ===")
        tetrads = build_tetrads(k, cat_vecs, n_anchors)
        print(f"  {len(tetrads):,} tetrads")

        if 1 not in args.skip:
            print("  Viz 1: sky density...")
            viz1_sky_density(tetrads, cat_vecs, cat, k, out_dir)

        if 2 not in args.skip:
            print("  Viz 2: coverage multiplicity (slow ~60s)...")
            viz2_multiplicity(tetrads, cat_vecs, cat, k, out_dir)

        if 3 not in args.skip:
            print("  Viz 3: star participation...")
            viz3_star_participation(tetrads, cat_vecs, cat, k, out_dir)

        if 4 not in args.skip:
            print("  Viz 4: span distribution (slow ~30s)...")
            viz4_span_distribution(tetrads, cat_vecs, cat, k, out_dir)

    print(f"\nAll done → {out_dir}")


if __name__ == "__main__":
    main()
