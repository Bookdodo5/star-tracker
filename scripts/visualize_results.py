"""
Generates visualizations for the Star Tracker pipeline results.

Outputs (written to outputs/):
  pipeline_overview.png   — full pipeline flowchart with timing at each step
  accuracy_comparison.png — TETRA vs Pyramid accuracy (synthetic vs real)
  error_distribution.png  — angular error histogram for successful real-image solves
  sky_coverage.png        — RA/DEC map of 20 test fields coloured by solve result
  timing_breakdown.png    — synthetic benchmark timing per algorithm step

Usage:
    python scripts/visualize_results.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
BATCH_CSV = OUTPUTS / "real_batch_latest.csv"
BENCH_CSV = OUTPUTS / "benchmark_latest.csv"

TETRA_COLOR  = "#2563eb"   # blue
PYRAMID_COLOR = "#dc2626"  # red
OK_COLOR     = "#16a34a"   # green
FAIL_COLOR   = "#9ca3af"   # grey
BG           = "#0f172a"   # dark background
PANEL        = "#1e293b"
TEXT         = "#f1f5f9"
GRID         = "#334155"


def style_dark(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)


# ── 1. Pipeline overview ───────────────────────────────────────────────────────

def plot_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    steps = [
        ("DSS Image\n(SkyView PPM)", 1.0,  3.0, "#7c3aed", "fetch_dss_image.py\nastroquery.SkyView\n~30–60 s/fetch"),
        ("Centroid\nExtraction",      3.5,  3.0, "#0891b2", "centroid_extract.cpp\nDoG → threshold\n→ morph open → CCL\nK=20 stars"),
        ("Camera\nModel",             6.0,  4.3, "#0369a1", "camera_model.c\npixel → unit vector\n(east-left convention)"),
        ("TETRA\nIdentifier",         8.5,  4.8, TETRA_COLOR, "identify_tetra.c\n4-star tetrad\nKD-tree lookup\n~5 ms/frame"),
        ("Pyramid\nIdentifier",       8.5,  1.2, PYRAMID_COLOR, "identify_pyramid.c\nseed pair + grow\nvoting\n~91 ms/frame"),
        ("Shared\nVerifier",          11.0, 3.0, "#b45309", "verify.c\nR^T × v_obs\ncatalog KD-tree\n< 0.01 ms"),
        ("Attitude\nOutput",          13.0, 3.0, "#15803d", "MatchResult\nRA, DEC, Roll\nRotation matrix"),
    ]

    # Draw boxes
    for label, x, y, color, note in steps:
        box = mpatches.FancyBboxPatch((x - 0.85, y - 0.6), 1.7, 1.2,
                                       boxstyle="round,pad=0.08", linewidth=1.5,
                                       edgecolor=color, facecolor=color + "33")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=TEXT)
        ax.text(x, y - 1.05, note, ha="center", va="top", fontsize=6.5,
                color="#94a3b8", linespacing=1.4)

    # Arrows: fetch → centroid → camera → TETRA/Pyramid → verifier → output
    arrow_kw = dict(arrowstyle="-|>", color="#64748b", lw=1.5)
    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(**arrow_kw))

    arr(1.85, 3.0, 2.65, 3.0)   # fetch → centroid
    arr(4.35, 3.0, 5.15, 3.0)   # centroid → camera (goes right then splits)
    # camera → TETRA
    ax.annotate("", xy=(7.65, 4.8), xytext=(6.85, 4.3),
                arrowprops=dict(**arrow_kw))
    # camera → Pyramid
    ax.annotate("", xy=(7.65, 1.2), xytext=(6.85, 1.7),
                arrowprops=dict(**arrow_kw))
    # TETRA → verifier
    ax.annotate("", xy=(10.15, 3.3), xytext=(9.35, 4.5),
                arrowprops=dict(**arrow_kw))
    # Pyramid → verifier
    ax.annotate("", xy=(10.15, 2.7), xytext=(9.35, 1.5),
                arrowprops=dict(**arrow_kw))
    arr(11.85, 3.0, 12.15, 3.0)  # verifier → output

    ax.set_title("Star Tracker Pipeline", color=TEXT, fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    out = OUTPUTS / "pipeline_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ── 2. Accuracy comparison ────────────────────────────────────────────────────

def plot_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))
    style_dark(fig, ax)

    categories  = ["Synthetic\n(100 fields)", "Real DSS\n(20 fields)"]
    tetra_acc   = [100.0, 100.0]
    pyramid_acc = [100.0, 30.0]

    x = np.arange(len(categories))
    w = 0.32
    b1 = ax.bar(x - w/2, tetra_acc,   w, label="TETRA",   color=TETRA_COLOR,   alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, pyramid_acc, w, label="Pyramid", color=PYRAMID_COLOR, alpha=0.85, zorder=3)

    for bar, val in [(bar, v) for bars, vals in [(b1, tetra_acc), (b2, pyramid_acc)]
                    for bar, v in zip(bars, vals)]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10,
                color=TEXT, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color=TEXT)
    ax.set_title("TETRA vs Pyramid Accuracy", color=TEXT, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=TEXT)
    ax.axhline(90, color="#fbbf24", linewidth=1.2, linestyle="--", label="Target (90%)", zorder=2)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    ax.grid(axis="x", visible=False)

    out = OUTPUTS / "accuracy_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ── 3. Angular error distribution ─────────────────────────────────────────────

def plot_errors(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    style_dark(fig, axes)

    tetra_err   = df.loc[df["tetra_correct"]   == True, "tetra_err_deg"].dropna() * 60   # → arcmin
    pyramid_err = df.loc[df["pyramid_correct"] == True, "pyramid_err_deg"].dropna() * 60

    bins = np.linspace(0, 3.0, 20)
    axes[0].hist(tetra_err,   bins=bins, color=TETRA_COLOR,   alpha=0.85, edgecolor=BG, zorder=3)
    axes[0].set_title("TETRA error distribution\n(20/20 correct)", color=TEXT, fontsize=11)
    axes[0].set_xlabel("Angular error (arcmin)", color=TEXT)
    axes[0].set_ylabel("Fields", color=TEXT)
    axes[0].axvline(float(tetra_err.mean()), color="#fbbf24", linewidth=1.5,
                    linestyle="--", label=f"Mean {tetra_err.mean():.2f}′")
    axes[0].legend(facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

    bins2 = np.linspace(0, 0.7, 15)
    axes[1].hist(pyramid_err, bins=bins2, color=PYRAMID_COLOR, alpha=0.85, edgecolor=BG, zorder=3)
    axes[1].set_title("Pyramid error distribution\n(6/20 correct)", color=TEXT, fontsize=11)
    axes[1].set_xlabel("Angular error (arcmin)", color=TEXT)
    axes[1].set_ylabel("Fields", color=TEXT)
    if len(pyramid_err):
        axes[1].axvline(float(pyramid_err.mean()), color="#fbbf24", linewidth=1.5,
                        linestyle="--", label=f"Mean {pyramid_err.mean():.2f}′")
        axes[1].legend(facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

    fig.suptitle("Angular Error on Correct Solves (Real DSS, 20 fields)", color=TEXT,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUTPUTS / "error_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ── 4. Sky coverage map ────────────────────────────────────────────────────────

def plot_sky(df: pd.DataFrame):
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.5)

    # Convert RA to [-π, π] for mollweide
    ra_rad  = np.radians(df["ra"].values - 180.0)
    dec_rad = np.radians(df["dec"].values)

    for i, row in df.iterrows():
        ra  = math.radians(row["ra"] - 180.0)
        dec = math.radians(row["dec"])
        tc  = row["tetra_correct"]
        pc  = row["pyramid_correct"]

        if tc and pc:
            color, marker, size, label = "#22c55e", "o", 120, "Both correct"
        elif tc:
            color, marker, size, label = TETRA_COLOR, "o", 90, "TETRA only"
        elif pc:
            color, marker, size, label = PYRAMID_COLOR, "^", 90, "Pyramid only"
        else:
            color, marker, size, label = FAIL_COLOR, "x", 60, "Both failed"

        ax.scatter(ra, dec, c=color, marker=marker, s=size, zorder=5,
                   edgecolors="white", linewidths=0.4)

    # Legend
    handles = [
        mpatches.Patch(color="#22c55e", label="Both correct"),
        mpatches.Patch(color=TETRA_COLOR, label="TETRA only"),
        mpatches.Patch(color=PYRAMID_COLOR, label="Pyramid only"),
        mpatches.Patch(color=FAIL_COLOR, label="Both failed"),
    ]
    legend = ax.legend(handles=handles, loc="lower right",
                       facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, fontsize=8)

    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title("Sky Coverage — 20 DSS Test Fields (FOV=10°)", color=TEXT,
                 fontsize=13, fontweight="bold", pad=15)
    ax.title.set_color(TEXT)

    out = OUTPUTS / "sky_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ── 5. Timing breakdown ────────────────────────────────────────────────────────

def plot_timing(bench: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    style_dark(fig, axes)

    def timing_bars(ax, db_col, verify_col, total_col, color, title):
        db_ms  = bench[db_col].mean()     / 1000.0
        ver_ms = bench[verify_col].mean() / 1000.0
        tot_ms = bench[total_col].mean()  / 1000.0

        labels = ["DB Search", "Verify", "Total"]
        vals   = [db_ms, ver_ms, tot_ms]
        bars   = ax.bar(labels, vals, color=[color, "#0891b2", "#475569"],
                        alpha=0.85, edgecolor=BG, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f} ms", ha="center", va="bottom", fontsize=9,
                    color=TEXT, fontweight="bold")
        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_ylabel("Time (ms)", color=TEXT)
        ax.set_ylim(0, max(vals) * 1.35 + 0.5)

    timing_bars(axes[0], "tetra_db_us",   "tetra_verify_us",   "tetra_total_us",
                TETRA_COLOR,   "TETRA — synthetic timing (100 fields)")
    timing_bars(axes[1], "pyramid_db_us", "pyramid_verify_us", "pyramid_total_us",
                PYRAMID_COLOR, "Pyramid — synthetic timing (100 fields)")

    fig.suptitle("Per-Step Timing (Synthetic, FOV=10°)", color=TEXT,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUTPUTS / "timing_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df    = pd.read_csv(BATCH_CSV)
    bench = pd.read_csv(BENCH_CSV)

    print("Generating visualizations...")
    plot_pipeline()
    plot_accuracy()
    plot_errors(df)
    plot_sky(df)
    plot_timing(bench)
    print("Done.")


if __name__ == "__main__":
    main()
