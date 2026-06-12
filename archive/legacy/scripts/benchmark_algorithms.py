from __future__ import annotations

from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import psutil
except ImportError:
    psutil = None

from src.star_tracker_core import (
    BatchConfig,
    OpenStarTrackerMatcher,
    PyramidMatcher,
    TetraMatcher,
    build_pair_database,
    build_tetra_database,
    load_catalog,
    run_batch,
    summarize_results,
)


def memory_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def timed(label: str, fn):
    before = memory_mb()
    started = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - started
    after = memory_mb()
    return value, {"step": label, "seconds": seconds, "memory_before_mb": before, "memory_after_mb": after}


def benchmark(name: str, make_matcher, top: int) -> dict[str, object]:
    matcher, setup = timed(f"{name}_setup", make_matcher)
    single, single_time = timed(f"{name}_single", lambda: matcher.identify(120.0, 15.0, 12.0, 6.5, top))
    batch, batch_time = timed(
        f"{name}_batch_20",
        lambda: run_batch(matcher.identify, BatchConfig(samples=20, fov_deg=12.0, mag_limit=6.5, max_stars_query=top)),
    )
    summary = summarize_results(batch)
    return {
        "algorithm": name,
        "setup_seconds": setup["seconds"],
        "single_seconds": single_time["seconds"],
        "batch20_seconds": batch_time["seconds"],
        "avg_identify_seconds": batch_time["seconds"] / 20.0,
        "memory_after_setup_mb": setup["memory_after_mb"],
        "memory_after_batch_mb": batch_time["memory_after_mb"],
        "single_outcome": single["outcome"],
        "batch_accuracy_pct": summary["accuracy_pct"],
    }


def main() -> None:
    catalog = load_catalog()

    results = [
        benchmark(
            "tetra",
            lambda: TetraMatcher(catalog, build_tetra_database(catalog, fov_deg=8.0, max_tetras_per_anchor=20)),
            10,
        ),
        benchmark(
            "openstartracker",
            lambda: OpenStarTrackerMatcher(catalog, *build_pair_database(catalog, "openstartracker")),
            10,
        ),
        benchmark(
            "pyramid",
            lambda: PyramidMatcher(catalog, *build_pair_database(catalog, "pyramid")),
            8,
        ),
    ]

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
