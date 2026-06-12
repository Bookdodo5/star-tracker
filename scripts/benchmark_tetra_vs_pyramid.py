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


def measure(fn):
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started, memory_mb()


def benchmark(name, make_matcher, top):
    matcher, setup_seconds, setup_memory = measure(make_matcher)
    single, single_seconds, single_memory = measure(lambda: matcher.identify(120.0, 15.0, 12.0, 6.5, top))
    batch, batch_seconds, batch_memory = measure(
        lambda: run_batch(
            matcher.identify,
            BatchConfig(samples=20, fov_deg=12.0, mag_limit=6.5, max_stars_query=top),
        )
    )
    summary = summarize_results(batch)
    return {
        "algorithm": name,
        "setup_seconds": setup_seconds,
        "single_seconds": single_seconds,
        "batch20_seconds": batch_seconds,
        "avg_identify_seconds": batch_seconds / 20,
        "memory_after_setup_mb": setup_memory,
        "memory_after_batch_mb": batch_memory,
        "single_outcome": single["outcome"],
        "batch_accuracy_pct": summary["accuracy_pct"],
    }


catalog = load_catalog()
results = [
    benchmark(
        "tetra",
        lambda: TetraMatcher(catalog, build_tetra_database(catalog, fov_deg=8.0, max_tetras_per_anchor=20)),
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
