# Star Tracker Experiments

This repo contains local Jupyter notebook experiments for star-identification algorithms using the Yale Bright Star Catalog file `catalog.bin`.

## Layout

- `notebooks/` contains the clean runnable notebooks.
- `src/star_tracker_core.py` contains shared catalog parsing, database builders, matchers, plotting, and evaluation helpers.
- `data/catalog.bin` is the catalog input copied from the original Tetra folder.
- `cache/` stores generated matcher databases so reruns are fast.
- `outputs/` stores generated plots, smoke-test logs, and experiment artifacts.
- `openstartracker/` is archived reference material; it is not part of the active comparison.
- `archive/legacy/` keeps the original `Tetra.ipynb` before cleanup.

## Notebooks

- `notebooks/Tetra.ipynb`
- `notebooks/Pyramid_StarTracker.ipynb`

Each notebook follows the same flow:

1. Load `catalog.bin`.
2. Build or load the cached algorithm database.
3. Run a single identification test.
4. Run a batch test.
5. Generate an FOV/magnitude accuracy matrix.
6. Print findings.

## Validation

Run:

```powershell
python scripts/smoke_test.py
```

The smoke test uses a small deterministic batch to catch path, database, and matcher regressions before running full notebook sweeps.
