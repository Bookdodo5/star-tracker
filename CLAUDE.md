# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build and validate local star-identification code:

- Centroid extraction from an image.
- Yale Bright Star Catalog from `data/catalog.bin`.
- Independent TETRA and Pyramid identifiers.
- C implementation suitable for later embedded integration.

Targets: ≥90% accuracy, ~3–5 Hz, attitude output as RA, DEC, roll, and rotation matrix.

## Quick start: `run.ps1`

`run.ps1` at the project root is the single entry point and wraps the workflows below:

```powershell
.\run.ps1 build                                   # configure + build all identifier and centroid targets
.\run.ps1 test [samples] [fov]                     # unit test + synthetic benchmark summary (default 100 10)
.\run.ps1 identify <image.png|.ppm> [fov]          # full pipeline: image -> centroids -> attitude
.\run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10      # download a DSS image for the field and identify it
```

The raw commands each subcommand wraps are documented below.

## Build Commands

**C identifier (release, all targets):**
```powershell
cmake -S .\identifier -B .\identifier\build-generated-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\identifier\build-generated-release --target demo_centroid_compare batch_synthetic_compare test_star_identifier
```

**Centroid pipeline:**
```powershell
cmake -S .\centroid -B .\centroid\build-mingw -G "MinGW Makefiles"
cmake --build .\centroid\build-mingw --target centroid_extract
```

**Regenerate C database sources** (run after changing Python-side DB generation):
```powershell
python .\identifier\tools\export_catalog_db.py
python .\identifier\tools\export_pyramid_db.py
python .\identifier\tools\export_tetra_db.py
```
Generated files are written to `identifier/generated/`.

## Test / Run Commands

**C unit test:**
```powershell
.\identifier\build-generated-release\test_star_identifier.exe
```
Expected output: `C star identifier tests passed`

**Python smoke test:**
```powershell
python scripts/smoke_test.py
```

**Full pipeline from PNG:**
```powershell
.\centroid\tools\png_to_ppm.ps1 .\centroid\test-image\10h16m56s-59-51-22.png
.\centroid\build-mingw\centroid_extract.exe .\centroid\test-image\10h16m56s-59-51-22.ppm .\centroid\test-image\stars.csv
.\identifier\build-generated-release\demo_centroid_compare.exe .\centroid\test-image\stars.csv 877 877 10
```

**Synthetic batch test / benchmark:**
```powershell
.\identifier\build-generated-release\batch_synthetic_compare.exe 100 10 10 .\outputs\c_batch_fov10.csv
```
Prints a per-algorithm summary table (Accuracy%, Mean_ms, DB_ms, Verify_ms, DB_MB) and always
writes per-image per-step timing to `outputs/benchmark_latest.csv` (camera/db/verify microseconds
for each algorithm; camera is 0 in synthetic mode, which builds observed vectors directly).

**Real image from DSS/SkyView (fetch + identify):**
```powershell
python .\scripts\fetch_dss_image.py --ra 83.8 --dec -5.4 --fov 10 --size 877 --output .\outputs\test_dss.ppm
.\centroid\build-mingw\centroid_extract.exe .\outputs\test_dss.ppm .\outputs\test_dss_stars.csv
.\identifier\build-generated-release\demo_centroid_compare.exe .\outputs\test_dss_stars.csv 877 877 10
```

**Real-image batch benchmark** (fetches many DSS fields, reports per-algorithm accuracy %):
```powershell
python .\scripts\batch_real_image_compare.py --count 15 --fov 10 --tolerance-deg 0.5
```
Caches images under `cache/real_images/`; writes `outputs/real_batch_latest.csv`.

**DSS centroid diagnostic** (projects catalog stars at a known attitude and matches detected centroids):
```powershell
python .\scripts\diagnose_dss_centroids.py --stars .\outputs\test_dss_stars.csv --ra 83.8 --dec -5.4 --fov 10 --size 877
```

**Parity harnesses** (diagnose C-vs-Python divergence on the Orion field):
```powershell
python .\scripts\parity_harness_tetra.py
python .\scripts\parity_harness_pyramid.py
```

**Controlled catalog-rendered image pipeline:**
```powershell
python .\scripts\render_catalog_test_image.py --output .\outputs\catalog_render.ppm --truth .\outputs\catalog_render_truth.csv --fov 10 --image-size 877
.\centroid\build-mingw\centroid_extract.exe .\outputs\catalog_render.ppm .\outputs\catalog_render_stars.csv
.\identifier\build-generated-release\demo_centroid_compare.exe .\outputs\catalog_render_stars.csv 877 877 10
```

## Architecture

### Layer overview

```
PNG/PPM image
    └── centroid/centroid_extract.cpp   → stars.csv  (pixel x, y, brightness)
            └── identifier/src/camera_model.c    → ObservedStar[] (unit vectors)
                    ├── identifier/src/identify_tetra.c   → MatchResult
                    └── identifier/src/identify_pyramid.c → MatchResult
                            └── identifier/src/verify.c   (shared, geometry-only)
```

Python layer (`src/star_tracker_core.py`, `scripts/`) handles catalog parsing, database construction, reference implementations, and batch evaluation. C layer handles the runtime identification.

### Repository layout

- `identifier/` — C identifier (source in `identifier/src`, headers in `identifier/include`, generated DBs in `identifier/generated`, DB generators in `identifier/tools`). Build dirs are git-ignored.
- `centroid/` — C++ centroid extractor (`centroid_extract.cpp` for detection, `centroid_cli.cpp` for CLI, `tools/png_to_ppm.ps1`). No HLS dependencies — all `#pragma HLS` directives and `ap_int.h` have been removed.
- `scripts/` — maintained Python: `fetch_dss_image.py`, `render_catalog_test_image.py`, `diagnose_dss_centroids.py`, `batch_real_image_compare.py`, `parity_harness_{tetra,pyramid}.py`, `{tetra,pyramid}_reference.py`, `verify_python_reference.py`, `smoke_test.py`.
- `src/star_tracker_core.py` — shared catalog/DB Python helpers.
- `data/catalog.bin` — Yale Bright Star Catalog.
- `docs/REPORT.md` — deep TETRA vs Pyramid analysis.
- `archive/` — retired notebooks and one-off scripts (git-ignored).
- `outputs/`, `cache/` — generated results and cached DSS images (git-ignored).

### Image orientation convention (critical)

The camera model (`identifier/src/camera_model.c`) expects the **physical astronomical convention**:
**north up, east left** — what a real camera and DSS/SkyView images produce when looking
outward at the celestial sphere. Pixel X increases to the right (west) and pixel Y increases
downward (south), so `pixel_to_unit_vector` negates **both** axes (`(cx-x)/fx`, `(cy-y)/fy`)
to recover a right-handed `(east, north, boresight)` frame aligned with the catalog.

If observed vectors come out chirally mirrored, `solve_attitude_triad` (proper rotations only)
cannot recover a rotation: TETRA fails outright and Pyramid can return a **false positive** with
a plausible score but wrong attitude. The synthetic renderer (`scripts/render_catalog_test_image.py`)
uses the same east-left/north-up convention so its images match real images.
`scripts/diagnose_dss_centroids.py` checks centroid-vs-catalog overlap under four orientations to
catch any future flip regression. Note `batch_synthetic_compare` bypasses the camera model entirely
(it builds observed vectors straight from catalog unit vectors), so its accuracy is independent of
the pixel orientation convention.

### Independence rule

TETRA and Pyramid must never confirm or reject each other. Both receive the same `ObservedStar[]`, produce separate `MatchResult` values, and call the shared `verify_attitude()` — which only checks catalog geometry and is algorithm-agnostic.

### Key C types (`identifier/include/star_types.h`)

- `DetectedStar` — raw pixel centroid output
- `ObservedStar` — unit-vector direction + brightness after camera model
- `CatalogStar` — HR number + Q15-quantized unit vector + magnitude
- `MatchResult` — matched HR IDs, residuals, attitude matrix, score, success flag

### Database encoding

- Unit vectors: Q15 format (int16 in range [-32767, 32767] → [-1, 1])
- Pair separations: uint16 codes (0–65535 mapping to 0–max_fov radians)
- Residuals: uint16 arcseconds, saturated at 65535

### Verifier thresholds (`identifier/include/verify.h`)

```c
#define VERIFY_MIN_INLIERS 6
#define VERIFY_MAX_RESIDUAL_ARCSEC 900u
#define VERIFY_MAX_MEAN_RESIDUAL_ARCSEC 150u
```

Success requires both `count >= VERIFY_MIN_INLIERS` **and** `mean_residual <= 150"`. The mean
residual gate is critical for real images: correct solves cluster at ~20–80" mean; false positives
at ~190–490". The score alone cannot separate them. The brightness/magnitude rank inversion check
is a **score penalty only** (−2000 each), not a hard gate — uncalibrated real images always have
some rank mismatches even on correct matches.

## Code Style

- Meaningful variable names; avoid `code`, `out`, `dir`.
- JavaDoc-style `/** ... */` comments on every function and non-obvious logic — written to be understandable without surrounding context.
- Print progress or ETA for any operation that takes longer than 10 seconds.

## Current Status and Known Issues

Both identifiers now meet the accuracy/timing targets on synthetic data and solve real
DSS images. Status as of the 2026-06 fix pass:

### TETRA

Working. `batch_synthetic_compare 100 10 10` and `100 15 10` both report **100%** at ~4–6 ms
mean (well under the 333 ms budget). The fix was DB coverage: the database is now built with
**dual field radii** (7.5° and 5° passes, deduplicated → ~1.47M tetrads) so both FOV=10 and
FOV=15 fields are covered, plus an early-exit once a high-confidence match is found.

### Pyramid

Working. `batch_synthetic_compare 100 10 10` and `100 15 10` both report **100%** at ~30–60 ms
mean. Two bugs were fixed: a per-grow branch budget (`PYRAMID_GROW_BRANCH_CAP`) so an early seed
pair cannot starve later ones, and a **signed** `pair_score()` (the previous `uint32_t` score
underflowed for well-voted correct pairs, ranking them last).

### Real DSS images

Measured by `scripts/batch_real_image_compare.py` over 13 DSS fields (FOV=10, 0.5° tolerance):

- **TETRA: 100%** (13/13), all within ~0.02° of truth. Reliable on real images.
- **Pyramid: ~23%** correct, with **zero false positives** — it solves rich fields and otherwise
  returns `success=false` rather than a confident wrong attitude.

What unlocked real-image solving:
1. **Image orientation convention** (see above) — real images were mirrored vs the old renderer.
2. **Centroid sensitivity** — the area floor was lowered (`centroid/centroid_extract.cpp`, 10→4) and the cap
   raised (K 10→20) so faint catalog stars are detected; real fields' brightest blobs include
   non-catalog objects (galaxies/blooming), so too few true stars were captured before.
3. **Verify over the full observed set** — both identifiers match patterns from the brightest
   query stars but count inliers across all detected stars, so correct solves reach
   `VERIFY_MIN_INLIERS` and outscore coincidental ones.
4. **Mean-residual gate** (`VERIFY_MAX_MEAN_RESIDUAL_ARCSEC`) — correct solves have low mean
   residual (~20–80″), false matches high (~190–490″); the score alone does not separate them.

Pyramid's remaining real-image weakness is its seed-pair voting: with the full-sky pair DB and
noisy real separations it does not reliably surface the correct candidate, so it abstains. TETRA
is the dependable solver; Pyramid is the independent cross-check.

A catalog 3D KD-tree was added (`catalog_kd_nodes[]` in `identifier/generated/catalog_db_generated.c`,
built by `export_catalog_db.py`, searched in `catalog_db.c`). The verifier now rotates each observed
star back to catalog frame and uses an O(log N) KD-tree lookup instead of an O(N) linear scan.
Result: verify time < 0.01 ms, TETRA total 0.42 ms (was 4.7 ms), real-image ~5 ms (was ~500 ms).
Both algorithms now meet the 3–5 Hz target.

### Legacy SkyView PNG (`centroid/test-image/10h16m56s-59-51-22.png`)

Still not validated. Contains SkyView overlays and diagonal artifacts and its FOV/WCS is
unconfirmed (diagnostics suggested ~20° with negative DEC, not the assumed 10°). Prefer the
DSS fetch path for real-image testing; do not treat this PNG as ground truth.

## Do Not Do

- Tune C constants without running diagnostics first.
- Treat the SkyView PNG as reliable ground truth until FOV/WCS/projection is confirmed.
- Claim `success=true` means correct attitude without checking against known truth.
- Make Pyramid and TETRA confirm or reject each other.
