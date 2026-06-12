# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build and validate local star-identification code:

- Centroid extraction from an image.
- Yale Bright Star Catalog from `data/catalog.bin`.
- Independent TETRA and Pyramid identifiers.
- C implementation suitable for later embedded integration.

Targets: ≥90% accuracy, ~3–5 Hz, attitude output as RA, DEC, roll, and rotation matrix.

## Build Commands

**C identifier (release, all targets):**
```powershell
cmake -S .\C -B .\C\build-generated-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\C\build-generated-release --target demo_centroid_compare batch_synthetic_compare test_star_identifier
```

**Centroid pipeline:**
```powershell
cmake -S .\Centroid -B .\Centroid\build-mingw -G "MinGW Makefiles"
cmake --build .\Centroid\build-mingw --target hls_preprocess_file_test
```

**Regenerate C database sources** (run after changing Python-side DB generation):
```powershell
python .\C\tools\export_catalog_db.py
python .\C\tools\export_pyramid_db.py
python .\C\tools\export_tetra_db.py
```
Generated files are written to `C/generated/`.

## Test / Run Commands

**C unit test:**
```powershell
.\C\build-generated-release\test_star_identifier.exe
```
Expected output: `C star identifier tests passed`

**Python smoke test:**
```powershell
python scripts/smoke_test.py
```

**Full pipeline from PNG:**
```powershell
.\Centroid\tools\png_to_ppm.ps1 .\Centroid\test-image\10h16m56s-59-51-22.png
.\Centroid\build-mingw\hls_preprocess_file_test.exe .\Centroid\test-image\10h16m56s-59-51-22.ppm .\Centroid\test-image\stars.csv
.\C\build-generated-release\demo_centroid_compare.exe .\Centroid\test-image\stars.csv 877 877 10
```

**Synthetic batch test:**
```powershell
.\C\build-generated-release\batch_synthetic_compare.exe 20 10 10 .\outputs\c_batch_fov10.csv
```

**Controlled catalog-rendered image pipeline:**
```powershell
python .\scripts\render_catalog_test_image.py --output .\outputs\catalog_render.ppm --truth .\outputs\catalog_render_truth.csv --fov 10 --image-size 877
.\Centroid\build-mingw\hls_preprocess_file_test.exe .\outputs\catalog_render.ppm .\outputs\catalog_render_stars.csv
.\C\build-generated-release\demo_centroid_compare.exe .\outputs\catalog_render_stars.csv 877 877 10
```

## Architecture

### Layer overview

```
PNG/PPM image
    └── Centroid/hls_preprocess.cpp     → stars.csv  (pixel x, y, brightness)
            └── C/src/camera_model.c    → ObservedStar[] (unit vectors)
                    ├── C/src/identify_tetra.c   → MatchResult
                    └── C/src/identify_pyramid.c → MatchResult
                            └── C/src/verify.c   (shared, geometry-only)
```

Python layer (`src/star_tracker_core.py`, notebooks) handles catalog parsing, database construction, and batch evaluation. C layer handles the runtime identification.

### Independence rule

TETRA and Pyramid must never confirm or reject each other. Both receive the same `ObservedStar[]`, produce separate `MatchResult` values, and call the shared `verify_attitude()` — which only checks catalog geometry and is algorithm-agnostic.

### Key C types (`C/include/star_types.h`)

- `DetectedStar` — raw pixel centroid output
- `ObservedStar` — unit-vector direction + brightness after camera model
- `CatalogStar` — HR number + Q15-quantized unit vector + magnitude
- `MatchResult` — matched HR IDs, residuals, attitude matrix, score, success flag

### Database encoding

- Unit vectors: Q15 format (int16 in range [-32767, 32767] → [-1, 1])
- Pair separations: uint16 codes (0–65535 mapping to 0–max_fov radians)
- Residuals: uint16 arcseconds, saturated at 65535

### Verifier thresholds (`C/include/verify.h`)

```c
#define VERIFY_MIN_INLIERS 6
#define VERIFY_MAX_RESIDUAL_ARCSEC 900u
```

Brightness/magnitude rank inversion check is active but unreliable on real (uncalibrated) images.

## Code Style

- Meaningful variable names; avoid `code`, `out`, `dir`.
- JavaDoc-style `/** ... */` comments on every function and non-obvious logic — written to be understandable without surrounding context.
- Print progress or ETA for any operation that takes longer than 10 seconds.

## Current Status and Known Issues

### TETRA

Not yet reliable. Root cause: database coverage is poor — most synthetic test fields have zero expected 4-star combinations in the DB. Attempts with different FOV/magnitude/cap settings have not reached target accuracy.

### Pyramid

Passes simple synthetic batch tests. Not yet robust on rendered catalog images. Known issues:
- Seed pair DB pruning can remove correct pairs; full DB makes `pyramid_db_generated.c` very large.
- Large branch budgets can explode runtime; global branch cap bounds it but may stop before the correct branch.
- Geometric voting in `identify_pyramid.c` is started but has not solved rendered-image true positives.

### Real image (`Centroid/test-image/10h16m56s-59-51-22.png`)

Contains SkyView overlays and diagonal artifacts — not a clean raw image. Diagnostics suggest the actual FOV is closer to 20° with negative DEC, not the assumed 10°. Identifier failures on this image should not be attributed solely to the algorithms until FOV/WCS is confirmed.

## Do Not Do

- Tune C constants without running diagnostics first.
- Treat the SkyView PNG as reliable ground truth until FOV/WCS/projection is confirmed.
- Claim `success=true` means correct attitude without checking against known truth.
- Make Pyramid and TETRA confirm or reject each other.
