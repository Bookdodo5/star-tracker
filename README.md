# Star Tracker

C implementation of the TETRA star-identification algorithm, with a C++ centroid
extractor and Python tooling for testing, evaluation, and data fetching. (A Pyramid
identifier was implemented and then retired — see [Pyramid retirement](#pyramid-retirement).)

**Current results:** TETRA 100% on synthetic data; TETRA 100% on real DSS images
(13 fields, FOV=10°, 0.5° tolerance); TETRA 97% within 0.5° on 102 KnacksatOrbit frames.
See [`docs/REPORT.md`](docs/REPORT.md) for the full TETRA-vs-Pyramid analysis (historical).

---

## Quick Start

```powershell
.\run.ps1 build                                    # configure + build all targets
.\run.ps1 test                                     # unit test + synthetic benchmark
.\run.ps1 identify <image.png|.ppm> [fov]          # full pipeline: image → attitude
.\run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10     # download a DSS image and identify it
```

Prerequisites: MinGW (GCC 13+), CMake 3.20+, Python 3.10+ with `numpy`, `astropy`,
`requests`, `pandas`.

---

## Repository Layout

```
run.ps1                       Single entry point (build / test / identify / fetch)
CLAUDE.md                     AI guidance for this codebase
data/catalog.bin              Yale Bright Star Catalog (~9,100 entries, binary)

identifier/                   C star identifier
    src/                      Algorithm implementations
        camera_model.c        Pixel → unit vector (east-left astronomical convention)
        identify_tetra.c      TETRA: 4-star pattern → KD-tree lookup → verify
        verify.c              Verifier (catalog geometry only)
        attitude.c            TRIAD solver for 2–4 correspondences
        star_math.c           Angular distance, Q15 ↔ float, matrix ops
        catalog_db.c          Catalog array + HR lookup + vector decode
        demo_* / batch_*      Demo and batch drivers
    include/                  Headers (star_types.h, verify.h, tetra_db.h)
    generated/                Auto-generated C arrays (catalog, TETRA KD-tree)
    tools/                    Python DB generators
        export_catalog_db.py  Write catalog_db_generated.c from data/catalog.bin
        export_tetra_db.py    Write tetra_db_generated.c  (~26 MB, takes ~10 min)
    CMakeLists.txt

centroid/                     C++ image → centroids extractor
    centroid_extract.cpp      DoG + threshold + morphological open + connected components
    centroid_cli.cpp          Command-line runner: PPM in → stars.csv out
    tools/png_to_ppm.ps1      Convert PNG to binary PPM (P6) via System.Drawing
    test-image/               Sample test images
    CMakeLists.txt

src/
    star_tracker_core.py      Host-side Python (never ships): catalog parsing, TETRA DB
                              builder, Python TETRA golden reference, synthetic eval harness

scripts/                      Python tools
    fetch_dss_image.py        Download a DSS2 Red FITS image and convert to PPM
    render_catalog_test_image.py  Render a synthetic star field from the catalog
    diagnose_dss_centroids.py     Test 4 centroid orientations vs projected catalog truth
    batch_real_image_compare.py   Batch real-image accuracy over many DSS fields
    eval_orbit_frames.py          Evaluate a folder of orbit frames vs output.txt truth
    smoke_test.py                 Quick regression check

docs/
    REPORT.md                 TETRA vs Pyramid deep analysis (Pyramid content is historical)

archive/                      Retired notebooks and one-off scripts (git-ignored)
    pyramid/                  Retired Pyramid identifier (C, generator, Python ref) — recoverable
```

---

## Build

Build all targets at once:

```powershell
.\run.ps1 build
```

Or manually:

```powershell
# C identifier (all targets)
cmake -S identifier -B identifier\build-generated-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build identifier\build-generated-release --target demo_centroid_compare batch_synthetic_compare test_star_identifier

# Centroid extractor
cmake -S centroid -B centroid\build-mingw -G "MinGW Makefiles"
cmake --build centroid\build-mingw --target centroid_extract
```

The generated C database files in `identifier/generated/` are pre-built and committed.
Regenerate them only if you change the catalog filtering or database parameters:

```powershell
python identifier\tools\export_catalog_db.py
python identifier\tools\export_tetra_db.py     # ~10 min, produces ~26 MB file
```

---

## Test

**Unit test** (checks C math helpers and verifier):

```powershell
.\run.ps1 test
# expected: "C star identifier tests passed" + synthetic benchmark table
```

Or directly:

```powershell
.\identifier\build-generated-release\test_star_identifier.exe
```

**Synthetic batch benchmark** (bypasses camera model; builds observed vectors from catalog):

```powershell
.\identifier\build-generated-release\batch_synthetic_compare.exe 100 10 10 outputs\c_batch_fov10.csv
```

Prints a TETRA summary table (Accuracy%, Mean_ms, DB_ms, Verify_ms, DB_MB) and writes per-image
timing to `outputs/benchmark_latest.csv`.

**Python smoke test:**

```powershell
python scripts\smoke_test.py
```

---

## Full Pipeline

### From a PNG or PPM image

```powershell
.\run.ps1 identify path\to\image.png 10
```

Steps it runs:
1. Convert PNG → PPM (if needed) via `centroid\tools\png_to_ppm.ps1`.
2. Run `centroid\build-mingw\centroid_extract.exe` → `outputs\<name>_stars.csv`.
3. Run `identifier\build-generated-release\demo_centroid_compare.exe` → attitude printed.

### Download a DSS field and identify it

```powershell
.\run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10
```

Equivalent manual steps:

```powershell
python scripts\fetch_dss_image.py --ra 83.8 --dec -5.4 --fov 10 --size 877 --output outputs\test_dss.ppm
.\centroid\build-mingw\centroid_extract.exe outputs\test_dss.ppm outputs\test_dss_stars.csv
.\identifier\build-generated-release\demo_centroid_compare.exe outputs\test_dss_stars.csv 877 877 10
```

### Batch real-image accuracy test

```powershell
python scripts\batch_real_image_compare.py --count 15 --fov 10 --tolerance-deg 0.5
```

Caches images in `cache/real_images/`; writes `outputs/real_batch_latest.csv`.

---

## Diagnostics

**Centroid orientation check** (verify east-left convention after any camera model change):

```powershell
python scripts\diagnose_dss_centroids.py --stars outputs\test_dss_stars.csv --ra 83.8 --dec -5.4 --fov 10 --size 877
```

Should report `mirror_x` orientation with the highest centroid match count and lowest pixel
error. Any other orientation winning means the camera model has a chirality bug.

**Controlled catalog-rendered image:**

```powershell
python scripts\render_catalog_test_image.py --output outputs\render.ppm --truth outputs\render_truth.csv --fov 10 --image-size 877
.\centroid\build-mingw\centroid_extract.exe outputs\render.ppm outputs\render_stars.csv
.\identifier\build-generated-release\demo_centroid_compare.exe outputs\render_stars.csv 877 877 10
```

---

## Architecture Notes

### Image orientation convention

The camera model (`identifier/src/camera_model.c`) uses the **physical astronomical
convention: north up, east left**. Pixel X increases right (west), pixel Y increases down
(south). `pixel_to_unit_vector` negates both axes to produce a right-handed
(east, north, boresight) frame aligned with the catalog. DSS images from `fetch_dss_image.py`
and the synthetic renderer both use the same convention.

If centroids look correct but the attitude is wrong or mirrored, run the orientation
diagnostic above — a chirality flip in the camera model is the most common culprit.

### Pyramid retirement

A Pyramid identifier was implemented as an independent cross-check, then retired: on real
images it solved only ~23% of fields (abstaining elsewhere, with zero false positives), while
TETRA solved 100%, so it earned its keep on neither accuracy nor the embedded ROM budget. All
Pyramid code now lives in `archive/pyramid/` (C source/headers, DB generator + generated array,
the `compare.c/h` harness, and the Python reference), recoverable via git history. TETRA is the
sole runtime identifier. `verify_attitude()` checks only catalog geometry and is algorithm-neutral.

### Database encoding

- Unit vectors: Q15 (int16 in [-32767, 32767] → [-1, 1])
- TETRA features: uint16 normalized edge ratios (0–65535)
- Residuals: uint16 arcseconds, saturated at 65535

---

## Performance

| Metric          | Target   | Synthetic       | Real DSS (Orion) |
|-----------------|----------|-----------------|------------------|
| Accuracy        | ≥ 90%    | 100%            | TETRA 100%       |
| TETRA time      | < 333 ms | 0.2–0.42 ms/frame | ~5 ms/frame ✅ |
| TETRA DB size   | < 50 MB  | ~9.8 MB in RAM (26 MB source) | — |

The June 2026 catalog KD-tree optimization reduced verify time from ~4 ms to < 0.01 ms
(~400× speedup), putting TETRA well within the 3–5 Hz real-image target.
