# Star Tracker

Real-time star identification: a camera/video/screen frame goes in, attitude
(RA, DEC, roll) comes out. A C++ centroid extractor feeds a C implementation of
the TETRA algorithm, compiled into one in-process library and driven from Python.

**Results:** TETRA solves real DSS fields to <0.02° and recovers an unknown FOV to
~0.5% via self-calibration; ~5 ms/frame (well inside the 3–5 Hz target).

---

## Quick Start

```powershell
python gui.py
```

1. **Build** — compiles the identification library (`live/build-mingw/libstar_live.dll`).
   Run once. Regenerates the baked star database first if it is missing.
2. **Run live** — pick a source, optionally enable **Self-calibrate FOV**, and read
   the attitude stream in the log.

Prerequisites: MinGW (GCC 13+), CMake 3.20+, Python 3.10+ with `numpy`, `opencv-python`
(and `mss` only for `--source screen`).

---

## Command line

The GUI is a thin face over these. Build:

```powershell
cmake -S live -B live/build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build live/build-mingw
```

Run live identification:

```powershell
python live_identify.py --source 0          --fov 17.75 --fov-search   # webcam, recover & lock FOV
python live_identify.py --source video.avi  --fov 10    --morph 0      # video file, keep faint stars
python live_identify.py --source screen     --fov 10    --show         # capture a monitor, draw overlay
```

Key flags: `--fov` (seed guess), `--fov-search` (recover the true FOV and lock it on the
first solve), `--morph` (0 keeps faint stars, 1 default), `--scale`, `--show`, `--save out.avi`,
`--monitor N` / `--region x,y,w,h` (screen source), `--cam-width/--cam-height`, `--list-monitors`.

**Non-square sensors (e.g. 7°×4°):** `--fov` is the **horizontal** FOV. Pixels are assumed
square (`fx=fy`), so the vertical FOV follows from the image height — pass `--fov 7` for a 7×4
camera and feed its native un-cropped resolution (`width/height ≈ 1.75` for 7×4). Add
`--fov-search` to recover the exact horizontal FOV from the first solve.

---

## Repository Layout

```
gui.py                        GUI: Build + Run live
live_identify.py              Live driver: capture (OpenCV) -> DLL -> attitude
data/catalog.bin              Yale Bright Star Catalog (binary)

live/
    live_identify.cpp         identify_frame / identify_frame_calibrate (the in-process core)
    CMakeLists.txt            Builds libstar_live.dll (centroid + identifier + baked DB)

centroid/
    centroid_extract.cpp      DoG + threshold + morphological open + connected components

identifier/
    src/                      camera_model, identify_tetra, verify, attitude, star_math, catalog_db
    include/                  Headers
    generated/                Auto-generated C arrays (catalog + TETRA KD-tree); built by tools/
    tools/                    export_catalog_db.py, export_tetra_db.py (regenerate the DB)

src/star_tracker_core.py      Host-side DB construction (imported by the export tools)
```

---

## How it works

- **FOV self-calibration.** TETRA's pattern lookup is scale-invariant (it matches edge
  *ratios*), so it finds the right star tetrad even when the seed FOV is far off. The matched
  catalog's true angles then pin the focal length directly — no FOV sweep. `--fov-search`
  recovers the true FOV on the first solve and locks it; later frames reuse it at full speed.

- **Image orientation.** The camera model (`identifier/src/camera_model.c`) uses the physical
  astronomical convention (north up, east left) and negates both pixel axes to produce a
  right-handed (east, north, boresight) frame aligned with the catalog. A chirality flip here
  makes the attitude come out mirrored.

- **Database encoding.** Unit vectors in Q15 (int16 → [-1, 1]); TETRA features as uint16
  normalized edge ratios; residuals as uint16 arcseconds.

To regenerate the baked database after changing catalog filtering or DB parameters:

```powershell
python identifier\tools\export_catalog_db.py
python identifier\tools\export_tetra_db.py     # ~10 min
```
