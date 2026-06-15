# Star Identifier C Core

This folder contains the C runtime plan for comparing TETRA and Pyramid independently.

## Rule

TETRA and Pyramid must never confirm or reject each other.

Both algorithms:

1. Receive the same `ObservedStar[]`.
2. Build their own hypothesis.
3. Call the shared `verify_attitude()` function.
4. Return separate `MatchResult` values.

The shared verifier only checks catalog geometry. It does not know which algorithm produced the hypothesis.

## Build

```powershell
cmake -S .\C -B .\C\build-mingw -G "MinGW Makefiles"
cmake --build .\C\build-mingw
.\C\build-mingw\test_star_identifier.exe
```

## Generated Databases

The runtime does not read CSV files. Generate C database sources from the notebook caches:

```powershell
python .\C\tools\export_catalog_db.py
python .\C\tools\export_pyramid_db.py
python .\C\tools\export_tetra_db.py
```

The generated files are written to `C/generated/`.

Applications should link those generated `.c` files together with `star_identifier`.

To compile the generated database library:

```powershell
cmake -S .\C -B .\C\build-generated -G "MinGW Makefiles"
cmake --build .\C\build-generated --target star_identifier_generated
```

## Run With Centroid Output

Build the demo executable:

```powershell
cmake -S .\C -B .\C\build-generated -G "MinGW Makefiles"
cmake --build .\C\build-generated --target demo_centroid_compare
```

Run it with a Centroid CSV:

```powershell
.\C\build-generated\demo_centroid_compare.exe .\Centroid\stars.csv 640 480 20
```

Arguments:

```text
demo_centroid_compare <stars.csv> <image_width> <image_height> <horizontal_fov_deg>
```

Use real camera calibration values when available. The demo currently derives a simple pinhole model from image size and horizontal FOV.

## Main Interfaces

```c
bool identify_tetra(const ObservedStar *obs, uint8_t obs_count, MatchResult *out);
bool identify_pyramid(const ObservedStar *obs, uint8_t obs_count, MatchResult *out);
CompareResult compare_tetra_pyramid(const ObservedStar *obs, uint8_t obs_count);
```

`compare_tetra_pyramid()` only runs both algorithms on the same input and records time. It does not combine their answers.
