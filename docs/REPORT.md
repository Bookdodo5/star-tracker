# TETRA vs Pyramid: Performance and Accuracy Analysis

## Overview

Both algorithms take the same input — a list of `ObservedStar` unit vectors sorted by
brightness descending — and independently produce a `MatchResult` with a rotation matrix,
inlier list, and success flag. They share one verifier (`verify.c`) that is completely
algorithm-neutral.

**Current results (100 synthetic fields, FOV=10°):**

| Algorithm | Accuracy | Mean total | DB search | Verify    | DB size  |
|-----------|----------|------------|-----------|-----------|----------|
| TETRA     | 100%     | 0.19 ms    | 0.19 ms   | < 0.01 ms | 9.79 MB  |
| Pyramid   | 100%     | 3.47 ms    | 3.47 ms   | < 0.01 ms | 4.07 MB  |

*Before catalog KD-tree: TETRA 4.7 ms (verify 4.1 ms), Pyramid 34 ms (verify 28 ms).*
*The catalog KD-tree (June 2026) reduced verify from O(N_catalog) to O(log N_catalog), ~400× speedup.*
*DB size reduction (June 2026): TETRA 44.8 MB → 9.79 MB (FOV 15°→10°, single field radius, MAX_FIELD_STARS 10→8 = 320K tetrads); Pyramid 9.0 MB → 4.07 MB (max_sep 15°→10°).*

**Centroid pipeline (877×877 image):**

| Step           | Time   | Notes |
|----------------|--------|-------|
| Grayscale      | ~3 ms  | RGB→Y integer weights |
| Integral image | ~4 ms  | 2-D prefix sum, built once |
| Blurs (×2)     | ~18 ms | Fine 3×3 + coarse 11×11, O(1)/pixel via integral image |
| DoG + threshold| ~5 ms  | |
| Morph open     | ~18 ms | Separable 3×3: 4 linear passes |
| CCL + centroid | ~8 ms  | Two-pass union-find |
| **Total**      | **~56 ms** | Was ~343 ms before optimization (6× speedup) |

**Real DSS images (20 fields, FOV=10°, 0.5° tolerance, seed=42):**

| Algorithm | Accuracy   | Mean error (correct) | Notes |
|-----------|------------|----------------------|-------|
| TETRA     | 20/20 100% | 0.013°               | ~5 ms/field; within 3–5 Hz budget |
| Pyramid   | 6/20   30% | 0.003°               | Zero false positives; abstains on hard fields |

Images fetched via `astroquery.SkyView` (DSS2 Red). FITS rows flipped on load so north is up,
matching the camera model's east-left/north-up convention.

---

## 1. How TETRA Works

### Core idea

TETRA matches a **4-star pattern (tetrad)** by its shape signature — five edge ratios that
are invariant to rotation and scale. The database stores every observable 4-combination from
the catalog, indexed by a KD-tree for nearest-neighbor lookup.

### Feature encoding

Given 4 observed unit vectors, compute all 6 pairwise angular distances in radians, sort
ascending, then normalize the 5 shortest by the longest edge:

```
feature[i] = round(edges[i] / edges[5] * 65535)   for i in 0..4
```

This 5-dimensional uint16 vector is rotation-invariant and scale-invariant (only ratios
matter). A degenerate tetrad (e.g., 4 collinear stars) with `edges[5] = 0` is discarded.

### Database: array KD-tree

The database (`tetra_kd_nodes[]`) is a flattened KD-tree of `TetraKdNode` records. Each
node stores:

- `f[5]` — five quantized edge ratios (uint16)
- `hr[4]` — the four catalog HR numbers
- `left`, `right` — child node indices (-1 = null)
- `axis` — split axis for this node (0–4)

**Size:** ~320K tetrads × 32 bytes per node ≈ **9.79 MB**. Built with a single 5° field
radius pass covering all tetrads observable in a 10° FOV, top-8 brightest stars per field.
(Prev: 1.47M tetrads / 44.8 MB with dual 7.5°+5° passes and top-10 stars.)

### Search and matching

1. For each 4-combination of the top-10 brightest observed stars (up to C(10,4) = 210
   tetrads), compute the query feature.
2. Traverse the KD-tree to find the 8 nearest database nodes within a visit budget of
   32,768 nodes.
3. For each candidate node, try all 24 permutations of the 4 catalog stars (because TETRA's
   sorted-edge feature does not preserve star ordering — any of the 24 orderings could match).
4. For each permutation, compute a TRIAD attitude matrix and check that the 4-star residual
   is ≤ 0.6° (`TETRA_PATTERN_MAX_RESIDUAL_RAD`).
5. If the pattern residual passes, call `verify_attitude()` against all detected stars.
6. Early-exit once a match scores ≥ 70,000.

### Why TETRA is fast on synthetic data

Most identification happens in the first few tetrads (the brightest stars form distinctive
patterns). The KD-tree find the nearest neighbor very quickly — fewer than 100 nodes visited
in typical cases. DB search takes only 0.6 ms; almost all time is in verification (4.1 ms).

### Why TETRA is slow on real DSS images

Real images produce 20 detected blobs rather than 10 clean synthetic stars. Many blobs are
not in the catalog (extended objects, blooming artifacts, sub-catalog stars). Each of the
C(10,4) = 210 tetrads is tested, and each calls `verify_attitude()`, which scans all 3,542
catalog stars for each of the 20 detected blobs — O(20 × 3,542) per verify call.
With ~70 verify calls per field, that is ~70 × 20 × 3,542 ≈ 5 million catalog lookups.

At 3–5 Hz the budget is 200–333 ms/frame; the current 0.5–1 s/frame needs a 3–5× speedup.

---

## 2. How Pyramid Works

### Core idea

Pyramid matches by growing a **4-star geometric chain** from a seed pair. Starting with the
angular separation of the two brightest observed stars, it retrieves candidate catalog pairs,
votes for each based on how many other observed stars are geometrically consistent, then
recursively extends each good seed pair to 4 stars. Identity is accepted when the verified
attitude scores enough inliers.

### Feature encoding

Observed separations are quantized on a linear scale:

```
sep_code = round(sep_rad / max_sep_rad * 65535)    max_sep_rad = 15°
```

Tolerances: seed lookup ±328 codes (~0.07°), growth lookup ±200 codes (~0.046°).

### Database: sorted pair array + adjacency list

Two structures coexist:

**`pyramid_pairs_by_sep[]`** — all catalog pairs within 15°, sorted by `sep_code`:

```c
typedef struct { uint16_t hr_a, hr_b, sep_code; } PairRow;
```

Seed lookup: binary-search to the tolerance window, scan pairs within it.

**`pyramid_neighbors_by_hr[]`** — for each catalog star, its neighbors sorted by `sep_code`:

```c
typedef struct { uint16_t hr_id, sep_code; } PairNeighbor;
```

Growth lookup: binary-search within one star's adjacency range — much cheaper than scanning
the full pairs array.

**Size:** ~3.49 million pair rows × 6 bytes + adjacency overhead ≈ **9.0 MB total**.

### Search and matching

1. Pre-compute all C(10,2) = 45 observed pairwise `sep_code` values into `angle_codes[][]`.
2. For each seed pair (i,j) of observed stars:
   a. Binary-search `pyramid_pairs_by_sep[]` for catalog pairs with matching separation.
   b. Score each candidate by checking how many other observed stars are geometrically
      consistent with it (voting). A well-voted pair ranks high.
   c. For each top-45 seed candidate (both orientations), call `grow()`.
3. `grow()` recursively adds observed stars by querying the adjacency list of the last
   matched catalog star, checking multi-edge geometry for each new candidate. Each grow
   attempt has a fresh branch budget of 400.
4. Once 4 stars are matched, call `verify_attitude()` against all detected stars.
5. Early-exit once a match scores ≥ 70,000.

### Why Pyramid is accurate on synthetic data

With clean separations and no contaminant blobs, the seed pair voting reliably surfaces the
correct pair near the top of the candidate list. The geometric growth chain then converges
quickly.

### Why Pyramid struggles on real DSS images

The full-sky pair database contains dozens of catalog pairs within the 0.07° seed tolerance
for any given observed separation. With noisy real separations and extra non-catalog blobs in
the detected set, the vote signal becomes diluted: the correct pair does not consistently
outrank coincidental pairs. The algorithm then spends its grow budget chasing wrong seeds and
runs out of budget before finding the correct pattern. Result: `success = false` (abstain)
rather than a wrong attitude.

---

## 3. Shared Verifier

`verify_attitude()` (`identifier/src/verify.c`) is called by both algorithms after candidate
attitude estimation. It:

1. For each detected star, finds the nearest catalog star after rotating by the candidate
   attitude (brute-force scan of all ~3,542 catalog stars).
2. Accepts the match if `residual ≤ VERIFY_MAX_RESIDUAL_ARCSEC` (900").
3. Computes `mean_residual_arcsec` over all accepted matches.
4. **Success gate:**
   - `count >= VERIFY_MIN_INLIERS` (6), AND
   - `mean_residual_arcsec <= VERIFY_MAX_MEAN_RESIDUAL_ARCSEC` (150")
5. **Score:** `count × 10000 − mean_residual × 10 − max_residual − inversions × 2000`

The mean-residual gate is critical for real images: a geometrically wrong attitude can
coincidentally gather 6 inliers under the generous 900" per-star cap, but only a correct
attitude achieves a mean below 150" (observed: correct ~20–80", false ~190–490").

### Verifier cost

**Before KD-tree:** O(N_observed × N_catalog) per call = O(20 × 9,096) ≈ 182,000 multiplies.
Called many times per field. This dominated both runtimes.

**After KD-tree (June 2026):** Each observed star rotates back to catalog frame (R^T × v_obs,
9 multiplies), then the catalog KD-tree finds the nearest star in O(log 9096) ≈ 13 operations.
Total: O(20 × 13) ≈ 260 multiplies per verify call. Measured speedup: ~400× on synthetic,
~100× on real DSS fields. TETRA real-image time dropped from ~500 ms to ~5 ms.

---

## 4. Data Structures and Memory

### CatalogStar (identifier/generated/catalog_db_generated.c)

```c
typedef struct {
    uint16_t hr;       // Harvard Revised catalog number
    int16_t  x, y, z; // Q15 unit vector: x = val / 32767.0
    int16_t  mag_q100; // Vmag × 100 (e.g., 350 = mag 3.50)
} CatalogStar;         // 10 bytes per star
```

~3,542 stars (Vmag ≤ 6.5) × 10 bytes = **35 KB**

### TetraKdNode (identifier/generated/tetra_db_generated.c)

```c
typedef struct {
    uint16_t f[5];     // five normalized edge ratios
    uint16_t hr[4];    // catalog HR numbers for the four stars
    int32_t  left, right;  // child node indices
    uint8_t  axis;         // split axis (0–4)
} TetraKdNode;             // ~27 bytes, padded to 32 bytes
```

~320K nodes × 32 bytes ≈ **9.79 MB** (reduced from 1.47M nodes / 44.8 MB)

### Pyramid structures (identifier/generated/pyramid_db_generated.c)

```c
typedef struct { uint16_t hr_a, hr_b, sep_code; } PairRow;      // 6 bytes
typedef struct { uint16_t hr_id, sep_code; }       PairNeighbor; // 4 bytes
```

~1.59M pair rows (max_sep ≤ 10°) × 6 + adjacency ≈ **4.07 MB** (reduced from 9.0 MB at 15°)

---

## 5. Accuracy Analysis

### Synthetic data (100%)

Both algorithms achieve 100% on synthetic fields because:
- Detected stars are exactly in the catalog (no contaminant blobs).
- Separations are noise-free.
- The camera model is bypassed (unit vectors built directly from catalog coordinates).

### Real DSS images

**TETRA 100%:** TETRA's shape-only matching is robust to the few-percent separation noise
from centroid jitter. The KD-tree always surfaces the correct tetrad near the top of its
candidate list as long as the field contains 4 catalog stars with separations ≤ 15°.

**Pyramid ~23%:** Pyramid's pair database is full-sky and matches pairs globally by angle
alone. On real images the seed tolerance window contains many more catalog pairs than on
synthetic data, the vote signal weakens, and the correct seed pair does not reliably rank
first. With the branch budget exhausted on wrong seeds, no correct 4-star pattern is found.

---

## 6. Problems Encountered and Fixes Applied

### Image chirality (east-left convention)

**Problem:** Real DSS images have east to the left (looking outward at the sky). The camera
model negated only the Y axis, so the X axis was mirrored. TETRA failed outright (proper
rotations only); Pyramid returned false positives.

**Fix:** `identifier/src/camera_model.c` now negates both axes:
```c
Vec3f raw = { (cx - x) / fx, (cy - y) / fy, 1.0f };
```
The synthetic renderer (`scripts/render_catalog_test_image.py`) uses `-east` as the X basis
vector for the same reason.

**Diagnostic:** `scripts/diagnose_dss_centroids.py` tests 4 orientations vs projected catalog
truth; the mirror_x orientation matched 7/10 centroids at 1.1 px mean, confirming the flip.

### TETRA DB coverage gaps

**Problem:** `batch_synthetic_compare` at FOV=15 showed 0% because tetrads for large fields
were not in the database (old scheme used a single 7.5° field radius).

**Fix:** DB generation (`identifier/tools/export_tetra_db.py`) now runs two passes at 7.5°
and 5° field radii, deduplicated, yielding ~1.47M tetrads vs ~590K before.

### Pyramid signed score underflow

**Problem:** `pair_score()` returned `uint32_t`. A well-voted correct pair had a large
negative value that wrapped to ~4 billion, ranking it last instead of first.

**Fix:** `pair_score()` now returns `int32_t`:
```c
return (int32_t)c->error + (int32_t)c->vote_error - (int32_t)c->votes * 256;
```

### Pyramid grow branch starvation

**Problem:** A single early seed pair could exhaust the branch budget, leaving later seed
pairs (possibly the correct one) with zero budget.

**Fix:** `state.branches = 0` is reset before each `grow()` call so every seed pair gets its
own fresh budget of 400 branches.

### Pyramid false positives on real images

**Problem:** Pyramid's score alone could not separate a correct match (77,070) from a false
one (82,544). The false match passed with `success = true` and ~30–167° of attitude error.

**Fix:** Added `VERIFY_MAX_MEAN_RESIDUAL_ARCSEC = 150` gate. Correct solves cluster at
20–80" mean residual; false positives at 190–490". Now Pyramid either abstains or is correct.

### Centroid sensitivity for real images

**Problem:** With area floor = 10 and K = 10, most faint catalog stars (mag 5.2–6.5) were
filtered. Real fields have non-catalog blobs occupying the brightest slots, leaving too few
true catalog stars for verification.

**Fix:** Area floor lowered to 4, K raised to 20. This recovers faint catalog stars and
ensures ≥6 inliers even in contaminated fields.

---

## 7. Proposed Improvements

### Priority 1: Faster verifier ✅ DONE (June 2026)

**Implemented:** 3D KD-tree over the 9,096 catalog unit vectors, built at Python generation
time and embedded as a static C array (`catalog_kd_nodes[]`, 254 KB). The verifier now
rotates each observed star back to catalog frame (R^T × v_obs) and searches the KD-tree
in O(log N_catalog) instead of scanning all 9,096 stars.

**Result:** verify step effectively 0 ms; TETRA total 0.42 ms (was 4.7 ms); real DSS images
~5 ms (was ~500–1000 ms). This meets the 3–5 Hz target with significant headroom.

### Priority 2: Smaller databases ✅ DONE (June 2026)

**TETRA:** 44.8 MB → **9.79 MB** (4.6× reduction, accuracy unchanged at 20/20 real DSS):
- FOV reduced 15° → 10°; single 5° field radius pass instead of dual 7.5°+5°.
- MAX_FIELD_STARS reduced 10 → 8: C(10,4)=210 → C(8,4)=70 combos per field → 883K → 320K tetrads.
- DB search time improved: 0.42 ms → 0.19 ms (fewer nodes to traverse).

**Pyramid:** 9.0 MB → **4.07 MB** (2.2× reduction, accuracy unchanged):
- MAX_FOV_DEG reduced 15° → 10°; eliminates ~half the pair rows outside the test FOV.
- DB search time improved: 5.04 ms → 3.47 ms.

Remaining options if further size reduction is needed:

**a) Compress features to 8-bit quantization**
Drop `f[5]` from uint16 to uint8. DB: −5 bytes/node × 320K ≈ −1.6 MB (→ ~8.2 MB). Needs
tolerance widening and accuracy re-check.

**b) Hash map instead of KD-tree**
Quantize the 5-feature to coarser bins, store in a flat hash table. O(1) average lookup.
Reduces TETRA further; requires collision handling.

### Priority 3: Pyramid real-image accuracy

The root cause is weak seed-pair discrimination when the full-sky database matches many
candidate pairs per observed separation. Two approaches:

**a) Reduce pair DB FOV ✅ DONE** — implemented (15° → 10°, see Priority 2 above).
Accuracy held at 6/20 = 30%; the vote-signal problem remains at 10° for hard fields.

**b) Geometric vote boost**
After the initial pair vote, compute a second-pass score using the actual angular residuals
of the voted third stars (not just presence/absence). Pairs where all voted stars fit tightly
(low residual) beat pairs where votes are geometrically loose.

**c) Seed-pair brightness prior**
Enforce that the seed pair catalog magnitudes are consistent with the two observed stars'
brightness ranks among all detected stars. This eliminates large fractions of the candidate
pool without any additional database.

### Priority 4: Embedded targets

For deployment on a microcontroller:

- The TETRA KD-tree now fits in flash at **9.79 MB** — within a mid-range STM32H7 flash (8 MB
  is tight; an H7 with external QSPI flash or PSRAM handles it). Further 8-bit quantization
  could bring it to ~8.2 MB if strict flash budget is required.
- The catalog (35 KB) and Pyramid DB (4.07 MB) fit in PSRAM easily.
- The verifier must be rewritten without floating-point if the MCU lacks an FPU; Q15
  integer arithmetic throughout is feasible since vectors are already stored as Q15.
