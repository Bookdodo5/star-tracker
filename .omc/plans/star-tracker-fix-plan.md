# Star Tracker Algorithm Fix & Pipeline Overhaul — Implementation Plan

**Status:** PENDING APPROVAL (consensus reached — Architect REVISE applied, Critic APPROVE WITH IMPROVEMENTS applied)
**Spec:** `.omc/specs/deep-interview-star-tracker-fix.md`
**Created:** 2026-06-12

---

## RALPLAN-DR Summary

### Principles
1. **Verify before diff** — pin one verified reference per algorithm before any comparative testing; diffing against an unverified reference produces uninterpretable results.
2. **Encoding-matched comparison** — Python and C use different encodings (float vs Q15/uint16); comparisons must happen at encoding-matched checkpoints, not line-by-line.
3. **Black-box accuracy first** — if encoding differences make step-level diffing ambiguous, final attitude accuracy against catalog ground truth is the authoritative signal.
4. **Independence rule** — TETRA and Pyramid reference pipelines must be kept fully separate; no shared fixture may let one algorithm's truth gate the other.
5. **Diagnose before fix** — never modify C code until the failure layer is identified by evidence, not speculation.

### Decision Drivers
1. **Reference ambiguity** — two "Python TETRA" files exist: `Tetra/Tetra.ipynb` (6 KB, user-verified) and `notebooks/Tetra.ipynb` (871 KB, Codex-written). No verified Pyramid baseline exists at all. Must pin references before any comparison.
2. **Encoding mismatch prevents line-level diffing** — C uses Q15-quantized unit vectors, uint16 separation codes, and `PYRAMID_MAX_BRANCHES=1200` early-exit; Python uses float pair_db with unbounded growth. Legitimate divergences at every intermediate step make line-level diffing uninformative.
3. **Unknown C failure mode** — C hasn't been run recently; root cause could be in DB generation, quantization, algorithm logic, or verification thresholds.

### Viable Options

#### Option A: Fix-in-place (read → patch)
**Approach:** Read existing C source, identify bugs by code inspection, apply targeted patches.
**Pros:** Fast if bugs are simple and obvious.
**Cons:** Without a reproducible comparison, "fixed" may mean "doesn't crash." Codex may have introduced multiple interacting bugs invisible to inspection.
**Invalidation:** The C failure mode is explicitly unknown — the code hasn't been run recently. Code inspection is only reliable when you know what you're looking for. With ≥4 potential bug sites (DB generation, Q15 quantization, algorithm logic, verification thresholds) and a Codex-written port of unknown fidelity, any patch based on inspection alone cannot be verified without running the result anyway. Option A requires the same regression testing as Option C but provides no diagnostic evidence to guide the fix — making it strictly weaker. Rejected because speculative patching produces unverifiable changes.

#### Option B: Reference-driven rewrite from Python
**Approach:** Treat verified Python as specification; rewrite C algorithms from scratch.
**Pros:** Clean slate; every line traces to a verified Python analogue.
**Cons:** High effort; scope expansion beyond what is a fix task; risks losing existing correct optimizations.
**Invalidation:** The task scope is a porting fix, not a new implementation. The C code may already have correct sections — full rewrite discards those and introduces new surface area for bugs. Additionally, no verified Pyramid Python baseline exists yet (notebooks/ is Codex-written), so the "trusted specification" premise of Option B does not hold for Pyramid. Option B would be the right choice if the C code were fundamentally misaligned in design — but checkpoint diffing (Option C) will determine this first, and a targeted rewrite of specific failing sections remains available as a fallback within Option C if needed. Rejected because scope and prerequisite (verified reference for both algorithms) are not satisfied.

#### Option C: Checkpoint diffing with parity harness (selected)
**Approach:** (1) Pin and verify one Python reference per algorithm. (2) Build a parity harness that runs Python through the same `export_*_db.py` quantization so both sides consume Q15/uint16 inputs. (3) Compare at three fixed checkpoints with tolerance bands: candidate set membership, pre-verify residual, final attitude matrix. (4) If checkpoint diffing is still ambiguous, fall back to black-box accuracy on synthetic fields as the authoritative signal.
**Pros:** Avoids false-positive "divergences" from encoding differences; any remaining divergence is a genuine bug. Black-box fallback is authoritative when encoding differences persist.
**Cons:** Requires upfront harness work (~half-day). More structured than Option A.
**Selection rationale:** Resolves the core weakness of naive side-by-side diffing (invalid port assumption). Option A is too speculative; Option B is out of scope.

**Independence guard:** TETRA and Pyramid parity harnesses are fully separate scripts; no shared fixture crosses algorithm boundaries.

---

## Requirements Summary

The C implementations of TETRA and Pyramid star identification produce wrong results. The root cause is unknown and must be diagnosed. The Python references are partially unverified (`notebooks/` was Codex-written). The fix must:

1. Pin one verified Python reference per algorithm (resolve the `Tetra/Tetra.ipynb` vs `notebooks/Tetra.ipynb` ambiguity).
2. Build a parity harness that runs both Python and C through encoding-matched inputs.
3. Use checkpoint diffing + black-box accuracy to identify and fix C failure mode(s).
4. Integrate a DSS/SkyView web API for real sky image validation.
5. Add a benchmarking suite (per-algorithm summary + per-step timing).
6. Reorganize the project into a clean, single-entry-point structure.

**Target:** ≥90% accuracy (target ≥95%) at 3–5 Hz on 10–15° FOV images, ≤0.5° attitude error.

---

## Acceptance Criteria

### Phase 0: Initial Build & Baseline Capture
- [ ] `C\build-generated-release\batch_synthetic_compare.exe 20 10 10` builds and runs without crash; output captured to `outputs\c_batch_baseline.csv`.
- [ ] `C\build-generated-release\test_star_identifier.exe` runs without crash; output captured.
- [ ] `outputs\baseline_summary.txt` records: TETRA accuracy%, Pyramid accuracy%, any build errors.

### Phase 1: Pin and Verify Python References
- [ ] The TETRA reference is explicitly chosen: `Tetra/Tetra.ipynb` (6 KB, user-verified) is the authoritative reference; `notebooks/Tetra.ipynb` (871 KB, Codex) is used as a candidate but must be validated against it.
- [ ] Running `scripts\verify_python_reference.py --algo tetra` with fixed seed (numpy 42) on 20 synthetic fields at FOV=10° produces ≥85% accuracy from `Tetra/Tetra.ipynb` core matching logic.
- [ ] Running `scripts\verify_python_reference.py --algo pyramid` with fixed seed on 20 synthetic fields at FOV=8° produces ≥85% accuracy from `notebooks/Pyramid_StarTracker.ipynb`. If < 85%, the notebook is corrected using the CLAUDE.md algorithm description and pair-angle geometry as specification; `Tetra/Tetra.ipynb` serves as structural template.
- [ ] Both references' accuracy numbers are committed to `outputs\python_reference_baseline.txt` as the fixed regression baseline.

### Phase 2: Parity Harness
- [ ] `scripts\parity_harness_tetra.py` exists and: (a) generates a fixed synthetic field, (b) runs Python TETRA through the same Q15 quantization as `C/tools/export_tetra_db.py`, (c) calls the C binary on the same field, (d) reports agreement at three checkpoints: candidate set overlap (Jaccard ≥ 0.5 expected), pre-verify best residual (within 2× quantization step), final attitude (within 1°).
- [ ] `scripts\parity_harness_pyramid.py` exists with equivalent checkpoints for Pyramid, kept strictly separate from TETRA harness.
- [ ] Running both harness scripts on the current (broken) C code produces a written divergence report in `outputs\divergence_report.txt` that names the failing checkpoint and the responsible code path.

### TETRA C Fix
- [ ] `batch_synthetic_compare.exe 100 10 10` reports TETRA accuracy ≥90%.
- [ ] `parity_harness_tetra.py` passes all three checkpoints on the fixed test field (Orion RA=83.8°, DEC=-5.4°).
- [ ] TETRA identification completes in ≤333ms per call on the 100-sample batch.

### Pyramid C Fix
- [ ] `batch_synthetic_compare.exe 100 10 10` reports Pyramid accuracy ≥90%.
- [ ] `parity_harness_pyramid.py` passes all three checkpoints on the fixed test field.
- [ ] Pyramid identification completes in ≤333ms per call.

### Real Image API
- [ ] `scripts\fetch_dss_image.py --ra 83.8 --dec -5.4 --fov 10 --size 877 --output outputs\test_dss.ppm` downloads a DSS2 Red image and saves it as PPM without error.
- [ ] Full pipeline on the fetched image returns success=true and attitude within 0.5° of RA=83.8°, DEC=-5.4°.
- [ ] `scripts\fetch_dss_image.py --batch 20 --fov 10 --output-dir outputs\dss_batch\` fetches 20 images and pipeline achieves ≥80% correct identifications (within 0.5°).

### Benchmarking Suite
- [ ] `run.ps1 test` prints a table: Algorithm | Accuracy% | Mean_ms | P99_ms | DB_MB | centroid_ms | db_ms | verify_ms.
- [ ] Results saved to `outputs\benchmark_latest.csv` (one row per test image).

### Project Reorganization
- [ ] `run.ps1 build` compiles all C targets without error from a fresh shell.
- [ ] `run.ps1 test` runs full benchmark and prints summary.
- [ ] `run.ps1 identify <image.png> [--fov N]` runs end-to-end pipeline.
- [ ] `run.ps1 fetch --ra X --dec Y --fov Z [--batch N]` fetches DSS image and runs pipeline.
- [ ] Dead/Codex-experiment code moved to `archive\` with a README.
- [ ] `CLAUDE.md` updated to reflect actual state.

---

## Implementation Steps

### Phase 0: Initial Build & Baseline Capture

**Step 0.1 — Build current C code**
```powershell
cmake -S .\C -B .\C\build-generated-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\C\build-generated-release --target demo_centroid_compare batch_synthetic_compare test_star_identifier
```
- Capture any build errors to `outputs\build_log.txt`.
- Files: `C\CMakeLists.txt`, `C\src\*.c`, `C\generated\*.c`

**Step 0.2 — Run existing C tests and capture baseline**
```powershell
.\C\build-generated-release\batch_synthetic_compare.exe 20 10 10 outputs\c_batch_baseline.csv
.\C\build-generated-release\test_star_identifier.exe
```
- Record TETRA accuracy%, Pyramid accuracy%, timing to `outputs\baseline_summary.txt`.
- Note if 0% or crash — do not attempt to interpret yet.

### Phase 1: Pin and Verify Python References

**Step 1.1 — Resolve the TETRA reference ambiguity**
- Two files claim to be the Python TETRA implementation:
  - `Tetra\Tetra.ipynb` (6 KB): user's own implementation, manually tested. User confirmed this is the one they verified.
  - `notebooks\Tetra.ipynb` (871 KB): Codex-written, more elaborate, unverified.
- **Decision rule** (apply in order, stop at first match):
  1. Run `git log --follow --oneline Tetra\Tetra.ipynb` — if this file predates `notebooks\Tetra.ipynb` in git history, it is the authoritative reference (user wrote it first, Codex extended it later).
  2. If git history is unclear, compare cell count: the 6 KB file has fewer cells (simpler = user's). Accept the simpler file.
  3. Record the chosen file's SHA: `git rev-parse HEAD:Tetra/Tetra.ipynb` → write to `outputs\python_reference_baseline.txt` alongside accuracy numbers.
- **Authoritative TETRA reference is `Tetra\Tetra.ipynb`** — this is the decision based on the user's confirmation.
- Action: Extract the core matching logic into `scripts\tetra_reference.py`. This becomes the canonical Python TETRA implementation.

**Step 1.2 — Verify and pin Pyramid reference**
- `notebooks\Pyramid_StarTracker.ipynb`: Codex-written, no prior verified version exists.
- Run independently. If accuracy ≥85% on 20-sample fixed-seed test: accept as reference.
- If < 85%: repair using the CLAUDE.md algorithm description + pair-angle geometry. Extract core logic into `scripts\pyramid_reference.py`.
- Either way, `scripts\pyramid_reference.py` is the canonical Python Pyramid implementation.

**Step 1.3 — Write reproducible reference test**
- File: `scripts\verify_python_reference.py`
- Fixed seed: numpy seed 42. Fixed parameters: 20 fields, FOV=10° for TETRA, FOV=8° for Pyramid, mag≤6.5.
- Prints accuracy for both algorithms. Saves to `outputs\python_reference_baseline.txt`.
- This file is the regression baseline; C must match or exceed it.

### Phase 2: Parity Harness

**Step 2.1 — Understand the quantization pipeline**
- Read `C\tools\export_tetra_db.py` and `C\tools\export_pyramid_db.py` to understand:
  - How unit vectors are encoded as Q15 int16: `round(v * 32767)` clamped to [-32767, 32767].
  - How pair separations are encoded as uint16: `round(sep_rad / max_sep_rad * 65535)`.
  - How TETRA features (f1-f5) are encoded as uint16.

**Step 2.2 — Build TETRA parity harness**
- File: `scripts\parity_harness_tetra.py`
- Fixed test field: Orion (RA=83.8°, DEC=-5.4°, FOV=10°, numpy seed 42).
- Step A: Run `scripts\tetra_reference.py` on the field; collect candidate tetrads and matched HR IDs.
- Step B: Run the same field through the Q15 quantization from `export_tetra_db.py`; verify the correct tetrads exist in the current `C\generated\tetra_db_generated.c` database (parse the array or check via the Python KD-tree on Q15-quantized features).
- Step C: Run C binary (`batch_synthetic_compare.exe` or `demo_centroid_compare.exe` on a synthetic centroid CSV for the test field); capture attitude output.
- Checkpoints with explicit pass tolerances:
  1. **Candidate set** — PASS if: the DB (verified via Python KD-tree on Q15-quantized features) contains ≥1 tetrad whose HR IDs include ≥3 of the top-5 brightest stars in the test field. FAIL = 0 matching tetrads → DB coverage problem.
  2. **Pre-verify residual** — PASS if: the C algorithm returns a best-candidate residual ≤ 1800 arcsec (2× `VERIFY_MAX_RESIDUAL_ARCSEC=900` from `C/include/verify.h`; factor-of-2 tolerance absorbs Q15 rounding). FAIL = residual > 1800 arcsec or no candidate found → quantization or algorithm logic problem.
  3. **Final attitude** — PASS if: C returns RA/DEC within 1° of the test field center (Orion RA=83.8°, DEC=-5.4°). FAIL = wrong attitude or success=false → verification or attitude computation problem.
- Output: `outputs\parity_tetra_report.txt` — pass/fail per checkpoint with numeric values + diagnosis.

**Step 2.3 — Build Pyramid parity harness**
- File: `scripts\parity_harness_pyramid.py`
- Same test field. Kept strictly separate from TETRA harness (separate script, separate output file).
- Checkpoints with explicit pass tolerances:
  1. **Seed pair coverage** — PASS if: the pair DB contains an entry whose two HR IDs match the two brightest stars in the test field (binary yes/no; parse `C\generated\pyramid_db_generated.c` or verify via Python re-running `export_pyramid_db.py` on the same field). FAIL = seed pair absent → DB pruning problem.
  2. **Branch budget** — PASS if: running C on the test field with `PYRAMID_MAX_BRANCHES=12000` (10× current cap, for diagnostic only) produces success=true. If it passes at 12000 but fails at 1200 → branch budget problem. If it still fails at 12000 → voting or logic problem.
  3. **Final attitude** — PASS if: C returns RA/DEC within 1° of the test field center. FAIL = wrong attitude or success=false even at 12000 branches → geometric voting or verification problem.
- Output: `outputs\parity_pyramid_report.txt` — pass/fail per checkpoint with numeric values + diagnosis.

**Step 2.4 — Run harnesses on current broken C code**
- Run both harness scripts; write `outputs\divergence_report.txt` summarizing which checkpoint fails and the likely root cause.
- This determines the fix priority (whichever algorithm has a shallower root cause is fixed first).

### Phase 3: Fix TETRA C Port

Based on the parity harness divergence report, target the failing checkpoint:

**If Checkpoint 1 fails (DB coverage)**:
- Problem: `C\tools\export_tetra_db.py` is generating a sparse database that misses most test fields.
- Fix: Adjust coverage parameters in `export_tetra_db.py` (FOV, magnitude limit, cap). Regenerate `C\generated\tetra_db_generated.c`.
- Key file: `C\tools\export_tetra_db.py`, parameters `MAX_FOV_DEG`, `MAG_LIMIT`, `CAP_PER_STAR`.

**If Checkpoint 2 fails (residual too high)**:
- Problem: Feature normalization or KD-tree query tolerance mismatch between Python and C.
- Fix: Compare `compute_features()` in `C\src\identify_tetra.c` against `scripts\tetra_reference.py`. Check Q15 rounding, feature ordering, and L1 distance threshold `TETRA_MAX_L1_DIST`.
- Key files: `C\src\identify_tetra.c` lines ~30-80.

**If Checkpoint 3 fails only (attitude wrong despite match)**:
- Problem: `C\src\verify.c` attitude computation or `verify_attitude()` threshold.
- Fix: Compare `verify_attitude()` implementation against Python verify logic.

**Regression gate after each change:**
```powershell
.\C\build-generated-release\batch_synthetic_compare.exe 100 10 10 outputs\c_batch_tetra_fix.csv
```
Target: TETRA ≥90%. Target ≥95%.

### Phase 4: Fix Pyramid C Port

Based on `outputs\parity_pyramid_report.txt`:

**If Checkpoint 1 fails (seed pair missing)**:
- Fix: Adjust `export_pyramid_db.py` pruning logic. Try building full DB first (remove pruning) to confirm this is the bottleneck; then re-add conservative pruning that retains the correct pair.
- Key file: `C\tools\export_pyramid_db.py`.

**If Checkpoint 2 fails (branch cap hit before correct hypothesis)**:
- Problem: Branch budget exhausted; correct hypothesis explored after cap.
- Fix options: (a) Increase `PYRAMID_MAX_BRANCHES` in `C\src\identify_pyramid.c`; (b) Improve branch ordering to prioritize more promising hypotheses first.
- Key file: `C\src\identify_pyramid.c`.

**If Checkpoint 2 fails (geometric voting incomplete)**:
- **Bug-vs-gap classification criterion** (apply before touching code):
  - Read `C\src\identify_pyramid.c` and search for `vote` or `geometric`. If a voting loop exists but is commented out or returns early unconditionally → **bug** (fix it).
  - Read `notebooks\Pyramid_StarTracker.ipynb` and identify the voting cell. If the Python has ≥5 lines of voting logic with no counterpart anywhere in `C\src\identify_pyramid.c` → **feature gap** (implement it).
  - If classification is ambiguous, treat as gap (safer: implementing from Python reference is correct by definition).
- **Bug fix path**: Uncomment / fix the existing voting code in `C\src\identify_pyramid.c`.
- **Gap fix path**: Implement the geometric voting loop from `notebooks\Pyramid_StarTracker.ipynb` (verified in Phase 1) into `C\src\identify_pyramid.c`. Keep the Python voting logic open side-by-side during implementation.

**Regression gate:**
```powershell
.\C\build-generated-release\batch_synthetic_compare.exe 100 10 10 outputs\c_batch_pyramid_fix.csv
```
Target: Pyramid ≥90%.

**Cross-check — independence rule**: After both fixes, run both algorithms on the same 100-field batch and confirm neither algorithm's result is used to gate the other's output (verify `C\src\identify_tetra.c` and `C\src\identify_pyramid.c` have no cross-references).

### Phase 5: Real Image API Integration

**Step 5.1 — Write DSS/SkyView fetch script**
- File: `scripts\fetch_dss_image.py`
- Use `astroquery.skyview` (`pip install astroquery`) or direct HTTP to SkyView.
- Survey: DSS2 Red (best star density at 10-15° FOV).
- Parameters: `--ra`, `--dec`, `--fov`, `--size` (pixels, default 877), `--output` (PPM path).
- FITS → PPM conversion: map intensity to 0-255 using percentile stretch (2nd–98th percentile).
- Handle SkyView rate limits gracefully (retry with 2s backoff).

**Step 5.2 — Validate on known fields**
- Test on Orion (RA=83.8°, DEC=-5.4°), M31 (RA=10.68°, DEC=41.27°), Pleiades (RA=56.75°, DEC=24.12°).
- Run full pipeline (`run.ps1 fetch --ra X --dec Y --fov 10`). All three must return success=true and attitude within 0.5°.

**Step 5.3 — Batch mode for accuracy measurement**
- `--batch N` mode: N random RA/DEC positions, fetch each, run pipeline, compare vs. known pointing.
- Save results to `outputs\dss_batch_results.csv`.

### Phase 6: Benchmarking Suite

**Step 6.1 — Add per-step timing to C**
- Modify `C\src\batch_synthetic_compare.c`: add `QueryPerformanceCounter` calls around:
  - Camera model (unit vector conversion): `C\src\camera_model.c`
  - DB search inside `identify_tetra()` / `identify_pyramid()`
  - Attitude verification: `C\src\verify.c`
- Output columns added to CSV: `tetra_camera_us`, `tetra_db_us`, `tetra_verify_us`, `pyramid_camera_us`, `pyramid_db_us`, `pyramid_verify_us`.

**Step 6.2 — DB memory constants**
- Add compile-time size reporting: sizeof of each generated array in `C\generated\*_db_generated.c`.
- Print: `TETRA DB: X.X MB | Pyramid DB: Y.Y MB | Catalog: Z KB`.

**Step 6.3 — Summary table and CSV**
- After batch run, `batch_synthetic_compare` prints summary table and saves `outputs\benchmark_latest.csv`.

### Phase 7: Project Reorganization

**Step 7.1 — Create run.ps1**
- File: `run.ps1` (project root)
- Subcommands:
  - `build`: cmake configure + build all targets; prints success/error clearly.
  - `test [--fov N] [--samples N]`: runs synthetic batch + catalog-rendered test; prints benchmark table; saves CSV.
  - `identify <image.png> [--fov N]`: centroid → C identifier; prints RA/DEC/roll and success status.
  - `fetch --ra X --dec Y --fov Z [--batch N]`: fetches DSS image(s); runs pipeline; reports accuracy.
- Each subcommand prints usage if arguments are missing or wrong.
- Prerequisite check at start: verify MinGW, Python, astroquery are in PATH.

**Step 7.2 — Archive dead Codex code**
- Create `archive\` directory with `README.md` (note: Codex experiments, not production code).
- Move any `.c`/`.h`/`.py` files in the root or `C\` that are not referenced by any `CMakeLists.txt`, `run.ps1`, or Python script.
- Do NOT delete; preserve for reference.

**Step 7.3 — Clean .gitignore and structure**
- Add to `.gitignore`: `C\build-*/`, `Centroid\build-*/`, `outputs\*.ppm`, `outputs\dss_batch\`.
- Add `outputs\.gitkeep`.

**Step 7.4 — Update CLAUDE.md**
- `## Current Status`: Update TETRA and Pyramid status to reflect post-fix accuracy.
- `## Build Commands`: Point to `run.ps1 build`.
- `## Test / Run Commands`: Point to `run.ps1 test / identify / fetch`.
- Add `## Python References` section documenting `Tetra/Tetra.ipynb` (authoritative TETRA) vs `notebooks/` (verified-after-Phase-1).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Pyramid geometric voting is a fundamental feature gap, not a bug | High | High | Phase 4 explicitly branches on this: if voting is incomplete, complete it using Python reference |
| Parity harness shows DB coverage failure (requires DB regeneration) | High | Medium | DB export scripts run in <5 min; regeneration is straightforward once parameters are adjusted |
| DSS SkyView returns sparse or poor-quality images at test positions | Medium | Medium | Test with 3 known bright fields first; fallback to catalog-rendered images if API unreliable |
| Q15 quantization differences cause parity harness false-pass at checkpoint 2 | Medium | Medium | Tolerance bands (2× quantization step) absorb expected rounding; genuine bugs produce larger divergences |
| Fixing shared `C/src/verify.c` for one algorithm breaks the other | Low | High | Run both algorithms' batch tests after every change to `verify.c`. **Rollback path:** before editing `verify.c`, run `git stash`; if either algorithm's accuracy drops after the change, `git stash pop` to restore, then re-diagnose. Never commit a `verify.c` change that regresses either algorithm's batch accuracy. |
| `Tetra/Tetra.ipynb` (6 KB) turns out to use a different FOV/mag range than expected | Low | Medium | Run it on the standard 20-field test first; if accuracy < 70%, investigate before using as reference |

---

## Verification Steps

1. **After Phase 0**: `outputs\baseline_summary.txt` records current C accuracy.
2. **After Phase 1**: `python scripts\verify_python_reference.py` prints ≥85% for both algorithms from pinned references.
3. **After Phase 2**: `outputs\divergence_report.txt` names the failing checkpoint and responsible code path for each algorithm.
4. **After Phase 3**: `batch_synthetic_compare.exe 100 10 10` → TETRA ≥90%. `parity_harness_tetra.py` → all 3 checkpoints pass.
5. **After Phase 4**: same → Pyramid ≥90%. `parity_harness_pyramid.py` → all 3 checkpoints pass.
6. **After Phase 5**: `run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10` returns success=true within 0.5°. DSS batch (20 images) ≥80%.
7. **After Phase 6**: `run.ps1 test` prints benchmark table with both algorithms' accuracy, timing, memory.
8. **After Phase 7**: `run.ps1 build && run.ps1 test && run.ps1 fetch --ra 83.8 --dec -5.4 --fov 10` all succeed from a fresh shell.
9. **Final regression**: `test_star_identifier.exe` still prints `C star identifier tests passed`.
10. **Independence check**: `grep -r "tetra\|pyramid" C\src\identify_tetra.c C\src\identify_pyramid.c` confirms no cross-algorithm calls.

---

## ADR

**Decision:** Fix C algorithms using a parity harness with encoding-matched checkpoint diffing (Option C), preceded by pinning verified Python references.

**Drivers:**
- Reference ambiguity requires explicit resolution before any comparative testing
- Q15/uint16 encoding mismatch makes line-level Python-C diffing uninformative
- Unknown C failure mode requires evidence-based, not speculation-based, fix strategy

**Alternatives considered:**
- **Fix-in-place (Option A)**: Rejected — too speculative without knowing failure layer; multiple potential bug sites; no regression safety
- **Reference-driven rewrite (Option B)**: Rejected — excessive scope for a porting fix; risk of losing existing correct code

**Why chosen:** Option C balances rigor (encoding-matched checkpoints eliminate false positives) with scope (maintains existing C structure, no rewrite). The parity harness produces a written diagnosis with a named failing checkpoint, giving the implementer a precise starting point rather than a code-reading exercise.

**Consequences:**
- Positive: Fix is evidence-based; each change is motivated by a measured divergence; regression safety is built in
- Positive: Parity harness scripts become permanent regression assets
- Negative: ~0.5 day upfront investment for harness scripts before any C is touched
- Negative: If the Pyramid geometric voting is fundamentally incomplete (not just buggy), Phase 4 is higher effort than a simple bug fix

**Follow-ups:**
- After fixes, consider adding the parity harness scripts to a CI check (`run.ps1 test-harness`) so future C changes are regression-tested
- If TETRA database coverage is the root cause, evaluate whether the 25 MB generated C file can be replaced with a runtime-loaded binary file for embedded deployment
- Real camera calibration (principal point, radial distortion) should replace the current pinhole approximation before production deployment

---

## Architect Review Notes Applied

*Revision from Architect (REVISE REQUIRED → applied):*
- **Reference ambiguity resolved**: `Tetra/Tetra.ipynb` (6 KB) explicitly named as authoritative TETRA reference; `notebooks/Tetra.ipynb` (871 KB) treated as candidate requiring validation.
- **Phase ordering corrected**: Phase 1 (Python verification) now strictly precedes all diffing; parity harness (Phase 2) built only after references are pinned.
- **Line-level diffing replaced**: Three encoding-matched checkpoints (candidate set, pre-verify residual, final attitude) with Q15/branch-cap tolerance bands replace naive line-level diffing.
- **Parity harness added**: Python runs through same `export_*_db.py` quantization as C before any comparison.
- **Independence guard added**: TETRA and Pyramid harnesses are separate scripts with separate output files; no shared fixture.

## Changelog

- v2: Applied all 5 Architect revision items (reference disambiguation, phase reordering, checkpoint diffing, parity harness, independence guard)
- v3: Applied all 5 Critic improvements (explicit checkpoint pass tolerances, expanded Option A/B invalidation rationale, TETRA reference decision rule with SHA recording, Pyramid bug-vs-gap classification criterion, verify.c rollback path via git stash)
