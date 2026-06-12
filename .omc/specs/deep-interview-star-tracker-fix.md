# Deep Interview Spec: Star Tracker Algorithm Fix & Pipeline Overhaul

## Metadata
- Interview ID: di-star-tracker-2026-06-12
- Rounds: 15
- Final Ambiguity Score: 24%
- Type: brownfield
- Generated: 2026-06-12
- Threshold: 0.20 (20%)
- Threshold Source: default
- Initial Context Summarized: no
- Status: BELOW_THRESHOLD_EARLY_EXIT (remaining gap is execution-discovery ambiguity, not requirements ambiguity)

---

## Clarity Breakdown

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.82 | 0.35 | 0.287 |
| Constraint Clarity | 0.75 | 0.25 | 0.1875 |
| Success Criteria | 0.72 | 0.25 | 0.18 |
| Context Clarity | 0.72 | 0.15 | 0.108 |
| **Total Clarity** | | | **0.7625** |
| **Ambiguity** | | | **24%** |

---

## Topology

| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| TETRA fix | active | Fix C port of TETRA 4-star identifier to match Python reference accuracy | Diagnose first, fix root cause in C port |
| Pyramid fix | active | Fix C port of Pyramid pair-based identifier to work on rendered and real images | Diagnose first; Pyramid has better synthetic baseline |
| Real image API | active | Integrate DSS/SkyView web API to fetch real photometric images at known RA/DEC/FOV | Python `astroquery`/HTTP; no local app required |
| Benchmarking suite | active | Per-algorithm accuracy/timing/memory table + per-step timing breakdown | Summary table + CSV; per-step timing for centroid→camera→DB→verify |
| End-to-end validation | active | Demonstrate correct attitude (within 0.5°) from real fetched images, batch-tested | ≥90% of fetched images identified correctly at ≥3 Hz |
| Project reorganization | active | Clean structure, remove Codex dead code, standardize build, one entry-point script | `run.ps1 build / test / identify <img> / fetch --ra X --dec Y --fov Z` |

---

## Goal

Fix the C implementations of TETRA and Pyramid star identification algorithms so they reach ≥90% (target ≥95%) identification accuracy at 3–5 Hz on images at 10–15° FOV, matching the verified Python reference implementations. Validate using DSS/SkyView real sky images at known pointings (≤0.5° attitude error tolerance). Deliver this inside a reorganized, easy-to-operate project with a single entry-point script.

---

## Constraints

- **Language**: Core algorithms stay in C (embedded deployment target). Python used only for testing infrastructure.
- **FOV**: 10–15° horizontal FOV.
- **Accuracy**: ≥90% identification rate (target ≥95%) on batch of ≥20 test images.
- **Speed**: ≤333ms per identification (≥3 Hz), target ≤200ms (5 Hz).
- **Attitude tolerance**: Within 0.5° (30 arcmin) of known ground truth for "correct".
- **Image source for testing**: DSS/SkyView HTTP API — no desktop app dependency.
- **Platform**: Windows (MinGW/PowerShell), later embedded.
- **Independence rule (from CLAUDE.md)**: TETRA and Pyramid must never confirm or reject each other.
- **Fix sequencing**: Diagnose C failure mode first; fix whichever algorithm is easier to fix first, then the other.
- **Python verification**: Verify `notebooks/` Codex-written accuracy before using them as a C reference. The `Tetra/Tetra.ipynb` is the only user-verified baseline.

---

## Non-Goals

- Rewriting the algorithms from scratch (fix the C port, not replace it).
- Adding camera calibration or distortion correction.
- GPU or SIMD optimization beyond basic performance.
- Changing the catalog source (Yale Bright Star Catalog stays).
- Building a GUI.
- Supporting image formats other than PPM (pipeline input) and PNG (user-supplied).
- Cross-platform build for Linux/macOS (Windows MinGW only for now).

---

## Acceptance Criteria

### Phase 0: Diagnosis
- [ ] Build the current C codebase and capture exact output of `batch_synthetic_compare` and `demo_centroid_compare` with the existing test image.
- [ ] Run the Codex Python notebooks (`notebooks/Tetra.ipynb`, `notebooks/Pyramid_StarTracker.ipynb`) and verify their accuracy claims are reproducible independently.
- [ ] Produce a written diagnosis: what is wrong in the C code vs the Python reference?

### TETRA fix
- [ ] TETRA C implementation achieves ≥90% on a 20-sample synthetic batch at FOV=10°, mag≤6.5.
- [ ] TETRA C output matches Python reference output on the same test field (same matched HR IDs ± residual tolerance).
- [ ] TETRA correctly identifies ≥80% of DSS/SkyView images fetched at known RA/DEC/FOV=10–15°.

### Pyramid fix
- [ ] Pyramid C implementation achieves ≥90% on a 20-sample synthetic batch at FOV=10°, mag≤6.5.
- [ ] Pyramid C output matches Python reference output on the same test field.
- [ ] Pyramid correctly identifies ≥80% of DSS/SkyView images fetched at known RA/DEC/FOV=10–15°.

### Real image API
- [ ] A Python script can fetch a DSS/SkyView image at given `--ra`, `--dec`, `--fov`, and `--size` parameters and save it as PPM.
- [ ] The script can fetch a batch of N images at random pointing directions and cache them locally.
- [ ] The batch accuracy of the full pipeline (centroid → C identifier) can be measured against the known pointing.

### Benchmarking suite
- [ ] A benchmark report prints a per-algorithm summary table: accuracy %, mean identification time (ms), peak DB memory (MB).
- [ ] The report includes per-step timing: centroid extraction, unit vector conversion, DB search, attitude verification.
- [ ] Report is also saved as CSV (one row per test image, columns: algorithm, correct, time_ms, memory_mb).

### End-to-end validation
- [ ] Running `run.ps1 fetch --ra 10.68 --dec 41.27 --fov 12` (M31 field) returns a valid attitude within 0.5° of the known pointing.
- [ ] Running `run.ps1 test` prints a benchmark summary showing ≥90% accuracy for at least one algorithm.

### Project reorganization
- [ ] Dead code and Codex experiments removed or clearly isolated.
- [ ] `run.ps1 build` compiles all C targets.
- [ ] `run.ps1 test` runs the full benchmark (synthetic + catalog-rendered + DSS batch).
- [ ] `run.ps1 identify <image.png>` runs the full pipeline and prints RA/DEC/roll.
- [ ] `run.ps1 fetch --ra X --dec Y --fov Z` fetches a DSS image and runs identification.
- [ ] CLAUDE.md updated to reflect actual project state.

---

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "Codex messed it up" = regression | Was there ever a working C version? | No — C code was never verified; Tetra.ipynb (user's) is the only confirmed baseline |
| TETRA Python notebooks show real accuracy | Codex may have faked test outputs | Verify notebooks/ before using as C reference |
| Real images require Stellarium | Stellarium needs to be running locally | Use DSS/SkyView HTTP API instead — no local app |
| Algorithms can't be tested without knowing failure mode | C output is currently unknown | Diagnosis is step 1 of execution; sequencing decided after diagnosis |
| TETRA and Pyramid should be fixed in parallel | May be better to sequence | Fix whichever has easier root cause first, then the other |

---

## Technical Context (Brownfield)

### Verified Python baseline
- `Tetra/Tetra.ipynb`: User's own TETRA implementation — 90% accuracy at FOV=12°, mag=6.5. **Trusted.**
- `notebooks/Tetra.ipynb`: Codex-written — claims 90% accuracy. **Unverified.**
- `notebooks/Pyramid_StarTracker.ipynb`: Codex-written — claims 100% accuracy at FOV=8°. **Unverified.**

### C layer (broken)
- `C/src/identify_tetra.c`: KD-tree 5-dimensional feature matching. Known issue: DB coverage sparse.
- `C/src/identify_pyramid.c`: Pair-based branching. Known issue: seed pair pruning removes correct pairs; geometric voting incomplete.
- `C/generated/`: Auto-generated C database files (pyramid_db=41MB, tetra_db=25MB, catalog_db=397KB).
- Current C test output: **unknown** — not run recently.

### Database parameters (current)
- TETRA: FOV=15°, mag≤6.5, cap=40, 315K KD-tree nodes.
- Pyramid: FOV=15°, mag≤6.5, 670K seed pairs.

### Camera model
- Simple pinhole: `fx = width / (2 * tan(fov_rad/2))`, principal point at image center.
- Real calibration: not available; FOV derived from image dimensions.

### Test infrastructure
- `batch_synthetic_compare.exe`: generates random synthetic fields, measures accuracy. Known to work.
- `render_catalog_test_image.py`: renders catalog-based test images with ground truth. Known to work.
- `demo_centroid_compare.exe`: runs full pipeline on a centroid CSV. Results unknown.

### Known hard constraint
- TETRA and Pyramid must never confirm or reject each other (see CLAUDE.md independence rule).

---

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| StarField | core domain | ra, dec, fov_deg, image_width, image_height | contains many DetectedStar |
| DetectedStar | core domain | x_pixel, y_pixel, brightness | belongs to StarField |
| ObservedStar | core domain | unit_vector, brightness | derived from DetectedStar via CameraModel |
| CatalogStar | core domain | hr_id, unit_vector_q15, magnitude | stored in CatalogDB |
| MatchResult | core domain | hr_ids[], attitude_matrix, residuals, score, success | produced by TETRA or Pyramid |
| Attitude | core domain | ra_deg, dec_deg, roll_deg, rotation_matrix_3x3 | extracted from MatchResult |
| CatalogDB | external system | catalog_db, pyramid_db, tetra_db | loaded by C identifiers |
| SkyAPI | external system | ra, dec, fov, format | queried by image fetch script |
| BenchmarkReport | supporting | accuracy_pct, mean_ms, peak_mb, per_step_ms | produced by benchmarking suite |

---

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 5 | 5 | - | - | N/A |
| 5 | 7 | 3 | 0 | 4 | 57% |
| 10 | 9 | 2 | 0 | 7 | 78% |
| 15 | 9 | 0 | 0 | 9 | 100% |

Ontology converged by round 10 — all 9 entities stable in the final 5 rounds.

---

## Interview Transcript

<details>
<summary>Full Q&A (15 rounds)</summary>

### Initial idea
**User:** Fix accuracy and efficiency — both TETRA and Pyramid were messed up by Codex. Want them to accurately process real star images (possibly from API for given RA/DEC/FOV). Target accuracy should match synthetic image accuracy. End product: correct camera attitude. Target FOV: 10-15 degrees.

### Round 0 (Topology)
**Q:** Topology confirmation — 4 components: TETRA fix, Pyramid fix, Real image API, End-to-end validation.
**A:** Change topology — add benchmarking and testing for accuracy, running time, and memory usage.
**Resolution:** Topology expanded to 5 components (then 6 after reorganization clarified).

### Round 1
**Q:** Were TETRA and Pyramid ever working correctly on any kind of image?
**A:** They were working with the Python notebook I tested (Tetra/Tetra.ipynb). Codex wrote Pyramid and optimized TETRA Python. Codex then ported to C. C tested on real image — results wrong. Codex tried to fix for hours, failed. Project state now unknown.
**Ambiguity:** ~75%

### Round 2
**Q:** What accuracy must the C port reach to be considered fixed?
**A:** ≥90% (target ≥95%) accuracy at 3-5 Hz on real images.
**Ambiguity:** ~65%

### Round 3
**Q:** How to verify accuracy on real images (ground truth problem)?
**A:** Fetch images at known RA/DEC and check output matches; catalog-rendered for debugging; manual visual sanity checks.
**Ambiguity:** ~55%

### Round 4 (Contrarian)
**Q:** What makes a real image API necessary vs catalog-rendered images?
**A:** Stellarium or planetarium-rendered images preferred.
**Ambiguity:** ~50%

### Round 5
**Q:** What does "reorganize" mean to you?
**A:** Clean file structure, remove dead Codex code, standardize build, make easy to operate without needing computer knowledge.
**Ambiguity:** ~48%

### Round 6 (Simplifier)
**Q:** Minimum "easy to operate" version?
**A:** One script that runs the full pipeline.
**Ambiguity:** ~45%

### Round 7
**Q:** Benchmarking: what steps to measure and what output format?
**A:** Per-algorithm summary table (accuracy, time, memory/DB) + per-step timing breakdown.
**Ambiguity:** ~40%

### Round 8 (Ontologist)
**Q:** What makes Stellarium necessary vs existing catalog-rendered pipeline?
**A:** Want to test on actual real camera images eventually; Stellarium is a stepping stone.
**Ambiguity:** ~38%

### Round 9
**Q:** Tolerance for "correct" attitude on real images?
**A:** Within 0.5° (30 arcmin) of the true pointing.
**Ambiguity:** ~35%

### Round 10
**Q:** Use Stellarium locally or web API?
**A:** DSS/SkyView web API — no desktop app needed.
**Ambiguity:** ~32%

### Round 11
**Q:** When you run the C code on test image, what actually happens?
**A:** Haven't run it recently — current output unknown.
**Ambiguity:** ~30%

### Round 12
**Q:** What commands should the entry-point script support?
**A:** build, test, identify <image.png>, fetch --ra X --dec Y --fov Z.
**Ambiguity:** ~27%

### Round 13
**Q:** Did Codex port from Python or write from scratch? Did C ever pass a test?
**A:** User asked for a port; actual C code is unknown quality. Only verified test: user's own Tetra.ipynb. Codex notebooks shown but not independently verified.
**Ambiguity:** ~26%

### Round 14
**Q:** Trust Codex notebooks as C reference, or rebuild from Tetra.ipynb?
**A:** Verify notebooks/ first before touching C.
**Ambiguity:** ~24%

### Round 15
**Q:** Fix TETRA and Pyramid in parallel or sequence?
**A:** Diagnose first, decide based on which is easier to fix.
**Ambiguity:** ~24% (early exit — remaining gap is execution-discovery)

</details>
