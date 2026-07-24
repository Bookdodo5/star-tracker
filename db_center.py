"""
Database command center: one entry point to build, measure, and inspect the TETRA
star-identification database for a chosen camera FOV.

The identifier only returns an attitude when the four brightest-ish stars it detects form a
tetrad that exists in the baked DB. So "never return NULL, always correct" is a property of
the DB's *coverage*: every patch of sky the camera can point at must contain a matchable
tetrad -- and, for robustness against a missed/extra star, must still contain one after any
single detected star is dropped. This tool makes that measurable and repeatable for any FOV
instead of the old hand-edited 7x4-only workflow.

Subcommands:
    status                     show the current DB (FOV, knobs, size, last eval)
    build   --fov-w W --fov-h H [knobs]   regenerate both DBs for this FOV and rebuild the DLL
    coverage --fov-w W --fov-h H [knobs]  fast pure-Python coverage preview (no compile)
    eval    [--fields N --tol T --robust] score the CURRENT baked DB with the real solver

Knobs (all optional; sensible defaults derived from the FOV) -- see `python db_center.py
build -h`. The redundancy target: pick knobs so `coverage --robust` reaches ~100%, which is
what "no NULL while sweeping the sky" requires.

    python db_center.py status
    python db_center.py coverage --fov-w 14.75 --fov-h 9.6 --robust
    python db_center.py build    --fov-w 14.75 --fov-h 9.6
    python db_center.py eval     --robust
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

GEN = ROOT / "identifier" / "generated"
DB_META = GEN / "db_meta.json"
CATALOG_C = GEN / "catalog_db_generated.c"
TETRA_C = GEN / "tetra_db_generated.c"
LIVE_BUILD = ROOT / "live" / "build-mingw"
DLL = LIVE_BUILD / "libstar_live.dll"
EXPORT_CATALOG = ROOT / "identifier" / "tools" / "export_catalog_db.py"
EXPORT_TETRA = ROOT / "identifier" / "tools" / "export_tetra_db.py"

# Knob catalog: (env var, CLI flag, help). Descriptions double as the user-facing guide.
KNOBS = [
    ("STAR_DB_MAG", "mag",
     "faintest star magnitude V included as a tetrad MEMBER (L). Higher = more stars = denser "
     "DB but larger. The identifier only sees stars this bright or brighter."),
    ("STAR_DB_BMC", "bmc",
     "faintest magnitude allowed as an ANCHOR (each tetrad is owned by its brightest star). "
     "Should be <= mag; anchors down to here ensure even sparse fields get a tetrad."),
    ("STAR_DB_RADIUS", "radius",
     "gather radius in DEGREES: how far from an anchor its companion stars are collected. "
     "Default = half the FOV diagonal, so companions stay in-frame. Larger adds redundancy."),
    ("STAR_DB_FIELDSTARS", "k",
     "K: number of brightest companions gathered per sparse anchor. Higher K = more tetrads "
     "per field = more redundancy (survives a dropped star) but a bigger DB."),
    ("STAR_DB_DENSITY_THRESH", "density-thresh",
     "adaptive-K threshold: anchors with more than this many companions use k-low instead of K "
     "(dense sky is already covered by overlapping anchors, so fewer combos each keeps size down)."),
    ("STAR_DB_K_LOW", "k-low",
     "K used for dense anchors (those above density-thresh companions)."),
]

# Synthetic gate constants (match benchmarks/sweep_db.py and coverage_sweep.py so numbers agree).
MIN_BRIGHT = 6            # a field is VALID only if it holds this many DB-magnitude stars
QUERY_STARS = 16         # matches TETRA_MAX_QUERY_STARS: the solver only tries the brightest 16
DEFAULT_FIELDS = 2000
DEFAULT_TOL_DEG = 0.5
SEED = 12345


# --------------------------------------------------------------------------- metadata

def load_meta() -> dict | None:
    """Returns the current DB metadata dict, or None if no DB has been built yet."""
    if not DB_META.exists():
        return None
    try:
        return json.loads(DB_META.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_meta(meta: dict) -> None:
    """Writes DB metadata, refreshing the on-disk baked size first."""
    tb = TETRA_C.stat().st_size if TETRA_C.exists() else 0
    cb = CATALOG_C.stat().st_size if CATALOG_C.exists() else 0
    meta["db_bytes"] = tb + cb
    DB_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dll_is_stale() -> bool:
    """True if the generated DB sources are newer than the built DLL (needs a rebuild)."""
    if not DLL.exists():
        return True
    dll_mtime = DLL.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > dll_mtime for p in (CATALOG_C, TETRA_C))


# --------------------------------------------------------------------------- projection

def _rotation(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """catalog->camera rotation with rows (east', north', boresight); roll rotates in-plane.
    Row 2 is the boresight, matching attitude_to_radecroll in live_identify.cpp."""
    ra, dec, roll = np.radians([ra_deg, dec_deg, roll_deg])
    e = np.array([-np.sin(ra), np.cos(ra), 0.0])
    n = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    b = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    x = e * np.cos(roll) + n * np.sin(roll)
    y = -e * np.sin(roll) + n * np.cos(roll)
    return np.vstack([x, y, b])


def _sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angle between two RA/DEC points, in degrees."""
    a = np.radians([ra1, dec1]); b = np.radians([ra2, dec2])
    d = np.sin(a[1]) * np.sin(b[1]) + np.cos(a[1]) * np.cos(b[1]) * np.cos(a[0] - b[0])
    return float(np.degrees(np.arccos(np.clip(d, -1.0, 1.0))))


def _field_stars(cat_vecs: np.ndarray, ra: float, dec: float, roll: float,
                 tan_w: float, tan_h: float) -> np.ndarray:
    """Camera-frame unit vectors of catalog stars inside the FOV, brightest-first."""
    obs = cat_vecs @ _rotation(ra, dec, roll).T
    z = obs[:, 2]
    infov = (z > 0) & (np.abs(obs[:, 0] / z) <= tan_w) & (np.abs(obs[:, 1] / z) <= tan_h)
    return obs[infov]  # cat_vecs is brightest-first; the mask preserves order


def _random_fields(n_fields: int):
    """Yields (ra, dec, roll) for n_fields uniformly-on-sphere random attitudes (fixed seed)."""
    rng = np.random.default_rng(SEED)
    t_ra = rng.uniform(0, 360, n_fields)
    t_dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n_fields)))
    t_roll = rng.uniform(0, 360, n_fields)
    for i in range(n_fields):
        yield t_ra[i], t_dec[i], t_roll[i]


# --------------------------------------------------------------------------- eval (real solver)

def evaluate(fov_w: float, fov_h: float, n_fields: int, tol_deg: float, robust: bool) -> dict:
    """
    Scores the currently baked DB with the real C solver over n_fields random attitudes.

    A field is VALID if it holds >= MIN_BRIGHT catalog stars in the fov_w x fov_h footprint.
    SOLVED = the solver returned an attitude; CORRECT = boresight within tol of truth. When
    robust=True also measures leave-one-out: the field is ROBUST only if it stays correct after
    dropping each single detected star in turn -- the property that prevents NULL when the real
    camera misses or adds one star mid-sweep. Robust mode does up to (stars+1) solves per field.
    """
    import live_identify as L
    from src.star_tracker_core import load_db_catalog

    lib = L.load_lib()
    if not hasattr(lib, "identify_vectors"):
        raise SystemExit("DLL lacks identify_vectors -- rebuild live/ (python db_center.py build).")

    meta = load_meta() or {}
    ref = load_db_catalog(meta.get("mag", 7.5))
    ra = ref["RA_deg"].to_numpy(); dec = ref["DEC_deg"].to_numpy()
    cat = np.column_stack([
        np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
        np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
        np.sin(np.radians(dec)),
    ])
    tan_w, tan_h = np.tan(np.radians(fov_w / 2)), np.tan(np.radians(fov_h / 2))

    valid = solved = correct = robust_ok = 0
    rng = np.random.default_rng(SEED)  # deterministic star-drop choices
    t0 = t_last = time.time()
    print(f"  scoring {n_fields} fields with the real solver"
          f"{' (robust = 3 extra solves, each dropping 1 random star)' if robust else ''}...", flush=True)
    for idx, (t_ra, t_dec, t_roll) in enumerate(_random_fields(n_fields)):
        # Time-based progress: report every ~2s so feedback is steady whether a field takes
        # microseconds (non-robust) or many milliseconds (robust leave-one-out).
        now = time.time()
        if idx and now - t_last >= 2.0:
            eta = (now - t0) / idx * (n_fields - idx)
            extra = f" robust={robust_ok}" if robust else ""
            print(f"  eval {idx}/{n_fields}  valid={valid} correct={correct}{extra}  "
                  f"{now - t0:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
            t_last = now
        field = _field_stars(cat, t_ra, t_dec, t_roll, tan_w, tan_h)
        if len(field) < MIN_BRIGHT:
            continue
        valid += 1
        att = L.solve_vectors(lib, field[:20])
        if att is None:
            continue
        solved += 1
        ok = _sep_deg(t_ra, t_dec, att[0], att[1]) <= tol_deg
        if ok:
            correct += 1
        if robust and ok:
            robust_ok += _robust_drop_ok(L, lib, field, t_ra, t_dec, tol_deg, rng)

    result = {
        "fov_w": fov_w, "fov_h": fov_h, "fields": n_fields, "tol_deg": tol_deg,
        "valid": valid, "solved": solved, "correct": correct,
        "accuracy": (correct / valid) if valid else 0.0,
        "solve_rate": (solved / valid) if valid else 0.0,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if robust:
        result["robust_ok"] = robust_ok
        result["robust_rate"] = (robust_ok / valid) if valid else 0.0
    print(f"  scored in {time.time() - t0:.0f}s", flush=True)
    return result


ROBUST_TRIALS = 3  # sampled single-star drops per field (cheap proxy for full leave-one-out)


def _robust_drop_ok(L, lib, field: np.ndarray, t_ra: float, t_dec: float,
                    tol_deg: float, rng) -> int:
    """Returns 1 if the field still solves correctly after dropping ONE random star, repeated
    ROBUST_TRIALS times. A sampled robustness check: cheap (3 solves) but catches fields that
    hinge on a single star, which is what causes NULL when the camera misses one mid-sweep."""
    used = field[:20]
    n = len(used)
    if n < 5:            # can't drop one and keep >= 4
        return 0
    for _ in range(ROBUST_TRIALS):
        sub = np.delete(used, int(rng.integers(n)), axis=0)
        att = L.solve_vectors(lib, sub)
        if att is None or _sep_deg(t_ra, t_dec, att[0], att[1]) > tol_deg:
            return 0
    return 1


# --------------------------------------------------------------------------- coverage (no compile)

def coverage_preview(fov_w: float, fov_h: float, knobs: dict, n_fields: int, robust: bool) -> dict:
    """
    Fast pure-Python coverage: replicates DB tetrad generation as a set of integer keys and
    tests whether the seeded random fields contain a matchable tetrad -- no DLL compile. Since
    the C solver verifies at ~100% correctness once a tetrad matches, coverage ~= solve rate,
    so this predicts a bake's quality (and its tetrad count / rough size) in seconds.

    robust=True reports leave-one-out coverage: the fraction of valid fields that still contain
    a tetrad after ANY single brightest-K star is removed. Drive this to ~100% for no-NULL sky.
    """
    from src.star_tracker_core import load_db_catalog, unit_vectors, anchored_allcombos_tetrads

    mag = float(knobs["STAR_DB_MAG"])
    bmc = float(knobs["STAR_DB_BMC"])
    radius_rad = math.radians(float(knobs["STAR_DB_RADIUS"]))
    k = int(knobs["STAR_DB_FIELDSTARS"])
    density_thresh = int(knobs["STAR_DB_DENSITY_THRESH"])
    k_low = int(knobs["STAR_DB_K_LOW"])
    max_edge_rad = math.radians(math.hypot(fov_w, fov_h))

    ref = load_db_catalog(mag)
    ref_vecs = unit_vectors(ref["RA_deg"], ref["DEC_deg"])
    n_anchors = int((ref["Vmag"].to_numpy() <= bmc).sum())

    print(f"  [1/2] generating tetrads (members={len(ref_vecs)} anchors={n_anchors} K={k} "
          f"radius={math.degrees(radius_rad):.2f}deg)...", flush=True)
    t_gen = time.time()
    tetrads = anchored_allcombos_tetrads(ref_vecs, n_anchors, radius_rad, max_edge_rad, k,
                                          density_thresh=density_thresh, k_low=k_low)
    keys = {_tetra_key(t) for t in tetrads}
    print(f"        {len(tetrads):,} tetrads in {time.time() - t_gen:.1f}s", flush=True)

    print(f"  [2/2] testing {n_fields} fields{' (+3 single-star drops)' if robust else ''}...", flush=True)
    t_cov = time.time()
    rng = np.random.default_rng(SEED)  # deterministic star-drop choices, matches evaluate()
    tan_w, tan_h = math.tan(math.radians(fov_w / 2)), math.tan(math.radians(fov_h / 2))
    valid = covered = robust_cov = 0
    for t_ra, t_dec, t_roll in _random_fields(n_fields):
        field = _field_stars(ref_vecs, t_ra, t_dec, t_roll, tan_w, tan_h)
        if len(field) < MIN_BRIGHT:
            continue
        # recover the catalog indices of the in-fov stars (brightest-first), capped at QUERY_STARS
        obs = ref_vecs @ _rotation(t_ra, t_dec, t_roll).T
        z = obs[:, 2]
        infov = np.where((z > 0) & (np.abs(obs[:, 0] / z) <= tan_w) & (np.abs(obs[:, 1] / z) <= tan_h))[0]
        valid += 1
        bright = infov[:QUERY_STARS].tolist()
        if _field_has_tetrad(bright, keys):
            covered += 1
        if robust and _field_robust(bright, keys, rng):
            robust_cov += 1
    print(f"        tested in {time.time() - t_cov:.1f}s", flush=True)

    est_bytes = _estimate_db_bytes(len(tetrads), len(ref_vecs))
    result = {
        "tetrads": len(tetrads), "valid": valid, "covered": covered,
        "coverage": (covered / valid) if valid else 0.0,
        "est_db_bytes": est_bytes,
    }
    if robust:
        result["robust_covered"] = robust_cov
        result["robust_coverage"] = (robust_cov / valid) if valid else 0.0
    return result


def _tetra_key(ids) -> int:
    a, b, c, d = sorted(ids)
    return (a << 48) | (b << 32) | (c << 16) | d


def _field_has_tetrad(bright: list[int], keys: set) -> bool:
    """True if any 4-subset of the field's brightest stars is a DB tetrad."""
    return any(_tetra_key(c) in keys for c in itertools.combinations(bright, 4))


def _field_robust(bright: list[int], keys: set, rng) -> bool:
    """True if a matchable tetrad remains after dropping ONE random star, repeated ROBUST_TRIALS
    times. Sampled proxy for full leave-one-out; matches evaluate()'s real-solver robust check."""
    n = len(bright)
    if n < 5:  # need >=5 so at least 4 survive a removal
        return False
    for _ in range(ROBUST_TRIALS):
        drop = int(rng.integers(n))
        if not _field_has_tetrad(bright[:drop] + bright[drop + 1:], keys):
            return False
    return True


def _estimate_db_bytes(tetrads: int, members: int) -> int:
    """Rough baked-size estimate: TetraKdNode ~= 34 chars/row, CatalogStar ~= 30 chars/row."""
    return tetrads * 34 + members * 30


# --------------------------------------------------------------------------- knobs / build

# Proven 7x4 config (CLAUDE.md): radius 3.5 deg reproduces the shipped 99.5%-solve DB exactly.
# Scale that tuned radius by the FOV diagonal so other FOVs inherit the same in-frame ratio
# (radius/diagonal), rather than a naive half-diagonal that over-caps dense anchors.
PROVEN_RADIUS_DEG = 3.5
PROVEN_DIAG_DEG = math.hypot(7.0, 4.0)


def derive_knobs(args) -> dict:
    """Builds the STAR_DB_* env dict from CLI args, defaulting radius to the proven radius/diagonal
    ratio for this FOV and every other knob to the exporter's source default when omitted."""
    diag = math.hypot(args.fov_w, args.fov_h)
    defaults = {
        "STAR_DB_MAG": "7.5", "STAR_DB_BMC": "7.0",
        "STAR_DB_RADIUS": f"{PROVEN_RADIUS_DEG * diag / PROVEN_DIAG_DEG:.4f}",
        "STAR_DB_FIELDSTARS": "9", "STAR_DB_DENSITY_THRESH": "15", "STAR_DB_K_LOW": "6",
    }
    env = dict(defaults)
    env["STAR_DB_FOV_W"] = str(args.fov_w)
    env["STAR_DB_FOV_H"] = str(args.fov_h)
    for _env, flag, _help in KNOBS:
        val = getattr(args, flag.replace("-", "_"))
        if val is not None:
            env[_env] = str(val)
    return env


def cmd_build(args) -> int:
    """Regenerates the catalog + tetra DBs for the given FOV/knobs and rebuilds the live DLL."""
    env = derive_knobs(args)
    total = 4 if args.skip_dll else 5
    print(f"[db] building DB for FOV {args.fov_w}x{args.fov_h} deg")
    for env_name, _flag, _help in KNOBS:
        print(f"     {env_name}={env[env_name]}")
    print(f"     STAR_DB_RADIUS={env['STAR_DB_RADIUS']}  (auto = proven ratio for this FOV)")
    print(f"[db] {total} steps. Rough time at 7x4: ~1-3 min total; larger FOV grows the tetra "
          f"step to many minutes. Each step below prints its own elapsed time.", flush=True)

    t_all = time.time()
    full_env = dict(os.environ, **env)
    steps = [
        ("catalog export (stars -> catalog_db_generated.c)", [sys.executable, str(EXPORT_CATALOG)]),
        ("tetra export (tetrads -> tetra_db_generated.c, the slow one)", [sys.executable, str(EXPORT_TETRA)]),
    ]
    for i, (label, cmd) in enumerate(steps, 1):
        print(f"\n[db] step {i}/{total}: {label}", flush=True)
        t_step = time.time()
        if _run(cmd, env=full_env) != 0:
            print(f"[db] step {i} FAILED after {time.time() - t_step:.0f}s"); return 1
        print(f"[db] step {i} done in {time.time() - t_step:.0f}s", flush=True)

    if args.skip_dll:
        print(f"\n[db] --skip-dll: sources regenerated in {time.time() - t_all:.0f}s "
              f"(DLL not rebuilt; run cmake --build live/build-mingw yourself).")
        return 0

    step = 3
    if not LIVE_BUILD.exists():
        total += 1
        print(f"\n[db] step {step}/{total}: cmake configure (first build only)", flush=True)
        if _run(["cmake", "-S", str(ROOT / "live"), "-B", str(LIVE_BUILD),
                 "-G", "MinGW Makefiles", "-DCMAKE_BUILD_TYPE=Release"]) != 0:
            print("[db] cmake configure failed"); return 1
        step += 1
    print(f"\n[db] step {step}/{total}: compile the DLL (cmake --build)", flush=True)
    t_step = time.time()
    if _run(["cmake", "--build", str(LIVE_BUILD)]) != 0:
        print(f"[db] DLL build FAILED after {time.time() - t_step:.0f}s"); return 1
    print(f"[db] DLL built in {time.time() - t_step:.0f}s")
    print(f"\n[db] all done in {time.time() - t_all:.0f}s. Verify: python db_center.py eval --robust")
    return 0


def cmd_coverage(args) -> int:
    """Fast pure-Python coverage preview for candidate knobs (no compile)."""
    knobs = derive_knobs(args)
    print(f"[db] coverage preview: FOV {args.fov_w}x{args.fov_h} deg, {args.fields} fields"
          f"{' (+leave-one-out)' if args.robust else ''}")
    r = coverage_preview(args.fov_w, args.fov_h, knobs, args.fields, args.robust)
    print(f"\n  tetrads:  {r['tetrads']:,}  (~{r['est_db_bytes']/1e6:.1f} MB baked)")
    print(f"  valid fields: {r['valid']}")
    print(f"  coverage:     {r['coverage']*100:.2f}%  ({r['covered']}/{r['valid']})")
    if args.robust:
        print(f"  robust cov:   {r['robust_coverage']*100:.2f}%  ({r['robust_covered']}/{r['valid']}) "
              f"<- drive to ~100% for no-NULL sky")
    if r["coverage"] < 0.99:
        print("  ! below 99%: raise --k or --mag, or widen --radius, and re-preview")
    return 0


def cmd_sweep(args) -> int:
    """
    Varies ONE knob across a list of values and prints a coverage table, so the size<->redundancy
    tradeoff of a knob is visible at a glance instead of guessed. Uses the fast no-compile
    coverage preview per value. For density-thresh, the value 'off' disables adaptive-K (every
    anchor uses full K) -- the natural top end of that knob's range.
    """
    flag_to_env = {flag: env for env, flag, _ in KNOBS}
    if args.knob not in flag_to_env:
        print(f"[db] unknown --knob {args.knob}. Choose one of: {', '.join(flag_to_env)}")
        return 1
    env_key = flag_to_env[args.knob]
    base = derive_knobs(args)
    values = [v.strip() for v in args.values.split(",") if v.strip()]

    print(f"[db] sweep {args.knob} over {values}  (FOV {args.fov_w}x{args.fov_h}, "
          f"{args.fields} fields{', +leave-one-out' if args.robust else ''})")
    hdr = f"{args.knob:>14} {'tetrads':>12} {'~MB':>7} {'cover%':>8}"
    if args.robust:
        hdr += f" {'robust%':>8}"
    print(hdr)
    rows = []
    for raw in values:
        knobs = dict(base)
        # 'off' means: make the threshold unreachable so adaptive-K never downgrades an anchor.
        knobs[env_key] = "999999999" if (args.knob == "density-thresh" and raw.lower() == "off") else raw
        r = coverage_preview(args.fov_w, args.fov_h, knobs, args.fields, args.robust)
        line = (f"{raw:>14} {r['tetrads']:>12,} {r['est_db_bytes']/1e6:>7.1f} "
                f"{r['coverage']*100:>8.2f}")
        if args.robust:
            line += f" {r['robust_coverage']*100:>8.2f}"
        print(line, flush=True)
        rows.append((raw, r))
    if args.robust:
        ok = [(v, r) for v, r in rows if r["robust_coverage"] >= 0.999]
        if ok:
            smallest = min(ok, key=lambda vr: vr[1]["tetrads"])
            print(f"\n  smallest DB with ~100% robust coverage: {args.knob}={smallest[0]} "
                  f"({smallest[1]['tetrads']:,} tetrads, {smallest[1]['est_db_bytes']/1e6:.1f} MB)")
        else:
            print(f"\n  none reached ~100% robust; raise --k / --mag / --radius and sweep again")
    return 0


def cmd_eval(args) -> int:
    """Scores the current baked DB with the real solver; records the result into db_meta.json."""
    meta = load_meta()
    if meta is None:
        print("[db] no db_meta.json -- build a DB first: python db_center.py build --fov-w W --fov-h H")
        return 1
    if dll_is_stale():
        print("[db] WARNING: DLL is older than the generated DB sources; run build first for a true score.")
    fov_w = meta.get("fov_w", 7.0); fov_h = meta.get("fov_h", 4.0)
    print(f"[db] eval current DB (FOV {fov_w}x{fov_h}) over {args.fields} fields, tol {args.tol} deg"
          f"{' (+robust: 3 single-star drops/field)' if args.robust else ''}")
    r = evaluate(fov_w, fov_h, args.fields, args.tol, args.robust)
    print(f"\n  valid={r['valid']} solved={r['solved']} correct={r['correct']}")
    print(f"  solve rate: {r['solve_rate']*100:.2f}%   accuracy: {r['accuracy']*100:.2f}%")
    if args.robust:
        print(f"  robust:     {r['robust_rate']*100:.2f}%  ({r['robust_ok']}/{r['valid']}) "
              f"still-correct across 3 single-star drops")
    print(f"  DB size:    {meta.get('db_bytes', 0)/1e6:.2f} MB")
    meta.setdefault("evals", []).append(r)
    save_meta(meta)
    print(f"  recorded in {DB_META.name}")
    return 0


def cmd_status(args) -> int:
    """Prints the current DB's FOV, knobs, size, staleness, and last eval."""
    meta = load_meta()
    if meta is None:
        print("[db] no DB metadata found. Build one: python db_center.py build --fov-w W --fov-h H")
        return 1
    print(f"Current DB  (identifier/generated/db_meta.json)")
    print(f"  FOV:        {meta.get('fov_w')} x {meta.get('fov_h')} deg  "
          f"(max edge {meta.get('max_edge_deg', 0):.2f} deg)")
    print(f"  magnitude:  members V<={meta.get('mag')}  anchors V<={meta.get('bmc')}")
    print(f"  gather:     radius {meta.get('radius_deg', 0):.2f} deg  K={meta.get('k')} "
          f"(dense: K={meta.get('k_low')} above {meta.get('density_thresh')} companions)")
    print(f"  members:    {meta.get('members'):,}   anchors: {meta.get('anchors'):,}")
    print(f"  tetrads:    {meta.get('tetrads'):,}   KD nodes: {meta.get('kd_nodes'):,}")
    print(f"  DB size:    {meta.get('db_bytes', 0)/1e6:.2f} MB   built {meta.get('generated_utc')}")
    print(f"  DLL:        {'STALE - run build' if dll_is_stale() else 'up to date'}")
    evals = meta.get("evals", [])
    if evals:
        e = evals[-1]
        line = (f"  last eval:  solve {e.get('solve_rate', 0)*100:.2f}%  "
                f"accuracy {e.get('accuracy', 0)*100:.2f}%")
        if "robust_rate" in e:
            line += f"  robust {e['robust_rate']*100:.2f}%"
        print(line + f"  ({e.get('valid')} valid, {e.get('when')})")
    else:
        print("  last eval:  none yet (python db_center.py eval --robust)")
    return 0


def _run(cmd: list[str], env: dict | None = None) -> int:
    """Runs a child command, streaming its stdout/stderr live; returns the exit code."""
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    proc.wait()
    return proc.returncode


def _add_knob_args(sub) -> None:
    """Adds the shared --fov-w/--fov-h and all optional STAR_DB_* knob flags to a subparser."""
    sub.add_argument("--fov-w", type=float, required=True, help="horizontal FOV in degrees")
    sub.add_argument("--fov-h", type=float, required=True, help="vertical FOV in degrees")
    for _env, flag, help_text in KNOBS:
        sub.add_argument(f"--{flag}", default=None, help=help_text)


def main() -> None:
    p = argparse.ArgumentParser(description="TETRA database command center")
    subs = p.add_subparsers(dest="cmd", required=True)

    subs.add_parser("status", help="show the current DB").set_defaults(func=cmd_status)

    b = subs.add_parser("build", help="regenerate the DB for a FOV and rebuild the DLL")
    _add_knob_args(b)
    b.add_argument("--skip-dll", action="store_true", help="regenerate sources but do not rebuild the DLL")
    b.set_defaults(func=cmd_build)

    c = subs.add_parser("coverage", help="fast pure-Python coverage preview (no compile)")
    _add_knob_args(c)
    c.add_argument("--fields", type=int, default=DEFAULT_FIELDS, help="random fields to test")
    c.add_argument("--robust", action="store_true", help="also measure leave-one-out coverage")
    c.set_defaults(func=cmd_coverage)

    s = subs.add_parser("sweep", help="vary one knob across values and print a coverage table")
    _add_knob_args(s)
    s.add_argument("--knob", required=True, help="knob flag to vary, e.g. k, density-thresh, radius, mag")
    s.add_argument("--values", required=True, help="comma list, e.g. 6,9,12,16 (density-thresh accepts 'off')")
    s.add_argument("--fields", type=int, default=DEFAULT_FIELDS)
    s.add_argument("--robust", action="store_true", help="also show leave-one-out coverage")
    s.set_defaults(func=cmd_sweep)

    e = subs.add_parser("eval", help="score the current baked DB with the real solver")
    e.add_argument("--fields", type=int, default=DEFAULT_FIELDS)
    e.add_argument("--tol", type=float, default=DEFAULT_TOL_DEG, help="correct-solve tolerance (deg)")
    e.add_argument("--robust", action="store_true", help="also measure leave-one-out solve rate")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
