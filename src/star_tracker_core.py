from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CATALOG_PATHS = (
    DATA_DIR / "catalog.bin",
)

TYCHO2_PATHS = (
    DATA_DIR / "tycho2.csv",
    DATA_DIR / "tycho2.parquet",
)

DEFAULT_DB_MAG_LIMIT = 6.5
DEFAULT_MAX_FOV_DEG = 20.0
DEFAULT_SEED = 42


@dataclass(frozen=True)
class BatchConfig:
    """
    Defines one repeatable batch test.

    Args:
        samples: Number of synthetic fields to test.
        fov_deg: Field-of-view diameter in degrees.
        mag_limit: Faintest visual magnitude included in each field.
        max_stars_query: Brightest observed stars used by the matcher.
        seed: Random seed used to sample field centers.
    """

    samples: int = 80
    fov_deg: float = 10.0
    mag_limit: float | None = 6.5
    max_stars_query: int = 12
    seed: int = DEFAULT_SEED


def ensure_dirs() -> None:
    """
    Creates project data, cache, and output folders if missing.
    """

    for path in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_catalog_path(path: str | Path | None = None) -> Path:
    """
    Finds the Yale Bright Star catalog binary used by the notebooks.

    Args:
        path: Optional explicit `catalog.bin` path.

    Returns:
        Existing catalog path.

    Raises:
        FileNotFoundError: If no usable catalog file exists.
    """

    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(candidate)
    for candidate in CATALOG_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find catalog.bin in data/ or Tetra/.")


def parse_ra(value: object) -> float:
    """
    Converts catalog right ascension text to degrees.

    Args:
        value: Catalog RA field formatted as HHMMSS-like text.

    Returns:
        Right ascension in degrees, or NaN for invalid input.
    """

    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if len(text) < 6:
        return np.nan
    try:
        hours = float(text[0:2])
        minutes = float(text[2:4])
        seconds = float(text[4:])
    except ValueError:
        return np.nan
    return (hours + minutes / 60.0 + seconds / 3600.0) * 15.0


def parse_dec(value: object) -> float:
    """
    Converts catalog declination text to degrees.

    Args:
        value: Catalog DEC field formatted as signed DDMMSS-like text.

    Returns:
        Declination in degrees, or NaN for invalid input.
    """

    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if len(text) < 6:
        return np.nan
    try:
        sign = -1.0 if text[0] == "-" else 1.0
        if text[0] in "+-":
            degrees = float(text[1:3])
            minutes = float(text[3:5])
            seconds = float(text[5:])
        else:
            degrees = float(text[0:2])
            minutes = float(text[2:4])
            seconds = float(text[4:])
    except ValueError:
        return np.nan
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def load_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """
    Loads `catalog.bin` into a clean star catalog table.

    Args:
        path: Optional explicit catalog path.

    Returns:
        DataFrame sorted by brightness with HR ID, RA, DEC, Vmag, and marker size.
    """

    catalog_path = resolve_catalog_path(path)
    column_specs = [(0, 4), (75, 83), (83, 90), (102, 107)]
    column_names = ["HR", "RA_str", "DEC_str", "Vmag_str"]
    raw = pd.read_fwf(catalog_path, colspecs=column_specs, names=column_names, header=None, dtype=str)

    df = raw.copy()
    df["RA_deg"] = df["RA_str"].apply(parse_ra)
    df["DEC_deg"] = df["DEC_str"].apply(parse_dec)
    df["Vmag"] = pd.to_numeric(df["Vmag_str"], errors="coerce")
    df["HR_clean"] = pd.to_numeric(df["HR"], errors="coerce")
    df = df.dropna(subset=["RA_deg", "DEC_deg", "Vmag", "HR_clean"]).copy()
    df["HR_clean"] = df["HR_clean"].astype(int)
    df = df.drop_duplicates("HR_clean").sort_values("Vmag").reset_index(drop=True)
    df["marker_size"] = np.clip((8.0 - df["Vmag"]) ** 2.0 * 0.6, 4.0, 120.0)
    return df[["HR_clean", "RA_deg", "DEC_deg", "Vmag", "marker_size"]]


def load_db_catalog(mag_limit: float | None = None) -> pd.DataFrame:
    """
    Loads the Tycho-2 catalog subset used to build the runtime DBs.

    Far denser than the Yale BSC (load_catalog), so small fields hold enough stars.
    HR_clean is reindexed 0..N-1 after the magnitude cut so the baked uint16 ids stay
    contiguous and below HR_INVALID (65535).

    Args:
        mag_limit: Faintest Johnson V to include (None = all in the cached file).

    Returns:
        DataFrame sorted by brightness: HR_clean(0..N-1), RA_deg, DEC_deg, Vmag, marker_size.
    """
    for candidate in TYCHO2_PATHS:
        if candidate.exists():
            path = candidate
            break
    else:
        raise FileNotFoundError(
            "Tycho-2 cache not found. Run: python benchmarks/fetch_tycho2.py --vmax 8.0 --out data/tycho2.csv"
        )

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if mag_limit is not None:
        df = df[df["Vmag"] <= mag_limit]
    df = df.sort_values("Vmag").reset_index(drop=True)
    if len(df) >= 0xFFFF:
        raise ValueError(
            f"{len(df)} stars at mag_limit={mag_limit} exceeds uint16 id space (<65535). "
            "Lower the magnitude limit or widen the baked id type to uint32."
        )
    df["HR_clean"] = np.arange(len(df), dtype=int)  # reindex 0..N-1 after the cut
    df["marker_size"] = np.clip((8.0 - df["Vmag"]) ** 2.0 * 0.6, 4.0, 120.0)
    return df[["HR_clean", "RA_deg", "DEC_deg", "Vmag", "marker_size"]]


def _fits_fov(pts: np.ndarray, fov_w_deg: float, fov_h_deg: float) -> bool:
    """
    True if the given unit vectors (a small star group) fit inside a fov_w x fov_h
    rectangle at SOME rotation. Projects to the tangent plane at the group's mean
    direction, then sweeps the rectangle orientation checking axis extents. Exact enough
    for <~8 deg groups and cheap for <=4 points.
    """
    center = pts.mean(axis=0)
    center = center / np.linalg.norm(center)
    ref = np.array([0.0, 0.0, 1.0]) if abs(center[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    east = np.cross(ref, center); east /= np.linalg.norm(east)
    north = np.cross(center, east)
    deg = 57.29577951308232
    px = (pts @ east) * deg
    py = (pts @ north) * deg
    # Vectorized over all rectangle orientations at once: extents per angle, then test fit.
    thetas = np.arange(0.0, math.pi / 2 + 1e-9, math.radians(3.0))
    c = np.cos(thetas)[:, None]; s = np.sin(thetas)[:, None]
    x = px[None, :] * c + py[None, :] * s        # (T, n)
    y = -px[None, :] * s + py[None, :] * c
    w = x.max(axis=1) - x.min(axis=1)            # (T,)
    h = y.max(axis=1) - y.min(axis=1)
    fits = ((w <= fov_w_deg) & (h <= fov_h_deg)) | ((w <= fov_h_deg) & (h <= fov_w_deg))
    return bool(fits.any())


def greedy_feasible_tetrads(
    vecs: np.ndarray,
    n_anchors: int,
    gather_rad: float,
    fov_w_deg: float,
    fov_h_deg: float,
    c1: int,
    c2: int,
    c3: int,
) -> list[tuple[int, int, int, int]]:
    """
    Greedy frame-feasible tetrad generation (anchor = brightest star).

    For each anchor a, walking dimmer stars brightest-first: keep the brightest c1 that fit a
    frame with a; for each, the brightest c2 (fainter) that fit a frame with {a,s2}; for each,
    the brightest c3 that fit with {a,s2,s3}. Every emitted tetra is therefore a real possible
    image (4 stars that fit one fov_w x fov_h frame), so no out-of-frame slots are wasted.
    Size is bounded by anchors * c1 * c2 * c3. Indices increase (a<s2<s3<s4) so each is unique.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(vecs)
    chord_gather = 2.0 * math.sin(gather_rad / 2.0)
    out: list[tuple[int, int, int, int]] = []
    progress_start = time.time()

    for a in range(n_anchors):
        if (a + 1) % 1000 == 0 or a == n_anchors - 1:
            elapsed = time.time() - progress_start
            frac = (a + 1) / n_anchors
            eta = elapsed / frac * (1.0 - frac) if frac > 0 else 0.0
            print(f"    greedy {a + 1}/{n_anchors} anchors | {len(out)} tetrads"
                  f" | elapsed {elapsed:.0f}s | eta {eta:.0f}s", flush=True)
        nb = sorted(j for j in tree.query_ball_point(vecs[a], chord_gather) if j > a)
        if len(nb) < 3:
            continue
        # Any two stars within the gather diagonal already fit one frame, so the brightest
        # c1 dimmer neighbours are the 2nd-star candidates directly (no fit test needed here).
        seconds = nb[:c1]
        for s2 in seconds:
            thirds = []
            for s3 in nb:
                if s3 <= s2:
                    continue
                if _fits_fov(vecs[[a, s2, s3]], fov_w_deg, fov_h_deg):
                    thirds.append(s3)
                    if len(thirds) == c2:
                        break
            for s3 in thirds:
                found = 0
                for s4 in nb:
                    if s4 <= s3:
                        continue
                    if _fits_fov(vecs[[a, s2, s3, s4]], fov_w_deg, fov_h_deg):
                        out.append((a, s2, s3, s4))
                        found += 1
                        if found == c3:
                            break
    return out


def anchored_allcombos_tetrads(
    vecs: np.ndarray,
    n_anchors: int,
    gather_rad: float,
    max_edge_rad: float,
    max_neighbours: int,
    density_thresh: int | None = None,
    k_low: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Anchored all-combinations generation: each tetra owned by its brightest star.

    For each anchor a, take the brightest `max_neighbours` dimmer stars within gather_rad and
    emit EVERY (a + 3 of them) whose max pairwise edge <= max_edge_rad. Small gather_rad (~the
    FOV half-diagonal) keeps the neighbour pool in-frame, so the combos are real frames; using
    every star as an anchor covers edge-anchor fields via a more central anchor's compact subset.
    Anchor is always the lowest index, so each 4-set is unique (no dedup needed).

    Adaptive-K: if density_thresh and k_low are set, anchors with more than density_thresh
    dimmer neighbours use k_low instead of max_neighbours. Dense sky regions are already covered
    by many overlapping anchors, so fewer combos per anchor are needed there.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(vecs)
    chord_gather = 2.0 * math.sin(gather_rad / 2.0)
    max_chord = 2.0 * math.sin(max_edge_rad / 2.0)
    out: list[tuple[int, int, int, int]] = []
    # Cache the local trio-index arrays per neighbour-count so combinations() runs once per size,
    # not once per anchor (there are only a handful of distinct sizes).
    trio_index_cache: dict[int, np.ndarray] = {}

    for a in range(n_anchors):
        all_nb = sorted(j for j in tree.query_ball_point(vecs[a], chord_gather) if j > a)
        k = (k_low if (density_thresh is not None and k_low is not None
                       and len(all_nb) > density_thresh) else max_neighbours)
        nb = all_nb[:k]
        m = len(nb)
        if m < 3:
            continue
        # Vectorized max-edge test: precompute the anchor->neighbour and neighbour->neighbour chord
        # distances once, then evaluate every (anchor + trio) at once. Same combos, same order as
        # itertools.combinations(nb, 3), so the emitted tetrad list is byte-identical to the scalar
        # version -- just far fewer Python-level numpy calls.
        pts = vecs[nb]                                      # (m, 3)
        anchor_dist = np.linalg.norm(pts - vecs[a], axis=1)  # (m,) anchor->neighbour
        nb_dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)  # (m, m) neighbour->neighbour
        trios = trio_index_cache.get(m)
        if trios is None:
            trios = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(m), 3)),
                                dtype=np.intp).reshape(-1, 3)
            trio_index_cache[m] = trios
        i, j, l = trios[:, 0], trios[:, 1], trios[:, 2]
        max_edge = np.maximum.reduce([
            anchor_dist[i], anchor_dist[j], anchor_dist[l],
            nb_dist[i, j], nb_dist[i, l], nb_dist[j, l],
        ])                                                  # (T,) largest of the 6 pairwise edges
        for t in np.nonzero(max_edge <= max_chord)[0]:      # ascending -> same order as before
            trio = trios[t]
            out.append((a, nb[trio[0]], nb[trio[1]], nb[trio[2]]))
    return out


def anchored_tetrads(
    vecs: np.ndarray,
    n_anchors: int,
    gather_rad: float,
    max_edge_rad: float,
    n_sectors: int,
    per_sector: int,
) -> list[tuple[int, int, int, int]]:
    """
    Anchored, direction-covering tetrad generation (shared by the exporter and the fast sweep).

    Each tetra is owned by its brightest star (the anchor). For every anchor a < n_anchors:
      - gather members strictly dimmer (index > a) within gather_rad (= FOV diagonal, so an
        edge anchor still reaches companions on the opposite side of the frame);
      - bin those companions by direction in the anchor's tangent plane into n_sectors angular
        sectors and keep the brightest `per_sector` in each — this guarantees coverage in every
        direction instead of letting globally-bright out-of-frame stars dominate;
      - emit every (anchor + 3 of the kept companions) whose max pairwise edge <= max_edge_rad
        (the 4 stars fit one frame). Anchor is always the lowest index, so each 4-set is unique.

    vecs must be unit vectors sorted brightest-first (index = brightness rank).
    Returns tetrads as 4-tuples of member indices.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(vecs)
    chord_gather = 2.0 * math.sin(gather_rad / 2.0)
    max_chord = 2.0 * math.sin(max_edge_rad / 2.0)
    out: list[tuple[int, int, int, int]] = []

    for a in range(n_anchors):
        anchor = vecs[a]
        neighbours = [j for j in tree.query_ball_point(anchor, chord_gather) if j > a]
        if len(neighbours) < 3:
            continue
        # Tangent-plane basis at the anchor for directional binning.
        ref = np.array([0.0, 0.0, 1.0]) if abs(anchor[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        east = np.cross(ref, anchor); east /= np.linalg.norm(east)
        north = np.cross(anchor, east)

        per_bin: dict[int, list[int]] = {}
        for j in neighbours:  # neighbours ascending index == brightest-first
            d = vecs[j] - anchor
            sector = int((math.atan2(float(north @ d), float(east @ d)) + math.pi) / (2 * math.pi) * n_sectors)
            sector = min(sector, n_sectors - 1)
            bucket = per_bin.setdefault(sector, [])
            if len(bucket) < per_sector:
                bucket.append(j)

        candidates = sorted(j for bucket in per_bin.values() for j in bucket)
        if len(candidates) < 3:
            continue
        for trio in itertools.combinations(candidates, 3):
            combo = (a, *trio)
            pts = vecs[list(combo)]
            ok = True
            for u in range(4):
                for v in range(u + 1, 4):
                    if np.linalg.norm(pts[u] - pts[v]) > max_chord:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append(combo)
    return out


def unit_vectors(ra_deg: np.ndarray | pd.Series, dec_deg: np.ndarray | pd.Series) -> np.ndarray:
    """
    Converts RA/DEC coordinates to 3D unit vectors.

    Args:
        ra_deg: Right ascension values in degrees.
        dec_deg: Declination values in degrees.

    Returns:
        Array with shape `(n, 3)`.
    """

    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes all pairwise angular distances between two vector sets.

    Args:
        a: First vector array with shape `(m, 3)`.
        b: Second vector array with shape `(n, 3)`.

    Returns:
        Distance matrix with shape `(m, n)` in radians.
    """

    return np.arccos(np.clip(np.asarray(a) @ np.asarray(b).T, -1.0, 1.0))


def angular_distance_pair(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the angular distance between two unit vectors.

    Args:
        a: First unit vector.
        b: Second unit vector.

    Returns:
        Angular distance in radians.
    """

    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def filter_stars_by_fov(
    catalog: pd.DataFrame,
    center_ra: float,
    center_dec: float,
    fov_deg: float,
    mag_limit: float | None = None,
) -> pd.DataFrame:
    """
    Selects visible catalog stars inside a square RA/DEC field.

    Args:
        catalog: Loaded star catalog.
        center_ra: Field center right ascension in degrees.
        center_dec: Field center declination in degrees.
        fov_deg: Field width and height in degrees.
        mag_limit: Optional faintest visual magnitude.

    Returns:
        Brightness-sorted visible star table.
    """

    ra_min = center_ra - fov_deg / 2.0
    ra_max = center_ra + fov_deg / 2.0
    if ra_min < 0.0:
        ra_mask = (catalog["RA_deg"] >= ra_min + 360.0) | (catalog["RA_deg"] <= ra_max)
    elif ra_max >= 360.0:
        ra_mask = (catalog["RA_deg"] >= ra_min) | (catalog["RA_deg"] <= ra_max - 360.0)
    else:
        ra_mask = (catalog["RA_deg"] >= ra_min) & (catalog["RA_deg"] <= ra_max)

    dec_min = max(-90.0, center_dec - fov_deg / 2.0)
    dec_max = min(90.0, center_dec + fov_deg / 2.0)
    dec_mask = (catalog["DEC_deg"] >= dec_min) & (catalog["DEC_deg"] <= dec_max)
    stars = catalog[ra_mask & dec_mask].copy()
    if mag_limit is not None:
        stars = stars[stars["Vmag"] <= mag_limit].copy()
    return stars.sort_values("Vmag").reset_index(drop=True)


def sorted_edge_feature(vectors4: np.ndarray) -> np.ndarray | None:
    """
    Builds the normalized five-edge TETRA feature for four stars.

    Args:
        vectors4: Four unit vectors with shape `(4, 3)`.

    Returns:
        Five shortest edge lengths normalized by the longest edge.
    """

    edges = []
    for i, j in itertools.combinations(range(4), 2):
        edges.append(angular_distance_pair(vectors4[i], vectors4[j]))
    edges = np.sort(np.asarray(edges, dtype=np.float64))
    if edges[-1] <= 0:
        return None
    return (edges[:5] / edges[-1]).astype(np.float32)


def build_tetra_database(
    catalog: pd.DataFrame,
    fov_deg: float = 8.0,
    mag_limit: float = DEFAULT_DB_MAG_LIMIT,
    max_tetras_per_anchor: int = 20,
    force: bool = False,
) -> pd.DataFrame:
    """
    Builds or loads the cached TETRA database.

    Args:
        catalog: Loaded star catalog.
        fov_deg: Maximum tetra diameter in degrees.
        mag_limit: Faintest catalog magnitude used for the database.
        max_tetras_per_anchor: Row cap per anchor star.
        force: Rebuild even when a cache file exists.

    Returns:
        DataFrame with four HR IDs and five normalized TETRA features.
    """

    ensure_dirs()
    cache_path = CACHE_DIR / f"tetra_fov{fov_deg:g}_mag{mag_limit:g}_cap{max_tetras_per_anchor}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    stars = catalog[catalog["Vmag"] <= mag_limit].sort_values("Vmag").reset_index(drop=True)
    vectors = unit_vectors(stars["RA_deg"], stars["DEC_deg"])
    hrs = stars["HR_clean"].to_numpy(np.int32)
    max_sep = math.radians(fov_deg)
    rows: list[dict[str, float | int]] = []

    for anchor in range(len(stars)):
        distances = angular_distance(vectors[anchor : anchor + 1], vectors).ravel()
        neighbors = np.where((distances <= max_sep) & (np.arange(len(stars)) > anchor))[0]
        if len(neighbors) < 3:
            continue
        added = 0
        for combo in itertools.combinations(neighbors[:30], 3):
            if added >= max_tetras_per_anchor:
                break
            idx = np.array((anchor, *combo), dtype=int)
            pair_dists = angular_distance(vectors[idx], vectors[idx])
            if pair_dists.max() > max_sep:
                continue
            feature = sorted_edge_feature(vectors[idx])
            if feature is None:
                continue
            rows.append(
                {
                    "hr_a": int(hrs[idx[0]]),
                    "hr_b": int(hrs[idx[1]]),
                    "hr_c": int(hrs[idx[2]]),
                    "hr_d": int(hrs[idx[3]]),
                    "f1": float(feature[0]),
                    "f2": float(feature[1]),
                    "f3": float(feature[2]),
                    "f4": float(feature[3]),
                    "f5": float(feature[4]),
                }
            )
            added += 1

    db = pd.DataFrame(rows).drop_duplicates(["hr_a", "hr_b", "hr_c", "hr_d"]).reset_index(drop=True)
    db.to_csv(cache_path, index=False)
    return db


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes one vector.

    Args:
        v: Input vector.

    Returns:
        Unit vector.
    """

    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("zero vector")
    return v / norm


def frame_from_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Builds an orthonormal frame from two star vectors.

    Args:
        a: First unit vector.
        b: Second unit vector.

    Returns:
        3x3 frame matrix.
    """

    e1 = normalize(a)
    e2 = normalize(np.cross(a, b))
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def rotation_from_pair(catalog_a: np.ndarray, catalog_b: np.ndarray, observed_a: np.ndarray, observed_b: np.ndarray) -> np.ndarray:
    """
    Estimates attitude from one catalog pair and one observed pair.

    Args:
        catalog_a: First catalog unit vector.
        catalog_b: Second catalog unit vector.
        observed_a: First observed unit vector.
        observed_b: Second observed unit vector.

    Returns:
        3x3 rotation matrix mapping catalog vectors to observed vectors.
    """

    r = frame_from_pair(observed_a, observed_b) @ frame_from_pair(catalog_a, catalog_b).T
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def wahba_rotation(catalog_vectors: np.ndarray, observed_vectors: np.ndarray) -> np.ndarray:
    """
    Solves Wahba's problem for a matched star pattern.

    Args:
        catalog_vectors: Catalog unit vectors.
        observed_vectors: Observed unit vectors in matching order.

    Returns:
        3x3 rotation matrix mapping catalog vectors to observed vectors.
    """

    h = observed_vectors.T @ catalog_vectors
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def score_rotation(
    rotation: np.ndarray,
    catalog_vectors: np.ndarray,
    catalog_hrs: np.ndarray,
    observed_vectors: np.ndarray,
    tolerance_deg: float = 0.75,
) -> tuple[float, list[tuple[int, int, float]], float]:
    """
    Scores an attitude by matching observed stars to rotated catalog stars.

    Args:
        rotation: Catalog-to-observed rotation matrix.
        catalog_vectors: Candidate catalog unit vectors.
        catalog_hrs: HR IDs aligned with `catalog_vectors`.
        observed_vectors: Observed unit vectors to verify.
        tolerance_deg: Nominal angular tolerance in degrees.

    Returns:
        Score, matched `(hr_id, observed_index, error_rad)` rows, and mean residual.
    """

    predicted = (rotation @ catalog_vectors.T).T
    tolerance = math.radians(tolerance_deg)
    used_catalog: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for obs_i, observed in enumerate(observed_vectors):
        residuals = angular_distance(predicted, observed[None, :]).ravel()
        for cat_i in np.argsort(residuals)[:8]:
            cat_i = int(cat_i)
            if cat_i in used_catalog:
                continue
            err = float(residuals[cat_i])
            if err <= 3.0 * tolerance:
                used_catalog.add(cat_i)
                matches.append((int(catalog_hrs[cat_i]), int(obs_i), err))
            break

    if not matches:
        return -1e9, [], np.inf
    errors = np.asarray([match[2] for match in matches])
    score = len(matches) * 10.0 - float(errors.mean() / tolerance) - float(errors.max() / tolerance) * 0.25
    return score, matches, float(errors.mean())


def identify_tetra(
    catalog: pd.DataFrame,
    database: pd.DataFrame,
    obs_ra: float,
    obs_dec: float,
    fov_deg: float,
    mag_limit: float | None,
    max_stars_query: int = 12,
    match_threshold: float = 0.003,
) -> dict[str, object]:
    """
    Identifies one field with the TETRA feature database.

    Args:
        catalog: Loaded star catalog.
        database: TETRA database from `build_tetra_database`.
        obs_ra: Field center right ascension in degrees.
        obs_dec: Field center declination in degrees.
        fov_deg: Field width and height in degrees.
        mag_limit: Optional faintest visual magnitude.
        max_stars_query: Brightest observed stars used for combinations.
        match_threshold: Maximum feature distance for early success.

    Returns:
        Result dictionary with outcome, matched HR IDs, and visible stars.
    """

    visible = filter_stars_by_fov(catalog, obs_ra, obs_dec, fov_deg, mag_limit)
    if len(visible) < 4:
        return _result("sparse", visible)

    db_features = database[["f1", "f2", "f3", "f4", "f5"]].to_numpy(np.float32)
    fov_hrs = set(visible["HR_clean"].astype(int))
    pool = visible.head(max_stars_query).reset_index(drop=True)
    best_score = np.inf
    best_ids: list[int] = []

    for combo in itertools.combinations(range(len(pool)), 4):
        vectors = unit_vectors(pool.loc[list(combo), "RA_deg"], pool.loc[list(combo), "DEC_deg"])
        feature = sorted_edge_feature(vectors)
        if feature is None:
            continue
        distances = np.linalg.norm(db_features - feature, axis=1)
        best_idx = int(np.argmin(distances))
        score = float(distances[best_idx])
        if score < best_score:
            row = database.iloc[best_idx]
            best_ids = sorted(int(row[col]) for col in ("hr_a", "hr_b", "hr_c", "hr_d"))
            best_score = score
        if score < match_threshold and all(hr in fov_hrs for hr in best_ids):
            return _result("correct", visible, best_score, best_ids)

    return _result("failure", visible, best_score if np.isfinite(best_score) else np.nan, best_ids)


class TetraMatcher:
    """
    Reusable TETRA matcher with preloaded feature vectors.

    Args:
        catalog: Loaded star catalog.
        database: TETRA database from `build_tetra_database`.
    """

    def __init__(self, catalog: pd.DataFrame, database: pd.DataFrame):
        self.catalog = catalog
        self.database = database
        self.features = database[["f1", "f2", "f3", "f4", "f5"]].to_numpy(np.float32)

    def identify(
        self,
        obs_ra: float,
        obs_dec: float,
        fov_deg: float,
        mag_limit: float | None,
        max_stars_query: int = 12,
        match_threshold: float = 0.003,
    ) -> dict[str, object]:
        """
        Identifies one field by nearest TETRA feature lookup.

        Args:
            obs_ra: Field center right ascension in degrees.
            obs_dec: Field center declination in degrees.
            fov_deg: Field width and height in degrees.
            mag_limit: Optional faintest visual magnitude.
            max_stars_query: Brightest observed stars used for combinations.
            match_threshold: Maximum feature distance for early success.

        Returns:
            Result dictionary with outcome, matched HR IDs, and visible stars.
        """

        visible = filter_stars_by_fov(self.catalog, obs_ra, obs_dec, fov_deg, mag_limit)
        if len(visible) < 4:
            return _result("sparse", visible)

        fov_hrs = set(visible["HR_clean"].astype(int))
        pool = visible.head(max_stars_query).reset_index(drop=True)
        best_score = np.inf
        best_ids: list[int] = []

        for combo in itertools.combinations(range(len(pool)), 4):
            vectors = unit_vectors(pool.loc[list(combo), "RA_deg"], pool.loc[list(combo), "DEC_deg"])
            feature = sorted_edge_feature(vectors)
            if feature is None:
                continue
            distances = np.linalg.norm(self.features - feature, axis=1)
            best_idx = int(np.argmin(distances))
            score = float(distances[best_idx])
            if score < best_score:
                row = self.database.iloc[best_idx]
                best_ids = sorted(int(row[col]) for col in ("hr_a", "hr_b", "hr_c", "hr_d"))
                best_score = score
            if score < match_threshold and all(hr in fov_hrs for hr in best_ids):
                return _result("correct", visible, best_score, best_ids)

        return _result("failure", visible, best_score if np.isfinite(best_score) else np.nan, best_ids)


def _result(
    outcome: str,
    visible: pd.DataFrame,
    score: float | None = np.nan,
    matched_ids: list[int] | None = None,
    mean_residual: float | None = np.nan,
    pattern_ids: list[int] | None = None,
) -> dict[str, object]:
    """
    Creates the standard result dictionary used by all matchers.

    Args:
        outcome: `"correct"`, `"failure"`, or `"sparse"`.
        visible: Visible stars for the tested field.
        score: Algorithm-specific match score.
        matched_ids: HR IDs verified after matching.
        mean_residual: Mean angular residual in radians.
        pattern_ids: Core pattern HR IDs used to estimate attitude.

    Returns:
        Result dictionary consumed by batch tests and plots.
    """

    return {
        "outcome": outcome,
        "correct": outcome == "correct",
        "n_stars": int(len(visible)),
        "score": score,
        "matched_ids": matched_ids or [],
        "pattern_ids": pattern_ids or [],
        "mean_residual_deg": math.degrees(mean_residual) if mean_residual is not None and np.isfinite(mean_residual) else np.nan,
        "visible": visible,
    }


def run_batch(
    identifier: Callable[[float, float, float, float | None, int], dict[str, object]],
    config: BatchConfig,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Runs one repeatable batch of synthetic field identifications.

    Args:
        identifier: Matcher function with signature `(ra, dec, fov, mag, max_stars)`.
        config: Batch sampling and query settings.
        show_progress: Print elapsed time and ETA for long runs.

    Returns:
        One result row per sampled field.
    """

    rng = np.random.default_rng(config.seed)
    rows = []
    started = time.perf_counter()
    next_progress_at = 1
    for sample_i in range(config.samples):
        obs_ra = float(rng.uniform(0.0, 360.0))
        obs_dec = float(rng.uniform(-70.0, 70.0))
        result = identifier(obs_ra, obs_dec, config.fov_deg, config.mag_limit, config.max_stars_query)
        rows.append(
            {
                "ra": obs_ra,
                "dec": obs_dec,
                "fov_deg": config.fov_deg,
                "mag_limit": "ALL" if config.mag_limit is None else config.mag_limit,
                "n_stars": result["n_stars"],
                "outcome": result["outcome"],
                "correct": result["correct"],
                "score": result["score"],
                "mean_residual_deg": result["mean_residual_deg"],
            }
        )
        completed = sample_i + 1
        should_report = (
            show_progress
            and config.samples > 1
            and (
                completed == config.samples
                or completed == next_progress_at
                or time.perf_counter() - started >= 10.0
            )
        )
        if should_report:
            elapsed = time.perf_counter() - started
            avg_sample_seconds = elapsed / completed
            eta_seconds = avg_sample_seconds * (config.samples - completed)
            valid = sum(row["outcome"] != "sparse" for row in rows)
            correct = sum(row["outcome"] == "correct" for row in rows)
            accuracy = correct / valid * 100.0 if valid else 0.0
            print(
                f"  batch {completed}/{config.samples} "
                f"({completed / config.samples * 100:.0f}%) | "
                f"accuracy={accuracy:.1f}% valid={valid} | "
                f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
                flush=True,
            )
            next_progress_at = min(config.samples, completed + max(1, config.samples // 10))
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> dict[str, float | int]:
    """
    Summarizes batch outcomes into accuracy counts.

    Args:
        results: Output from `run_batch`.

    Returns:
        Counts for total, sparse, valid, correct, failure, and accuracy percent.
    """

    valid = results[results["outcome"] != "sparse"]
    correct = int((valid["outcome"] == "correct").sum())
    failure = int((valid["outcome"] == "failure").sum())
    return {
        "total": int(len(results)),
        "sparse": int((results["outcome"] == "sparse").sum()),
        "valid": int(len(valid)),
        "correct": correct,
        "failure": failure,
        "accuracy_pct": float(correct / len(valid) * 100.0) if len(valid) else 0.0,
    }
