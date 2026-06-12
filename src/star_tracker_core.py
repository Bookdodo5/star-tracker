from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CATALOG_PATHS = (
    DATA_DIR / "catalog.bin",
    PROJECT_ROOT / "Tetra" / "catalog.bin",
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


@dataclass
class PairDatabase:
    """
    Stores catalog star pairs for Pyramid seed lookup.

    The rows are sorted by `sep_code`, so candidate lookup is a binary-search
    range scan. This mirrors the intended C structure:
    `{ uint16_t hr_a, hr_b, sep_code; }`.

    Args:
        hr_a: First HR identifier for each pair.
        hr_b: Second HR identifier for each pair.
        sep_code: Quantized angular separation in the range [0, 65535].
        max_fov_deg: Maximum separation represented by `sep_code`.
    """

    hr_a: np.ndarray
    hr_b: np.ndarray
    sep_code: np.ndarray
    max_fov_deg: float

    @property
    def max_sep_rad(self) -> float:
        """
        Returns the largest pair separation in radians.
        """

        return math.radians(self.max_fov_deg)

    def __len__(self) -> int:
        return int(len(self.sep_code))

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Returns a readable preview of the first database rows.

        Args:
            n: Maximum number of rows to return.

        Returns:
            DataFrame with HR IDs and decoded separation.
        """

        n = min(n, len(self))
        sep_rad = self.sep_code[:n].astype(np.float64) / 65535.0 * self.max_sep_rad
        return pd.DataFrame(
            {
                "hr_a": self.hr_a[:n],
                "hr_b": self.hr_b[:n],
                "sep_rad": sep_rad,
                "sep_deg": np.degrees(sep_rad),
            }
        )

    def save(self, path: Path) -> None:
        """
        Saves the compact pair database as compressed NumPy arrays.

        Args:
            path: Output `.npz` path.
        """

        np.savez_compressed(
            path,
            hr_a=self.hr_a.astype(np.uint16),
            hr_b=self.hr_b.astype(np.uint16),
            sep_code=self.sep_code.astype(np.uint16),
            max_fov_deg=np.asarray([self.max_fov_deg], dtype=np.float32),
        )

    def save_csv(self, path: Path) -> None:
        """
        Saves the pair database as a normal CSV file.

        Args:
            path: Output `.csv` path.
        """

        sep_rad = self.sep_code.astype(np.float64) / 65535.0 * self.max_sep_rad
        pd.DataFrame(
            {
                "hr_a": self.hr_a.astype(np.uint16),
                "hr_b": self.hr_b.astype(np.uint16),
                "sep_rad": sep_rad.astype(np.float32),
            }
        ).to_csv(path, index=False)

    @classmethod
    def load(cls, path: Path) -> "PairDatabase":
        """
        Loads a compressed NumPy pair database.

        Args:
            path: Input `.npz` path.

        Returns:
            PairDatabase ready for lookup.
        """

        data = np.load(path)
        return cls(
            hr_a=data["hr_a"].astype(np.uint16),
            hr_b=data["hr_b"].astype(np.uint16),
            sep_code=data["sep_code"].astype(np.uint16),
            max_fov_deg=float(data["max_fov_deg"][0]),
        )

    @classmethod
    def load_csv(cls, path: Path, max_fov_deg: float) -> "PairDatabase":
        """
        Loads a CSV pair database and quantizes separations.

        Args:
            path: Input `.csv` path.
            max_fov_deg: Maximum separation represented by the CSV.

        Returns:
            PairDatabase sorted in file order.
        """

        df = pd.read_csv(path)
        max_sep = math.radians(max_fov_deg)
        sep_code = np.rint(df["sep_rad"].to_numpy(np.float64) / max_sep * 65535.0).astype(np.uint16)
        return cls(
            hr_a=df["hr_a"].to_numpy(np.uint16),
            hr_b=df["hr_b"].to_numpy(np.uint16),
            sep_code=sep_code,
            max_fov_deg=max_fov_deg,
        )

    def candidates(self, separation: float, tolerance_deg: float, limit: int) -> pd.DataFrame:
        """
        Finds catalog pairs near one observed angular separation.

        Args:
            separation: Observed pair separation in radians.
            tolerance_deg: Allowed separation error in degrees.
            limit: Maximum candidate rows returned.

        Returns:
            Candidate pair rows sorted near the requested separation.
        """

        code = self._encode_separation(separation)
        tolerance_code = max(1, int(round(math.radians(tolerance_deg) / self.max_sep_rad * 65535.0)))
        lo = np.searchsorted(self.sep_code, max(0, code - tolerance_code), side="left")
        hi = np.searchsorted(self.sep_code, min(65535, code + tolerance_code), side="right")
        indices = np.arange(lo, hi)
        if len(indices) > limit:
            delta = np.abs(self.sep_code[indices].astype(np.int32) - code)
            indices = indices[np.argsort(delta)[:limit]]
        sep_rad = self.sep_code[indices].astype(np.float64) / 65535.0 * self.max_sep_rad
        return pd.DataFrame(
            {
                "hr_a": self.hr_a[indices].astype(np.int32),
                "hr_b": self.hr_b[indices].astype(np.int32),
                "sep_rad": sep_rad,
            }
        )

    def _encode_separation(self, separation: float) -> int:
        code = int(round(separation / self.max_sep_rad * 65535.0))
        return max(0, min(65535, code))


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


def build_pair_database(
    catalog: pd.DataFrame,
    algorithm_name: str,
    max_fov_deg: float = DEFAULT_MAX_FOV_DEG,
    mag_limit: float = DEFAULT_DB_MAG_LIMIT,
    min_separation_deg: float = 0.25,
    max_stars: int | None = None,
    max_pairs_per_star: int = 15,
    separation_bin_deg: float = 0.25,
    max_pairs_per_star_bin: int = 2,
    force: bool = False,
) -> tuple[PairDatabase, pd.DataFrame]:
    """
    Builds or loads the Pyramid seed-pair database.

    Args:
        catalog: Loaded star catalog.
        algorithm_name: Cache prefix; use `"pyramid"` for CSV output.
        max_fov_deg: Maximum pair separation stored in the database.
        mag_limit: Faintest catalog magnitude used for the database.
        min_separation_deg: Minimum stored pair separation.
        max_stars: Optional cap on catalog stars before pair generation.
        max_pairs_per_star: Maximum rows kept per anchor star.
        separation_bin_deg: Bin width used to keep pair angles diverse.
        max_pairs_per_star_bin: Maximum rows kept per anchor star per bin.
        force: Rebuild even when a cache file exists.

    Returns:
        Pair database and the catalog stars used to build it.
    """

    ensure_dirs()
    use_csv_cache = algorithm_name == "pyramid"
    extension = "csv" if use_csv_cache else "npz"
    cache_name = (
        f"{algorithm_name}_pairs_fov{max_fov_deg:g}_mag{mag_limit:g}"
        f"_min{min_separation_deg:g}_stars{'all' if max_stars is None else max_stars}"
        f"_cap{max_pairs_per_star}_bin{max_pairs_per_star_bin}.{extension}"
    )
    cache_path = CACHE_DIR / cache_name
    stars = catalog[catalog["Vmag"] <= mag_limit].sort_values("Vmag")
    if max_stars is not None:
        stars = stars.head(max_stars)
    stars = stars.reset_index(drop=True)
    if cache_path.exists() and not force:
        if use_csv_cache:
            return PairDatabase.load_csv(cache_path, max_fov_deg), stars
        return PairDatabase.load(cache_path), stars

    vectors = unit_vectors(stars["RA_deg"], stars["DEC_deg"])
    hrs = stars["HR_clean"].to_numpy(np.uint16)
    max_sep = math.radians(max_fov_deg)
    min_sep = math.radians(min_separation_deg)
    separation_bin = math.radians(separation_bin_deg)
    hr_a_parts = []
    hr_b_parts = []
    sep_parts = []
    chunk = 256
    for start in range(0, len(stars), chunk):
        end = min(start + chunk, len(stars))
        seps = angular_distance(vectors[start:end], vectors)
        for local_i, i in enumerate(range(start, end)):
            mask = (np.arange(len(stars)) > i) & (seps[local_i] >= min_sep) & (seps[local_i] <= max_sep)
            js = np.where(mask)[0]
            if len(js) == 0:
                continue
            pair_order = np.argsort(seps[local_i, js])
            js = js[pair_order]
            kept = []
            bin_counts: dict[int, int] = {}
            for j in js:
                bin_id = int(seps[local_i, j] / separation_bin)
                if bin_counts.get(bin_id, 0) >= max_pairs_per_star_bin:
                    continue
                kept.append(j)
                bin_counts[bin_id] = bin_counts.get(bin_id, 0) + 1
                if len(kept) >= max_pairs_per_star:
                    break
            js = np.asarray(kept, dtype=int)
            if len(js) == 0:
                continue
            sep_code = np.rint(seps[local_i, js] / max_sep * 65535.0).astype(np.uint16)
            hr_a_parts.append(np.full(len(js), hrs[i], dtype=np.uint16))
            hr_b_parts.append(hrs[js].astype(np.uint16))
            sep_parts.append(sep_code)
    if hr_a_parts:
        hr_a = np.concatenate(hr_a_parts)
        hr_b = np.concatenate(hr_b_parts)
        sep_code = np.concatenate(sep_parts)
        order = np.argsort(sep_code, kind="mergesort")
        db = PairDatabase(hr_a=hr_a[order], hr_b=hr_b[order], sep_code=sep_code[order], max_fov_deg=max_fov_deg)
    else:
        db = PairDatabase(
            hr_a=np.array([], dtype=np.uint16),
            hr_b=np.array([], dtype=np.uint16),
            sep_code=np.array([], dtype=np.uint16),
            max_fov_deg=max_fov_deg,
        )
    if use_csv_cache:
        db.save_csv(cache_path)
    else:
        db.save(cache_path)
    return db, stars


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


class PairDatabaseContext:
    """
    Holds pair database lookup state shared by pair-based matchers.

    Args:
        pair_db: Pair database used for seed lookup.
        stars: Catalog stars aligned with `pair_db`.
    """

    def __init__(self, pair_db: PairDatabase, stars: pd.DataFrame):
        self.pair_db = pair_db
        self.stars = stars
        self.vectors = unit_vectors(stars["RA_deg"], stars["DEC_deg"])
        self.hrs = stars["HR_clean"].to_numpy(np.int32)
        self.magnitudes = stars["Vmag"].to_numpy(np.float32)
        self.hr_to_idx = {int(hr): i for i, hr in enumerate(self.hrs)}

    def candidates(self, separation: float, tolerance_deg: float, limit: int) -> pd.DataFrame:
        """
        Returns pair candidates near one observed separation.

        Args:
            separation: Observed separation in radians.
            tolerance_deg: Allowed separation error in degrees.
            limit: Maximum number of candidates.

        Returns:
            Candidate pair rows.
        """

        return self.pair_db.candidates(separation, tolerance_deg, limit)


class OpenStarTrackerMatcher:
    def __init__(self, catalog: pd.DataFrame, pair_db: PairDatabase, stars: pd.DataFrame):
        self.catalog = catalog
        self.context = PairDatabaseContext(pair_db, stars)
        self.catalog_limit = min(900, len(self.context.hrs))

    def identify(
        self,
        obs_ra: float,
        obs_dec: float,
        fov_deg: float,
        mag_limit: float | None,
        max_stars_query: int = 12,
        sep_tolerance_deg: float = 0.08,
    ) -> dict[str, object]:
        visible = filter_stars_by_fov(self.catalog, obs_ra, obs_dec, fov_deg, mag_limit)
        if len(visible) < 3:
            return _result("sparse", visible)

        observed = unit_vectors(visible["RA_deg"], visible["DEC_deg"])
        observed_magnitudes = visible["Vmag"].to_numpy(np.float32)
        fov_hrs = set(visible["HR_clean"].astype(int))
        top_n = min(max_stars_query, len(visible))
        best = {"score": -1e9, "ids": [], "pattern_ids": [], "residual": np.inf}
        shortlist: list[dict[str, object]] = []
        tolerance = math.radians(sep_tolerance_deg)
        catalog_slice = slice(0, self.catalog_limit)
        limited_vectors = self.context.vectors[catalog_slice]
        limited_hrs = self.context.hrs[catalog_slice]

        for oi, oj in itertools.combinations(range(top_n), 2):
            obs_sep = angular_distance_pair(observed[oi], observed[oj])
            obs_to_a = np.asarray([angular_distance_pair(observed[oi], observed[k]) for k in range(top_n)])
            obs_to_b = np.asarray([angular_distance_pair(observed[oj], observed[k]) for k in range(top_n)])
            for row in self.context.candidates(obs_sep, sep_tolerance_deg, 100).itertuples(index=False):
                ia = self.context.hr_to_idx.get(int(row.hr_a))
                ib = self.context.hr_to_idx.get(int(row.hr_b))
                if ia is None or ib is None:
                    continue
                for seed_a, seed_b in ((ia, ib), (ib, ia)):
                    cat_to_a = angular_distance(limited_vectors, self.context.vectors[seed_a][None, :]).ravel()
                    cat_to_b = angular_distance(limited_vectors, self.context.vectors[seed_b][None, :]).ravel()
                    mapped = [
                        (int(self.context.hrs[seed_a]), oi),
                        (int(self.context.hrs[seed_b]), oj),
                    ]
                    for obs_k in range(top_n):
                        if obs_k in (oi, oj):
                            continue
                        err = np.abs(cat_to_a - obs_to_a[obs_k]) + np.abs(cat_to_b - obs_to_b[obs_k])
                        best_idx = int(np.argmin(err))
                        if float(err[best_idx]) <= 2.0 * tolerance:
                            mapped.append((int(limited_hrs[best_idx]), obs_k))
                    ids = sorted({hr for hr, _ in mapped})
                    errors = []
                    for left, right in itertools.combinations(mapped, 2):
                        left_hr, left_obs = left
                        right_hr, right_obs = right
                        left_idx = self.context.hr_to_idx[left_hr]
                        right_idx = self.context.hr_to_idx[right_hr]
                        obs_pair_sep = angular_distance_pair(observed[left_obs], observed[right_obs])
                        cat_pair_sep = angular_distance_pair(self.context.vectors[left_idx], self.context.vectors[right_idx])
                        errors.append(abs(obs_pair_sep - cat_pair_sep))
                    residual = float(np.mean(errors)) if errors else np.inf
                    mag_error = float(
                        np.mean(
                            [
                                abs(float(self.context.magnitudes[self.context.hr_to_idx[hr]]) - float(observed_magnitudes[obs_id]))
                                for hr, obs_id in mapped
                            ]
                        )
                    )
                    score = len(ids) * 15.0 - residual / tolerance - mag_error * 4.0
                    shortlist.append({"score": score, "seed_a": seed_a, "seed_b": seed_b, "oi": oi, "oj": oj, "ids": ids, "residual": residual})
                    if score > best["score"]:
                        best = {"score": score, "ids": ids, "residual": residual}

        for item in sorted(shortlist, key=lambda x: float(x["score"]), reverse=True)[:120]:
            rotation = rotation_from_pair(
                self.context.vectors[int(item["seed_a"])],
                self.context.vectors[int(item["seed_b"])],
                observed[int(item["oi"])],
                observed[int(item["oj"])],
            )
            _, matches, residual = score_rotation(rotation, self.context.vectors, self.context.hrs, observed[:top_n])
            mag_error = 0.0
            if matches:
                mag_error = float(
                    np.mean(
                        [
                            abs(float(self.context.magnitudes[self.context.hr_to_idx[hr]]) - float(observed_magnitudes[obs_i]))
                            for hr, obs_i, _ in matches
                        ]
                    )
                )
            score = len(matches) * 10.0 - residual / math.radians(0.75) * 25.0 - mag_error * 4.0 + float(item["score"]) * 0.01
            if score > best["score"]:
                best = {"score": score, "ids": sorted({m[0] for m in matches}), "residual": residual}

        visible_hits = sum(1 for hr in best["ids"] if hr in fov_hrs)
        correct = visible_hits >= 3 and visible_hits / max(1, len(best["ids"])) >= 0.70
        return _result("correct" if correct else "failure", visible, best["score"], best["ids"], best["residual"])


class PyramidMatcher:
    """
    Pyramid matcher that grows pair seeds into a verified star pattern.

    Args:
        catalog: Loaded star catalog.
        pair_db: Pair database from `build_pair_database`.
        stars: Catalog stars aligned with `pair_db`.
    """

    def __init__(self, catalog: pd.DataFrame, pair_db: PairDatabase, stars: pd.DataFrame):
        self.catalog = catalog
        self.context = PairDatabaseContext(pair_db, stars)

    def identify(
        self,
        obs_ra: float,
        obs_dec: float,
        fov_deg: float,
        mag_limit: float | None,
        max_stars_query: int = 10,
        target_size: int = 4,
        seed_tolerance_deg: float = 0.10,
        grow_tolerance_deg: float = 0.18,
    ) -> dict[str, object]:
        """
        Identifies one field with the Pyramid algorithm.

        The matcher chooses observed seed pairs, looks up catalog pairs by
        separation, grows each seed into a `target_size` pattern, estimates
        attitude with Wahba rotation, and verifies extra stars.

        Args:
            obs_ra: Field center right ascension in degrees.
            obs_dec: Field center declination in degrees.
            fov_deg: Field width and height in degrees.
            mag_limit: Optional faintest visual magnitude.
            max_stars_query: Brightest observed stars used by the matcher.
            target_size: Number of stars in the core Pyramid pattern.
            seed_tolerance_deg: Pair lookup tolerance in degrees.
            grow_tolerance_deg: Pattern growth tolerance in degrees.

        Returns:
            Result dictionary with matched IDs, core `pattern_ids`, and branch count.
        """

        visible = filter_stars_by_fov(self.catalog, obs_ra, obs_dec, fov_deg, mag_limit)
        if len(visible) < target_size:
            return _result("sparse", visible)

        observed = unit_vectors(visible["RA_deg"], visible["DEC_deg"])
        fov_hrs = set(visible["HR_clean"].astype(int))
        top_n = min(max_stars_query, len(visible))
        obs_indices = list(range(top_n))
        best = {"score": -1e9, "ids": [], "residual": np.inf}
        branches = 0
        grow_tolerance = math.radians(grow_tolerance_deg)

        def next_catalog_candidates(next_obs: int, obs_ids: list[int], cat_ids: list[int]) -> list[int]:
            errors = np.zeros(len(self.context.hrs), dtype=np.float64)
            for obs_id, cat_id in zip(obs_ids, cat_ids):
                cat_idx = self.context.hr_to_idx[cat_id]
                obs_sep = angular_distance_pair(observed[next_obs], observed[obs_id])
                cat_sep = angular_distance(self.context.vectors, self.context.vectors[cat_idx][None, :]).ravel()
                errors += np.abs(cat_sep - obs_sep)
            for cat_id in cat_ids:
                errors[self.context.hr_to_idx[cat_id]] = np.inf
            order = np.argsort(errors)
            threshold = max(1, len(obs_ids)) * grow_tolerance
            return [int(self.context.hrs[i]) for i in order[:45] if errors[i] <= threshold]

        def evaluate(obs_ids: list[int], cat_ids: list[int]) -> None:
            nonlocal best
            catalog_vectors = np.asarray([self.context.vectors[self.context.hr_to_idx[hr]] for hr in cat_ids])
            observed_vectors = np.asarray([observed[i] for i in obs_ids])
            rotation = wahba_rotation(catalog_vectors, observed_vectors)
            score, matches, residual = score_rotation(rotation, self.context.vectors, self.context.hrs, observed[:top_n])
            if score > best["score"]:
                best = {
                    "score": score,
                    "ids": sorted({m[0] for m in matches}),
                    "pattern_ids": sorted(cat_ids),
                    "residual": residual,
                }

        def grow(obs_ids: list[int], cat_ids: list[int], remaining_obs: list[int]) -> None:
            nonlocal branches
            branches += 1
            if len(obs_ids) == target_size:
                evaluate(obs_ids, cat_ids)
                return
            if not remaining_obs or branches > 1200:
                return
            next_obs = remaining_obs[0]
            for candidate_hr in next_catalog_candidates(next_obs, obs_ids, cat_ids):
                grow([*obs_ids, next_obs], [*cat_ids, candidate_hr], remaining_obs[1:])

        for oi, oj in itertools.combinations(obs_indices, 2):
            obs_sep = angular_distance_pair(observed[oi], observed[oj])
            for row in self.context.candidates(obs_sep, seed_tolerance_deg, 45).itertuples(index=False):
                for seed_pair in ((int(row.hr_a), int(row.hr_b)), (int(row.hr_b), int(row.hr_a))):
                    remaining_obs = [idx for idx in obs_indices if idx not in (oi, oj)]
                    grow([oi, oj], list(seed_pair), remaining_obs)
                    if branches > 1200:
                        break
                if branches > 1200:
                    break
            if branches > 1200:
                break

        visible_hits = sum(1 for hr in best["ids"] if hr in fov_hrs)
        correct = visible_hits >= target_size and visible_hits / max(1, len(best["ids"])) >= 0.70
        result = _result(
            "correct" if correct else "failure",
            visible,
            best["score"],
            best["ids"],
            best["residual"],
            best["pattern_ids"],
        )
        result["branches"] = branches
        return result


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


def run_confusion_matrix(
    identifier: Callable[[float, float, float, float | None, int], dict[str, object]],
    fov_values: list[float],
    mag_limits: list[float | None],
    samples: int,
    max_stars_query: int,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tests accuracy across FOV and magnitude settings.

    Args:
        identifier: Matcher function with signature `(ra, dec, fov, mag, max_stars)`.
        fov_values: FOV values to test.
        mag_limits: Magnitude limits to test.
        samples: Number of fields per matrix cell.
        max_stars_query: Brightest observed stars used per query.
        seed: Random seed reused across cells.

    Returns:
        Accuracy matrix and detailed per-cell summary table.
    """

    rows = []
    summaries = []
    total_cells = len(fov_values) * len(mag_limits)
    completed_cells = 0
    matrix_started = time.perf_counter()
    for fov in fov_values:
        row = {"FOV": fov}
        for mag_limit in mag_limits:
            started = time.perf_counter()
            label = "ALL" if mag_limit is None else f"<={mag_limit:g}"
            print(
                f"[{completed_cells + 1}/{total_cells}] "
                f"Running FOV={fov:g}, magnitude={label}, samples={samples} ...",
                flush=True,
            )
            results = run_batch(
                identifier,
                BatchConfig(samples=samples, fov_deg=fov, mag_limit=mag_limit, max_stars_query=max_stars_query, seed=seed),
                show_progress=False,
            )
            summary = summarize_results(results)
            elapsed = time.perf_counter() - started
            completed_cells += 1
            total_elapsed = time.perf_counter() - matrix_started
            avg_cell_seconds = total_elapsed / completed_cells
            remaining_cells = total_cells - completed_cells
            eta_seconds = avg_cell_seconds * remaining_cells
            row[label] = summary["accuracy_pct"]
            summaries.append({"FOV": fov, "Magnitude": label, "seconds": elapsed, **summary})
            print(
                f"  done {completed_cells}/{total_cells} "
                f"({completed_cells / total_cells * 100:.0f}%) | "
                f"accuracy={summary['accuracy_pct']:.1f}% valid={summary['valid']} | "
                f"cell={elapsed:.1f}s elapsed={total_elapsed:.1f}s eta={eta_seconds:.1f}s",
                flush=True,
            )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def plot_catalog(catalog: pd.DataFrame, output_name: str = "catalog_map.png") -> Path:
    """
    Plots the loaded catalog on RA/DEC axes.

    Args:
        catalog: Loaded star catalog.
        output_name: PNG filename under `outputs`.

    Returns:
        Saved image path.
    """

    ensure_dirs()
    fig, ax = plt.subplots(figsize=(14, 7))
    points = ax.scatter(
        catalog["RA_deg"],
        catalog["DEC_deg"],
        s=catalog["marker_size"],
        c=catalog["Vmag"],
        cmap="viridis_r",
        alpha=0.8,
        edgecolors="none",
    )
    ax.set_title("Yale Bright Star Catalog parsed from catalog.bin")
    ax.set_xlabel("Right Ascension (degrees)")
    ax.set_ylabel("Declination (degrees)")
    ax.set_xlim(360.0, 0.0)
    ax.set_ylim(-90.0, 90.0)
    ax.grid(True, linestyle=":", alpha=0.3)
    colorbar = fig.colorbar(points, ax=ax, label="Visual magnitude")
    colorbar.ax.invert_yaxis()
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=180)
    return path


def plot_single_result(result: dict[str, object], output_name: str, title: str) -> Path:
    """
    Plots one field identification result.

    Red circles show the core pattern. Orange circles show extra stars verified
    by the estimated attitude.

    Args:
        result: Matcher result dictionary.
        output_name: PNG filename under `outputs`.
        title: Plot title.

    Returns:
        Saved image path.
    """

    ensure_dirs()
    visible = result["visible"]
    assert isinstance(visible, pd.DataFrame)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(visible["RA_deg"], visible["DEC_deg"], s=visible["marker_size"] * 4.0, c=visible["Vmag"], cmap="viridis_r", alpha=0.8)
    pattern_ids = result.get("pattern_ids") or result["matched_ids"]
    verified_ids = [hr for hr in result["matched_ids"] if hr not in pattern_ids]
    verified = visible[visible["HR_clean"].isin(verified_ids)]
    pattern = visible[visible["HR_clean"].isin(pattern_ids)]
    if len(verified):
        ax.scatter(verified["RA_deg"], verified["DEC_deg"], s=120, facecolors="none", edgecolors="orange", linewidths=1.5)
    if len(pattern):
        ax.scatter(pattern["RA_deg"], pattern["DEC_deg"], s=190, facecolors="none", edgecolors="red", linewidths=2.2)
    ax.invert_xaxis()
    ax.set_title(title)
    ax.set_xlabel("RA degrees")
    ax.set_ylabel("DEC degrees")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=180)
    return path


def plot_confusion_matrix(confusion: pd.DataFrame, title: str, output_name: str) -> Path:
    """
    Plots the FOV/magnitude accuracy matrix.

    Args:
        confusion: Accuracy matrix from `run_confusion_matrix`.
        title: Plot title.
        output_name: PNG filename under `outputs`.

    Returns:
        Saved image path.
    """

    ensure_dirs()
    labels = [col for col in confusion.columns if col != "FOV"]
    values = confusion[labels].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(confusion)), [f"{x:g}" for x in confusion["FOV"]])
    ax.set_xlabel("Magnitude limit")
    ax.set_ylabel("FOV degrees")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] < 55.0 else "black"
            ax.text(col, row, f"{values[row, col]:.1f}%", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="Accuracy on non-sparse fields")
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=180)
    return path


def findings(summary: pd.DataFrame) -> str:
    """
    Converts confusion-matrix summaries into a short written finding.

    Args:
        summary: Detailed summary table from `run_confusion_matrix`.

    Returns:
        Human-readable analysis text.
    """

    valid_summary = summary[summary["valid"] > 0].copy()
    if valid_summary.empty:
        return "All tested cells were sparse. Increase FOV, magnitude depth, or sample count."
    best = valid_summary.sort_values("accuracy_pct", ascending=False).iloc[0]
    worst = valid_summary.sort_values("accuracy_pct", ascending=True).iloc[0]
    sparse = summary.sort_values("sparse", ascending=False).iloc[0]
    return "\n".join(
        [
            f"Best accuracy: {best['accuracy_pct']:.2f}% at FOV={best['FOV']} deg, magnitude={best['Magnitude']}.",
            f"Lowest non-sparse accuracy: {worst['accuracy_pct']:.2f}% at FOV={worst['FOV']} deg, magnitude={worst['Magnitude']}.",
            f"Most sparse setting: FOV={sparse['FOV']} deg, magnitude={sparse['Magnitude']} with {int(sparse['sparse'])}/{int(sparse['total'])} sparse samples.",
            "Accuracy improves when the field has enough bright stars and drops when the query becomes either sparse or too ambiguous.",
        ]
    )
