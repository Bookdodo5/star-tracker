"""
Retired Pyramid reference implementation (Python).

Moved out of src/star_tracker_core.py when Pyramid was retired from the active
workflow. TETRA is the sole maintained identifier. This module is recoverable:
it imports the still-shared helpers from src.star_tracker_core, so to bring
Pyramid back, re-inline these blocks (PairDatabase, build_pair_database,
PairDatabaseContext, PyramidMatcher) into the core module.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.star_tracker_core import (
    CACHE_DIR,
    DEFAULT_DB_MAG_LIMIT,
    DEFAULT_MAX_FOV_DEG,
    _result,
    angular_distance,
    angular_distance_pair,
    ensure_dirs,
    filter_stars_by_fov,
    score_rotation,
    unit_vectors,
    wahba_rotation,
)


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
