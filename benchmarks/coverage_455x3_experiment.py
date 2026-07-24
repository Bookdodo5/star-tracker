"""
One-time coverage check: current generated TETRA DB vs a 4.55 x 3.0 degree camera.

Coverage here means: among random sky pointings with enough visible stars, the
brightest observed stars contain at least one tetrad present in the current DB.
"""
from __future__ import annotations

import itertools
import math
import re
import sys
import time
import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.star_tracker_core import load_db_catalog, unit_vectors  # noqa: E402

DEFAULT_CAMERA_FOV_W = 4.55
DEFAULT_CAMERA_FOV_H = 3.0
DB_FOV_W = 7.0
DB_FOV_H = 4.0
MAG_LIMIT = 7.5
QUERY_STARS = 16
MIN_VISIBLE = 6
SEED = 12345
GENERATED_DB = ROOT / "identifier" / "generated" / "tetra_db_generated.c"
LIVE_DLL = ROOT / "live" / "build-mingw" / "libstar_live.dll"
RAD_TO_DEG = 57.29577951308232


def key(star_ids: tuple[int, int, int, int]) -> int:
    """Packs four sorted catalog indices into one integer key."""
    a, b, c, d = sorted(star_ids)
    return (a << 48) | (b << 32) | (c << 16) | d


def camera_basis(ra_deg: float, dec_deg: float, roll_deg: float) -> np.ndarray:
    """Builds the camera-to-catalog basis for one random pointing."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    roll = math.radians(roll_deg)
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    bore = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    x_axis = east * math.cos(roll) + north * math.sin(roll)
    y_axis = -east * math.sin(roll) + north * math.cos(roll)
    return np.vstack([x_axis, y_axis, bore])


def angular_error_deg(ra_a: float, dec_a: float, ra_b: float, dec_b: float) -> float:
    """Returns boresight angular error between two RA/DEC directions."""
    ra1 = math.radians(ra_a)
    dec1 = math.radians(dec_a)
    ra2 = math.radians(ra_b)
    dec2 = math.radians(dec_b)
    dot = (
        math.cos(dec1) * math.cos(ra1) * math.cos(dec2) * math.cos(ra2)
        + math.cos(dec1) * math.sin(ra1) * math.cos(dec2) * math.sin(ra2)
        + math.sin(dec1) * math.sin(dec2)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def load_identifier():
    """Loads the live DLL and returns a tiny identify_vectors wrapper."""
    if not LIVE_DLL.exists():
        raise FileNotFoundError(f"Missing {LIVE_DLL}; build live/ first")
    lib = ctypes.CDLL(str(LIVE_DLL))
    double_ptr = ctypes.POINTER(ctypes.c_double)
    lib.identify_vectors.restype = ctypes.c_int
    lib.identify_vectors.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
    ]

    def solve_vectors(xyz: np.ndarray) -> tuple[float, float, float] | None:
        """Runs the actual C TETRA identifier on brightest-first observed unit vectors."""
        arr = np.ascontiguousarray(xyz[:20], dtype=np.float32)
        ra, dec, roll, qw, qx, qy, qz = (ctypes.c_double() for _ in range(7))
        rc = lib.identify_vectors(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(arr.shape[0]),
            ctypes.byref(ra),
            ctypes.byref(dec),
            ctypes.byref(roll),
            ctypes.byref(qw),
            ctypes.byref(qx),
            ctypes.byref(qy),
            ctypes.byref(qz),
        )
        return None if rc != 1 else (ra.value, dec.value, roll.value)

    return solve_vectors


def load_generated_db_keys() -> set[int]:
    """Reads the baked generated TETRA DB and keeps only each node's four star ids."""
    keys: set[int] = set()
    hr_pattern = re.compile(r"\}, \{(\d+), (\d+), (\d+), (\d+)\},")
    with GENERATED_DB.open("r", encoding="utf-8") as file:
        for line in file:
            match = hr_pattern.search(line)
            if match:
                keys.add(key(tuple(int(match.group(i)) for i in range(1, 5))))
    return keys


def measure_coverage(
    keys: set[int],
    vecs: np.ndarray,
    samples: int,
    fov_w: float,
    fov_h: float,
    solve_vectors,
    correct_deg: float,
) -> dict[str, float]:
    """Runs sampled fields through the real identifier and counts where it breaks."""
    rng = np.random.default_rng(SEED)
    tan_w = math.tan(math.radians(fov_w * 0.5))
    tan_h = math.tan(math.radians(fov_h * 0.5))
    valid = 0
    no_tetra = 0
    solved = 0
    correct = 0
    wrong = 0
    total_visible = 0
    started = time.time()

    for sample_index in range(samples):
        if sample_index and sample_index % 1000 == 0:
            done = sample_index / samples
            eta = (time.time() - started) / done * (1.0 - done)
            print(f"progress {sample_index}/{samples} fields, eta {eta:.0f}s", flush=True)

        true_ra = float(rng.uniform(0.0, 360.0))
        true_dec = float(math.degrees(math.asin(rng.uniform(-1.0, 1.0))))
        true_roll = float(rng.uniform(0.0, 360.0))
        basis = camera_basis(true_ra, true_dec, true_roll)
        observed = vecs @ basis.T
        z = observed[:, 2]
        in_fov = np.where((z > 0.0) & (np.abs(observed[:, 0] / z) <= tan_w) & (np.abs(observed[:, 1] / z) <= tan_h))[0]
        total_visible += len(in_fov)
        if len(in_fov) < 4:
            continue

        valid += 1
        query = in_fov[:QUERY_STARS].tolist()
        if not any(key(combo) in keys for combo in itertools.combinations(query, 4)):
            no_tetra += 1

        attitude = solve_vectors(observed[in_fov])
        if attitude is None:
            continue
        solved += 1
        if angular_error_deg(true_ra, true_dec, attitude[0], attitude[1]) <= correct_deg:
            correct += 1
        else:
            wrong += 1

    return {
        "samples": samples,
        "valid": valid,
        "no_tetra": no_tetra,
        "null": valid - solved,
        "has_tetra_null": (valid - solved) - no_tetra,
        "solved": solved,
        "correct": correct,
        "wrong": wrong,
        "valid_pct": 100.0 * valid / samples,
        "no_tetra_pct": 100.0 * no_tetra / valid if valid else 0.0,
        "null_pct": 100.0 * (valid - solved) / valid if valid else 0.0,
        "has_tetra_null_pct": 100.0 * ((valid - solved) - no_tetra) / valid if valid else 0.0,
        "solve_pct": 100.0 * solved / valid if valid else 0.0,
        "correct_pct": 100.0 * correct / valid if valid else 0.0,
        "break_pct": 100.0 * (valid - correct) / valid if valid else 0.0,
        "sky_correct_pct": 100.0 * correct / samples,
        "mean_visible": total_visible / samples,
    }


def main() -> None:
    """Runs the one-time 4.55 x 3.0 FOV coverage experiment."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fov-w", type=float, default=DEFAULT_CAMERA_FOV_W)
    parser.add_argument("--fov-h", type=float, default=DEFAULT_CAMERA_FOV_H)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--correct-deg", type=float, default=0.5)
    args = parser.parse_args()

    catalog = load_db_catalog(MAG_LIMIT)
    vecs = unit_vectors(catalog["RA_deg"], catalog["DEC_deg"])

    print("loading generated DB key set...", flush=True)
    keys = load_generated_db_keys()
    print(f"db_tetrads={len(keys):,}", flush=True)

    solve_vectors = load_identifier()
    result = measure_coverage(keys, vecs, args.samples, args.fov_w, args.fov_h, solve_vectors, args.correct_deg)
    print(
        f"camera_fov={args.fov_w}x{args.fov_h}deg "
        f"correct_deg={args.correct_deg} "
        f"samples={result['samples']:.0f} valid={result['valid']:.0f} "
        f"no_tetra={result['no_tetra']:.0f} null={result['null']:.0f} "
        f"has_tetra_but_null={result['has_tetra_null']:.0f} solved={result['solved']:.0f} "
        f"correct={result['correct']:.0f} wrong={result['wrong']:.0f} "
        f"valid_fields={result['valid_pct']:.2f}% "
        f"no_tetra_of_valid={result['no_tetra_pct']:.2f}% "
        f"null_of_valid={result['null_pct']:.2f}% "
        f"has_tetra_but_null_of_valid={result['has_tetra_null_pct']:.2f}% "
        f"solve_of_valid={result['solve_pct']:.2f}% "
        f"correct_of_valid={result['correct_pct']:.2f}% "
        f"break_of_valid={result['break_pct']:.2f}% "
        f"sky_correct={result['sky_correct_pct']:.2f}% "
        f"mean_visible={result['mean_visible']:.2f}"
    )


if __name__ == "__main__":
    main()
