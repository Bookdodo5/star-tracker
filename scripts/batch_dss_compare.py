"""
Batch real-image accuracy test on DSS2 Red fields at a rectangular FOV (default 7x4 deg).

Fetches N RANDOM sky fields from SkyView, runs the in-process TETRA pipeline (FOV
self-calibration), and reports how many solve within --tolerance of the requested
center. Images are cached under cache/real_images/ (named by RA/DEC) so reruns are offline.

    python scripts/batch_dss_compare.py --count 20 --fov-w 7 --fov-h 4 --tolerance 0.5

FOV note: the camera model uses fx=fy, so only the horizontal FOV (--fov-w) is passed
to the solver; --fov-h only sizes the fetched image (height px = width px * fov_h/fov_w).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import live_identify as L  # noqa: E402

CACHE = ROOT / "cache" / "real_images"


def random_fields(count, seed):
    """Returns `count` (ra, dec) deg pairs uniform on the sphere (DEC via arcsin for equal area)."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, count)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, count)))
    return list(zip(ra.tolist(), dec.tolist()))


def field_name(ra, dec):
    """Cache/file name encoding the pointing, e.g. ra123.45_dec-06.78."""
    return f"ra{ra:06.2f}_dec{dec:+06.2f}"


def angular_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle angle between two RA/DEC points, in degrees."""
    a = SkyCoord(ra1, dec1, unit="deg")
    b = SkyCoord(ra2, dec2, unit="deg")
    return a.separation(b).deg


def fetch_field(name, ra, dec, fov_w, fov_h, width_px):
    """Returns a north-up grayscale BGR uint8 image (cached as viewable PNG), fetching DSS2 Red if needed."""
    height_px = int(round(width_px * fov_h / fov_w))
    cache_path = CACHE / f"dss_{name}_{fov_w:g}x{fov_h:g}_{width_px}.png"
    if cache_path.exists():
        return cv2.imread(str(cache_path))
    coord = SkyCoord(ra, dec, unit="deg")
    images = SkyView.get_images(
        position=coord, survey="DSS2 Red",
        width=fov_w * u.deg, height=fov_h * u.deg,
        pixels=f"{width_px},{height_px}",
    )
    image = np.flipud(images[0][0].data.astype(np.float32))  # FITS row0 is south -> flip to north-up
    lo, hi = float(image.min()), float(image.max())
    scaled = (((image - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
              if hi > lo else np.zeros_like(image, dtype=np.uint8))
    bgr = np.stack([scaled, scaled, scaled], axis=-1)  # gray -> 3-channel for the BGR pipeline
    CACHE.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), bgr)
    return bgr


def main():
    p = argparse.ArgumentParser(description="Batch DSS accuracy test at a rectangular FOV")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--fov-w", type=float, default=7.0, help="horizontal FOV (deg), passed to the solver")
    p.add_argument("--fov-h", type=float, default=4.0, help="vertical FOV (deg), only sizes the fetched image")
    p.add_argument("--width-px", type=int, default=700)
    p.add_argument("--tolerance", type=float, default=0.5, help="max boresight error to count as correct (deg)")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for the random pointings (reproducible)")
    p.add_argument("--fixed", action="store_true",
                   help="solve at fixed --fov-w (no FOV search); correct for a known-FOV camera and "
                        "rejects spurious matches at the wrong scale")
    p.add_argument("--morph", type=int, default=0)
    args = p.parse_args()

    lib = L.load_lib()
    fields = random_fields(args.count, args.seed)
    print(f"Testing {len(fields)} random DSS fields at {args.fov_w:g}x{args.fov_h:g} deg, "
          f"tolerance {args.tolerance} deg (seed {args.seed})\n")

    solved = correct = 0
    errors = []
    for ra, dec in fields:
        name = field_name(ra, dec)
        try:
            bgr = fetch_field(name, ra, dec, args.fov_w, args.fov_h, args.width_px)
        except Exception as e:
            print(f"{name:22s} FETCH FAILED: {e}")
            continue
        if args.fixed:
            att = L.solve(lib, bgr, args.fov_w, morph=args.morph)
            rec_fov = args.fov_w
        else:
            rec_fov, att = L.calibrate_fov(lib, bgr, args.fov_w, morph=args.morph)
        if att is None:
            print(f"{name:22s} truth=({ra:7.2f},{dec:6.2f})  NO SOLVE")
            continue
        solved += 1
        err = angular_sep_deg(ra, dec, att[0], att[1])
        errors.append(err)
        ok = err <= args.tolerance
        correct += ok
        print(f"{name:22s} truth=({ra:7.2f},{dec:6.2f})  solved=({att[0]:7.2f},{att[1]:6.2f})  "
              f"err={err:5.3f} deg  fov={rec_fov:5.3f}  {'OK' if ok else 'MISS'}")

    n = len(fields)
    print(f"\nsolved {solved}/{n}   within {args.tolerance} deg: {correct}/{n} "
          f"({100.0 * correct / n:.0f}%)")
    if errors:
        print(f"mean error {sum(errors) / len(errors):.3f} deg   max {max(errors):.3f} deg")


if __name__ == "__main__":
    main()
