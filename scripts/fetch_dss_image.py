"""
Fetches a DSS2 Red image from SkyView via astroquery and writes it as a PPM
file ready for the centroid pipeline.

Usage:
    python scripts/fetch_dss_image.py --ra 83.8 --dec -5.4 --fov 10 --size 400 \
        --output outputs/test_dss.ppm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.skyview import SkyView


def fetch_image(ra: float, dec: float, fov_deg: float, pixels: int) -> np.ndarray:
    """Downloads a DSS2 Red field and returns a float32 2-D array (north up, east left)."""
    coord = SkyCoord(ra, dec, unit='deg')
    images = SkyView.get_images(
        position=coord,
        survey='DSS2 Red',
        radius=(fov_deg / 2) * u.deg,
        pixels=f'{pixels},{pixels}',
    )
    hdu = images[0][0]
    image = hdu.data.astype(np.float32)
    # FITS row 0 is south (bottom); flip so row 0 is north (top) as the camera model expects
    return np.flipud(image)


def image_to_ppm(image: np.ndarray) -> bytes:
    """Converts a float32 2-D array to an 8-bit grayscale PPM (P6)."""
    lo, hi = image.min(), image.max()
    if hi > lo:
        scaled = ((image - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        scaled = np.zeros_like(image, dtype=np.uint8)
    h, w = scaled.shape
    rgb = np.stack([scaled, scaled, scaled], axis=-1)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + rgb.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch DSS2 Red image from SkyView and save as PPM")
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--fov", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("outputs/test_dss.ppm"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching DSS2 Red: RA={args.ra} DEC={args.dec} FOV={args.fov}° size={args.size}px ...")
    image = fetch_image(args.ra, args.dec, args.fov, args.size)
    print(f"  Image shape: {image.shape}  min={image.min():.1f}  max={image.max():.1f}")

    ppm_bytes = image_to_ppm(image)
    args.output.write_bytes(ppm_bytes)
    print(f"  Written {len(ppm_bytes)} bytes → {args.output}")


if __name__ == "__main__":
    main()
