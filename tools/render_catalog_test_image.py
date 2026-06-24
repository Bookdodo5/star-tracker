from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CatalogStar:
    """
    Stores one catalog star with sky position, magnitude, and unit vector.
    """

    hr_id: int
    magnitude: float
    vector: tuple[float, float, float]


@dataclass(frozen=True)
class RenderedStar:
    """
    Stores one rendered catalog star and its pixel position.
    """

    hr_id: int
    x: float
    y: float
    magnitude: float


def parse_ra(text: str) -> float | None:
    """
    Converts fixed-width Yale catalog RA text to degrees.
    """

    text = text.strip()
    if len(text) < 6:
        return None
    try:
        return (float(text[:2]) + float(text[2:4]) / 60.0 + float(text[4:]) / 3600.0) * 15.0
    except ValueError:
        return None


def parse_dec(text: str) -> float | None:
    """
    Converts fixed-width Yale catalog DEC text to degrees.
    """

    text = text.strip()
    if len(text) < 6:
        return None
    try:
        sign = -1.0 if text[0] == "-" else 1.0
        body = text[1:] if text[0] in "+-" else text
        return sign * (float(body[:2]) + float(body[2:4]) / 60.0 + float(body[4:]) / 3600.0)
    except ValueError:
        return None


def unit_vector(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    """
    Converts RA/DEC to a unit vector in catalog coordinates.
    """

    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    return (
        math.cos(dec_rad) * math.cos(ra_rad),
        math.cos(dec_rad) * math.sin(ra_rad),
        math.sin(dec_rad),
    )


def load_catalog(catalog_path: Path, magnitude_limit: float) -> list[CatalogStar]:
    """
    Loads catalog stars bright enough to render.
    """

    stars: list[CatalogStar] = []
    for line in catalog_path.read_text(errors="ignore").splitlines():
        try:
            hr_id = int(line[:4])
            ra_deg = parse_ra(line[75:83])
            dec_deg = parse_dec(line[83:90])
            magnitude = float(line[102:107])
        except ValueError:
            continue
        if ra_deg is None or dec_deg is None or magnitude > magnitude_limit:
            continue
        stars.append(CatalogStar(hr_id, magnitude, unit_vector(ra_deg, dec_deg)))
    return sorted(stars, key=lambda star: star.magnitude)


def dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    """
    Computes a 3D dot product.
    """

    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Returns a unit-length copy of a 3D vector.
    """

    norm = math.sqrt(dot(vector, vector))
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def camera_basis(center_ra_deg: float, center_dec_deg: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """
    Builds catalog-to-camera basis vectors matching the physical astronomical
    image convention: north up, east left, pixel Y increasing downward.
    Pixel X increases to the right (west) and pixel Y increases downward (south),
    so the X basis is -east and the Y basis is -north.
    """

    center = unit_vector(center_ra_deg, center_dec_deg)
    ra_rad = math.radians(center_ra_deg)
    dec_rad = math.radians(center_dec_deg)
    east = (-math.sin(ra_rad), math.cos(ra_rad), 0.0)
    north = (-math.sin(dec_rad) * math.cos(ra_rad), -math.sin(dec_rad) * math.sin(ra_rad), math.cos(dec_rad))
    return normalize((-east[0], -east[1], -east[2])), normalize((-north[0], -north[1], -north[2])), normalize(center)


def project_stars(
    stars: list[CatalogStar],
    center_ra_deg: float,
    center_dec_deg: float,
    image_size: int,
    horizontal_fov_deg: float,
) -> list[RenderedStar]:
    """
    Projects catalog stars through the same pinhole camera model used by the C identifier.
    """

    camera_x, camera_y, camera_z = camera_basis(center_ra_deg, center_dec_deg)
    focal_length_px = (image_size * 0.5) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    center_px = (image_size - 1) * 0.5
    rendered: list[RenderedStar] = []
    for star in stars:
        cam_x = dot(camera_x, star.vector)
        cam_y = dot(camera_y, star.vector)
        cam_z = dot(camera_z, star.vector)
        if cam_z <= 0.0:
            continue
        pixel_x = focal_length_px * cam_x / cam_z + center_px
        pixel_y = focal_length_px * cam_y / cam_z + center_px
        if 2 <= pixel_x < image_size - 2 and 2 <= pixel_y < image_size - 2:
            rendered.append(RenderedStar(star.hr_id, pixel_x, pixel_y, star.magnitude))
    return rendered


def add_star(image: list[int], image_size: int, star: RenderedStar) -> None:
    """
    Adds one small synthetic star blob into the RGB image buffer.
    """

    amplitude = max(90.0, min(255.0, 255.0 - star.magnitude * 22.0))
    sigma = 1.2
    for y_offset in range(-3, 4):
        for x_offset in range(-3, 4):
            pixel_x = int(round(star.x)) + x_offset
            pixel_y = int(round(star.y)) + y_offset
            if pixel_x < 0 or pixel_x >= image_size or pixel_y < 0 or pixel_y >= image_size:
                continue
            weight = math.exp(-(x_offset * x_offset + y_offset * y_offset) / (2.0 * sigma * sigma))
            value = int(amplitude * weight)
            index = (pixel_y * image_size + pixel_x) * 3
            image[index] = min(255, image[index] + value)
            image[index + 1] = min(255, image[index + 1] + value)
            image[index + 2] = min(255, image[index + 2] + value)


def write_ppm(path: Path, image: list[int], image_size: int) -> None:
    """
    Writes an RGB P6 PPM image.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(f"P6\n{image_size} {image_size}\n255\n".encode("ascii"))
        file.write(bytes(image))


def write_truth(path: Path, rendered_stars: list[RenderedStar]) -> None:
    """
    Writes the rendered HR IDs and ideal pixel positions.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["hr_id", "x", "y", "magnitude"])
        for star in rendered_stars:
            writer.writerow([star.hr_id, f"{star.x:.3f}", f"{star.y:.3f}", f"{star.magnitude:.2f}"])


def main() -> None:
    """
    Renders a catalog-backed test image with known attitude for full pipeline testing.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=PROJECT_ROOT / "data" / "catalog.bin", type=Path)
    parser.add_argument("--output", default=PROJECT_ROOT / "outputs" / "catalog_render.ppm", type=Path)
    parser.add_argument("--truth", default=PROJECT_ROOT / "outputs" / "catalog_render_truth.csv", type=Path)
    parser.add_argument("--center-ra", default=(10 + 16 / 60 + 56 / 3600) * 15, type=float)
    parser.add_argument("--center-dec", default=-(59 + 51 / 60 + 22 / 3600), type=float)
    parser.add_argument("--image-size", default=877, type=int)
    parser.add_argument("--fov", default=10.0, type=float)
    parser.add_argument("--magnitude-limit", default=6.5, type=float)
    args = parser.parse_args()

    stars = load_catalog(args.catalog, args.magnitude_limit)
    rendered_stars = project_stars(stars, args.center_ra, args.center_dec, args.image_size, args.fov)
    image = [8] * (args.image_size * args.image_size * 3)
    for star in rendered_stars:
        add_star(image, args.image_size, star)
    write_ppm(args.output, image, args.image_size)
    write_truth(args.truth, rendered_stars)
    print(f"rendered_stars={len(rendered_stars)} output={args.output} truth={args.truth}")


if __name__ == "__main__":
    main()
