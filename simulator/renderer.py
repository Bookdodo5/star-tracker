"""
Star-field renderer for the simulator.

Uses the *same* pinhole projection the C identifier expects — the camera basis and pixel
math are taken from ``tools/render_catalog_test_image.py`` (one source of truth for the
camera model / chirality), but the per-star loop is replaced with a single numpy matmul so
it stays fast on the ~26k-star Tycho-2 catalog at video rate.

Catalog: **Tycho-2** (``data/tycho2.csv``, columns ``HR_clean,RA_deg,DEC_deg,Vmag``) — the
same catalog the runtime DB is built from, so the displayed field matches what the tracker
can identify. Default display magnitude limit 7.5 (the DB member limit).

Brightness: physics-based Pogson (flux ∝ 10^(-0.4·mag)) through a display ``gamma`` so faint
stars stay above the centroid floor. Roll and configurable aberrations (noise/blur/streak)
are applied before JPEG encode.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
import render_catalog_test_image as R  # noqa: E402  (camera_basis + unit_vector, reused)

from .state import DEFAULT_CONFIG  # noqa: E402

FLOOR = 90.0        # min amplitude for a drawn star (keeps faint stars centroid-able)
MAG_REF = 2.0       # magnitude at which PSF sigma is config["blob_size"]
SIGMA_MIN = 0.6     # smallest PSF sigma regardless of config (faint stars still centroid-able)


def pogson_amplitude(mag, config: dict):
    """
    Maps magnitude(s) to a pixel amplitude via Pogson flux through a display gamma.

    ``flux_norm = 10^(-0.4*(mag - MAG_REF))`` (1.0 at MAG_REF), then
    ``amp = saturation_cap * (gain*flux_norm)^(1/gamma)`` clamped to [FLOOR, saturation_cap].
    Accepts a scalar or a numpy array. The un-clamped region follows the true flux ratio.
    """
    mag = np.asarray(mag, dtype=float)
    flux_norm = np.power(10.0, -0.4 * (mag - MAG_REF))
    amp = config["saturation_cap"] * np.power(config["gain"] * flux_norm, 1.0 / config["gamma"])
    return np.clip(amp, FLOOR, config["saturation_cap"])


class Renderer:
    """Renders a Tycho-2 star field for a given ``(ra, dec, roll)`` attitude as JPEG bytes."""

    def __init__(self, image_size: int = 877, fov_deg: float = 10.0,
                 magnitude_limit: float = 7.5, catalog_path: Path | None = None):
        self.image_size = image_size
        self.fov_deg = fov_deg
        catalog_path = catalog_path or (PROJECT_ROOT / "data" / "tycho2.csv")
        ids, ras, decs, mags, vecs = [], [], [], [], []
        with open(catalog_path, newline="") as f:
            for row in csv.DictReader(f):
                mag = float(row["Vmag"])
                if mag > magnitude_limit:
                    continue
                ra, dec = float(row["RA_deg"]), float(row["DEC_deg"])
                ids.append(int(float(row["HR_clean"])))
                ras.append(ra); decs.append(dec); mags.append(mag)
                vecs.append(R.unit_vector(ra, dec))
        self._ids = np.array(ids)
        self._ra = np.array(ras)
        self._dec = np.array(decs)
        self._mag = np.array(mags)
        self._vecs = np.array(vecs)  # (N, 3) catalog unit vectors
        self._id_to_row = {hr: i for i, hr in enumerate(ids)}

    def hr_lookup(self, hr_id: int) -> tuple[float, float]:
        """Returns ``(ra_deg, dec_deg)`` for a Tycho reindexed id (``point_at HR<id>``)."""
        row = self._id_to_row.get(hr_id)
        if row is None:
            raise ValueError(f"id {hr_id} not in catalog (mag limit)")
        return float(self._ra[row]), float(self._dec[row])

    def _project(self, ra: float, dec: float, roll_deg: float, fov_deg: float | None = None):
        """Vectorized pinhole projection → (xs, ys, mags) of in-frame stars. Matches R.project_stars.

        ``fov_deg`` overrides the construction FOV (the display FOV is live-adjustable for
        alignment); ``observed_vectors`` leaves it None so the headless path stays fixed.
        """
        fov_deg = self.fov_deg if fov_deg is None else fov_deg
        basis = np.array(R.camera_basis(ra, dec))          # rows: camera_x, camera_y, camera_z
        cam = self._vecs @ basis.T                         # (N, 3): cam_x, cam_y, cam_z
        z = cam[:, 2]
        focal = (self.image_size * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        center = (self.image_size - 1) * 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = focal * cam[:, 0] / z + center
            ys = focal * cam[:, 1] / z + center
        if roll_deg % 360.0 != 0.0:
            theta = math.radians(roll_deg)
            dx, dy = xs - center, ys - center
            xs = center + dx * math.cos(theta) - dy * math.sin(theta)
            ys = center + dx * math.sin(theta) + dy * math.cos(theta)
        inb = (z > 0) & (xs >= 2) & (xs < self.image_size - 2) & (ys >= 2) & (ys < self.image_size - 2)
        return xs[inb], ys[inb], self._mag[inb]

    def observed_vectors(self, ra: float, dec: float, roll: float = 0.0,
                         pixel_noise: float = 0.0, seed: int | None = None) -> np.ndarray:
        """
        Camera-frame unit vectors of the in-frame stars, brightest first — the headless
        equivalent of what the tracker sees, for feeding ``solve_vectors`` (no image/camera).

        Goes attitude → pixels (the correct projection) → the C camera model's
        ``pixel_to_unit_vector`` ((cx-x)/f, (cy-y)/f, 1), so the convention matches the real
        pipeline exactly. ``pixel_noise`` (px, gaussian) injects centroid positional error.
        """
        xs, ys, mags = self._project(ra, dec, roll)
        order = np.argsort(mags)          # brightest (lowest mag) first
        xs, ys = xs[order], ys[order]
        if pixel_noise > 0.0:
            rng = np.random.default_rng(seed)
            xs = xs + rng.normal(0.0, pixel_noise, xs.shape)
            ys = ys + rng.normal(0.0, pixel_noise, ys.shape)
        focal = (self.image_size * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)
        center = (self.image_size - 1) * 0.5
        vx = (center - xs) / focal
        vy = (center - ys) / focal
        vz = np.ones_like(vx)
        vecs = np.stack([vx, vy, vz], axis=1)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def _draw(self, img, xs, ys, mags, config: dict) -> None:
        """Adds a small gaussian blob per star; brighter stars are larger (PSF bloom)."""
        amps = pogson_amplitude(mags, config)
        base_sigma = config.get("blob_size", DEFAULT_CONFIG["blob_size"])
        bloom = config.get("blob_scale", DEFAULT_CONFIG["blob_scale"])
        sigmas = np.maximum(SIGMA_MIN, base_sigma + bloom * (MAG_REF - mags))
        for x, y, amp, sigma in zip(xs, ys, amps, sigmas):
            radius = max(2, int(math.ceil(3 * sigma)))
            x0, y0 = int(round(x)), int(round(y))
            xlo, xhi = max(0, x0 - radius), min(self.image_size, x0 + radius + 1)
            ylo, yhi = max(0, y0 - radius), min(self.image_size, y0 + radius + 1)
            gx = np.arange(xlo, xhi) - x
            gy = (np.arange(ylo, yhi) - y)[:, None]
            blob = amp * np.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma))
            region = img[ylo:yhi, xlo:xhi].astype(np.float32) + blob[..., None]
            img[ylo:yhi, xlo:xhi] = np.clip(region, 0, 255).astype(np.uint8)

    def _aberrations(self, img, config: dict):
        """Applies streak → defocus blur → sensor noise (each off when its param is 0)."""
        streak_len = config.get("streak_len", 0.0)
        if streak_len >= 1.0:
            length = int(streak_len)
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2, :] = 1.0 / length
            kernel = _rotate_kernel(kernel, config.get("streak_angle", 0.0))
            img = cv2.filter2D(img, -1, kernel)
        blur_sigma = config.get("blur_sigma", 0.0)
        if blur_sigma > 0.0:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        noise_sigma = config.get("noise_sigma", 0.0)
        if noise_sigma > 0.0:
            noise = np.random.normal(0.0, noise_sigma, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def render(self, ra: float, dec: float, roll: float = 0.0,
               roll_sign: float = 1.0, config: dict | None = None) -> bytes:
        """Renders the field at ``(ra, dec, roll)`` with ``config`` and returns JPEG bytes."""
        # With no config, render at the constructor's FOV — NOT DEFAULT_CONFIG's fixed 10°, which
        # would silently ignore Renderer(fov_deg=…) and render every field at 10°. The live path
        # always passes an explicit config (seeded from --fov), so this only affects direct calls.
        config = config or dict(DEFAULT_CONFIG, fov_deg=self.fov_deg)
        fov = config.get("fov_deg", self.fov_deg)
        img = np.full((self.image_size, self.image_size, 3), 8, np.uint8)
        xs, ys, mags = self._project(ra, dec, roll_sign * roll, fov)
        self._draw(img, xs, ys, mags, config)
        img = self._aberrations(img, config)
        self._overlay(img, config, fov)  # alignment aids stay crisp (drawn after aberrations)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return jpg.tobytes()

    def _overlay(self, img, config: dict, fov_deg: float) -> None:
        """Draws the alignment aids (all off by default, each gated by its config toggle):

        - ``grid``: gnomonic angular grid at ``grid_spacing_deg`` off boresight — an angular ruler.
        - ``guide``: the camera-frame rectangle at ``guide_fov_deg`` × ``guide_aspect``. When the
          display FOV is wider than the camera FOV this sits inside the field (overscan), giving a
          target to fill: aim the camera so it sees only stars, no rectangle edge.
        - ``crosshair``: a small boresight cross at frame centre.

        Colours are BGR (OpenCV). These are for the human eye during setup — the tracker's
        centroider would read the lines as stars, so toggle them off before scoring.
        """
        if not (config.get("grid") or config.get("guide") or config.get("crosshair")):
            return
        size = self.image_size
        focal = (size * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        center = (size - 1) * 0.5
        if config.get("grid"):
            spacing = max(0.25, config.get("grid_spacing_deg", 2.0))
            k = 1
            while True:
                off = focal * math.tan(math.radians(k * spacing))
                if off >= size:  # this ring and every wider one is off-screen
                    break
                for signed in (off, -off):
                    p = int(round(center + signed))
                    if 0 <= p < size:
                        cv2.line(img, (p, 0), (p, size - 1), (70, 100, 70), 1)
                        cv2.line(img, (0, p), (size - 1, p), (70, 100, 70), 1)
                k += 1
            c = int(round(center))  # brighter boresight axes
            cv2.line(img, (c, 0), (c, size - 1), (95, 140, 95), 1)
            cv2.line(img, (0, c), (size - 1, c), (95, 140, 95), 1)
        if config.get("guide"):
            gfov = max(0.1, config.get("guide_fov_deg", fov_deg))
            aspect = max(0.1, config.get("guide_aspect", 4.0 / 3.0))
            half_w = focal * math.tan(math.radians(gfov) * 0.5)
            half_h = half_w / aspect
            cv2.rectangle(img, (int(round(center - half_w)), int(round(center - half_h))),
                          (int(round(center + half_w)), int(round(center + half_h))),
                          (30, 160, 230), 1)  # amber
        if config.get("crosshair"):
            c = int(round(center))
            arm = max(6, size // 40)
            cv2.line(img, (c - arm, c), (c + arm, c), (220, 200, 120), 1)  # cyan
            cv2.line(img, (c, c - arm), (c, c + arm), (220, 200, 120), 1)


def flash_jpeg(image_size: int, color: tuple[int, int, int]) -> bytes:
    """Full-frame solid-color JPEG for the sync visual check."""
    img = np.zeros((image_size, image_size, 3), np.uint8)
    img[:, :] = (color[2], color[1], color[0])  # OpenCV is BGR
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


def _rotate_kernel(kernel, angle_deg: float):
    """Rotates a square kernel about its centre (for a directional streak)."""
    if angle_deg % 360.0 == 0.0:
        return kernel
    size = kernel.shape[0]
    m = cv2.getRotationMatrix2D((size / 2.0 - 0.5, size / 2.0 - 0.5), angle_deg, 1.0)
    rotated = cv2.warpAffine(kernel, m, (size, size))
    total = rotated.sum()
    return rotated / total if total > 0 else kernel


def _demo() -> None:
    """Self-check: Pogson flux ratio (AC4), config changes brightness (AC5), aberrations (AC6)."""
    cfg = dict(DEFAULT_CONFIG)
    # AC4: un-clamped flux ratio mag4 vs mag6 is 10^0.8 ≈ 6.31 (gamma=1 = no display curve; high
    # saturation_cap so both stay between FLOOR and the cap).
    lin = dict(cfg, gamma=1.0, gain=1.0, saturation_cap=10000.0)
    ratio = float(pogson_amplitude(4.0, lin) / pogson_amplitude(6.0, lin))
    assert abs(ratio - 6.3096) < 0.01, ratio
    # Blob size scales with magnitude across the whole range, not just brighter than MAG_REF.
    bs, bl = cfg["blob_size"], cfg["blob_scale"]
    s2, s4, s8 = (max(SIGMA_MIN, bs + bl * (MAG_REF - m)) for m in (2.0, 4.0, 8.0))
    assert s2 > s4 > s8 == SIGMA_MIN, (s2, s4, s8)
    # AC5: different gain/gamma give different mean intensity on the same field.
    r = Renderer(image_size=200, fov_deg=10.0, magnitude_limit=7.5)
    import numpy as _np
    a = _np.frombuffer(cv2.imdecode(_np.frombuffer(r.render(83.8, -5.4, config=dict(cfg, gain=1.0)), _np.uint8), 1), _np.uint8)
    b = _np.frombuffer(cv2.imdecode(_np.frombuffer(r.render(83.8, -5.4, config=dict(cfg, gain=0.2)), _np.uint8), 1), _np.uint8)
    assert a.mean() != b.mean(), (a.mean(), b.mean())
    # AC6: zero aberration = identity; noise changes the frame.
    clean = r.render(83.8, -5.4, config=cfg)
    noised = r.render(83.8, -5.4, config=dict(cfg, noise_sigma=25.0))
    assert clean == r.render(83.8, -5.4, config=cfg), "zero-aberration render not deterministic"
    assert noised != clean, "noise had no effect"
    assert r.render(83.8, -5.4, config=dict(cfg, blob_size=2.5)) != clean, "blob_size had no effect"
    assert r.render(83.8, -5.4, config=dict(cfg, blob_scale=0.4)) != clean, "blob_scale had no effect"
    # Alignment overlays: each toggle must change the frame; runtime FOV changes projection.
    for key, val in (("grid", 1.0), ("guide", 1.0), ("crosshair", 1.0)):
        assert r.render(83.8, -5.4, config=dict(cfg, **{key: val})) != clean, f"{key} overlay had no effect"
    assert r.render(83.8, -5.4, config=dict(cfg, fov_deg=20.0)) != clean, "runtime FOV had no effect"
    print(f"renderer.py self-check passed ({len(r._ids)} Tycho stars ≤7.5)")


if __name__ == "__main__":
    _demo()
