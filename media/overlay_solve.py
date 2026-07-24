"""
Solve one image (with FOV search) and draw a diagnostic overlay:

  * cyan  ring   = TETRA detected centroid (what the solver saw)
  * red   +      = Tycho-2 catalog star projected through the solved attitude (in FOV)
  * green ring   = cross-checked = a detected centroid that lands on a catalog star

Everything is reconstructed from the DLL's own solve (attitude quaternion + recovered
FOV) and its own centroid detector, projected back with the exact inverse of the C
camera model (`pixel_to_unit_vector` = ((cx-x)/f,(cy-y)/f,1)), so no roll/chirality
guessing. Run:

    python media/overlay_solve.py cache/FVIDEO/UntitledNULL.png --fov 10 --fov-search
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live_identify as L  # noqa: E402  (needs ROOT on path)


def quat_to_matrix(qw, qx, qy, qz):
    """catalog->camera rotation matrix from a unit quaternion (w,x,y,z)."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_catalog(path, mag_limit):
    """Tycho-2 rows (RA,DEC,Vmag) as unit vectors + mags, brighter than mag_limit."""
    vecs, mags = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mag = float(row["Vmag"])
            if mag > mag_limit:
                continue
            ra, dec = np.radians(float(row["RA_deg"])), np.radians(float(row["DEC_deg"]))
            vecs.append((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
            mags.append(mag)
    return np.array(vecs), np.array(mags)


def project(vecs, mags, rot, width, height, fov_deg):
    """Project catalog unit vectors to pixels via the inverse C camera model. Returns (xs,ys,mags)."""
    focal = (width * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    cam = vecs @ rot.T                      # catalog->camera
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = cx - focal * cam[:, 0] / z
        ys = cy - focal * cam[:, 1] / z
    inb = (z > 0) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return xs[inb], ys[inb], mags[inb]


def detect_centroids(lib, bgr, morph):
    """Runs the DLL's own centroid detector; returns list of (x,y) the solver used."""
    lib.detect_centroids.restype = ctypes.c_int
    lib.detect_centroids.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_uint16),
                                     ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    h, w = rgb.shape[:2]
    out_x = (ctypes.c_uint16 * 64)()
    out_y = (ctypes.c_uint16 * 64)()
    n = lib.detect_centroids(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), w, h, morph,
                             out_x, out_y, 64)
    return [(out_x[i], out_y[i]) for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--fov", type=float, default=10.0)
    p.add_argument("--fov-search", action="store_true")
    p.add_argument("--morph", type=int, default=0, help="0 keeps faint real stars")
    p.add_argument("--mag-limit", type=float, default=7.5)
    p.add_argument("--match-px", type=float, default=8.0, help="centroid<->catalog match tolerance")
    p.add_argument("--catalog", default=str(ROOT / "data" / "tycho2.csv"))
    p.add_argument("--out", default=str(ROOT / "outputs" / "overlay_solve.png"))
    args = p.parse_args()

    lib = L.load_lib()
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = bgr.shape[:2]

    if args.fov_search:
        fov, att = L.calibrate_fov(lib, bgr, args.fov, args.morph)
    else:
        att, fov = L.solve(lib, bgr, args.fov, args.morph), args.fov
    if att is None:
        raise SystemExit("no solve")
    ra, dec, roll, (qw, qx, qy, qz) = att
    print(f"SOLVE  RA={ra:.3f}  DEC={dec:.3f}  ROLL={roll:.3f}  FOV={fov:.3f}")

    rot = quat_to_matrix(qw, qx, qy, qz)
    vecs, mags = load_catalog(args.catalog, args.mag_limit)
    xs, ys, cmags = project(vecs, mags, rot, w, h, fov)
    centroids = detect_centroids(lib, bgr, args.morph)
    print(f"catalog-in-FOV={len(xs)}  centroids={len(centroids)}")

    # draw on a brightened copy so faint stars stay visible under the markers
    canvas = cv2.convertScaleAbs(bgr, alpha=1.6, beta=0)

    # red + for every catalog star in the FOV (size by brightness)
    for x, y, m in zip(xs, ys, cmags):
        r = int(np.clip(9 - m, 3, 9))
        cv2.drawMarker(canvas, (int(round(x)), int(round(y))), (0, 0, 255),
                       cv2.MARKER_CROSS, r * 2, 1)

    # detected centroids: green if a catalog star is within match-px, else cyan
    cat_pts = np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2))
    matched = 0
    for (cxp, cyp) in centroids:
        color, ok = (255, 255, 0), False
        if len(cat_pts):
            d = np.hypot(cat_pts[:, 0] - cxp, cat_pts[:, 1] - cyp)
            if d.min() <= args.match_px:
                color, ok = (0, 255, 0), True
        cv2.circle(canvas, (cxp, cyp), 10, color, 2)
        matched += ok
    print(f"cross-checked (centroid on catalog star)={matched}/{len(centroids)}")

    # legend + attitude text
    lines = [
        f"RA={ra:.2f} DEC={dec:.2f} ROLL={roll:.2f} FOV={fov:.2f}",
        f"green=cross-checked({matched})  cyan=detected  red=catalog(<={args.mag_limit:g})",
    ]
    for i, t in enumerate(lines):
        cv2.putText(canvas, t, (8, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
