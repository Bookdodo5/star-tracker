"""
Overlay detected centroids (numbered by brightness rank, colored red->yellow:
red = brightest, yellow = dimmest) and projected catalog stars (white crosses + Vmag)
onto cached DSS field images. RA/DEC are parsed from the filename and the catalog is
projected at that true attitude (roll=0, north-up east-left convention).

    python benchmarks/mark_centroids.py                 # all cache/real_images/*_7x4_1400.png
    python benchmarks/mark_centroids.py --glob '*_700.png'
Outputs to outputs/marked/.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import live_identify as L  # noqa: E402
from src.star_tracker_core import load_db_catalog  # noqa: E402

FOV_W = 7.0
NAME_RE = re.compile(r"dss_ra([0-9.]+)_dec([+-][0-9.]+)_")


def rank_color(rank: int, n: int):
    """Red (brightest) -> yellow (dimmest): ramp the green channel up with rank. BGR."""
    g = int(255 * rank / max(1, n - 1))
    return (0, g, 255)


def mark(lib, cat_vecs, cat_vmag, img_path: Path, out_dir: Path):
    m = NAME_RE.search(img_path.name)
    if not m:
        return None
    ra_deg, dec_deg = float(m.group(1)), float(m.group(2))
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    out = bgr.copy()

    # detected centroids (already brightness-sorted: index 0 = brightest)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    ox = (ctypes.c_uint16 * 64)(); oy = (ctypes.c_uint16 * 64)()
    nd = lib.detect_centroids(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), w, h, 0, ox, oy, 64)

    # catalog projected at the field's true attitude (roll 0)
    r, d = math.radians(ra_deg), math.radians(dec_deg)
    east = np.array([-math.sin(r), math.cos(r), 0.0])
    north = np.array([-math.sin(d) * math.cos(r), -math.sin(d) * math.sin(r), math.cos(d)])
    bore = np.array([math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d)])
    foc = (w * 0.5) / math.tan(math.radians(FOV_W) / 2)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    es, ns, bs = cat_vecs @ east, cat_vecs @ north, cat_vecs @ bore
    for i in range(len(cat_vecs)):
        if bs[i] <= 0:
            continue
        x = cx - foc * es[i] / bs[i]; y = cy - foc * ns[i] / bs[i]
        if -10 <= x < w + 10 and -10 <= y < h + 10:
            xi, yi = int(round(x)), int(round(y))
            cv2.drawMarker(out, (xi, yi), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.putText(out, f"m{cat_vmag[i]:.1f}", (xi + 8, yi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for rank in range(nd):
        c = rank_color(rank, nd)
        x, y = int(ox[rank]), int(oy[rank])
        cv2.circle(out, (x, y), 11, c, 2)
        cv2.putText(out, f"{rank + 1}", (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)

    cv2.putText(out, "detected #=brightness rank (1=brightest)  red->yellow=bright->dim   white X=catalog Vmag",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    for k in range(nd):
        cv2.rectangle(out, (10 + k * 16, 40), (10 + k * 16 + 15, 55), rank_color(k, nd), -1)

    out_dir.mkdir(parents=True, exist_ok=True)
    op = out_dir / f"marked_{img_path.stem}.png"
    cv2.imwrite(str(op), out)
    return op, nd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="*_7x4_1400.png", help="image filename pattern in cache/real_images")
    p.add_argument("--out", default="outputs/marked")
    args = p.parse_args()

    lib = L.load_lib()
    lib.detect_centroids.restype = ctypes.c_int
    lib.detect_centroids.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
    cat = load_db_catalog(7.5)
    ra = np.radians(cat.RA_deg.to_numpy()); dec = np.radians(cat.DEC_deg.to_numpy())
    cat_vecs = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    cat_vmag = cat.Vmag.to_numpy()

    imgs = sorted((ROOT / "cache" / "real_images").glob(args.glob))
    print(f"marking {len(imgs)} images -> {args.out}")
    for img in imgs:
        res = mark(lib, cat_vecs, cat_vmag, img, ROOT / args.out)
        if res:
            print(f"  {res[0].name}  (detected {res[1]})", flush=True)


if __name__ == "__main__":
    main()
