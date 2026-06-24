"""
Still-image and video identification on the in-process library.

Three modes, all reusing the same DLL path as live_identify.py:
    python identify.py image  <img>     [--fov F] [--fov-search] [--morph M]
    python identify.py batch  <folder>  [--fov F] [--fov-search] [--morph M] [--glob '*.ppm']
    python identify.py video  <video>   [--fov F] [--fov-search] [--morph M] [--every N]

--fov-search recovers the true FOV from the first solve (and, in video mode, locks it
for the remaining frames). Attitude is printed as RA / DEC / roll in degrees.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2

import live_identify as L

ROOT = Path(__file__).resolve().parent


def _solve_frame(lib, bgr, fov, fov_search, morph):
    """Returns (attitude, fov_used). attitude is (ra,dec,roll) or None."""
    if fov_search:
        rec_fov, att = L.calibrate_fov(lib, bgr, fov, morph)
        return att, (rec_fov if rec_fov is not None else fov)
    return L.solve(lib, bgr, fov, morph), fov


def _fmt(att, fov):
    if att is None:
        return "NULL"
    return f"RA={att[0]:8.3f}  DEC={att[1]:8.3f}  ROLL={att[2]:8.3f}  (FOV={fov:.3f})"


def identify_image(lib, path, fov, fov_search, morph):
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"{path}: cannot read image")
        return False
    att, used = _solve_frame(lib, bgr, fov, fov_search, morph)
    print(f"{path}: {_fmt(att, used)}")
    return att is not None


def identify_batch(lib, folder, pattern, fov, fov_search, morph):
    files = sorted(glob.glob(str(Path(folder) / pattern)))
    if not files:
        print(f"no files matching {pattern} in {folder}")
        return
    solved = 0
    for f in files:
        if identify_image(lib, f, fov, fov_search, morph):
            solved += 1
    print(f"\nsolved {solved}/{len(files)}")


def identify_video(lib, path, fov, fov_search, morph, every):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"{path}: cannot open video")
        return
    locked = not fov_search  # once self-cal solves, lock the FOV and stop searching
    used = fov
    i = solved = shown = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % every:
            continue
        shown += 1
        if not locked:
            rec_fov, att = L.calibrate_fov(lib, bgr, fov, morph)
            if att is not None:
                used, locked = rec_fov, True
                print(f"[locked FOV = {used:.3f}]")
        else:
            att = L.solve(lib, bgr, used, morph)
        solved += att is not None
        print(f"frame {i:5d} | {_fmt(att, used)}")
    cap.release()
    print(f"\nsolved {solved}/{shown} sampled frames")


def main():
    p = argparse.ArgumentParser(description="Identify attitude from a still image, a folder, or a video")
    p.add_argument("mode", choices=["image", "batch", "video"])
    p.add_argument("path", help="image file, folder, or video file")
    p.add_argument("--fov", type=float, default=10.0, help="FOV seed in degrees")
    p.add_argument("--fov-search", action="store_true", help="recover the true FOV from the first solve")
    p.add_argument("--morph", type=int, default=0, help="centroid morph passes (0 = keep faint stars)")
    p.add_argument("--glob", default="*.ppm", help="batch mode: filename pattern")
    p.add_argument("--every", type=int, default=1, help="video mode: process every Nth frame")
    args = p.parse_args()

    if not L.DLL.exists():
        sys.exit(f"Missing {L.DLL} -- build it first (gui.py Build, or cmake build of live/).")
    lib = L.load_lib()

    if args.mode == "image":
        identify_image(lib, args.path, args.fov, args.fov_search, args.morph)
    elif args.mode == "batch":
        identify_batch(lib, args.path, args.glob, args.fov, args.fov_search, args.morph)
    else:
        identify_video(lib, args.path, args.fov, args.fov_search, args.morph, max(1, args.every))


if __name__ == "__main__":
    main()
