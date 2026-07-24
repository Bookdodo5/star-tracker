"""
Overlays the detector's centroids on every frame of a video and writes a new video.

Uses the same C centroid detector the solver uses (detect_centroids in
libstar_live.dll), so the labels are exactly what the pipeline sees -- not a
Python reimplementation.

    python benchmarks/label_centroids_video.py cache/cam.mp4 outputs/cam_centroids.mp4 --morph 0
"""
import argparse
import ctypes
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLL = ROOT / "live" / "build-mingw" / "libstar_live.dll"
MAX_OUT = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--morph", type=int, default=0,
                    help="centroid morphological-open passes (0 = camera, keep faint stars)")
    args = ap.parse_args()

    lib = ctypes.CDLL(str(DLL))
    lib.detect_centroids.restype = ctypes.c_int
    lib.detect_centroids.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int,
    ]

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    out_x = (ctypes.c_uint16 * MAX_OUT)()
    out_y = (ctypes.c_uint16 * MAX_OUT)()
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        n = lib.detect_centroids(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                 w, h, args.morph, out_x, out_y, MAX_OUT)
        for i in range(n):
            x, y = int(out_x[i]), int(out_y[i])
            cv2.circle(bgr, (x, y), 10, (0, 255, 0), 1)
            cv2.putText(bgr, str(i), (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"frame {frame_idx} | {n} centroids", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        out.write(bgr)
        frame_idx += 1
        if total and frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()
    out.release()
    print(f"wrote {args.output} ({frame_idx} frames)")


if __name__ == "__main__":
    main()
