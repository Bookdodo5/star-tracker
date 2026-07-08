"""
Real-time star identification from a video file or webcam, in-process.

Per frame: capture -> (ctypes) star_live.identify_frame -> "RA=.. DEC=.. ROLL=.."
in degrees, or "NULL" when nothing solves. No subprocess, no PPM/CSV files: the
whole centroid -> TETRA chain runs inside one process via the star_live DLL, which
is loaded (with its database) exactly once.

OpenCV is only the frame source -- the portable core is identify_frame(), the same
entry point a firmware loop would call. Build the DLL first:
    cmake -S live -B live/build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
    cmake --build live/build-mingw

Usage:
    python scripts/live_identify.py --source 0                 # notebook camera (index 0)
    python scripts/live_identify.py --source path/to/video.avi # video file
    python scripts/live_identify.py --source screen            # capture the whole screen
    python scripts/live_identify.py --source screen --region 100,100,877,877  # a sub-region
    python scripts/live_identify.py --source video.avi --fov 17.75 --scale 0.5 --show

The notebook camera will print NULL constantly -- correct, it cannot see real stars.
Screen capture is handy for pointing the pipeline at Stellarium or a star image on screen.
"""
import argparse
import ctypes
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DLL = ROOT / "live" / "build-mingw" / "libstar_live.dll"


def load_lib():
    """Loads star_live.dll and declares the identify_frame signature."""
    if not DLL.exists():
        sys.exit(f"Missing {DLL} -- build it:\n"
                 "  cmake -S live -B live/build-mingw -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Release\n"
                 "  cmake --build live/build-mingw")
    lib = ctypes.CDLL(str(DLL))
    D = ctypes.POINTER(ctypes.c_double)
    lib.identify_frame.restype = ctypes.c_int
    lib.identify_frame.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_float,
        D, D, D, D, D, D, D,
        ctypes.c_int,
    ]
    lib.identify_frame_calibrate.restype = ctypes.c_int
    lib.identify_frame_calibrate.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_float,
        D, D, D, D, D, D, D,
        D, ctypes.c_int,
    ]
    if hasattr(lib, "identify_vectors"):
        lib.identify_vectors.restype = ctypes.c_int
        lib.identify_vectors.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            D, D, D, D, D, D, D,
        ]
    return lib


def solve_vectors(lib, xyz):
    """Solves attitude from observed unit vectors (brightest-first, shape (n,3)).
    Returns (ra, dec, roll, (qw,qx,qy,qz)) or None."""
    import numpy as np
    arr = np.ascontiguousarray(xyz, dtype=np.float32)
    n = arr.shape[0]
    ra, dec, roll, qw, qx, qy, qz = (ctypes.c_double() for _ in range(7))
    rc = lib.identify_vectors(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
                              ctypes.byref(ra), ctypes.byref(dec), ctypes.byref(roll),
                              ctypes.byref(qw), ctypes.byref(qx), ctypes.byref(qy), ctypes.byref(qz))
    if rc != 1:
        return None
    return (ra.value, dec.value, roll.value, (qw.value, qx.value, qy.value, qz.value))


def solve(lib, bgr, fov, morph=1):
    """Runs the in-process pipeline on one BGR frame. Returns (ra, dec, roll, (qw,qx,qy,qz)) or None.

    morph is the centroid morphological-open strength: 1 = satellite default,
    0 = camera (keep small/faint stars), N = repeat. See extract_centroids.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    h, w = rgb.shape[:2]
    ra, dec, roll, qw, qx, qy, qz = (ctypes.c_double() for _ in range(7))
    rc = lib.identify_frame(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            w, h, ctypes.c_float(fov),
                            ctypes.byref(ra), ctypes.byref(dec), ctypes.byref(roll),
                            ctypes.byref(qw), ctypes.byref(qx), ctypes.byref(qy), ctypes.byref(qz), morph)
    if rc != 1:
        return None
    return (ra.value, dec.value, roll.value, (qw.value, qx.value, qy.value, qz.value))


def calibrate_fov(lib, bgr, seed_fov, morph=1):
    """Recovers the true FOV from one frame via C self-calibration; returns (fov, attitude) or (None, None).

    attitude is (ra, dec, roll, (qw,qx,qy,qz)). The TETRA feature lookup is scale-invariant, so it
    finds the right tetrad even when seed_fov is far off; the matched catalog's true angles then pin
    the focal length directly. One C call replaces the old multi-solve FOV grid sweep. Use this once
    to lock a fixed camera/screen FOV.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    h, w = rgb.shape[:2]
    ra, dec, roll, qw, qx, qy, qz, fov_out = (ctypes.c_double() for _ in range(8))
    rc = lib.identify_frame_calibrate(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                      w, h, ctypes.c_float(seed_fov),
                                      ctypes.byref(ra), ctypes.byref(dec), ctypes.byref(roll),
                                      ctypes.byref(qw), ctypes.byref(qx), ctypes.byref(qy), ctypes.byref(qz),
                                      ctypes.byref(fov_out), morph)
    if rc != 1:
        return None, None
    return fov_out.value, (ra.value, dec.value, roll.value, (qw.value, qx.value, qy.value, qz.value))


def list_monitors():
    """Returns each monitor's bounds as (left, top, right, bottom), in virtual-desktop
    coordinates, via the Windows user32 API -- no extra dependency. Order matches
    EnumDisplayMonitors (monitor 1 is usually the primary)."""
    import ctypes
    from ctypes import wintypes

    monitors = []
    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
        ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def callback(hmon, hdc, lprect, lparam):
        r = lprect.contents
        monitors.append((r.left, r.top, r.right, r.bottom))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(callback), 0)
    return monitors


def screen_reader(region=None, monitor=None):
    """Returns a read() like cv2.VideoCapture: grabs the screen (or a region) as a BGR frame.

    monitor is a 1-based screen number (see --list-monitors); region is (x, y, w, h) in
    that monitor's local pixels, or None for the whole monitor. With neither, grabs the
    primary screen. Uses PIL.ImageGrab (already installed via Pillow), no extra dependency.
    """
    from PIL import ImageGrab

    ox, oy = 0, 0
    bbox = None
    if monitor is not None:
        mons = list_monitors()
        if not (1 <= monitor <= len(mons)):
            sys.exit(f"Monitor {monitor} not found; {len(mons)} detected. Use --list-monitors.")
        left, top, right, bottom = mons[monitor - 1]
        ox, oy = left, top
        bbox = (left, top, right, bottom)
    if region:
        bbox = (ox + region[0], oy + region[1], ox + region[0] + region[2], oy + region[1] + region[3])

    def read():
        # all_screens=True is required to reach monitors past the primary one
        rgb = np.asarray(ImageGrab.grab(bbox=bbox, all_screens=True))  # PIL gives RGB
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return read


def main():
    parser = argparse.ArgumentParser(description="Real-time in-process TETRA star identification")
    parser.add_argument("--source", default="0",
                        help="webcam index (e.g. 0), path to a video file, or 'screen'")
    parser.add_argument("--monitor", type=int, help="for --source screen: 1-based monitor number to capture")
    parser.add_argument("--list-monitors", action="store_true", help="print detected monitors and exit")
    parser.add_argument("--region", help="for --source screen: x,y,w,h sub-region within the chosen screen")
    parser.add_argument("--fov", type=float, default=17.75, help="horizontal field of view in degrees")
    parser.add_argument("--fov-search", action="store_true",
                        help="self-calibrate FOV from --fov as a seed (the scale-invariant TETRA match "
                             "recovers the true focal length in one solve), then lock it. "
                             "Use when the true FOV is unknown (fixed camera/screen).")
    parser.add_argument("--morph", type=int, default=1,
                        help="centroid morphological-open passes: 1 = satellite default (3x3 open), "
                             "0 = camera (skip the open so small/faint stars survive), N = repeat for noisy sensors.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="downsample each frame by this factor before centroiding (e.g. 0.5). "
                             "FOV is unchanged, so attitude stays valid; only centroid precision drops.")
    parser.add_argument("--cam-width", type=int, help="request this capture width from a webcam")
    parser.add_argument("--cam-height", type=int, help="request this capture height from a webcam")
    parser.add_argument("--quiet", action="store_true",
                        help="only print frames that solve (suppress NULL / no-solve lines)")
    parser.add_argument("--timing", action="store_true",
                        help="print per-frame centroid timing from the C pipeline (off by default). "
                             "Requires the DLL built with this change.")
    parser.add_argument("--show", action="store_true", help="show the video window with an attitude overlay")
    parser.add_argument("--save", help="write the first captured frame to this path (e.g. outputs/cap.ppm) and exit, "
                                       "so you can run it through the standalone centroid/identify pipeline")
    args = parser.parse_args()

    if args.timing:  # read once by the DLL's getenv on the first centroid call
        os.environ["STAR_CENTROID_TIMING"] = "1"

    if args.list_monitors:
        for i, (l, t, r, b) in enumerate(list_monitors(), 1):
            print(f"monitor {i}: {r - l}x{b - t} at ({l},{t})")
        return

    lib = load_lib()
    cap = None
    if args.source == "screen":
        region = tuple(int(v) for v in args.region.split(",")) if args.region else None
        read_frame = screen_reader(region, args.monitor)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"Could not open source: {args.source}")
        if isinstance(source, int):  # webcam: ask the driver for a lower native resolution
            if args.cam_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            if args.cam_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        read_frame = cap.read

    if args.save:
        ok, bgr = read_frame()
        if not ok:
            sys.exit("Could not read a frame to save")
        if args.scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(args.save, bgr)
        h, w = bgr.shape[:2]
        print(f"[live] saved {w}x{h} frame to {args.save}")
        if cap is not None:
            cap.release()
        return

    print(f"[live] source={args.source} fov={args.fov} scale={args.scale}  (Ctrl+C to stop)")
    # ponytail: 2 consecutive calibrations that agree on FOV required, to reject a lucky false lock.
    # Agreement is on recovered FOV (the camera intrinsic, constant) NOT attitude: the field may be
    # sweeping, so consecutive solves point elsewhere while the true FOV stays put.
    CALIB_CONFIRM = 2
    CALIB_AGREE_PCT = 0.05  # max fractional FOV difference between consecutive calibration results
    if args.fov_search:
        print(f"[live] FOV self-calibration from seed {args.fov}; will lock after {CALIB_CONFIRM} agreeing solves")
    locked = not args.fov_search
    calib_window = []  # (fov, att) from recent consecutive calibration successes
    frame_i, t_prev = 0, time.time()
    try:
        while True:
            ok, bgr = read_frame()
            if not ok:
                break  # end of video file (a webcam/screen keeps returning frames)
            frame_i += 1
            now = time.time()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            if args.scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            if not locked:  # self-calibrate to recover the true FOV; require CALIB_CONFIRM agreeing results
                found, att = calibrate_fov(lib, bgr, args.fov, morph=args.morph)
                if found:
                    if calib_window and abs(found - calib_window[-1][0]) > CALIB_AGREE_PCT * calib_window[-1][0]:
                        calib_window.clear()  # FOV disagreement: restart the confirmation window
                    calib_window.append((found, att))
                    if len(calib_window) < CALIB_CONFIRM:
                        if not args.quiet:
                            print(f"frame {frame_i:4d} | calibrating... ({len(calib_window)}/{CALIB_CONFIRM} agree)", flush=True)
                        continue
                    fov_locked = sum(f for f, _ in calib_window) / len(calib_window)
                    args.fov, locked = fov_locked, True
                    att = calib_window[-1][1]
                    print(f"[live] locked FOV = {fov_locked:.3f} deg (confirmed {CALIB_CONFIRM} frames)", flush=True)
                else:
                    if not args.quiet:
                        print(f"frame {frame_i:4d} | self-calibration found no solve; retrying next frame", flush=True)
                    continue
            else:
                att = solve(lib, bgr, args.fov, args.morph)
            if att:
                qw, qx, qy, qz = att[3]
                print(f"frame {frame_i:4d} | RA={att[0]:8.3f}  DEC={att[1]:8.3f}  ROLL={att[2]:8.3f}  "
                      f"Q=({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f})   ({fps:.1f} fps)", flush=True)
            elif not args.quiet:
                print(f"frame {frame_i:4d} | NULL                                   ({fps:.1f} fps)", flush=True)
            if args.show:
                label = f"RA={att[0]:.2f} DEC={att[1]:.2f} ROLL={att[2]:.2f}" if att else "NULL"
                cv2.putText(bgr, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0) if att else (0, 0, 255), 2)
                try:
                    cv2.imshow("live_identify", bgr)
                    if cv2.waitKey(1) == 27:  # Esc
                        break
                except cv2.error:
                    print("[live] --show: OpenCV has no GUI support; install opencv-python (not headless). Continuing without display.")
                    args.show = False
    except KeyboardInterrupt:
        print("\n[live] stopped")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
