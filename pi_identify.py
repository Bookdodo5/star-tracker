#!/usr/bin/env python3
"""
Pi live star identification: Sentech GigE camera → centroid_extract → TETRA → RA/DEC/ROLL.

Subprocess approach — no C changes: stapipy grabs frames, this script writes PPM to
a tmpdir, shells out to centroid_extract then demo_centroid_compare, and parses stdout.

Usage:
    python pi_identify.py --fov 10
    python pi_identify.py --fov 10 --morph 0          # real night-sky point-source stars
    python pi_identify.py --fov 10 --scale 0.5        # downscale 2× before centroiding
    python pi_identify.py --fov 10 --fov-search       # sweep FOV ×0.5–×2 and lock on first solve
    python pi_identify.py --fov 10 --frames 1         # single shot
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import stapipy as st

ROOT = os.path.expanduser("~/src/star-tracker")
CENTROID_BIN = os.path.join(ROOT, "centroid", "build-pi", "centroid_extract")
IDENTIFY_BIN = os.path.join(ROOT, "identifier", "build-pi", "demo_centroid_compare")

# FOV sweep: try this many evenly-spaced values from seed×LO to seed×HI
_FOV_STEPS = 20
_FOV_LO, _FOV_HI = 0.5, 2.0


def _check_bins():
    missing = [b for b in (CENTROID_BIN, IDENTIFY_BIN) if not os.path.isfile(b)]
    if missing:
        sys.exit("Missing binaries — build on Pi first:\n"
                 + "\n".join(f"  {b}" for b in missing))


def _resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        h, w = gray.shape
        img = Image.fromarray(gray).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.array(img)


def _to_ppm(data: bytes, pfi, width: int, height: int, scale: float):
    """Convert raw Sentech buffer → P6 PPM bytes + (out_width, out_height)."""
    if pfi.each_component_total_bit_count > 8:
        arr = np.frombuffer(data, np.uint16)
        arr = (arr >> (pfi.each_component_valid_bit_count - 8)).astype(np.uint8)
    else:
        arr = np.frombuffer(data, np.uint8)

    if pfi.is_mono:
        gray = arr.reshape(height, width)
    else:
        arr = arr.reshape(height, width)
        q = ((arr[0::2, 0::2].astype(np.uint16) + arr[0::2, 1::2]
              + arr[1::2, 0::2] + arr[1::2, 1::2]) // 4).astype(np.uint8)
        gray = q.repeat(2, axis=0).repeat(2, axis=1)[:height, :width]

    if scale != 1.0:
        gray = _resize_gray(gray, scale)

    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)
    return f"P6\n{w} {h}\n255\n".encode() + rgb.tobytes(), w, h


def _parse_attitude(stdout: str):
    for line in stdout.splitlines():
        if "attitude_ra_deg=" in line:
            try:
                kv = dict(tok.split("=") for tok in line.split() if "=" in tok)
                return (float(kv["attitude_ra_deg"]),
                        float(kv["attitude_dec_deg"]),
                        float(kv["attitude_roll_deg"]))
            except (KeyError, ValueError):
                pass
    return None


def _run_identify(csv_path: str, width: int, height: int, fov: float):
    r = subprocess.run([IDENTIFY_BIN, csv_path, str(width), str(height), str(fov)],
                       capture_output=True, text=True)
    return _parse_attitude(r.stdout)


def _centroid(ppm_bytes: bytes, morph: int, tmp: str):
    """Write PPM and run centroid_extract. Returns csv_path or None on error."""
    ppm = os.path.join(tmp, "frame.ppm")
    csv = os.path.join(tmp, "stars.csv")
    with open(ppm, "wb") as f:
        f.write(ppm_bytes)
    r = subprocess.run([CENTROID_BIN, ppm, csv, str(morph)], capture_output=True)
    if r.returncode != 0:
        print("[centroid]", r.stderr.decode(errors="replace").strip(), file=sys.stderr)
        return None
    return csv


def identify_frame(ppm_bytes: bytes, width: int, height: int, fov: float, morph: int):
    """Single-FOV identification. Returns (ra, dec, roll) or None."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = _centroid(ppm_bytes, morph, tmp)
        if csv is None:
            return None
        return _run_identify(csv, width, height, fov)


def identify_frame_fov_search(ppm_bytes: bytes, width: int, height: int,
                               seed_fov: float, morph: int):
    """Sweep FOVs seed×LO…seed×HI; return ((ra,dec,roll), locked_fov) or (None, None)."""
    fovs = [seed_fov * (_FOV_LO + (_FOV_HI - _FOV_LO) * i / (_FOV_STEPS - 1))
            for i in range(_FOV_STEPS)]
    with tempfile.TemporaryDirectory() as tmp:
        csv = _centroid(ppm_bytes, morph, tmp)
        if csv is None:
            return None, None
        for fov in fovs:
            att = _run_identify(csv, width, height, fov)
            if att:
                return att, fov
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Sentech GigE → TETRA attitude")
    parser.add_argument("--fov", type=float, default=10.0,
                        help="horizontal FOV in degrees (seed if --fov-search)")
    parser.add_argument("--fov-search", action="store_true",
                        help="sweep FOV ×0.5–×2.0 around seed, lock on first solve")
    parser.add_argument("--morph", type=int, default=0,
                        help="centroid morph passes: 0=real stars (default), 1+=satellite blobs")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="downscale frame before centroiding (e.g. 0.5 = half resolution)")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames to process (0 = infinite, Ctrl+C to stop)")
    args = parser.parse_args()

    _check_bins()

    st.initialize()
    st_system = st.create_system()
    st_device = st_system.create_first_device()
    print(f"Camera: {st_device.info.display_name}")
    print(f"FOV={args.fov}°{'  fov-search=ON' if args.fov_search else ''}  "
          f"morph={args.morph}  scale={args.scale}  (Ctrl+C to stop)")

    st_datastream = st_device.create_datastream()
    grab_count = args.frames if args.frames > 0 else 2**62
    st_datastream.start_acquisition(grab_count)
    st_device.acquisition_start()

    frame_i = 0
    t0 = time.monotonic()
    fov = args.fov
    fov_locked = not args.fov_search

    try:
        while st_datastream.is_grabbing:
            with st_datastream.retrieve_buffer() as st_buffer:
                if not st_buffer.info.is_image_present:
                    continue
                st_image = st_buffer.get_image()
                w, h = st_image.width, st_image.height
                pfi = st.get_pixel_format_info(st_image.pixel_format)
                if not (pfi.is_mono or pfi.is_bayer):
                    print("[warn] unsupported pixel format, skipping")
                    continue
                ppm, w, h = _to_ppm(st_image.get_image_data(), pfi, w, h, args.scale)

            frame_i += 1
            fps = frame_i / max(time.monotonic() - t0, 1e-6)

            if not fov_locked:
                att, found_fov = identify_frame_fov_search(ppm, w, h, fov, args.morph)
                if att:
                    fov = found_fov
                    fov_locked = True
                    print(f"[fov-search] locked FOV = {fov:.3f}°")
                else:
                    print(f"frame {frame_i:4d} | NULL (searching FOV {args.fov * _FOV_LO:.1f}–"
                          f"{args.fov * _FOV_HI:.1f}°)   ({fps:.2f} fps)")
                    continue

            else:
                att = identify_frame(ppm, w, h, fov, args.morph)

            if att:
                print(f"frame {frame_i:4d} | RA={att[0]:9.4f}  DEC={att[1]:8.4f}  "
                      f"ROLL={att[2]:8.3f}  ({fps:.2f} fps)")
            else:
                print(f"frame {frame_i:4d} | NULL                                         "
                      f"({fps:.2f} fps)")

    except KeyboardInterrupt:
        print("\n[pi_identify] stopped")
    finally:
        st_device.acquisition_stop()
        st_datastream.stop_acquisition()


if __name__ == "__main__":
    main()
