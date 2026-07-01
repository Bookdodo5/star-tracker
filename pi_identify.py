#!/usr/bin/env python3
"""
Pi live star identification: Sentech GigE camera → centroid_extract → TETRA → RA/DEC/ROLL.

Subprocess approach — no C changes: stapipy grabs frames, this script writes PPM to
a tmpdir, shells out to centroid_extract then demo_centroid_compare, and parses stdout.

Usage:
    python pi_identify.py --fov 10
    python pi_identify.py --fov 10 --morph 0    # real night-sky (point-source stars)
    python pi_identify.py --fov 10 --frames 1   # single shot
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


def _check_bins():
    missing = [b for b in (CENTROID_BIN, IDENTIFY_BIN) if not os.path.isfile(b)]
    if missing:
        sys.exit(
            "Missing binaries — build on Pi first:\n"
            + "\n".join(f"  {b}" for b in missing)
        )


def _to_ppm_bytes(data: bytes, pfi, width: int, height: int) -> bytes:
    """Convert raw Sentech buffer to an 8-bit binary P6 PPM byte string."""
    if pfi.each_component_total_bit_count > 8:
        arr = np.frombuffer(data, np.uint16)
        arr = (arr >> (pfi.each_component_valid_bit_count - 8)).astype(np.uint8)
    else:
        arr = np.frombuffer(data, np.uint8)

    if pfi.is_mono:
        gray = arr.reshape(height, width)
    else:
        # bayer: average 2×2 quads → gray, repeat pixels back to full resolution
        arr = arr.reshape(height, width)
        q = ((arr[0::2, 0::2].astype(np.uint16) + arr[0::2, 1::2]
              + arr[1::2, 0::2] + arr[1::2, 1::2]) // 4).astype(np.uint8)
        gray = q.repeat(2, axis=0).repeat(2, axis=1)[:height, :width]

    rgb = np.stack([gray, gray, gray], axis=-1)  # centroid_extract expects P6
    return f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes()


def _parse_attitude(stdout: str):
    """Extract (ra, dec, roll) from demo_centroid_compare output, or None."""
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


def identify_frame(ppm_bytes: bytes, width: int, height: int, fov: float, morph: int):
    """Returns (ra, dec, roll) or None. Uses tmp files; releases camera buffer first."""
    with tempfile.TemporaryDirectory() as tmp:
        ppm = os.path.join(tmp, "frame.ppm")
        csv = os.path.join(tmp, "stars.csv")
        with open(ppm, "wb") as f:
            f.write(ppm_bytes)
        r = subprocess.run([CENTROID_BIN, ppm, csv, str(morph)], capture_output=True)
        if r.returncode != 0:
            print("[centroid]", r.stderr.decode(errors="replace").strip(), file=sys.stderr)
            return None
        r = subprocess.run([IDENTIFY_BIN, csv, str(width), str(height), str(fov)],
                           capture_output=True, text=True)
        return _parse_attitude(r.stdout)


def main():
    parser = argparse.ArgumentParser(description="Sentech GigE → TETRA attitude")
    parser.add_argument("--fov", type=float, default=10.0,
                        help="horizontal FOV in degrees")
    parser.add_argument("--morph", type=int, default=0,
                        help="centroid morph passes: 0=real stars (default), 1+=satellite blobs")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames to process (0 = infinite, Ctrl+C to stop)")
    args = parser.parse_args()

    _check_bins()

    st.initialize()
    st_system = st.create_system()
    st_device = st_system.create_first_device()
    print(f"Camera: {st_device.info.display_name}")
    print(f"FOV={args.fov}°  morph={args.morph}  (Ctrl+C to stop)")

    st_datastream = st_device.create_datastream()
    grab_count = args.frames if args.frames > 0 else 2**62
    st_datastream.start_acquisition(grab_count)
    st_device.acquisition_start()

    frame_i = 0
    t0 = time.monotonic()

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
                ppm = _to_ppm_bytes(st_image.get_image_data(), pfi, w, h)
            # buffer released — now run the subprocesses
            frame_i += 1
            att = identify_frame(ppm, w, h, args.fov, args.morph)
            fps = frame_i / max(time.monotonic() - t0, 1e-6)
            if att:
                print(f"frame {frame_i:4d} | RA={att[0]:9.4f}  DEC={att[1]:8.4f}  ROLL={att[2]:8.3f}  ({fps:.2f} fps)")
            else:
                print(f"frame {frame_i:4d} | NULL                                         ({fps:.2f} fps)")
    except KeyboardInterrupt:
        print("\n[pi_identify] stopped")
    finally:
        st_device.acquisition_stop()
        st_datastream.stop_acquisition()


if __name__ == "__main__":
    main()
