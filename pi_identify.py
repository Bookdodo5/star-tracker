#!/usr/bin/env python3
"""
Pi live star identification: Sentech GigE camera → centroid_extract → TETRA → RA/DEC/ROLL.

Subprocess approach — no C changes: stapipy grabs frames, this script writes PPM to
a tmpdir, shells out to centroid_extract then demo_centroid_compare, and parses stdout.

Also serves a live MJPEG preview on http://<pi-ip>:8080 (--stream, enabled by default).

Usage:
    python pi_identify.py --fov 7.569
    python pi_identify.py --fov 10 --morph 0          # real night-sky point-source stars
    python pi_identify.py --fov 10 --scale 0.5        # downscale 2× before centroiding
    python pi_identify.py --fov 10 --fov-search       # calibrate FOV from seed, lock on first solve
    python pi_identify.py --fov 10 --no-stream        # disable MJPEG server
    python pi_identify.py --fov 10 --frames 1         # single shot
"""
import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import stapipy as st

ROOT = os.path.expanduser("~/src/star-tracker")
CENTROID_BIN = os.path.join(ROOT, "centroid", "build-pi", "centroid_extract")
IDENTIFY_BIN = os.path.join(ROOT, "identifier", "build-pi", "demo_centroid_compare")

# Shared state for the MJPEG server
_latest_jpeg = [b""]
_latest_att = [None]
_jpeg_lock = threading.Lock()


def _check_bins():
    missing = [b for b in (CENTROID_BIN, IDENTIFY_BIN) if not os.path.isfile(b)]
    if missing:
        sys.exit("Missing binaries — build on Pi first:\n"
                 + "\n".join(f"  {b}" for b in missing))


def _resize_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _to_gray(data: bytes, pfi, width: int, height: int) -> np.ndarray:
    if pfi.each_component_total_bit_count > 8:
        arr = np.frombuffer(data, np.uint16)
        arr = (arr >> (pfi.each_component_valid_bit_count - 8)).astype(np.uint8)
    else:
        arr = np.frombuffer(data, np.uint8)
    if pfi.is_mono:
        return arr.reshape(height, width)
    arr = arr.reshape(height, width)
    q = ((arr[0::2, 0::2].astype(np.uint16) + arr[0::2, 1::2]
          + arr[1::2, 0::2] + arr[1::2, 1::2]) // 4).astype(np.uint8)
    return q.repeat(2, axis=0).repeat(2, axis=1)[:height, :width]


def _gray_to_ppm(gray: np.ndarray) -> tuple:
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)
    return f"P6\n{w} {h}\n255\n".encode() + rgb.tobytes(), w, h


def _update_jpeg(gray: np.ndarray, att):
    """Encode frame + attitude overlay as JPEG for the MJPEG stream."""
    small = cv2.resize(gray, (812, 618))
    bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    if att:
        qw, qx, qy, qz = att[3]
        label = f"RA={att[0]:.3f}  DEC={att[1]:.3f}  ROLL={att[2]:.2f}  Q=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f})"
        cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(bgr, "NULL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _jpeg_lock:
        _latest_jpeg[0] = jpg.tobytes()


def _update_jpeg_raw(gray: np.ndarray):
    """Encode the frame as JPEG with no overlay (raw camera preview)."""
    small = cv2.resize(gray, (812, 618))
    _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _jpeg_lock:
        _latest_jpeg[0] = jpg.tobytes()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _jpeg_lock:
                    data = _latest_jpeg[0]
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                time.sleep(0.05)
        except Exception:
            pass


def _start_mjpeg_server(port: int):
    server = HTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[stream] http://10.90.37.15:{port}")


def _parse_attitude(stdout: str):
    kv = {}
    for line in stdout.splitlines():
        if "attitude_ra_deg=" in line or "attitude_qw=" in line:
            kv.update(tok.split("=") for tok in line.split() if "=" in tok)
    if "attitude_ra_deg" not in kv:
        return None
    try:
        return (float(kv["attitude_ra_deg"]),
                float(kv["attitude_dec_deg"]),
                float(kv["attitude_roll_deg"]),
                (float(kv["attitude_qw"]), float(kv["attitude_qx"]),
                 float(kv["attitude_qy"]), float(kv["attitude_qz"])))
    except (KeyError, ValueError):
        return None


def _parse_calibrated_fov(stdout: str):
    for line in stdout.splitlines():
        if "calibrated_fov_deg=" in line:
            try:
                kv = dict(tok.split("=") for tok in line.split() if "=" in tok)
                return float(kv["calibrated_fov_deg"])
            except (KeyError, ValueError):
                pass
    return None


def _run_identify(csv_path: str, width: int, height: int, fov: float, calibrate: bool = False):
    cmd = [IDENTIFY_BIN]
    if calibrate:
        cmd.append("--calibrate")
    cmd += [csv_path, str(width), str(height), str(fov)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    att = _parse_attitude(r.stdout)
    if calibrate:
        return att, _parse_calibrated_fov(r.stdout)
    return att


def _centroid(ppm_bytes: bytes, morph: int, tmp: str):
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
    with tempfile.TemporaryDirectory() as tmp:
        csv = _centroid(ppm_bytes, morph, tmp)
        if csv is None:
            return None
        return _run_identify(csv, width, height, fov)


def identify_frame_calibrate(ppm_bytes: bytes, width: int, height: int,
                              seed_fov: float, morph: int):
    with tempfile.TemporaryDirectory() as tmp:
        csv = _centroid(ppm_bytes, morph, tmp)
        if csv is None:
            return None, None
        return _run_identify(csv, width, height, seed_fov, calibrate=True)


def main():
    parser = argparse.ArgumentParser(description="Sentech GigE → TETRA attitude")
    parser.add_argument("--fov", type=float, default=10.0,
                        help="horizontal FOV in degrees (seed if --fov-search)")
    parser.add_argument("--fov-search", action="store_true",
                        help="calibrate FOV from seed, lock on first solve")
    parser.add_argument("--stream-only", action="store_true",
                        help="only serve the raw camera MJPEG preview; skip centroid/identify")
    parser.add_argument("--morph", type=int, default=0,
                        help="centroid morph passes: 0=real stars (default), 1+=satellite blobs")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="downscale frame before centroiding (e.g. 0.5 = half resolution)")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames to process (0 = infinite, Ctrl+C to stop)")
    parser.add_argument("--stream", action="store_true", default=True,
                        help="serve MJPEG preview on port 8080 (default: on)")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="disable MJPEG preview server")
    parser.add_argument("--port", type=int, default=8080,
                        help="MJPEG server port (default: 8080)")
    args = parser.parse_args()

    if not args.stream_only:
        _check_bins()

    if args.stream or args.stream_only:
        _start_mjpeg_server(args.port)

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

    CALIB_CONFIRM = 2
    CALIB_AGREE_PCT = 0.05  # max fractional FOV difference between consecutive calibrations
    frame_i = 0
    t0 = t_prev = time.monotonic()
    fov = args.fov
    fov_locked = not args.fov_search
    calib_window = []  # (fov, att) from recent consecutive calibration successes

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
                gray = _to_gray(st_image.get_image_data(), pfi, w, h)

            if args.scale != 1.0:
                gray = _resize_gray(gray, args.scale)
                h, w = gray.shape

            frame_i += 1
            now = time.monotonic()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            elapsed = now - t0

            if args.stream_only:
                _update_jpeg_raw(gray)
                if frame_i % 30 == 0:
                    print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | stream-only ({fps:.1f} fps)", flush=True)
                continue

            ppm, w, h = _gray_to_ppm(gray)

            if not fov_locked:
                att, found_fov = identify_frame_calibrate(ppm, w, h, fov, args.morph)
                if att:
                    fov = found_fov
                    fov_locked = True
                    print(f"[fov-search] locked FOV = {fov:.3f}°", flush=True)
                else:
                    print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | NULL (fov-search seed={args.fov}°)   ({fps:.2f} fps)", flush=True)
                    if args.stream:
                        _update_jpeg(gray, None)
                    continue
            else:
                att = identify_frame(ppm, w, h, fov, args.morph)

            if args.stream:
                _update_jpeg(gray, att)

            if att:
                qw, qx, qy, qz = att[3]
                print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | RA={att[0]:9.4f}  DEC={att[1]:8.4f}  "
                      f"ROLL={att[2]:8.3f}  Q=({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f})  ({fps:.2f} fps)", flush=True)
            else:
                print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | NULL                                         "
                      f"({fps:.2f} fps)", flush=True)

    except KeyboardInterrupt:
        print("\n[pi_identify] stopped", flush=True)
    finally:
        st_device.acquisition_stop()
        st_datastream.stop_acquisition()


if __name__ == "__main__":
    main()
