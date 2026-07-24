#!/usr/bin/env python3
"""
Pi live star identification: Sentech GigE camera -> in-process centroid+TETRA -> RA/DEC/ROLL.

In-process approach: stapipy grabs frames; the whole centroid -> TETRA chain runs inside
this process via the star_live shared library (loaded once through live_identify.py — the
same solver the PC driver uses). No PPM temp files, no subprocess spawns: ~30 ms/solve
instead of seconds, which is what makes real-time HIL detumble rates possible. Build once
on the Pi:

    cmake -S live -B live/build-pi -DCMAKE_BUILD_TYPE=Release
    cmake --build live/build-pi

Also serves a live MJPEG preview on http://<pi-ip>:8080 (--stream, enabled by default).

Usage:
    python pi_identify.py --fov 7.569
    python pi_identify.py --fov 10 --morph 0          # real night-sky point-source stars
    python pi_identify.py --fov 10 --scale 0.5        # downscale 2x before centroiding
    python pi_identify.py --fov 10 --fov-search       # calibrate FOV from seed, lock after 2 agreeing solves
    python pi_identify.py --fov 10 --no-stream        # disable MJPEG server
    python pi_identify.py --fov 10 --frames 1         # single shot
"""
import argparse
import ctypes
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

import live_identify as L  # shared in-process solver: load_lib / solve / calibrate_fov

st = None  # stapipy, imported in main() so the helpers stay importable off-Pi (tests/CI)

# Shared state for the MJPEG server
_latest_jpeg = [b""]
_jpeg_lock = threading.Lock()


class _Tee:
    """Duplicates a text stream to a log file, so everything printed to the
    console (attitude lines, warnings) is also recorded on disk. stdout must
    stay live because simulator/feed.py parses it -- hence tee, not redirect."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()


def _tee_output_to(log_path):
    """Appends all stdout/stderr of this process to log_path (line-buffered)."""
    log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


def _set_node(nodemap, name, value, is_enum=False):
    """Best-effort set of a camera GenICam node; warns instead of crashing on unknown nodes."""
    try:
        node = nodemap.get_node(name)
        if is_enum:
            st.PyIEnumeration(node).set_symbolic_value(str(value))
        else:
            st.PyIFloat(node).value = float(value)
        print(f"[camera] {name} = {value}", flush=True)
    except Exception as exc:  # node name differs across firmware / not writable; don't abort the run
        print(f"[camera] could not set {name} ({exc})", flush=True)


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


def _detect_centroids(lib, gray: np.ndarray, morph: int):
    """Runs the DLL's own centroid detector (the same one the solver uses); returns [(x, y)]."""
    rgb = np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))
    h, w = gray.shape
    lib.detect_centroids.restype = ctypes.c_int
    lib.detect_centroids.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_uint16),
                                     ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
    out_x = (ctypes.c_uint16 * 64)()
    out_y = (ctypes.c_uint16 * 64)()
    n = lib.detect_centroids(rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), w, h, morph,
                             out_x, out_y, 64)
    return [(out_x[i], out_y[i]) for i in range(n)]


def _save_snapshot(path: str, gray: np.ndarray, lib, morph: int):
    """Save one native frame + a centroid overlay and print diagnostics (brightness, star count)."""
    h, w = gray.shape
    cv2.imwrite(path, gray)
    print(f"[snapshot] saved {path}  size={w}x{h}  "
          f"min={int(gray.min())} max={int(gray.max())} mean={gray.mean():.1f}", flush=True)
    centroids = _detect_centroids(lib, gray, morph)
    print(f"[snapshot] centroids found: {len(centroids)}", flush=True)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cx, cy in centroids:
        cv2.circle(overlay, (int(cx), int(cy)), 12, (0, 255, 0), 2)
    over_path = path.rsplit(".", 1)[0] + "_centroids.png"
    cv2.imwrite(over_path, overlay)
    print(f"[snapshot] overlay saved {over_path}", flush=True)


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


def _lan_ip() -> str:
    """Best-effort LAN IP of this machine (no packets sent; just picks the outbound interface)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _start_mjpeg_server(port: int):
    server = HTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[stream] http://{_lan_ip()}:{port}")


def main():
    parser = argparse.ArgumentParser(description="Sentech GigE -> TETRA attitude")
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
    parser.add_argument("--snapshot", metavar="PATH",
                        help="grab one native frame, save it (+centroid overlay) to PATH, print stats, exit")
    parser.add_argument("--exposure", type=float, metavar="US",
                        help="fixed exposure time in microseconds (disables auto-exposure); raise to detect more stars")
    parser.add_argument("--gain", type=float, metavar="DB",
                        help="fixed analog gain (disables auto-gain)")
    parser.add_argument("--log", default="run.log", metavar="PATH",
                        help="append all console output to this file (default: run.log; '' to disable). "
                             "Tail it live with media/serve_log.py")
    args = parser.parse_args()

    if args.log:
        _tee_output_to(args.log)

    lib = None
    if not args.stream_only:
        lib = L.load_lib()   # exits with build instructions if the .so is missing

    if args.stream or args.stream_only:
        _start_mjpeg_server(args.port)

    global st
    import stapipy as st
    st.initialize()
    st_system = st.create_system()
    st_device = st_system.create_first_device()
    print(f"Camera: {st_device.info.display_name}")
    print(f"FOV={args.fov} deg{'  fov-search=ON' if args.fov_search else ''}  "
          f"morph={args.morph}  scale={args.scale}  (Ctrl+C to stop)")
    if not args.fov_search:
        L.warn_fov_mismatch(args.fov)

    if args.exposure is not None or args.gain is not None:
        nodemap = st_device.remote_port.nodemap
        if args.exposure is not None:
            _set_node(nodemap, "ExposureAuto", "Off", is_enum=True)
            _set_node(nodemap, "ExposureTime", args.exposure)
        if args.gain is not None:
            _set_node(nodemap, "GainAuto", "Off", is_enum=True)
            _set_node(nodemap, "Gain", args.gain)

    st_datastream = st_device.create_datastream()
    grab_count = args.frames if args.frames > 0 else 2**62
    st_datastream.start_acquisition(grab_count)
    st_device.acquisition_start()

    RETRIEVE_TIMEOUT_MS = 5000  # cap the C grab wait so Ctrl-C is handled and stalls self-recover
    CALIB_CONFIRM = 2
    CALIB_AGREE_PCT = 0.05  # max fractional FOV difference between consecutive calibrations
    frame_i = 0
    t0 = t_prev = time.monotonic()
    fov = args.fov
    fov_locked = not args.fov_search
    calib_window = []  # (fov, att) from recent consecutive calibration successes
    last_att = None  # most recent successful solve; held through NULL frames (real-time coasting)

    try:
        while st_datastream.is_grabbing:
            try:
                buffer_ctx = st_datastream.retrieve_buffer(RETRIEVE_TIMEOUT_MS)
            except TypeError:  # older binding: retrieve_buffer takes no timeout arg
                buffer_ctx = st_datastream.retrieve_buffer()
            except Exception as exc:  # timeout / transient grab error: don't hang, retry
                print(f"[warn] no frame in {RETRIEVE_TIMEOUT_MS} ms; retrying ({exc})", flush=True)
                continue
            with buffer_ctx as st_buffer:
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

            if args.snapshot:
                _save_snapshot(args.snapshot, gray, lib, args.morph)
                break

            # The solver takes an RGB frame; the camera is mono, so replicate the channel.
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if not fov_locked:
                found_fov, att = L.calibrate_fov(lib, bgr, fov, args.morph)
                if att:
                    if calib_window and abs(found_fov - calib_window[-1][0]) > CALIB_AGREE_PCT * calib_window[-1][0]:
                        calib_window.clear()
                    calib_window.append((found_fov, att))
                    if len(calib_window) < CALIB_CONFIRM:
                        print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | calibrating FOV "
                              f"({len(calib_window)}/{CALIB_CONFIRM} agree, last={found_fov:.3f} deg)   "
                              f"({fps:.2f} fps)", flush=True)
                        if args.stream:
                            _update_jpeg(gray, None)
                        continue
                    fov = sum(candidate_fov for candidate_fov, _ in calib_window) / len(calib_window)
                    fov_locked = True
                    att = calib_window[-1][1]
                    print(f"[fov-search] locked FOV = {fov:.3f} deg "
                          f"(confirmed {CALIB_CONFIRM} frames)", flush=True)
                else:
                    calib_window.clear()
                    print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | NULL (fov-search seed={args.fov} deg)   ({fps:.2f} fps)", flush=True)
                    if args.stream:
                        _update_jpeg(gray, None)
                    continue
            else:
                att = L.solve(lib, bgr, fov, args.morph)

            if args.stream:
                _update_jpeg(gray, att)

            if att:
                last_att = att
                qw, qx, qy, qz = att[3]
                print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | RA={att[0]:9.4f}  DEC={att[1]:8.4f}  "
                      f"ROLL={att[2]:8.3f}  Q=({qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f})  ({fps:.2f} fps)", flush=True)
            elif last_att is not None:
                # no solve this frame: coast on the last attitude, marked as held (stale)
                print(f"frame {frame_i:4d} | t={elapsed:7.2f}s | NULL (hold) RA={last_att[0]:9.4f}  DEC={last_att[1]:8.4f}  "
                      f"ROLL={last_att[2]:8.3f}                                  ({fps:.2f} fps)", flush=True)
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
