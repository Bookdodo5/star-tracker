"""
MJPEG stream + control server for the simulator.

Serves, on one port over the local network:
    ``GET  /``               the fullscreen viewer page (simulator/web/viewer.html)
    ``GET  /stream``         a multipart MJPEG stream of the latest rendered frame
    ``GET  /status``         live metrics + current render config (JSON)
    ``POST /command``        body {"line": "point_at 83.8 -5.4"} → injected live
    ``POST /config``         body of render params (gain/gamma/noise_sigma/…) → merged
    ``POST /tracker/start``  start the tracker child   (needs a controller)
    ``POST /tracker/stop``   stop the tracker child    (needs a controller)
    ``POST /calibrate-delay`` measure + set pipeline_delay (needs a controller)

The phone opens ``http://<host>:<port>/``; the CLI and GUI drive the JSON routes. This
mirrors the MJPEG pattern proven in pi_identify.py — HTTP-native, no build step, no Vercel
(an HTTPS host would trip mixed-content blocking on the HTTP stream).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_WEB = Path(__file__).resolve().parent / "web"
_VIEWER_HTML = (_WEB / "viewer.html").read_bytes()
_CONTROL_HTML = (_WEB / "control.html").read_bytes()


class FrameBuffer:
    """Holds the latest JPEG frame; thread-safe single-slot swap."""

    def __init__(self) -> None:
        self._jpeg = b""
        self._lock = threading.Lock()

    def set(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg

    def get(self) -> bytes:
        with self._lock:
            return self._jpeg


def start_server(buffer: FrameBuffer, state=None, port: int = 8090, controller=None) -> ThreadingHTTPServer:
    """
    Starts the MJPEG + control server in a background thread and returns it.

    ``state`` is a SimState (None = display-only, control routes return 503). ``controller``
    is an optional object with ``start_tracker()`` / ``stop_tracker()`` / ``calibrate_delay()``
    for the tracker/calibrate routes; when absent those routes return 501.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path.startswith("/stream"):
                self._serve_stream()
            elif self.path.startswith("/status"):
                if state is None:
                    return self._json({"error": "no state"}, 503)
                self._json({"metrics": state.get_metrics(), "config": state.get_config(),
                            "tracker_lines": state.get_tracker_lines()})
            elif self.path.startswith("/control"):
                self._html(_CONTROL_HTML)
            else:
                self._html(_VIEWER_HTML)

        def _html(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if state is None:
                return self._json({"error": "no state"}, 503)
            if self.path.startswith("/command"):
                line = self._read_json().get("line", "").strip()
                if not line:
                    return self._json({"error": "missing 'line'"}, 400)
                state.put_command(line)
                self._json({"ok": True, "queued": line})
            elif self.path.startswith("/config"):
                self._json({"ok": True, "config": state.update_config(self._read_json())})
            elif self.path.startswith("/tracker/start"):
                self._controller_call(controller and controller.start_tracker)
            elif self.path.startswith("/tracker/stop"):
                self._controller_call(controller and controller.stop_tracker)
            elif self.path.startswith("/calibrate-delay"):
                self._controller_call(controller and controller.calibrate_delay)
            elif self.path.startswith("/flash-check"):
                if controller is not None:
                    self._controller_call(controller.flash_check)
                else:
                    # No camera: just flash the display so you can visually verify rendering.
                    def _flash():
                        for color in [(180, 0, 0), (0, 180, 0), (0, 0, 180)]:
                            state.set_flash(color)
                            time.sleep(0.5)
                        state.set_flash(None)
                    threading.Thread(target=_flash, daemon=True).start()
                    self._json({"ok": True, "result": "display flashed (no camera to detect round-trip)"})
            else:
                self._json({"error": "unknown route"}, 404)

        def _controller_call(self, fn):
            if fn is None:
                return self._json({"error": "no controller"}, 501)
            try:
                self._json({"ok": True, "result": fn()})
            except Exception as exc:  # surface controller errors as 500 JSON, don't crash the server
                self._json({"error": str(exc)}, 500)

        def _serve_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    data = buffer.get()
                    if data:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
