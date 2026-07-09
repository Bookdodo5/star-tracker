"""
Serve FVIDEO/vid.mp4 to any browser on the LAN with native play/pause/seek.

    python serve_video.py [--port 8095] [--file FVIDEO/vid.mp4]

Then open http://<this-machine's-LAN-IP>:8095/ on any device on the same WiFi.
"""
from __future__ import annotations

import argparse
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_INDEX = b"""<!doctype html>
<title>vid.mp4</title>
<body style="margin:0;background:#000;display:flex;align-items:center;justify-content:center;height:100vh">
<video src="/vid.mp4" controls autoplay style="max-width:100%;max-height:100%"></video>
</body>"""

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def make_handler(video_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # ponytail: quiet by default, re-enable for debugging
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_INDEX)))
                self.end_headers()
                self.wfile.write(_INDEX)
                return
            if self.path != "/vid.mp4":
                self.send_error(404)
                return
            self._serve_video()

        def _serve_video(self):
            size = video_path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            status = 200
            if range_header:
                m = _RANGE_RE.match(range_header)
                if m:
                    status = 206
                    start = int(m.group(1)) if m.group(1) else 0
                    end = int(m.group(2)) if m.group(2) else size - 1

            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            try:
                with video_path.open("rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # ponytail: client seeked/closed mid-stream — normal for <video>, not an error

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--file", default="cache/FVIDEO/vid.mp4")
    args = parser.parse_args()

    video_path = Path(args.file).resolve()
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(video_path))
    print(f"serving {video_path} on port {args.port} — open http://<this-machine-LAN-IP>:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
