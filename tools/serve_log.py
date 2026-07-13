#!/usr/bin/env python3
"""
Serve the tail of a log file as an auto-refreshing web page.

    python serve_log.py [--file run.log] [--port 8081] [--lines 40]

Open http://<this-machine-LAN-IP>:8081/ ; it reloads every 1s and shows the last N lines.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
from pathlib import Path


def make_handler(path: Path, lines: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            try:
                with path.open(errors="replace") as f:
                    tail = "".join(deque(f, maxlen=lines))
            except FileNotFoundError:
                tail = f"(waiting for {path} ...)"
            body = ("<!doctype html><meta http-equiv=refresh content=1>"
                    "<body style='margin:0;background:#111;color:#9f9;"
                    "font:12px/1.4 monospace'><pre>" + tail + "</pre>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # ponytail: browser closed mid-write on refresh — normal
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="run.log")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--lines", type=int, default=40)
    args = p.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(Path(args.file), args.lines))
    print(f"serving {args.file} on http://<LAN-IP>:{args.port}/ (last {args.lines} lines, 1s refresh)")
    server.serve_forever()


if __name__ == "__main__":
    main()
