"""
CLI client for the running simulator's HTTP control API — the scriptable remote
(interactive use is easier via the web page at http://<host>:8090/control).

    python -m simulator.control point_at 83.8 -5.4        # send a live command
    python -m simulator.control roll 3 forever
    python -m simulator.control --status                  # print live metrics + config
    python -m simulator.control --config gain=2.0 noise_sigma=5
    python -m simulator.control --calibrate               # measure + set pipeline delay
    python -m simulator.control --start | --stop          # tracker child

    python -m simulator.control --host 10.0.0.5:8090 point_at 83.8 -5.4
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_HOST = "127.0.0.1:8090"


def _post(host: str, path: str, obj: dict) -> dict:
    req = urllib.request.Request(f"http://{host}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get(host: str, path: str) -> dict:
    with urllib.request.urlopen(f"http://{host}{path}", timeout=10) as r:
        return json.loads(r.read())


def parse_config(pairs: list[str]) -> dict:
    """Turns ``["gain=2.0", "noise_sigma=5"]`` into ``{"gain":2.0, "noise_sigma":5.0}``."""
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key.strip()] = float(value)
    return out


def join_command(tokens: list[str]) -> str:
    """Joins CLI tokens into a command line (``["point_at","83.8","-5.4"] -> "point_at 83.8 -5.4"``)."""
    return " ".join(tokens)


# --- thin API wrappers (importable for scripting) ---
def send_command(host: str, line: str) -> dict:
    return _post(host, "/command", {"line": line})


def get_status(host: str) -> dict:
    return _get(host, "/status")


def set_config(host: str, config: dict) -> dict:
    return _post(host, "/config", config)


def calibrate(host: str) -> dict:
    return _post(host, "/calibrate-delay", {})


def flash_check(host: str) -> dict:
    return _post(host, "/flash-check", {})


def tracker(host: str, action: str) -> dict:
    return _post(host, f"/tracker/{action}", {})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Simulator control client")
    p.add_argument("--host", default=DEFAULT_HOST, help="host:port (default 127.0.0.1:8090)")
    p.add_argument("--status", action="store_true", help="print live status")
    p.add_argument("--config", nargs="+", metavar="k=v", help="set render config params")
    p.add_argument("--calibrate", action="store_true", help="measure + set pipeline delay")
    p.add_argument("--flash", action="store_true", help="RGB flash liveness check (visual round-trip)")
    p.add_argument("--start", action="store_true", help="start the tracker child")
    p.add_argument("--stop", action="store_true", help="stop the tracker child")
    p.add_argument("command", nargs="*", help="a live command, e.g. point_at 83.8 -5.4")
    args = p.parse_args(argv)

    if args.status:
        print(json.dumps(get_status(args.host), indent=2))
    elif args.config:
        print(set_config(args.host, parse_config(args.config)))
    elif args.calibrate:
        print(calibrate(args.host))
    elif args.flash:
        print(flash_check(args.host))
    elif args.start:
        print(tracker(args.host, "start"))
    elif args.stop:
        print(tracker(args.host, "stop"))
    elif args.command:
        print(send_command(args.host, join_command(args.command)))
    else:
        p.error("nothing to do (give a command, --status, --config, --calibrate, --start/--stop)")


def _demo() -> None:
    """Self-check: pure request-mapping helpers (no network)."""
    assert parse_config(["gain=2.0", "noise_sigma=5"]) == {"gain": 2.0, "noise_sigma": 5.0}
    assert join_command(["point_at", "83.8", "-5.4"]) == "point_at 83.8 -5.4"
    assert join_command(["roll", "3", "forever"]) == "roll 3 forever"
    print("control.py self-check passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _demo()
    else:
        main()
