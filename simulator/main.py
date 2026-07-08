"""
Star simulator entry point (Pi-side orchestrator).

Wires: attitude source -> renderer -> MJPEG stream (phone displays it), an HTTP control API
(live commands / render config / status / tracker start-stop / delay calibration), and an
optional tracker feed -> comparator (score the tracker's attitude against commanded truth).

    # Display + control (open http://<this-host>:8090/ on the phone; drive it with the GUI/CLI):
    python -m simulator.main --commands simulator/examples/point_scan.txt --fov 10

    # Let the simulator spawn + score the tracker (Start/Stop from the GUI, or --autostart):
    python -m simulator.main --fov 10 --tracker "python pi_identify.py --fov 10 --no-stream" --autostart

    # Or pipe a tracker in on stdin instead of spawning it:
    python pi_identify.py --fov 10 | python -m simulator.main --fov 10 --compare-stdin

The phone is a passive display; all control is here. If the truth is *moving*, accuracy
numbers are only trustworthy over static holds — see simulator/comparator.py.
"""
from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .commands import parse_commands
from .comparator import Comparator, TruthTimeline, estimate_delay
from .feed import parse_line
from .renderer import Renderer, flash_jpeg
from .source import CommandQueueSource, ReplaySource
from .state import SimState
from .stream_server import FrameBuffer, start_server

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lan_ip() -> str:
    """Best-effort LAN IP of this machine (the address the phone must open)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _lan_ip_candidates() -> list[str]:
    """All this host's IPv4 addresses (the default-route guess may be a VPN tunnel, not WiFi)."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ips.discard("127.0.0.1")
    guess = _lan_ip()
    # Put the default-route guess first, then the rest — one of these is the WiFi address.
    return [guess] + sorted(ips - {guess})


def _build_source(args, renderer: Renderer):
    """
    Returns an attitude source. Command / static / default all use a CommandQueueSource so
    live commands always work; --replay uses a fixed ReplaySource (live commands don't apply).
    """
    start = (args.start_ra, args.start_dec, args.start_roll)
    if args.replay:
        return ReplaySource(Path(args.replay))
    if args.static is not None:
        ra, dec, roll = args.static
        return CommandQueueSource(parse_commands(f"point_at {ra} {dec} {roll}"), start)
    commands = []
    if args.commands:
        commands = parse_commands(Path(args.commands).read_text(), renderer.hr_lookup)
    return CommandQueueSource(commands, start)


class TrackerController:
    """
    Manages the tracker child process and delay calibration. Given to the HTTP server so the
    GUI can Start/Stop the tracker and trigger calibration remotely. Minimal by design: start
    = Popen + reader thread; stop = terminate; a dead child just flips ``tracker_running``.
    """

    def __init__(self, state: SimState, comparator: Comparator, timeline: TruthTimeline,
                 tracker_cmd: str | None, t0: float, preview_url: str = "http://127.0.0.1:8080/"):
        self._state = state
        self._comparator = comparator
        self._timeline = timeline
        self._cmd = tracker_cmd
        self._t0 = t0
        self._preview_url = preview_url
        self._proc = None
        self._estimates: list[tuple[float, tuple[float, float, float]]] = []

    def ingest(self, line: str) -> None:
        """Parses one tracker stdout line and, if it's an attitude, scores + records it."""
        est = parse_line(line)
        if est is None:
            return
        t_recv = time.monotonic() - self._t0
        self._estimates.append((t_recv, est))
        comparison = self._comparator.add_estimate(t_recv, est)
        metrics = {"est": est}
        if comparison is not None:
            metrics.update(pointing_err_deg=round(comparison.pointing_err_deg, 4),
                           roll_err_deg=round(comparison.roll_err_deg, 4),
                           sync_ok=True)
        self._state.update_metrics(metrics)

    def _read_loop(self) -> None:
        for line in self._proc.stdout:
            self.ingest(line)
        self._state.update_metrics({"tracker_running": False})

    def start_tracker(self) -> str:
        if self._cmd is None:
            raise RuntimeError("no --tracker command configured")
        if self._proc is not None and self._proc.poll() is None:
            return "already running"
        self._proc = subprocess.Popen(shlex.split(self._cmd), stdout=subprocess.PIPE,
                                      text=True, bufsize=1)
        self._state.update_metrics({"tracker_running": True})
        threading.Thread(target=self._read_loop, daemon=True).start()
        return "started"

    def stop_tracker(self) -> str:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._state.update_metrics({"tracker_running": False})
        return "stopped"

    def flash_check(self) -> dict:
        """
        RGB flash liveness check: fills the phone screen red→green→blue and watches the
        tracker's camera-preview MJPEG for each colour. Confirms the whole optical loop
        (render→display→camera→preview) is alive and returns the visual round-trip time.
        This is a sanity check, NOT the pipeline delay (see calibrate_delay for that).
        """
        from .sync import detect_flash
        try:
            round_trip = detect_flash(self._state.set_flash, self._preview_url)
        finally:
            self._state.set_flash(None)  # never leave the screen stuck on a flash
        if round_trip is None:
            return {"ok": False, "reason": f"flash not seen in preview at {self._preview_url} "
                                           "(camera not pointed at the screen, or preview not running)"}
        return {"ok": True, "visual_round_trip_s": round(round_trip, 3)}

    def calibrate_delay(self, offset_deg: float = 8.0, settle_s: float = 4.0) -> dict:
        """
        Measures pipeline delay by commanding a known RA step and timing when the tracker's
        estimate follows (comparator.estimate_delay over the stdout stream — the quantity that
        actually needs compensating). Sets the comparator delay on success.
        """
        truth = self._state.get_metrics().get("truth")
        if truth is None:
            return {"ok": False, "reason": "no current truth"}
        new_ra = (truth[0] + offset_deg) % 360.0
        new_pointing = (new_ra, truth[1])
        step_time = time.monotonic() - self._t0
        self._estimates = [e for e in self._estimates if e[0] >= step_time]  # only post-step
        self._state.put_command(f"point_at {new_ra} {truth[1]} {truth[2]}")  # via mailbox (render thread injects)
        time.sleep(settle_s)
        delay = estimate_delay(self._timeline, list(self._estimates), step_time, new_pointing)
        if delay is None:
            return {"ok": False, "reason": "estimate never reached the commanded step"}
        self._comparator.set_delay(delay)
        self._state.update_metrics({"delay": round(delay, 4)})
        return {"ok": True, "delay": round(delay, 4)}


def main() -> None:
    p = argparse.ArgumentParser(description="Star simulator: render + stream + control + score")
    p.add_argument("--fov", type=float, default=10.0, help="horizontal FOV in degrees")
    p.add_argument("--image-size", type=int, default=877, help="rendered frame size (px, square)")
    p.add_argument("--mag-limit", type=float, default=7.5, help="faintest Tycho magnitude to render")
    p.add_argument("--catalog", type=Path, default=None, help="catalog CSV (default data/tycho2.csv)")
    p.add_argument("--port", type=int, default=8090, help="stream/viewer/control HTTP port")
    p.add_argument("--fps", type=float, default=10.0, help="render/stream rate")
    p.add_argument("--roll-sign", type=float, default=1.0, help="flip visual roll direction (+1/-1)")

    src = p.add_argument_group("attitude source (default: live commands from --start-*)")
    src.add_argument("--commands", help="command script file (see simulator/examples)")
    src.add_argument("--replay", help="CSV timeline t,ra,dec,roll to replay (no live commands)")
    src.add_argument("--static", nargs=3, type=float, metavar=("RA", "DEC", "ROLL"),
                     help="start at a fixed attitude (still accepts live commands)")
    src.add_argument("--start-ra", type=float, default=83.8, help="initial RA")
    src.add_argument("--start-dec", type=float, default=-5.4, help="initial DEC")
    src.add_argument("--start-roll", type=float, default=0.0, help="initial roll")

    sc = p.add_argument_group("scoring (optional)")
    sc.add_argument("--tracker", help="tracker command the simulator can spawn (Start/Stop via API)")
    sc.add_argument("--preview-url", default="http://127.0.0.1:8080/",
                    help="tracker camera-preview MJPEG URL for the flash liveness check")
    sc.add_argument("--autostart", action="store_true", help="start --tracker immediately")
    sc.add_argument("--compare-stdin", action="store_true", help="score tracker lines piped on stdin")
    sc.add_argument("--pipeline-delay", type=float, default=0.30, help="initial delay seconds")
    sc.add_argument("--score-roll-sign", type=float, default=1.0, help="align tracker roll sign (+1/-1)")
    sc.add_argument("--score-roll-offset", type=float, default=0.0, help="align tracker roll offset (deg)")
    sc.add_argument("--csv", type=Path, default=None, help="comparison CSV (default outputs/simulator_run_<ts>.csv)")
    args = p.parse_args()

    renderer = Renderer(args.image_size, args.fov, args.mag_limit, args.catalog)
    source = _build_source(args, renderer)
    state = SimState(pipeline_delay=args.pipeline_delay)

    timeline = TruthTimeline()
    scoring = args.compare_stdin or args.tracker
    comparator = controller = None
    if scoring:
        csv_path = args.csv or (PROJECT_ROOT / "outputs" / f"simulator_run_{int(time.time())}.csv")
        comparator = Comparator(timeline, args.pipeline_delay, args.score_roll_sign,
                                args.score_roll_offset, csv_path)

    t0 = time.monotonic()
    if comparator is not None:
        controller = TrackerController(state, comparator, timeline, args.tracker, t0, args.preview_url)

    buffer = FrameBuffer()
    start_server(buffer, state, args.port, controller)
    print("[simulator] OPEN ONE OF THESE ON THE PHONE (same WiFi) — try each until it loads:")
    for ip in _lan_ip_candidates():
        print(f"    http://{ip}:{args.port}/")
    print("[simulator] (the first is a guess and may be a VPN tunnel; the WiFi one is usually 192.168.* or 172.*)")
    print(f"[simulator] FOV={args.fov}  size={args.image_size}  fps={args.fps}  source={type(source).__name__}")
    if comparator is not None:
        print(f"[simulator] scoring -> {csv_path}  (delay={args.pipeline_delay}s)")

    if args.compare_stdin and controller is not None:
        threading.Thread(target=lambda: [controller.ingest(l) for l in sys.stdin], daemon=True).start()
    if args.autostart and controller is not None:
        print(f"[simulator] {controller.start_tracker()} tracker: {args.tracker}")

    can_inject = isinstance(source, CommandQueueSource)
    period = 1.0 / max(args.fps, 0.1)
    frames = 0
    try:
        while True:
            now = time.monotonic()
            t = now - t0
            if can_inject:
                for line in state.drain_commands():
                    try:
                        source.inject(parse_commands(line, renderer.hr_lookup)[0], t)
                    except (ValueError, IndexError) as exc:
                        print(f"[simulator] bad command {line!r}: {exc}", file=sys.stderr)
            (ra, dec, roll), moving = source.attitude(t)
            timeline.record(t, ra, dec, roll, moving)
            # Fill priority: sync flash (calibration) > command blank/flash > rendered stars.
            fill = state.get_flash()
            if fill is None and can_inject:
                fill = source.display_color()
            if fill is not None:
                buffer.set(flash_jpeg(args.image_size, fill))
            else:
                buffer.set(renderer.render(ra, dec, roll, args.roll_sign, state.get_config()))
            frames += 1
            state.update_metrics({"fps": round(frames / max(t, 1e-6), 2),
                                  "truth": (round(ra, 4), round(dec, 4), round(roll, 4))})
            dt = period - (time.monotonic() - now)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[simulator] stopped")
    finally:
        if controller is not None:
            controller.stop_tracker()
        if comparator is not None:
            print(f"[simulator] summary (static holds only): {comparator.summary(static_only=True)}")
            comparator.close()


if __name__ == "__main__":
    main()
