"""
Scenario runner — the declarative scoreboard (UVM's idea).

A scenario file bundles a virtual-camera config, a command script, and PASS/FAIL
expectations. It runs headless against the SoftwareDUT (no phone/camera) and prints a
verdict, so it doubles as a regression test for any C-side change.

Format (``#`` comments allowed):

    config fov=8 mag=6.5 image_size=512 pixel_noise=0.3
    commands
      point_at 83.8 -5.4
      hold 1
      lost_in_space 30 1 7
    end
    expect solve_pct>80 mean_pointing_deg<0.1

Run:  python -m simulator.scenario simulator/scenarios/orion.scn
"""
from __future__ import annotations

import argparse
import operator
import sys
from pathlib import Path

from .attitude import angular_sep_deg, roll_diff_deg
from .commands import Resolver, parse_commands
from .dut import SoftwareDUT
from .renderer import Renderer

_OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}


def parse_scenario(text: str) -> tuple[dict, str, list[tuple]]:
    """Splits a scenario into (config dict, command-script text, expectations)."""
    config, expectations, command_lines = {}, [], []
    in_commands = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0].lower()
        if head == "config":
            for pair in line.split()[1:]:
                key, _, value = pair.partition("=")
                config[key] = float(value)
        elif head == "commands":
            in_commands = True
        elif head == "end":
            in_commands = False
        elif head == "expect":
            for token in line.split()[1:]:
                for symbol in (">=", "<=", "==", ">", "<"):
                    if symbol in token:
                        metric, value = token.split(symbol)
                        expectations.append((metric, symbol, float(value)))
                        break
        elif in_commands:
            command_lines.append(line)
    return config, "\n".join(command_lines), expectations


def run_scenario(text: str) -> dict:
    """Runs a scenario headless against the SoftwareDUT and returns the metrics dict."""
    config, command_text, _ = parse_scenario(text)
    renderer = Renderer(image_size=int(config.get("image_size", 877)),
                        fov_deg=config.get("fov", 10.0),
                        magnitude_limit=config.get("mag", 7.5))
    dut = SoftwareDUT(renderer)
    commands = parse_commands(command_text, renderer.hr_lookup)
    if any(c.duration == float("inf") for c in commands):
        raise ValueError("scenario commands must be finite (no 'forever')")

    # Step virtual time; collect one test point per static (held) attitude segment.
    resolver = Resolver(commands, (0.0, 0.0, 0.0))
    total = sum(c.duration for c in commands) + 0.1
    dt, t, last_truth, segments = 0.25, 0.0, None, []
    while t <= total:
        (ra, dec, roll), moving = resolver.attitude(t)
        truth = (round(ra, 4), round(dec, 4), round(roll, 4))
        if not moving and truth != last_truth:
            segments.append(truth)
            last_truth = truth
        elif moving:
            last_truth = None
        t += dt

    solved, point_errs, roll_errs = 0, [], []
    for truth in segments:
        est = dut.solve(truth, config)
        if est is not None:
            solved += 1
            point_errs.append(angular_sep_deg((truth[0], truth[1]), (est[0], est[1])))
            roll_errs.append(abs(roll_diff_deg(truth[2], est[2])))
    n = len(segments)
    return {
        "test_points": n,
        "solved": solved,
        "solve_pct": 100.0 * solved / n if n else 0.0,
        "mean_pointing_deg": sum(point_errs) / len(point_errs) if point_errs else float("nan"),
        "max_pointing_deg": max(point_errs) if point_errs else float("nan"),
        "mean_roll_deg": sum(roll_errs) / len(roll_errs) if roll_errs else float("nan"),
    }


def evaluate(metrics: dict, expectations: list[tuple]) -> tuple[bool, list[str]]:
    """Checks metrics against expectations; returns (all_passed, per-line report)."""
    all_passed, lines = True, []
    for metric, symbol, target in expectations:
        actual = metrics.get(metric)
        ok = actual is not None and _OPS[symbol](actual, target)
        all_passed &= ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {metric} {symbol} {target}  (actual={actual})")
    return all_passed, lines


def main() -> None:
    p = argparse.ArgumentParser(description="Run a headless star-tracker scenario")
    p.add_argument("scenario", type=Path, help="scenario file (.scn)")
    args = p.parse_args()
    text = args.scenario.read_text(encoding="utf-8")
    _, _, expectations = parse_scenario(text)
    metrics = run_scenario(text)
    passed, report = evaluate(metrics, expectations)
    print(f"Scenario: {args.scenario.name}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("\n".join(report) if report else "  (no expectations)")
    verdict = "PASS" if passed else "FAIL"
    print(f"VERDICT: {verdict}")
    sys.exit(0 if passed else 1)


def _demo() -> None:
    """Self-check: a dense-field scenario solves and passes its expectations."""
    text = (
        "config fov=10 mag=7.5 image_size=877\n"
        "commands\n"
        "point_at 83.8 -5.4\n"
        "hold 1\n"
        "point_at 101.3 -16.7\n"
        "hold 1\n"
        "end\n"
        "expect solve_pct>=100 mean_pointing_deg<0.1\n"
    )
    metrics = run_scenario(text)
    assert metrics["test_points"] == 2, metrics
    _, _, expectations = parse_scenario(text)
    passed, _ = evaluate(metrics, expectations)
    assert passed, metrics
    # A failing expectation must be reported as FAIL.
    bad_pass, _ = evaluate(metrics, [("mean_pointing_deg", "<", 0.0)])
    assert not bad_pass
    print(f"scenario.py self-check passed (solve_pct={metrics['solve_pct']:.0f}, "
          f"mean_err={metrics['mean_pointing_deg']:.4f}°)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _demo()
    else:
        main()
