"""
Command primitives that drive the simulated attitude over time.

The operator issues an ordered list of commands; the :class:`Resolver` turns
``(command_queue, elapsed_time)`` into the current ``(ra, dec, roll)`` attitude and a
``moving`` flag (True while the active command has a nonzero angular rate). This is the
whole "satellite motion" model for Phase 2 — attitude-only kinematics, no orbit.

Commands (parsed from a text file or REPL, one per line, ``#`` comments allowed):

    point_at <ra> <dec> [roll]     snap boresight to a sky position (instant)
    point_at HR<id>                snap to a catalog star by HR id (needs a lookup)
    hold <seconds>                 stay put for N seconds (settle window for scoring)
    slew <axis> <delta_deg> <rate> move by delta about axis (ra|dec|roll) at rate deg/s
    roll <rate> <seconds|forever>  spin about boresight at rate deg/s
    tumble <ra/s> <dec/s> <roll/s> [seconds|forever]   3-axis constant-rate tumble
    lost_in_space <n> <hold_s> [seed]  jump to n random attitudes, hold each (solve-rate test)
    blank <seconds>                dark frame (no stars) — dark-frame / no-signal test
    flash <r> <g> <b> <seconds>    solid RGB frame — scripted marker / liveness

New motion types are added by writing one more parse branch + `_eval` case — nothing else changes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional


INF = float("inf")


@dataclass
class Command:
    """One motion command: a kind, its parameters, how long it runs, and an optional display fill."""
    kind: str
    params: dict
    duration: float                       # seconds; INF for 'forever'
    display_color: Optional[tuple] = None  # (r,g,b) to fill the whole frame (blank/flash), else None


def parse_commands(text: str, hr_lookup: Optional[Callable[[int], tuple[float, float]]] = None) -> list[Command]:
    """
    Parses a command script into :class:`Command` objects.

    ``hr_lookup`` maps an HR id to ``(ra_deg, dec_deg)`` for ``point_at HR<id>``.
    Raises ``ValueError`` with the offending line on any malformed command.
    """
    commands: list[Command] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0].lower()
        try:
            if kind == "point_at":
                if parts[1].upper().startswith("HR"):
                    if hr_lookup is None:
                        raise ValueError("point_at HR<id> needs a catalog lookup")
                    ra, dec = hr_lookup(int(parts[1][2:]))
                    roll = float(parts[2]) if len(parts) > 2 else None
                else:
                    ra, dec = float(parts[1]), float(parts[2])
                    roll = float(parts[3]) if len(parts) > 3 else None
                commands.append(Command("point_at", {"ra": ra, "dec": dec, "roll": roll}, 0.0))
            elif kind == "hold":
                commands.append(Command("hold", {}, float(parts[1])))
            elif kind == "slew":
                axis = parts[1].lower()
                if axis not in ("ra", "dec", "roll"):
                    raise ValueError(f"slew axis must be ra|dec|roll, got {axis!r}")
                delta, rate = float(parts[2]), float(parts[3])
                if rate <= 0:
                    raise ValueError("slew rate must be > 0")
                commands.append(Command("slew", {"axis": axis, "delta": delta, "rate": rate},
                                        abs(delta) / rate))
            elif kind == "roll":
                rate = float(parts[1])
                forever = len(parts) > 2 and parts[2].lower() == "forever"
                dur = INF if forever else float(parts[2])
                commands.append(Command("roll", {"rate": rate}, dur))
            elif kind == "tumble":
                ra_r, dec_r, roll_r = float(parts[1]), float(parts[2]), float(parts[3])
                forever = len(parts) > 4 and parts[4].lower() == "forever"
                dur = INF if forever else (float(parts[4]) if len(parts) > 4 else INF)
                commands.append(Command("tumble", {"ra": ra_r, "dec": dec_r, "roll": roll_r}, dur))
            elif kind == "lost_in_space":
                n, hold_s = int(parts[1]), float(parts[2])
                seed = int(parts[3]) if len(parts) > 3 else 0
                commands.extend(_lost_in_space(n, hold_s, seed))
            elif kind == "blank":
                commands.append(Command("blank", {}, float(parts[1]), display_color=(8, 8, 8)))
            elif kind == "flash":
                color = (int(parts[1]), int(parts[2]), int(parts[3]))
                commands.append(Command("flash", {}, float(parts[4]), display_color=color))
            else:
                raise ValueError(f"unknown command {kind!r}")
        except (IndexError, ValueError) as exc:
            raise ValueError(f"line {lineno}: {raw!r}: {exc}") from exc
    return commands


def _lost_in_space(n: int, hold_s: float, seed: int) -> list[Command]:
    """
    Expands into n (point_at random attitude → hold) pairs. Uniform on the sphere; the
    seed makes it reproducible. Each target's truth is recorded frame-by-frame in the
    comparator/truth timeline during its hold, so accuracy per target is scored there.
    """
    rng = random.Random(seed)
    out: list[Command] = []
    for _ in range(n):
        ra = rng.uniform(0.0, 360.0)
        dec = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))  # uniform on sphere, not in dec
        roll = rng.uniform(0.0, 360.0)
        out.append(Command("point_at", {"ra": ra, "dec": dec, "roll": roll}, 0.0))
        out.append(Command("hold", {}, hold_s))
    return out


class Resolver:
    """
    Resolves an ordered command queue into the current attitude given wall-clock time.

    Call :meth:`attitude` repeatedly with a monotonically increasing timestamp. The
    resolver advances through finished commands, carrying the end attitude of each as the
    start of the next, and returns ``((ra, dec, roll), moving)``.
    """

    def __init__(self, commands: list[Command], start_attitude: tuple[float, float, float]):
        self._queue = list(commands)
        self._start_att = start_attitude          # attitude at the start of the active command
        self._index = 0                            # index of the active command
        self._cmd_start_t: Optional[float] = None  # timestamp the active command began
        self._active: Optional[Command] = None     # command evaluated on the last attitude() call

    def display_color(self) -> Optional[tuple]:
        """Full-frame fill colour of the active command (blank/flash), else None."""
        return self._active.display_color if self._active else None

    def _end_attitude(self, cmd: Command, start: tuple[float, float, float]) -> tuple[float, float, float]:
        """Attitude when ``cmd`` has fully completed (used to seed the next command)."""
        return self._eval(cmd, start, cmd.duration if cmd.duration != INF else 0.0)[0]

    def _eval(self, cmd: Command, start: tuple[float, float, float], elapsed: float):
        """Returns ``(attitude, moving)`` for one command at ``elapsed`` seconds in."""
        ra, dec, roll = start
        if cmd.kind == "point_at":
            p = cmd.params
            return (p["ra"], p["dec"], p["roll"] if p["roll"] is not None else roll), False
        if cmd.kind == "hold":
            return (ra, dec, roll), False
        if cmd.kind == "slew":
            axis, rate = cmd.params["axis"], cmd.params["rate"]
            sign = 1.0 if cmd.params["delta"] >= 0 else -1.0
            moved = sign * min(rate * elapsed, abs(cmd.params["delta"]))
            if axis == "ra":
                ra = (ra + moved) % 360.0
            elif axis == "dec":
                dec = max(-90.0, min(90.0, dec + moved))
            else:
                roll = (roll + moved) % 360.0
            still_moving = rate * elapsed < abs(cmd.params["delta"])
            return (ra, dec, roll), still_moving
        if cmd.kind == "roll":
            roll = (roll + cmd.params["rate"] * elapsed) % 360.0
            return (ra, dec, roll), True
        if cmd.kind == "tumble":
            ra = (ra + cmd.params["ra"] * elapsed) % 360.0
            dec = max(-90.0, min(90.0, dec + cmd.params["dec"] * elapsed))
            roll = (roll + cmd.params["roll"] * elapsed) % 360.0
            return (ra, dec, roll), True
        if cmd.kind in ("blank", "flash"):        # hold attitude; the frame is filled by display_color
            return (ra, dec, roll), False
        raise ValueError(f"unhandled command kind {cmd.kind!r}")

    def inject(self, command: Command, now: float) -> None:
        """
        Interrupts the current motion with ``command`` starting now, seeded from the current
        attitude (live-command semantics). Discards the remaining queue.

        Order matters: capture the current attitude *before* mutating, then reset all four
        bookkeeping fields, else the injected command sees a stale ``elapsed`` and a slew/roll
        jumps to its end instantly.
        """
        att, _ = self.attitude(now)          # (a) current resolved attitude, before mutation
        self._queue = [command]              # (b)
        self._index = 0                      # (c)
        self._start_att = att                # (d)
        self._cmd_start_t = now              # (e)

    def attitude(self, now: float) -> tuple[tuple[float, float, float], bool]:
        """Returns ``((ra, dec, roll), moving)`` at monotonic time ``now``."""
        if self._cmd_start_t is None:
            self._cmd_start_t = now
        # Advance past any commands that have finished by 'now'.
        while self._index < len(self._queue):
            cmd = self._queue[self._index]
            elapsed = now - self._cmd_start_t
            if elapsed < cmd.duration:
                self._active = cmd
                return self._eval(cmd, self._start_att, elapsed)
            # Command done: carry its end attitude forward and move on.
            self._start_att = self._end_attitude(cmd, self._start_att)
            self._cmd_start_t += cmd.duration
            self._index += 1
        # Queue exhausted: hold the last attitude.
        self._active = None
        return self._start_att, False


def _demo() -> None:
    """Self-check: point_at snaps, roll advances, slew reaches its target then holds."""
    cmds = parse_commands("point_at 100 0 0\nhold 2\nslew ra 10 5\nroll 1 forever")
    r = Resolver(cmds, (0.0, 0.0, 0.0))
    (ra, dec, roll), moving = r.attitude(0.0)          # point_at (instant) -> hold
    assert (ra, dec, roll) == (100.0, 0.0, 0.0) and not moving
    (ra, _, _), moving = r.attitude(1.0)               # still holding
    assert ra == 100.0 and not moving
    (ra, _, _), moving = r.attitude(2.0 + 1.0)         # 1s into a 5deg/s slew, 2s hold done
    assert abs(ra - 105.0) < 1e-9 and moving, ra
    (ra, _, _), moving = r.attitude(2.0 + 2.0)         # slew of 10deg/5 = 2s done -> target
    assert abs(ra - 110.0) < 1e-9, ra
    (_, _, roll), moving = r.attitude(2.0 + 2.0 + 3.0)  # 3s into roll @1deg/s
    assert abs(roll - 3.0) < 1e-9 and moving, roll
    # inject: interrupt a hold mid-way; the new slew must start from the interrupted attitude.
    r2 = Resolver(parse_commands("point_at 100 20 0\nhold 100"), (0.0, 0.0, 0.0))
    r2.attitude(0.0)                                     # snap to (100,20)
    r2.inject(parse_commands("slew ra 10 5")[0], 5.0)    # interrupt at t=5
    (ra, dec, _), moving = r2.attitude(5.0 + 1.0)        # 1s into the injected slew
    assert abs(ra - 105.0) < 1e-9 and abs(dec - 20.0) < 1e-9 and moving, (ra, dec)
    # tumble: 3-axis constant rate.
    r3 = Resolver(parse_commands("point_at 10 0 0\ntumble 1 0.5 2 forever"), (0, 0, 0))
    r3.attitude(0.0)
    (ra, dec, roll), moving = r3.attitude(2.0)           # 2s of tumble
    assert abs(ra - 12.0) < 1e-9 and abs(dec - 1.0) < 1e-9 and abs(roll - 4.0) < 1e-9 and moving, (ra, dec, roll)
    # blank/flash: attitude holds, display_color is exposed.
    r4 = Resolver(parse_commands("point_at 10 0 0\nflash 255 0 0 2\nblank 2"), (0, 0, 0))
    r4.attitude(0.0); r4.attitude(0.001)                 # into the flash
    assert r4.display_color() == (255, 0, 0), r4.display_color()
    r4.attitude(2.5)                                     # into the blank
    assert r4.display_color() == (8, 8, 8), r4.display_color()
    # lost_in_space: expands to n point_at+hold, reproducible by seed.
    lis = parse_commands("lost_in_space 5 3 42")
    assert len(lis) == 10 and lis[0].kind == "point_at" and lis[1].kind == "hold"
    assert parse_commands("lost_in_space 5 3 42")[0].params == lis[0].params  # seed reproducible
    print("commands.py self-check passed")


if __name__ == "__main__":
    _demo()
