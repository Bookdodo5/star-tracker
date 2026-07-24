"""
One front door for the simulator:

    python -m simulator serve ...       the live rig: render + phone stream + web control
                                        (+ HIL scoring with --tracker); args = main.py's
    python -m simulator control ...     remote CLI for a running rig (point_at / --status / ...)
    python -m simulator detumble        headless closed-loop detumble demo (vectors-in;
                                        --image = full render→centroid→TETRA chain;
                                        --hil [host] = hardware-in-the-loop via OpticalDUT
                                        against a running `serve --tracker ...`;
                                        --attitude RA DEC ROLL / --omega WRA WDEC WROLL
                                        set start attitude / initial tumble rates;
                                        --target RA DEC ROLL = detumble AND slew to that
                                        attitude and stabilize there)
    python -m simulator selftest        run every module's built-in self-check

For a detumble against a *physically* tumbling body (torque-free rigid body, Euler's equation)
plus sky-path / attitude figures, run ``python media/plot_free_detumble.py`` — the plant there
is ``freebody.FreeRigidBody`` and the controller ``freebody.BodyRateController``, rather than
the decoupled-angle-rate ``dynamics.RigidBody`` this demo uses.

``python -m simulator.<module>`` still runs an individual module's self-check directly.
"""
from __future__ import annotations

import sys

USAGE = "usage: python -m simulator {serve|control|detumble|selftest} [args...]"


def main() -> None:
    verb = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.argv = [f"python -m simulator {verb}"] + sys.argv[2:]  # verbs re-parse their own args
    if verb == "serve":
        from . import main as serve
        serve.main()
    elif verb == "control":
        from . import control
        control.main()
    elif verb == "detumble":
        from . import dynamics
        if "--hil" in sys.argv:   # hardware-in-the-loop against a running `serve` (+ --tracker)
            def _flag(name: str, default: float) -> float:
                return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

            def _flag3(name: str, default):
                """Three-float flag, e.g. --omega 1.5 0.3 2.0 (ra dec roll)."""
                if name not in sys.argv:
                    return default
                i = sys.argv.index(name)
                return tuple(float(v) for v in sys.argv[i + 1:i + 4])
            i = sys.argv.index("--hil")
            nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
            host = nxt if nxt and not nxt.startswith("-") else "127.0.0.1:8090"
            attitude = _flag3("--attitude", None)   # start attitude ra dec roll (default: current)
            if "--stare" in sys.argv:  # stop-and-stare: virtual physics pauses per solve (plumbing check)
                from .dut import OpticalDUT
                dut = OpticalDUT(host, settle_s=_flag("--settle", 1.5),   # must exceed pipeline delay
                                 roll_sign=_flag("--roll-sign", 1.0))     # -1 if roll axis diverges
                body = dynamics.RigidBody(attitude=attitude or (83.8, -5.4, 0.0),
                                          omega=_flag3("--omega", (2.0, 1.5, 2.5)))
                history = dynamics.run_detumble(
                    dut, body, dynamics.RateController(gain=2.0, alpha=0.35),
                    duration_s=_flag("--duration", 30.0), dt=0.5,
                    on_step=lambda t, tr, est, w: print(
                        f"  t={t:5.1f}s  |w|={w:5.2f} deg/s  est={'ok' if est else 'NULL'}", flush=True))
                print(f"HIL stop-and-stare detumble |w|: {history[0]:.2f} -> {history[-1]:.3f} deg/s")
            else:  # real-time: the displayed field tumbles continuously while the loop fights it
                if attitude is not None:  # point the rig at the start attitude before tumbling
                    from . import control as ctl
                    ctl.send_command(host, f"point_at {attitude[0]} {attitude[1]} {attitude[2]}")
                dynamics.run_hil_detumble(host,
                                          gain=_flag("--gain", 0.3),
                                          duration_s=_flag("--duration", 90.0),
                                          omega0=_flag3("--omega", (1.5, 0.3, 2.0)),
                                          roll_sign=_flag("--roll-sign", 1.0),
                                          target=_flag3("--target", None),  # point+stabilize here
                                          kp=_flag("--kp", 0.08),  # keep 1/kp above the ~7s meas. lag
                                          slew_max=_flag("--slew-max", 1.5))
        else:
            dynamics._demo(full_image_pipeline="--image" in sys.argv)
    elif verb == "selftest":
        # Cheap, dependency-free checks first; the last three need the catalog CSV + DLL.
        from . import (attitude, commands, comparator, control, feed, state, sync, renderer,
                       freebody, dut, dynamics)
        for module in (attitude, commands, comparator, control, feed, state, sync, renderer,
                       freebody, dut, dynamics):
            module._demo()
        print("simulator selftest: all modules passed")
    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
