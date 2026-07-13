"""
One front door for the simulator:

    python -m simulator serve ...       the live rig: render + phone stream + web control
                                        (+ HIL scoring with --tracker); args = main.py's
    python -m simulator control ...     remote CLI for a running rig (point_at / --status / ...)
    python -m simulator detumble        headless closed-loop detumble demo (vectors-in;
                                        --image = full render→centroid→TETRA chain;
                                        --hil [host] = hardware-in-the-loop via OpticalDUT
                                        against a running `serve --tracker ...`)
    python -m simulator selftest        run every module's built-in self-check

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
            i = sys.argv.index("--hil")
            nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
            host = nxt if nxt and not nxt.startswith("-") else "127.0.0.1:8090"
            if "--stare" in sys.argv:  # stop-and-stare: virtual physics pauses per solve (plumbing check)
                from .dut import OpticalDUT
                dut = OpticalDUT(host, settle_s=_flag("--settle", 1.5),   # must exceed pipeline delay
                                 roll_sign=_flag("--roll-sign", 1.0))     # -1 if roll axis diverges
                body = dynamics.RigidBody(omega=(2.0, 1.5, 2.5))
                history = dynamics.run_detumble(
                    dut, body, dynamics.RateController(gain=2.0, alpha=0.35),
                    duration_s=_flag("--duration", 30.0), dt=0.5,
                    on_step=lambda t, tr, est, w: print(
                        f"  t={t:5.1f}s  |w|={w:5.2f} deg/s  est={'ok' if est else 'NULL'}", flush=True))
                print(f"HIL stop-and-stare detumble |w|: {history[0]:.2f} -> {history[-1]:.3f} deg/s")
            else:  # real-time: the displayed field tumbles continuously while the loop fights it
                dynamics.run_hil_detumble(host,
                                          gain=_flag("--gain", 0.3),
                                          duration_s=_flag("--duration", 90.0),
                                          roll_sign=_flag("--roll-sign", 1.0))
        else:
            dynamics._demo(full_image_pipeline="--image" in sys.argv)
    elif verb == "selftest":
        # Cheap, dependency-free checks first; the last three need the catalog CSV + DLL.
        from . import attitude, commands, comparator, control, feed, state, sync, renderer, dut, dynamics
        for module in (attitude, commands, comparator, control, feed, state, sync, renderer, dut, dynamics):
            module._demo()
        print("simulator selftest: all modules passed")
    else:
        sys.exit(USAGE)


if __name__ == "__main__":
    main()
