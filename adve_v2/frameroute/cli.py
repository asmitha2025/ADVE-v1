"""
frameroute.cli — command line entry point.

    python -m frameroute route   VIDEO --budget-per-hour 240
    python -m frameroute signals VIDEO
    python -m frameroute cost    VIDEO --usd-per-call 0.002
    python -m frameroute gate1   VIDEO
    python -m frameroute gate2   VIDEO

`cost` is the one to run first when talking to a customer: it needs no GPU,
no model and no API key, because it only counts the calls a policy would
make and multiplies by a price you supply. That is enough to size the
saving on their own footage before anyone spends anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_signals(a) -> int:
    from .signals import analyze_video
    track = analyze_video(
        a.video, stride=a.stride, max_frames=a.max_frames,
        weights=a.weights, progress=True,
    )
    print(json.dumps(track.summary(), indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": track.summary(),
                 "signals": [s.as_dict() for s in track.signals]},
                f, indent=2,
            )
        print(f"wrote {a.out}")
    return 0


def _cmd_route(a) -> int:
    from .router import FrameRouter
    r = FrameRouter(
        min_gap_sec=a.min_gap, max_gap_sec=a.max_gap,
        hard_trigger=None if a.no_trigger else a.hard_trigger,
    )
    sel, track = r.route(
        a.video, budget=a.budget, budget_per_hour=a.budget_per_hour,
        policy=a.policy, weights=a.weights, stride=a.stride,
        max_frames=a.max_frames,
    )
    print(f"video            {Path(a.video).name}")
    print(f"duration         {sel.duration_sec:.1f}s")
    print(f"frames analyzed  {sel.frames_analyzed}")
    print(f"signal cost      {sel.signal_cost_ms:.2f} ms/frame "
          f"({track.summary()['analyze_fps']:.0f} fps analysis)")
    print(f"calls selected   {sel.n_calls}  ({sel.calls_per_hour:.0f}/hour)")
    print(f"reduction        {100.0 * (1 - sel.n_calls / max(sel.frames_analyzed,1)):.1f}% "
          f"vs one call per analyzed frame")
    print(f"reasons          {sel.reason_counts()}")
    print(f"guarantees       {json.dumps(sel.guarantees(track))}")
    if a.out:
        sel.to_json(a.out, track)
        print(f"wrote {a.out}")
    return 0


def _cmd_cost(a) -> int:
    """Price a routing decision without running any model."""
    from .router import FrameRouter
    from .policies import UniformFPS, UniformN, SceneCut, RouterPolicy
    from .signals import analyze_video

    track = analyze_video(a.video, stride=a.stride, max_frames=a.max_frames,
                          weights=a.weights)
    hours = track.duration_sec / 3600.0
    budget = a.budget or max(2, int(round(a.budget_per_hour * hours)))

    rows = []
    for pol in (UniformN(), UniformFPS(1.0), SceneCut(),
                RouterPolicy(policy="coverage"), RouterPolicy(policy="hybrid")):
        picks = pol.select(track, budget=budget)
        n = len(picks)
        rows.append((pol.name, n, n * a.usd_per_call))
    full = track.n_analyzed
    rows.insert(0, ("full_compute", full, full * a.usd_per_call))

    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{Path(a.video).name} — {track.duration_sec:.0f}s, "
          f"{track.n_analyzed} frames analyzed, ${a.usd_per_call} per call")
    print(f"budget: {budget} calls "
          f"({a.budget_per_hour:g}/hour x {hours:.3f} hours)\n")
    print(f"{'policy':<{w}}{'calls':>8}{'cost':>12}{'vs full':>10}")
    print("-" * (w + 30))
    for name, n, cost in rows:
        ratio = f"{full / max(n,1):.1f}x" if name != "full_compute" else "—"
        print(f"{name:<{w}}{n:>8}{'$' + format(cost, '.2f'):>12}{ratio:>10}")
    print("\nThese are counted calls, not estimates. What they are WORTH "
          "depends on retrieval parity — run gate1 before quoting the saving.\n")
    return 0


def _cmd_gate1(a) -> int:
    from bench.parity import main as gate1_main
    sys.argv = ["parity", "--video", a.video] + a.rest
    gate1_main()
    return 0


def _cmd_gate2(a) -> int:
    from bench.routing_bench import main as gate2_main
    sys.argv = ["routing_bench", "--video", a.video] + a.rest
    gate2_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="frameroute",
        description="Budget-constrained frame selection for video AI pipelines.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("video")
        p.add_argument("--stride", type=int, default=1)
        p.add_argument("--max-frames", type=int, default=None)
        p.add_argument("--weights", default="default",
                       help="default | static_cam | surveillance | ego_motion")

    p = sub.add_parser("signals", help="compute and dump change signals")
    common(p); p.add_argument("--out", default="")
    p.set_defaults(fn=_cmd_signals)

    p = sub.add_parser("route", help="select frames under a budget")
    common(p)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--budget-per-hour", type=float, default=None)
    p.add_argument("--policy", default="coverage",
                   help="coverage | peak | hybrid")
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--max-gap", type=float, default=30.0)
    p.add_argument("--hard-trigger", type=float, default=0.55)
    p.add_argument("--no-trigger", action="store_true")
    p.add_argument("--out", default="")
    p.set_defaults(fn=_cmd_route)

    p = sub.add_parser("cost", help="price policies without running a model")
    common(p)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--budget-per-hour", type=float, default=240.0)
    p.add_argument("--usd-per-call", type=float, default=0.002)
    p.set_defaults(fn=_cmd_cost)

    p = sub.add_parser("gate1", help="run the retrieval parity experiment")
    p.add_argument("video")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=_cmd_gate1)

    p = sub.add_parser("gate2", help="run the routing benchmark")
    p.add_argument("video")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=_cmd_gate2)

    return ap


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
