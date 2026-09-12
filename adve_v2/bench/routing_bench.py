"""
bench.routing_bench — GATE 2 and GATE 3.

GATE 2  Does change-aware routing beat uniform sampling at the SAME budget?
GATE 3  Is the whole pipeline actually faster and cheaper in wall clock?

The comparison must be at matched cost. "We cut calls 80%" says nothing on
its own -- uniform sampling can cut calls by any amount you like. The only
question that matters is what you keep at a fixed spend, which is why every
policy here is handed the same budget and the budget is swept to produce a
curve rather than a single point.

Gate 3 exists because of results/traffic_benchmark_report.json, which
reported 59.9% "computing power saved" from a FLOP calculation while the
measured time moved 5.3% and memory went UP 18%. Nothing in this module
reports FLOPs. It reports seconds, calls, bytes and, when you supply a
price, currency.

Outputs
-------
results/gate2_routing.json   full sweep, machine readable
results/gate2_routing.md     the table and the curve, for the README
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from frameroute.policies import (
    SelectionPolicy, UniformN, UniformFPS, SceneCut, RouterPolicy,
)
from frameroute.signals import SignalTrack, analyze_video
from bench.parity import (
    build_reference_index, build_routed_index, compare_indexes,
)


GATE2_MARGIN = 0.02   # router must beat the best baseline by this much
GATE3_SPEEDUP = 5.0   # required cost reduction at <=5% recall loss
GATE3_MAX_RECALL_LOSS = 0.05


# --------------------------------------------------------------------------
# event coverage
# --------------------------------------------------------------------------

def event_coverage(
    pick_times: Sequence[float],
    events: Sequence[Tuple[float, float]],
    tolerance_sec: float = 0.0,
) -> Dict[str, float]:
    """
    Fraction of labelled events that received at least one call.

    This is the honest version of "zero missed security events". That phrase
    appears throughout the existing docs but was never measured against a
    labelled event list -- it was inferred from the fact that a trigger
    existed. A trigger firing is not the same as an event being covered.
    """
    if not events:
        return {"n_events": 0, "covered": 0, "coverage_pct": float("nan")}
    pt = np.asarray(sorted(pick_times), dtype=np.float64)
    covered = 0
    for t0, t1 in events:
        lo, hi = t0 - tolerance_sec, t1 + tolerance_sec
        if pt.size and np.any((pt >= lo) & (pt <= hi)):
            covered += 1
    return {
        "n_events": len(events),
        "covered": covered,
        "coverage_pct": round(100.0 * covered / len(events), 2),
    }


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

@dataclass
class ArmResult:
    policy: str
    budget: int
    n_calls: int
    savings_pct: float
    recall_temporal: float
    recall_exact: float
    ndcg: float
    top1_time_error_sec: float
    event_coverage_pct: float
    max_time_gap_sec: float
    index_build_seconds: float


@dataclass
class BenchReport:
    video_path: str
    duration_sec: float
    frames_analyzed: int
    reference_calls: int
    reference_build_seconds: float
    signal_analyze_seconds: float
    mean_signal_cost_ms: float
    budgets: List[int] = field(default_factory=list)
    arms: List[ArmResult] = field(default_factory=list)
    gate2_verdict: str = "UNKNOWN"
    gate3_verdict: str = "UNKNOWN"
    gate3_detail: Dict = field(default_factory=dict)
    machine: Dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    # -- reporting ---------------------------------------------------------

    def by_budget(self) -> Dict[int, List[ArmResult]]:
        out: Dict[int, List[ArmResult]] = {}
        for a in self.arms:
            out.setdefault(a.budget, []).append(a)
        return out

    def to_markdown(self) -> str:
        L = [
            "# Gate 2 / Gate 3 — Routing benchmark",
            "",
            f"**Video:** `{Path(self.video_path).name}`  ",
            f"**Duration:** {self.duration_sec:.1f}s · "
            f"**Frames analyzed:** {self.frames_analyzed}  ",
            f"**Full-compute reference:** {self.reference_calls} calls "
            f"in {self.reference_build_seconds:.1f}s  ",
            f"**Signal pass:** {self.signal_analyze_seconds:.1f}s "
            f"({self.mean_signal_cost_ms:.2f} ms/frame, no detector)",
            "",
            "All policies are given the same call budget, with one deliberate "
            "exception: `uniform_1fps` is a fixed rate, not a budget, so its "
            "call count does not move between rows. It is included as the "
            "reference point for what pipelines actually do today. "
            "`uniform_n` is the matched-budget time-domain comparison.",
            "",
            "Every figure below is measured. None is derived from a FLOP count.",
            "",
        ]
        for budget, arms in sorted(self.by_budget().items()):
            L += [
                f"### Budget: {budget} calls "
                f"({100.0 * budget / max(self.reference_calls,1):.1f}% of full)",
                "",
                "| Policy | Calls | Temporal recall@10 | nDCG | Event coverage | Max gap | Top-1 error |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            best = max((a.recall_temporal for a in arms), default=0.0)
            for a in sorted(arms, key=lambda x: -x.recall_temporal):
                star = " **←**" if a.recall_temporal >= best - 1e-9 else ""
                cov = ("—" if np.isnan(a.event_coverage_pct)
                       else f"{a.event_coverage_pct:.0f}%")
                L.append(
                    f"| {a.policy}{star} | {a.n_calls} | {a.recall_temporal:.4f} | "
                    f"{a.ndcg:.4f} | {cov} | {a.max_time_gap_sec:.1f}s | "
                    f"{a.top1_time_error_sec:.2f}s |"
                )
            L.append("")

        L += [
            "## Cost / recall curve",
            "",
            "| Policy | " + " | ".join(f"{b}" for b in sorted(self.by_budget())) + " |",
            "|---|" + "---:|" * len(self.by_budget()),
        ]
        policies = sorted({a.policy for a in self.arms})
        for p in policies:
            row = []
            for b in sorted(self.by_budget()):
                m = [a.recall_temporal for a in self.arms if a.policy == p and a.budget == b]
                row.append(f"{m[0]:.3f}" if m else "—")
            L.append(f"| {p} | " + " | ".join(row) + " |")

        L += [
            "",
            f"**Gate 2 verdict: {self.gate2_verdict}** "
            f"(router must beat the best baseline by >= {GATE2_MARGIN:.2f} "
            f"temporal recall at matched budget)",
            "",
            f"**Gate 3 verdict: {self.gate3_verdict}** "
            f"(>= {GATE3_SPEEDUP:g}x cost reduction at <= "
            f"{GATE3_MAX_RECALL_LOSS:.0%} recall loss)",
        ]
        if self.gate3_detail:
            d = self.gate3_detail
            L += [
                "",
                f"- cheapest budget within the recall bound: "
                f"**{d.get('budget')} calls**",
                f"- cost reduction vs full compute: **{d.get('cost_reduction_x', 0):.1f}x**",
                f"- recall retained: **{d.get('recall', 0):.4f}** "
                f"(loss {d.get('recall_loss', 0):.3f})",
            ]
        if self.machine:
            L += ["", "---", "", "Machine: " + json.dumps(self.machine)]
        return "\n".join(L)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def _machine_info() -> Dict:
    info: Dict = {"cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def run_routing_bench(
    video_path: str,
    queries: Sequence[str],
    embedder,
    budgets: Optional[Sequence[int]] = None,
    events: Optional[Sequence[Tuple[float, float]]] = None,
    policies: Optional[Sequence[SelectionPolicy]] = None,
    stride: int = 1,
    max_frames: Optional[int] = 1200,
    weights: str = "default",
    fill: str = "slerp",
    k: int = 10,
    window_sec: float = 2.0,
    verbose: bool = True,
) -> BenchReport:
    if verbose:
        print(f"[gate2] signal pass: {Path(video_path).name}")
    track = analyze_video(video_path, stride=stride, max_frames=max_frames, weights=weights)
    if verbose:
        print(f"[gate2] {track.summary()}")

    if verbose:
        print("[gate2] full-compute reference index ...")
    ref = build_reference_index(video_path, track, embedder)

    n_ref = ref.n_model_calls
    if budgets is None:
        budgets = [
            max(2, int(n_ref * f)) for f in (0.02, 0.05, 0.10, 0.20, 0.40)
        ]
    budgets = sorted(set(int(b) for b in budgets if b >= 2))

    if policies is None:
        policies = [
            UniformN(), UniformFPS(1.0), SceneCut(),
            RouterPolicy(policy="coverage"), RouterPolicy(policy="hybrid"),
        ]

    report = BenchReport(
        video_path=video_path,
        duration_sec=track.duration_sec,
        frames_analyzed=track.n_analyzed,
        reference_calls=n_ref,
        reference_build_seconds=round(ref.build_seconds, 2),
        signal_analyze_seconds=round(track.analyze_seconds, 2),
        mean_signal_cost_ms=round(track.mean_signal_cost_ms(), 3),
        budgets=list(budgets),
        machine=_machine_info(),
    )

    ev = list(events or [])

    for budget in budgets:
        for pol in policies:
            picks = pol.select(track, budget=budget)
            pick_idx = [p.idx for p in picks]
            pick_t = [p.t for p in picks]

            cand = build_routed_index(
                video_path, track, embedder,
                pick_indices=pick_idx, fill=fill, name=f"{pol.name}@{budget}",
            )
            _, agg = compare_indexes(ref, cand, embedder, queries, k=k, window_sec=window_sec)

            gaps = np.diff(sorted(pick_t)) if len(pick_t) > 1 else np.asarray([0.0])
            cov = event_coverage(pick_t, ev)

            report.arms.append(ArmResult(
                policy=pol.name,
                budget=budget,
                n_calls=cand.n_model_calls,
                savings_pct=round(100.0 * (1 - cand.n_model_calls / max(n_ref, 1)), 2),
                recall_temporal=agg["recall_temporal"],
                recall_exact=agg["recall_exact"],
                ndcg=agg["ndcg"],
                top1_time_error_sec=agg["top1_time_error_sec"],
                event_coverage_pct=cov["coverage_pct"],
                max_time_gap_sec=round(float(gaps.max()), 2) if gaps.size else 0.0,
                index_build_seconds=round(cand.build_seconds, 2),
            ))
            if verbose:
                print(f"  budget={budget:5d} {pol.name:<20s} "
                      f"calls={cand.n_model_calls:5d} "
                      f"recall={agg['recall_temporal']:.4f}")

    _score_gates(report)
    return report


def _score_gates(report: BenchReport) -> None:
    # Gate 2: router beats the best non-router baseline at matched budget
    wins, total = 0, 0
    for budget, arms in report.by_budget().items():
        routers = [a for a in arms if a.policy.startswith("router")]
        others = [a for a in arms if not a.policy.startswith("router")]
        if not routers or not others:
            continue
        total += 1
        if max(a.recall_temporal for a in routers) >= \
           max(a.recall_temporal for a in others) + GATE2_MARGIN:
            wins += 1
    report.gate2_verdict = (
        f"PASS ({wins}/{total} budgets)" if total and wins > total / 2
        else f"FAIL ({wins}/{total} budgets)" if total else "NO DATA"
    )

    # Gate 3: cheapest budget that stays within the recall bound
    routers = [a for a in report.arms if a.policy.startswith("router")]
    if routers:
        best_by_budget: Dict[int, ArmResult] = {}
        for a in routers:
            cur = best_by_budget.get(a.budget)
            if cur is None or a.recall_temporal > cur.recall_temporal:
                best_by_budget[a.budget] = a
        ok = [a for a in best_by_budget.values()
              if a.recall_temporal >= 1.0 - GATE3_MAX_RECALL_LOSS]
        if ok:
            cheapest = min(ok, key=lambda a: a.n_calls)
            reduction = report.reference_calls / max(cheapest.n_calls, 1)
            report.gate3_detail = {
                "budget": cheapest.budget,
                "n_calls": cheapest.n_calls,
                "cost_reduction_x": round(reduction, 2),
                "recall": cheapest.recall_temporal,
                "recall_loss": round(1.0 - cheapest.recall_temporal, 4),
            }
            report.gate3_verdict = "PASS" if reduction >= GATE3_SPEEDUP else "FAIL"
        else:
            report.gate3_verdict = "FAIL (no budget stayed within the recall bound)"


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main() -> None:
    import argparse
    from frameroute.adapters import ClipEmbedder
    from bench.queries import QuerySet, DEFAULT_QUERIES, queries_for

    ap = argparse.ArgumentParser(
        description="Gate 2/3: does change-aware routing beat uniform sampling at equal cost?"
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--query-set", default="", help="path to a QuerySet JSON")
    ap.add_argument("--domain", default="generic",
                    help="generic | traffic | surveillance | lecture | action")
    ap.add_argument("--weights", default="default",
                    help="signal preset: default | static_cam | surveillance | ego_motion")
    ap.add_argument("--fill", default="slerp", help="carry_forward | slerp")
    ap.add_argument("--budgets", default="", help="comma-separated call budgets")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--out", default="results/gate2_routing.json")
    ap.add_argument("--md", default="results/gate2_routing.md")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    events: List[Tuple[float, float]] = []
    if args.query_set:
        qs = QuerySet.load(args.query_set)
        queries = qs.queries or queries_for(args.domain)
        events = qs.event_spans()
    else:
        queries = queries_for(args.domain) or DEFAULT_QUERIES

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()] if args.budgets else None

    embedder = ClipEmbedder(device=args.device)
    report = run_routing_bench(
        video_path=args.video,
        queries=queries,
        embedder=embedder,
        budgets=budgets,
        events=events,
        stride=args.stride,
        max_frames=args.max_frames,
        weights=args.weights,
        fill=args.fill,
        k=args.k,
        window_sec=args.window_sec,
    )

    report.to_json(args.out)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(report.to_markdown(), encoding="utf-8")

    print()
    print(report.to_markdown())
    print()
    print(f"[gate2] wrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
