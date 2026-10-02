"""
bench.audit — the Frame Budget Audit in one command.

GO-TO-MARKET.md §3 is three commands and 30 minutes of attention. This module
is the same audit as one command that writes the one-page deliverable from §4,
so a prospect can be answered in minutes and the numbers can never be quoted
without their caveats attached.

    python -m bench.audit --video CUSTOMER.mp4 --price-per-call 0.005 \
        --volume-hours 500 --out results/CUSTOMER_audit.json

What it runs:
  1. retrieval vs independent ground truth  (bench.grounded_eval, OCR-mined)
  2. end-to-end indexing latency            (bench.latency)
  3. calls and money at the price you give  (counted, never estimated)

Honesty rules this module enforces:
  - OCR ground truth is optional. When it is unavailable, the audit says the
    saving is NOT quality-safe instead of quietly quoting a reduction.
  - The router-vs-uniform delta is always printed with its real sign. If
    uniform wins, the one-pager says so — the saving still stands; the picker
    just is not the reason.
  - Every report carries the sample size and the downstream-model caveat.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# The same margin bench.routing_bench.py uses to call a Gate-2 win.
ROUTER_MARGIN = 0.02


def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def _fmt_price(x: float) -> str:
    """Per-call prices live below a cent; 2 decimals would round 0.005 to 0.01."""
    return "$" + f"{x:.6f}".rstrip("0").rstrip(".")


@dataclass
class AuditReport:
    video: str
    generated_at: str
    duration_sec: float
    frames_analyzed: int
    domain: str
    price_per_call_usd: float
    volume_hours_per_month: Optional[float]

    # retrieval (None => not measured; never guess)
    retrieval_available: bool = False
    retrieval_note: str = ""
    n_queries: int = 0
    k: int = 5
    hit_full: float = 0.0
    hit_routed: float = 0.0
    hit_uniform: float = 0.0
    precision_full: float = 0.0
    precision_routed: float = 0.0
    precision_uniform: float = 0.0

    # calls
    full_calls: int = 0
    routed_calls: int = 0
    uniform_calls: int = 0

    # latency (optional)
    latency_available: bool = False
    latency_note: str = ""
    signal_ms_per_frame: float = 0.0
    routed_hour_minutes: float = 0.0
    full_hour_minutes: float = 0.0
    speedup_vs_full: float = 0.0

    # money
    usd_full_per_hour: float = 0.0
    usd_routed_per_hour: float = 0.0
    usd_saved_per_hour: float = 0.0
    saved_pct: float = 0.0
    usd_saved_per_month: Optional[float] = None

    verdict: str = ""
    caveats: List[str] = field(default_factory=list)

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def sec_per_call(self) -> float:
        return self.duration_sec / max(self.routed_calls, 1)

    @property
    def frames_per_call(self) -> float:
        return self.frames_analyzed / max(self.routed_calls, 1)

    @property
    def routed_per_hour(self) -> float:
        hours = self.duration_sec / 3600.0
        return self.routed_calls / hours if hours else 0.0

    @property
    def full_per_hour(self) -> float:
        hours = self.duration_sec / 3600.0
        return self.full_calls / hours if hours else 0.0

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    # ── the customer-facing page ─────────────────────────────────────────
    def one_pager(self) -> str:
        L = [
            f"# Frame Budget Audit — {Path(self.video).name}",
            "",
            f"**Video:** `{Path(self.video).name}`  ",
            f"**Footage:** {self.duration_sec:,.0f}s · {self.frames_analyzed:,} frames analysed"
            f" · domain: {self.domain}  ",
            f"**Price used:** {_fmt_price(self.price_per_call_usd)} per model call"
            + (f" · **volume:** {self.volume_hours_per_month:,.0f} video-hours/month"
               if self.volume_hours_per_month else "")
            + "  ",
            f"**Generated:** {self.generated_at}",
            "",
            "## Your safe budget",
            "",
            f"**1 model call every {self.sec_per_call:.1f} seconds of video** "
            f"({self.frames_per_call:.1f} frames per call).",
            "You currently send ~1 fps (a call every second).",
            "",
            "## Calls and money",
            "",
            "| | calls / video-hour | $ / video-hour |",
            "|---|---:|---:|",
            f"| full compute (every analysed frame) | {self.full_per_hour:,.0f} | "
            f"{_fmt_usd(self.usd_full_per_hour)} |",
            f"| frameroute | {self.routed_per_hour:,.0f} | "
            f"{_fmt_usd(self.usd_routed_per_hour)} |",
            "",
            f"**Saving: {_fmt_usd(self.usd_saved_per_hour)} per video-hour "
            f"({self.saved_pct:.0f}% fewer calls)**"
            + (f" — {_fmt_usd(self.usd_saved_per_month)} per month at "
               f"{self.volume_hours_per_month:,.0f} video-hours."
               if self.usd_saved_per_month is not None else "."),
            "",
        ]

        if self.retrieval_available:
            L += [
                "## Retrieval at that budget",
                "",
                f"Scored against the video's own slide text ({self.n_queries} OCR-mined "
                f"queries, top-{self.k}) — ground truth neither index produced.",
                "",
                f"| arm | calls | hit@{self.k} | precision@{self.k} |",
                "|---|---:|---:|---:|",
                f"| full | {self.full_calls} | {self.hit_full:.3f} | {self.precision_full:.3f} |",
                f"| **frameroute** | {self.routed_calls} | **{self.hit_routed:.3f}** | "
                f"{self.precision_routed:.3f} |",
                f"| uniform (same calls) | {self.uniform_calls} | {self.hit_uniform:.3f} | "
                f"{self.precision_uniform:.3f} |",
                "",
                f"**{self.verdict}**",
                "",
            ]
        else:
            reason = ("skipped by request (`--skip-retrieval`)"
                      if self.retrieval_note == "skipped by flag" else self.retrieval_note)
            L += [
                "## Retrieval",
                "",
                f"**Not measured** — {reason}.",
                "",
                "The saving above is a counted call reduction, **not** a quality-safe "
                "result, until retrieval is measured on slide/screen content.",
                "",
            ]

        if self.latency_available:
            if self.speedup_vs_full >= 1.05:
                speed = (f"- routed is **{self.speedup_vs_full:.1f}×** faster than "
                         "encoding every frame")
            else:
                speed = (f"- routed speedup on this sample: {self.speedup_vs_full:.2f}× — "
                         "within noise; do not quote it (re-measure on a longer clip)")
            L += [
                "## Speed",
                "",
                f"- change signal: **{self.signal_ms_per_frame:.2f} ms/frame** (CPU, hot path)",
                f"- 1 hour of video indexed in **{self.routed_hour_minutes:.1f} min** "
                f"(full compute: {self.full_hour_minutes:.1f} min)",
                speed,
                "",
            ]
        elif self.latency_note:
            reason = ("skipped by request (`--skip-latency`)"
                      if self.latency_note == "skipped by flag" else self.latency_note)
            L += ["## Speed", "", f"Not measured — {reason}.", ""]

        L += [
            "## Method",
            "",
            "- Budget: change-aware routing (`frameroute`, equal-accumulated-novelty) "
            "with hard-trigger and max-gap guarantees.",
            "- Quality: retrieval against ground truth from the video's own on-screen "
            "text (OCR), not a model grading itself.",
            "- Cost: counted model calls × the price you supplied. No estimates, no FLOPs.",
            "",
            "## Caveats",
            "",
        ]
        L += [f"- {c}" for c in self.caveats]
        L += [
            "",
            "## What we do not claim",
            "",
            "- That the router beats uniform sampling. On some footage it does, on some "
            "it ties or loses — the table above shows which, for this video.",
            "- That the saving is quality-safe when retrieval was not measured.",
            "",
        ]
        return "\n".join(L)


# --------------------------------------------------------------------------
# verdict + caveats — the honest part
# --------------------------------------------------------------------------

def _verdict(rep: AuditReport) -> tuple[str, List[str]]:
    caveats: List[str] = []

    if not rep.retrieval_available:
        verdict = ("Retrieval parity was not measured on this footage — do not quote "
                   "the saving as quality-safe.")
        if rep.retrieval_note == "skipped by flag":
            caveats.append(
                "Retrieval was skipped by request, so the saving is counted calls only."
            )
        else:
            caveats.append(
                f"Retrieval could not be measured ({rep.retrieval_note}). No OCR-mined "
                "queries were available; ask for slide/screen-heavy footage."
            )
    else:
        d_uniform = rep.hit_routed - rep.hit_uniform
        d_full = rep.hit_routed - rep.hit_full
        if d_uniform >= ROUTER_MARGIN:
            verdict = (f"Skipping frames beat uniform sampling at the same call count "
                       f"on this footage ({rep.hit_routed:.3f} vs {rep.hit_uniform:.3f} "
                       f"hit@{rep.k}, {d_uniform:+.3f}).")
        elif d_uniform <= -ROUTER_MARGIN:
            verdict = (f"Uniform sampling beat change-aware routing at the same call "
                       f"count on this footage ({rep.hit_uniform:.3f} vs "
                       f"{rep.hit_routed:.3f} hit@{rep.k}). The saving comes from "
                       "sending fewer frames, not from the picker.")
        else:
            verdict = (f"Router tied uniform sampling at the same call count on this "
                       f"footage ({rep.hit_routed:.3f} vs {rep.hit_uniform:.3f} "
                       f"hit@{rep.k}). The saving comes from sending fewer frames.")
        caveats.append(
            f"Router vs uniform on this video: {d_uniform:+.3f} hit@{rep.k} "
            f"({rep.hit_routed:.3f} vs {rep.hit_uniform:.3f}) at the same call count."
        )
        if d_full < -0.05:
            caveats.append(
                f"Routed hit@{rep.k} was {abs(d_full):.3f} below full compute "
                f"({rep.hit_routed:.3f} vs {rep.hit_full:.3f}); do not claim "
                "'same answers' on this footage."
            )

    caveats += [
        f"Sample size: {rep.n_queries} queries on {rep.duration_sec:,.0f}s "
        f"({rep.frames_analyzed:,} analysed frames). A signal, not yet a published benchmark.",
        "The saving depends entirely on the downstream model's price per call. At "
        f"{_fmt_price(rep.price_per_call_usd)}/call it is {_fmt_usd(rep.usd_saved_per_hour)} "
        "per video-hour; ask the customer for their real rate before quoting.",
    ]
    if rep.retrieval_available:
        caveats.append(
            "OCR-grounded hit rates are modest in absolute terms because CLIP reads text "
            "in images poorly; the comparison between arms is fair, the absolute number "
            "is not a product quality target."
        )
    return verdict, caveats


# --------------------------------------------------------------------------
# assembly — pure function over stage reports, so it can be tested offline
# --------------------------------------------------------------------------

def assemble(
    video: str,
    *,
    track=None,
    grounded=None,
    latency=None,
    budget: Optional[int] = None,
    price_per_call: float = 0.005,
    volume_hours: Optional[float] = None,
    domain: str = "generic",
    retrieval_note: str = "",
    latency_note: str = "",
) -> AuditReport:
    """
    Compose an AuditReport from already-run stage reports. Any stage may be
    missing; the report degrades honestly rather than inventing numbers.
    """
    if grounded is None and track is None:
        raise ValueError("assemble needs either a grounded report or an analyzed track")

    if grounded is not None:
        duration = grounded.duration_sec
        frames = grounded.frames_analyzed
        full_calls = grounded.full["model_calls"]
        routed_calls = grounded.routed["model_calls"]
        uniform_calls = grounded.uniform["model_calls"]
    else:
        from frameroute.policies import RouterPolicy, UniformN
        duration = track.duration_sec
        frames = track.n_analyzed
        b = budget or max(2, frames // 5)
        routed_calls = len(set(p.idx for p in RouterPolicy(policy="coverage").select(track, budget=b)))
        uniform_calls = len(set(p.idx for p in UniformN(routed_calls).select(track, budget=routed_calls)))
        full_calls = frames

    rep = AuditReport(
        video=video,
        generated_at=time.strftime("%Y-%m-%d %H:%M"),
        duration_sec=duration,
        frames_analyzed=frames,
        domain=domain,
        price_per_call_usd=price_per_call,
        volume_hours_per_month=volume_hours,
        full_calls=full_calls,
        routed_calls=routed_calls,
        uniform_calls=uniform_calls,
        retrieval_note=retrieval_note,
        latency_note=latency_note,
    )

    if grounded is not None:
        rep.retrieval_available = True
        rep.n_queries = grounded.n_queries
        rep.k = grounded.k
        rep.hit_full = grounded.full["hit_at_k"]
        rep.hit_routed = grounded.routed["hit_at_k"]
        rep.hit_uniform = grounded.uniform["hit_at_k"]
        rep.precision_full = grounded.full["precision_at_k"]
        rep.precision_routed = grounded.routed["precision_at_k"]
        rep.precision_uniform = grounded.uniform["precision_at_k"]

    if latency is not None:
        rep.latency_available = True
        rep.signal_ms_per_frame = latency.signal_ms_per_frame
        rep.routed_hour_minutes = latency.routed_hour_minutes
        rep.full_hour_minutes = latency.full_hour_minutes
        rep.speedup_vs_full = latency.speedup_vs_full

    rep.usd_full_per_hour = round(rep.full_per_hour * price_per_call, 2)
    rep.usd_routed_per_hour = round(rep.routed_per_hour * price_per_call, 2)
    rep.usd_saved_per_hour = round(rep.usd_full_per_hour - rep.usd_routed_per_hour, 2)
    rep.saved_pct = round(100.0 * (1.0 - routed_calls / max(full_calls, 1)), 1)
    if volume_hours:
        rep.usd_saved_per_month = round(rep.usd_saved_per_hour * volume_hours, 2)

    rep.verdict, rep.caveats = _verdict(rep)
    return rep


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------

def _budget_per_hour(video: str, budget: Optional[int]) -> float:
    """Convert a call budget over the video into calls-per-hour, cheaply."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = total / fps if total else 0.0
    if duration <= 0:
        return float(budget or 240)
    b = budget or max(2, int(total // 5))
    return b / (duration / 3600.0)


def run_audit(
    video: str,
    budget: Optional[int] = None,
    max_frames: int = 400,
    k: int = 5,
    n_queries: int = 20,
    weights: str = "default",
    device: Optional[str] = None,
    price_per_call: float = 0.005,
    volume_hours: Optional[float] = None,
    domain: str = "generic",
    skip_retrieval: bool = False,
    skip_latency: bool = False,
    retrieval_fn=None,
    latency_fn=None,
) -> AuditReport:
    """Run every stage that can run, then assemble the one-pager."""
    grounded = None
    retrieval_note = ""
    if not skip_retrieval:
        fn = retrieval_fn
        if fn is None:
            from bench.grounded_eval import run as fn  # type: ignore[no-redef]
        print(f"[audit] 1/3 retrieval vs OCR ground truth "
              f"(max_frames={max_frames}, queries={n_queries})", flush=True)
        try:
            grounded = fn(video, budget, max_frames, k, n_queries, weights, device)
        except Exception as e:
            retrieval_note = str(e)[:300] or e.__class__.__name__
            print(f"[audit] retrieval stage unavailable: {retrieval_note}", flush=True)
    else:
        retrieval_note = "skipped by flag"

    latency = None
    latency_note = ""
    if not skip_latency:
        fn = latency_fn
        if fn is None:
            from bench.latency import run as fn  # type: ignore[no-redef]
        bph = _budget_per_hour(video, budget)
        print(f"[audit] 2/3 latency (max_frames={max_frames}, budget={bph:.0f}/hour)", flush=True)
        try:
            latency = fn(video, max_frames, 1, bph, weights, device)
        except Exception as e:
            latency_note = str(e)[:300] or e.__class__.__name__
            print(f"[audit] latency stage unavailable: {latency_note}", flush=True)
    else:
        latency_note = "skipped by flag"

    # The track is only needed for call counts when retrieval could not run.
    track = None
    if grounded is None:
        from frameroute.signals import analyze_video
        print("[audit] analysing signals for call counts", flush=True)
        track = analyze_video(video, stride=1, max_frames=max_frames, weights=weights)

    print("[audit] 3/3 assembling the one-pager", flush=True)
    return assemble(
        video, track=track, grounded=grounded, latency=latency, budget=budget,
        price_per_call=price_per_call, volume_hours=volume_hours, domain=domain,
        retrieval_note=retrieval_note, latency_note=latency_note,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Frame Budget Audit: calls, money, retrieval and speed in one command."
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--budget", type=int, default=None,
                    help="model calls for the routed index (default: analysed frames / 5)")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--weights", default="default",
                    help="signal preset: default | static_cam | surveillance | ego_motion")
    ap.add_argument("--device", default=None)
    ap.add_argument("--domain", default="generic",
                    help="generic | traffic | surveillance | lecture | action")
    ap.add_argument("--price-per-call", type=float, default=0.005,
                    help="USD per downstream model call (ask the customer; do not guess)")
    ap.add_argument("--volume-hours", type=float, default=None,
                    help="customer's monthly video-hours, for the monthly projection")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="call counts + latency only; the saving is then NOT quality-safe")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--out", default="", help="write the full report JSON here")
    ap.add_argument("--md", default="", help="write the one-page markdown here")
    a = ap.parse_args()

    rep = run_audit(
        a.video, budget=a.budget, max_frames=a.max_frames, k=a.k,
        n_queries=a.n_queries, weights=a.weights, device=a.device,
        price_per_call=a.price_per_call, volume_hours=a.volume_hours,
        domain=a.domain, skip_retrieval=a.skip_retrieval, skip_latency=a.skip_latency,
    )
    page = rep.one_pager()
    print()
    print(page)
    if a.out:
        rep.to_json(a.out)
        print(f"[audit] wrote {a.out}")
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(page, encoding="utf-8")
        print(f"[audit] wrote {a.md}")

    # Non-zero only when the audit could not measure anything useful.
    if not rep.retrieval_available and not rep.latency_available:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
