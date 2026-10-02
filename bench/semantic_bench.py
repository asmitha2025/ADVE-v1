"""
bench.semantic_bench — does CONTENT-space novelty beat pixel novelty?

The router has no demonstrated edge over uniform sampling on lectures: it won
one clip (+25%) and lost the other (-13%) using pixel-space change. This
experiment tests the obvious next hypothesis: on slide-heavy content, the
thing that changes is the TEXT, and pixel histograms are a poor proxy for it.

Four arms, scored against OCR-mined ground truth (neither index produces it):

    full          every analysed frame, the reference
    uniform       equal-time sampling, matched to the routed call counts
    router        change-aware routing on pixel novelty (today's product)
    router+text   change-aware routing on max(pixel, text change)

If `router+text` beats `router` and `uniform` at matched calls on a lecture,
content-space routing is the product's edge. If it does not, that is the
result and the README says so.

    python -m bench.semantic_bench --video LECTURE.mp4 --budget 80 \
        --max-frames 400 --n-queries 20

Cost note: OCR runs on every analysed frame here because the labels come from
the same pass. Production would OCR a sparse subset (frameroute.semantics
supports that) — this harness deliberately overpays to keep the label set and
the signal in one measurement.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from frameroute.semantics import (
    TextObservation, fuse_track, text_change_scores, text_novelty_series,
)


@dataclass
class SemanticReport:
    video: str
    frames_analyzed: int
    duration_sec: float
    budget: int
    k: int
    n_queries: int
    text_weight: float
    text_observations: int = 0
    text_observations_nonempty: int = 0
    text_changes_detected: int = 0
    full: Dict = field(default_factory=dict)
    uniform: Dict = field(default_factory=dict)
    router: Dict = field(default_factory=dict)
    router_text: Dict = field(default_factory=dict)
    router_vs_uniform: float = 0.0
    text_vs_uniform: float = 0.0
    text_vs_router: float = 0.0

    def table(self) -> str:
        def row(name, a):
            return (f"  {name:<18}{a['model_calls']:>7}"
                    f"{a['hit_at_k']:>12.3f}{a['precision_at_k']:>14.3f}")
        L = [
            "",
            f"  CONTENT-SPACE ROUTING — {Path(self.video).name}",
            f"  {self.frames_analyzed} frames analysed · {self.duration_sec:.0f}s · "
            f"{self.n_queries} OCR-grounded queries · top-{self.k} · "
            f"text weight {self.text_weight:g}",
            f"  text observations: {self.text_observations_nonempty}/"
            f"{self.text_observations} non-empty · "
            f"{self.text_changes_detected} changes detected",
            "  " + "-" * 56,
            f"  {'arm':<18}{'calls':>7}{'hit@k':>12}{'precision@k':>14}",
            row("full", self.full),
            row("uniform", self.uniform),
            row("router", self.router),
            row("router+text", self.router_text),
            "  " + "-" * 56,
            f"  router vs uniform:      {self.router_vs_uniform:+.3f} hit@k",
            f"  router+text vs uniform: {self.text_vs_uniform:+.3f} hit@k",
            f"  router+text vs router:  {self.text_vs_router:+.3f} hit@k",
            "",
        ]
        return "\n".join(L)


def run(video: str, budget: Optional[int], max_frames: int, k: int,
        n_queries: int, weights: str, device: Optional[str],
        text_weight: float = 1.0, text_cache: str = "") -> SemanticReport:
    import cv2
    from frameroute.policies import RouterPolicy, UniformN
    from frameroute.adapters import ClipEmbedder
    from frameroute.signals import analyze_video
    from bench.parity import build_reference_index, build_routed_index, read_frames
    from bench.grounded_eval import ocr_frames, mine_labels, score_index

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    stride = max(1, total // max_frames) if total else 1

    print(f"[semantic] analyzing (stride={stride}, max_frames={max_frames})", flush=True)
    track = analyze_video(video, stride=stride, max_frames=max_frames, weights=weights)

    print(f"[semantic] decoding {track.n_analyzed} frames once", flush=True)
    frames = read_frames(video, [s.idx for s in track.signals])

    cache_path = Path(text_cache) if text_cache else None
    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        frame_text = {int(k): v for k, v in raw.items()}
        print(f"[semantic] loaded OCR cache ({len(frame_text)} frames) from {cache_path}",
              flush=True)
    else:
        print("[semantic] OCR for ground truth AND for the text signal …", flush=True)
        t0 = time.perf_counter()
        frame_text = ocr_frames(frames, device=device or "cuda")
        print(f"[semantic] OCR took {time.perf_counter()-t0:.0f}s", flush=True)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(frame_text, ensure_ascii=False),
                                  encoding="utf-8")
            print(f"[semantic] cached OCR text to {cache_path}", flush=True)

    labels = mine_labels(frame_text, n_queries=n_queries)
    if not labels:
        raise RuntimeError("no OCR-grounded queries found — try a slide-heavy video")
    print(f"[semantic] {len(labels)} queries: {list(labels)[:8]}", flush=True)

    # Text observations ON the analysed frames (same pass as the labels).
    t_of = {s.idx: s.t for s in track.signals}
    obs = [TextObservation(t=t_of[i], text=frame_text.get(i, ""))
           for i in sorted(frame_text)]
    sem = text_novelty_series(obs, track.time_array())
    fused = fuse_track(track, sem, weight=text_weight)
    n_changes = int((text_change_scores(obs) > 0).sum())

    emb = ClipEmbedder(device=device)
    b = budget or max(2, track.n_analyzed // 5)

    print(f"[semantic] building indexes (budget={b})", flush=True)
    full = build_reference_index(video, track, emb, frames=frames)

    def picks_of(t):
        return sorted(set(p.idx for p in
                          RouterPolicy(policy="coverage").select(t, budget=b)))

    r_picks = picks_of(track)
    t_picks = picks_of(fused)
    n_match = max(len(r_picks), len(t_picks))
    u_picks = sorted(set(p.idx for p in UniformN(n_match).select(track, budget=n_match)))

    routed = build_routed_index(video, track, emb, pick_indices=r_picks,
                                fill="slerp", name="router", frames=frames)
    text_arm = build_routed_index(video, fused, emb, pick_indices=t_picks,
                                  fill="slerp", name="router_text", frames=frames)
    uniform = build_routed_index(video, track, emb, pick_indices=u_picks,
                                 fill="slerp", name="uniform", frames=frames)

    s_full = score_index(full, labels, emb, k=k)
    s_uni = score_index(uniform, labels, emb, k=k)
    s_r = score_index(routed, labels, emb, k=k)
    s_t = score_index(text_arm, labels, emb, k=k)

    return SemanticReport(
        video=video, frames_analyzed=track.n_analyzed, duration_sec=track.duration_sec,
        budget=b, k=k, n_queries=len(labels), text_weight=text_weight,
        text_observations=len(obs),
        text_observations_nonempty=sum(1 for o in obs if o.text),
        text_changes_detected=n_changes,
        full=asdict(s_full), uniform=asdict(s_uni),
        router=asdict(s_r), router_text=asdict(s_t),
        router_vs_uniform=round(s_r.hit_at_k - s_uni.hit_at_k, 4),
        text_vs_uniform=round(s_t.hit_at_k - s_uni.hit_at_k, 4),
        text_vs_router=round(s_t.hit_at_k - s_r.hit_at_k, 4),
    )


def summarize(paths: List[str]) -> str:
    """Aggregate result JSONs into the table the README quotes."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.append(d)
    if not rows:
        return "(no result files)"
    rows.sort(key=lambda d: Path(d["video"]).name)

    lines = [
        "| lecture | calls | full | uniform | router | router+text |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in rows:
        name = Path(d["video"]).name
        calls = d["router"]["model_calls"]
        lines.append(
            f"| {name} | {calls} | {d['full']['hit_at_k']:.3f} | "
            f"{d['uniform']['hit_at_k']:.3f} | {d['router']['hit_at_k']:.3f} | "
            f"{d['router_text']['hit_at_k']:.3f} |"
        )
    n = len(rows)
    mean = lambda k: sum(d[k] for d in rows) / n  # noqa: E731
    lines += [
        "",
        f"Mean over {n} lectures (hit@5, matched calls): "
        f"router {mean('router_vs_uniform'):+.3f} vs uniform, "
        f"router+text {mean('text_vs_uniform'):+.3f} vs uniform, "
        f"text-vs-pixel {mean('text_vs_router'):+.3f}.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Does content-space (OCR) novelty beat pixel novelty at matched calls?"
    )
    ap.add_argument("--video", default="")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--weights", default="default")
    ap.add_argument("--device", default=None)
    ap.add_argument("--text-weight", type=float, default=1.0,
                    help="fusion weight on the text signal (floor is weight*text change)")
    ap.add_argument("--text-cache", default="",
                    help="JSON file to cache OCR text; re-runs reuse it instead of paying "
                         "the OCR pass again")
    ap.add_argument("--summary", default="",
                    help="glob of result JSONs to aggregate instead of running "
                         "(e.g. 'results/semantic_*.json')")
    ap.add_argument("--out", default="results/semantic_bench.json")
    ap.add_argument("--md", default="results/semantic_bench.md")
    a = ap.parse_args()

    if a.summary:
        import glob
        table = summarize(sorted(glob.glob(a.summary)))
        print(table)
        Path("results/semantic_summary.md").write_text(table + "\n", encoding="utf-8")
        print("[semantic] wrote results/semantic_summary.md")
        return 0

    if not a.video:
        ap.error("--video is required unless --summary is used")

    rep = run(a.video, a.budget, a.max_frames, a.k, a.n_queries,
              a.weights, a.device, text_weight=a.text_weight,
              text_cache=a.text_cache)
    print(rep.table())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    Path(a.md).write_text(
        "# Content-space routing benchmark\n\n```\n" + rep.table() + "\n```\n",
        encoding="utf-8",
    )
    print(f"[semantic] wrote {a.out} and {a.md}")
    print("[semantic] publish this result whatever it says.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
