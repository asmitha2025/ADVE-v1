"""
bench.parity — GATE 1: does skipping frames preserve retrieval?

This is the experiment the ADVE project never ran, and everything else in
the repo is downstream of its answer.

Every accuracy number in docs/ is cosine similarity between a reconstructed
embedding and the true CLIP embedding. That is the wrong metric, for a
specific reason: in CLIP space two unrelated frames from the same fixed
camera routinely sit at 0.85-0.95, because the background dominates the
vector. A mean of 0.9753 therefore tells you almost nothing about whether
search still works. It is a number that cannot fail.

The number that can fail is retrieval parity:

    given a query, does the cheap index return the same moments
    as the index that spent full compute on every frame?

That is what this module measures. If parity holds, the efficiency story is
real and defensible. If it does not, four generations of reconstructor were
optimising a metric that never mattered -- and it is much better to learn
that in week one than in year three.

Two modes
---------
oracle    (no labels needed, runnable today)
          The full-compute index is the reference. Parity = overlap between
          the cheap index's top-k and the reference's top-k, measured with a
          temporal tolerance so that returning frame 1043 instead of 1041
          is not scored as an error.

labelled  (the version that goes in a public benchmark)
          Human-labelled relevant time ranges per query. Both indexes are
          scored independently against ground truth. Slower to prepare,
          immune to the criticism that the oracle is itself just CLIP.

Fill strategies for the frames that did NOT get a call
------------------------------------------------------
carry_forward   reuse the last selected embedding. What a routed pipeline
                actually yields with no model at all. THE BASELINE TO BEAT.
slerp           spherical interpolation between the two neighbouring
                selected embeddings. Zero parameters, zero training.
adve            plug in the trained DeltaReconstructorV3 / UALW.

The comparison that matters is adve vs slerp. The repo has never run it. If
a trained GRU cannot beat spherical interpolation between its own anchors,
the training pipeline is not earning its complexity -- and slerp should ship
instead, because it has no checkpoint, no drift and no failure mode.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from frameroute.adapters import CountingEmbedder, ExactIndex
from frameroute.signals import SignalTrack, analyze_video


# --------------------------------------------------------------------------
# frame access
# --------------------------------------------------------------------------

def read_frames(video_path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """
    Sequential read of the requested frame indices. Sequential rather than
    seek-based because seeking in long-GOP H.264 is both slow and, on some
    builds, inexact -- and an off-by-a-few-frames read would silently
    corrupt a parity measurement.
    """
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    out: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    ptr, idx = 0, 0
    while ptr < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx == wanted[ptr]:
            out[idx] = frame
            ptr += 1
        idx += 1
    cap.release()
    return out


# --------------------------------------------------------------------------
# fills
# --------------------------------------------------------------------------

def _slerp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Spherical interpolation on the unit sphere; falls back to lerp when close."""
    a = a / max(float(np.linalg.norm(a)), 1e-8)
    b = b / max(float(np.linalg.norm(b)), 1e-8)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9995:
        out = a + w * (b - a)
    else:
        theta = np.arccos(dot)
        st = np.sin(theta)
        out = (np.sin((1 - w) * theta) / st) * a + (np.sin(w * theta) / st) * b
    return out / max(float(np.linalg.norm(out)), 1e-8)


def fill_carry_forward(
    all_idx: List[int], pick_idx: List[int], pick_vecs: np.ndarray, **_
) -> np.ndarray:
    """Every unselected frame inherits the most recent selected embedding."""
    order = np.argsort(pick_idx)
    pi = np.asarray(pick_idx)[order]
    pv = pick_vecs[order]
    pos = np.searchsorted(pi, np.asarray(all_idx), side="right") - 1
    pos = np.clip(pos, 0, len(pi) - 1)
    return pv[pos]


def fill_slerp(
    all_idx: List[int], pick_idx: List[int], pick_vecs: np.ndarray, **_
) -> np.ndarray:
    """Interpolate between the bracketing selected embeddings."""
    order = np.argsort(pick_idx)
    pi = np.asarray(pick_idx, dtype=np.float64)[order]
    pv = pick_vecs[order]
    out = np.zeros((len(all_idx), pv.shape[1]), dtype=np.float32)
    for row, i in enumerate(all_idx):
        r = int(np.searchsorted(pi, i, side="right"))
        lo = max(0, r - 1)
        hi = min(len(pi) - 1, r)
        if lo == hi or pi[hi] == pi[lo]:
            out[row] = pv[lo]
        else:
            w = (i - pi[lo]) / (pi[hi] - pi[lo])
            out[row] = _slerp(pv[lo], pv[hi], float(np.clip(w, 0.0, 1.0)))
    return out


FILLS: Dict[str, Callable] = {
    "carry_forward": fill_carry_forward,
    "slerp": fill_slerp,
}


# --------------------------------------------------------------------------
# indexes
# --------------------------------------------------------------------------

@dataclass
class BuiltIndex:
    name: str
    index: ExactIndex
    frame_idx: List[int]
    times: List[float]
    n_model_calls: int
    build_seconds: float
    ledger: Dict = field(default_factory=dict)


def build_reference_index(
    video_path: str, track: SignalTrack, embedder, batch: int = 64,
    frames: Optional[Dict[int, np.ndarray]] = None,
) -> BuiltIndex:
    """Full compute: an embedding for every analyzed frame. The upper bound.

    Pass `frames` (a pre-decoded {idx: frame} cache) to avoid re-decoding the
    video — the sweep decodes once and reuses it across every index build."""
    t0 = time.perf_counter()
    idxs = [s.idx for s in track.signals]
    times = [s.t for s in track.signals]
    counting = CountingEmbedder(embedder)

    vecs: List[np.ndarray] = []
    for i in range(0, len(idxs), batch):
        chunk = idxs[i:i + batch]
        frames_map = ({j: frames[j] for j in chunk if j in frames}
                      if frames is not None else read_frames(video_path, chunk))
        chunk_frames = [frames_map[j] for j in chunk if j in frames_map]
        if chunk_frames:
            vecs.append(counting.embed_frames(chunk_frames))
    V = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, embedder.dim), np.float32)
    n = V.shape[0]

    return BuiltIndex(
        name="reference_full",
        index=ExactIndex(V, [{"frame_idx": idxs[i], "t": times[i]} for i in range(n)]),
        frame_idx=idxs[:n],
        times=times[:n],
        n_model_calls=counting.ledger.frame_calls,
        build_seconds=time.perf_counter() - t0,
        ledger=counting.ledger.as_dict(),
    )


def build_routed_index(
    video_path: str,
    track: SignalTrack,
    embedder,
    pick_indices: Sequence[int],
    fill: str = "carry_forward",
    name: Optional[str] = None,
    custom_fill: Optional[Callable] = None,
    frames: Optional[Dict[int, np.ndarray]] = None,
) -> BuiltIndex:
    """
    Spend model calls only on `pick_indices`; synthesise the rest.

    The resulting index has the same number of rows as the reference, which
    is what makes the comparison fair: both indexes can return any moment in
    the video, they just differ in how much compute produced each row.

    Pass `frames` (a pre-decoded {idx: frame} cache) to skip re-decoding.
    """
    t0 = time.perf_counter()
    all_idx = [s.idx for s in track.signals]
    times = [s.t for s in track.signals]

    picks = sorted(set(int(i) for i in pick_indices if i in set(all_idx)))
    if not picks:
        picks = [all_idx[0]]

    counting = CountingEmbedder(embedder)
    frames_map = ({i: frames[i] for i in picks if i in frames}
                  if frames is not None else read_frames(video_path, picks))
    ordered = [i for i in picks if i in frames_map]
    pv = counting.embed_frames([frames_map[i] for i in ordered])

    fn = custom_fill or FILLS.get(fill)
    if fn is None:
        raise ValueError(f"unknown fill {fill!r}; choose from {sorted(FILLS)} or pass custom_fill")

    V = fn(all_idx=all_idx, pick_idx=ordered, pick_vecs=pv, track=track)
    V = np.asarray(V, dtype=np.float32)

    return BuiltIndex(
        name=name or f"routed_{fill}",
        index=ExactIndex(V, [{"frame_idx": all_idx[i], "t": times[i]} for i in range(len(all_idx))]),
        frame_idx=all_idx,
        times=times,
        n_model_calls=counting.ledger.frame_calls,
        build_seconds=time.perf_counter() - t0,
        ledger=counting.ledger.as_dict(),
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    recall_exact: float
    recall_temporal: float
    ndcg: float
    top1_time_error_sec: float
    ref_times: List[float]
    cand_times: List[float]


def _dcg(gains: Sequence[float]) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def compare_indexes(
    reference: BuiltIndex,
    candidate: BuiltIndex,
    embedder,
    queries: Sequence[str],
    k: int = 10,
    window_sec: float = 2.0,
) -> Tuple[List[QueryResult], Dict[str, float]]:
    """
    Oracle-mode parity.

    recall_exact      strict frame-id overlap of the two top-k sets
    recall_temporal   a candidate hit counts if it lands within window_sec of
                      ANY reference hit -- the honest metric, because
                      neighbouring frames are the same moment
    ndcg              candidate ordering scored against reference similarity
                      as graded relevance
    top1_time_error   how far the best result moved, in seconds
    """
    qvecs = embedder.embed_text(list(queries))
    per_query: List[QueryResult] = []

    for qi, q in enumerate(queries):
        r_ids, r_scores = reference.index.search(qvecs[qi], k=k)
        c_ids, _ = candidate.index.search(qvecs[qi], k=k)

        r_times = [reference.times[i] for i in r_ids]
        c_times = [candidate.times[i] for i in c_ids]
        r_frames = {reference.frame_idx[i] for i in r_ids}
        c_frames = [candidate.frame_idx[i] for i in c_ids]

        exact = len(r_frames & set(c_frames)) / max(len(r_frames), 1)

        hits = sum(
            1 for ct in c_times if any(abs(ct - rt) <= window_sec for rt in r_times)
        )
        temporal = hits / max(len(r_times), 1)

        # graded relevance from the reference's own similarity scores
        rel = {reference.times[i]: float(s) for i, s in zip(r_ids, r_scores)}
        gains = []
        for ct in c_times:
            best = 0.0
            for rt, s in rel.items():
                if abs(ct - rt) <= window_sec:
                    best = max(best, s)
            gains.append(best)
        ideal = sorted(rel.values(), reverse=True)
        ndcg = _dcg(gains) / _dcg(ideal) if ideal and _dcg(ideal) > 0 else 0.0

        t1err = abs(c_times[0] - r_times[0]) if c_times and r_times else float("nan")

        per_query.append(QueryResult(
            query=q,
            recall_exact=round(exact, 4),
            recall_temporal=round(temporal, 4),
            ndcg=round(float(np.clip(ndcg, 0.0, 1.0)), 4),
            top1_time_error_sec=round(float(t1err), 3),
            ref_times=[round(t, 2) for t in r_times],
            cand_times=[round(t, 2) for t in c_times],
        ))

    agg = {
        "recall_exact": round(float(np.mean([r.recall_exact for r in per_query])), 4),
        "recall_temporal": round(float(np.mean([r.recall_temporal for r in per_query])), 4),
        "ndcg": round(float(np.mean([r.ndcg for r in per_query])), 4),
        "top1_time_error_sec": round(
            float(np.nanmean([r.top1_time_error_sec for r in per_query])), 3
        ),
        "n_queries": len(per_query),
        "k": k,
        "window_sec": window_sec,
    }
    return per_query, agg


def score_labelled(
    built: BuiltIndex,
    embedder,
    labelled: Dict[str, List[Tuple[float, float]]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Labelled-mode scoring: precision@k and interval recall against
    human-marked relevant time ranges. Independent of any oracle.
    """
    queries = list(labelled.keys())
    if not queries:
        return {}
    qvecs = embedder.embed_text(queries)

    precisions, interval_recalls = [], []
    for qi, q in enumerate(queries):
        ids, _ = built.index.search(qvecs[qi], k=k)
        times = [built.times[i] for i in ids]
        spans = labelled[q]

        hit = [any(s <= t <= e for s, e in spans) for t in times]
        precisions.append(sum(hit) / max(len(hit), 1))

        covered = sum(
            1 for s, e in spans if any(s <= t <= e for t in times)
        )
        interval_recalls.append(covered / max(len(spans), 1))

    return {
        "precision_at_k": round(float(np.mean(precisions)), 4),
        "interval_recall": round(float(np.mean(interval_recalls)), 4),
        "n_queries": len(queries),
        "k": k,
    }


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

GATE1_THRESHOLD = 0.95   # mean temporal recall parity required to pass


@dataclass
class ParityReport:
    video_path: str
    frames_analyzed: int
    duration_sec: float
    reference_calls: int
    arms: Dict[str, Dict] = field(default_factory=dict)
    verdict: str = "UNKNOWN"
    threshold: float = GATE1_THRESHOLD

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def to_markdown(self) -> str:
        L = [
            "# Gate 1 — Retrieval Parity",
            "",
            f"**Video:** `{Path(self.video_path).name}`  ",
            f"**Frames analyzed:** {self.frames_analyzed}  ",
            f"**Duration:** {self.duration_sec:.1f}s  ",
            f"**Reference model calls:** {self.reference_calls}",
            "",
            "Parity is measured against a full-compute index using exact "
            "(not approximate) search, with a temporal tolerance so that "
            "neighbouring frames count as the same moment.",
            "",
            "| Arm | Calls | Saved | Temporal recall@10 | Exact recall | nDCG | Top-1 error | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|:--|",
        ]
        for name, a in self.arms.items():
            m = a.get("metrics", {})
            saved = a.get("savings_pct", 0.0)
            tr = m.get("recall_temporal", 0.0)
            mark = "PASS" if tr >= self.threshold else "FAIL"
            L.append(
                f"| {name} | {a.get('n_model_calls', 0)} | {saved:.1f}% | "
                f"{tr:.4f} | {m.get('recall_exact', 0):.4f} | "
                f"{m.get('ndcg', 0):.4f} | {m.get('top1_time_error_sec', 0):.2f}s | {mark} |"
            )
        L += [
            "",
            f"**Gate 1 verdict: {self.verdict}** "
            f"(pass requires temporal recall >= {self.threshold})",
            "",
            "Savings are a count of model calls that did not happen. They are "
            "not a wall-clock claim; measure that separately with "
            "`bench/routing_bench.py`, which times the whole pipeline.",
        ]
        return "\n".join(L)


def run_gate1(
    video_path: str,
    queries: Sequence[str],
    embedder,
    arms: Optional[Dict[str, Dict]] = None,
    stride: int = 1,
    max_frames: Optional[int] = 1200,
    k: int = 10,
    window_sec: float = 2.0,
    weights: str = "default",
    budget_per_hour: float = 240.0,
    verbose: bool = True,
) -> ParityReport:
    """
    Run the parity experiment end to end.

    Default arms cover the question that matters:
      router+carry_forward   no reconstruction model at all
      router+slerp           zero-parameter interpolation
      uniform_1fps+carry     what the industry does today

    Add an "adve" arm with custom_fill=<your reconstructor> to find out
    whether four generations of training beat spherical interpolation.
    """
    from frameroute.policies import RouterPolicy, UniformFPS

    if verbose:
        print(f"[gate1] analyzing signals: {Path(video_path).name}")
    track = analyze_video(video_path, stride=stride, max_frames=max_frames, weights=weights)
    if verbose:
        print(f"[gate1] {track.summary()}")

    if verbose:
        print("[gate1] building full-compute reference index ...")
    ref = build_reference_index(video_path, track, embedder)

    if arms is None:
        router_picks = [p.idx for p in RouterPolicy(policy="coverage").select(
            track, budget=max(2, int(round(budget_per_hour * track.duration_sec / 3600.0)))
        )]
        uniform_picks = [p.idx for p in UniformFPS(1.0).select(track)]
        arms = {
            "router+carry_forward": {"picks": router_picks, "fill": "carry_forward"},
            "router+slerp":         {"picks": router_picks, "fill": "slerp"},
            "uniform_1fps+carry":   {"picks": uniform_picks, "fill": "carry_forward"},
        }

    report = ParityReport(
        video_path=video_path,
        frames_analyzed=track.n_analyzed,
        duration_sec=track.duration_sec,
        reference_calls=ref.n_model_calls,
    )

    worst = 1.0
    for name, spec in arms.items():
        if verbose:
            print(f"[gate1] arm: {name}")
        cand = build_routed_index(
            video_path, track, embedder,
            pick_indices=spec["picks"],
            fill=spec.get("fill", "carry_forward"),
            custom_fill=spec.get("custom_fill"),
            name=name,
        )
        _, agg = compare_indexes(ref, cand, embedder, queries, k=k, window_sec=window_sec)
        saved = 100.0 * (1.0 - cand.n_model_calls / max(ref.n_model_calls, 1))
        report.arms[name] = {
            "n_model_calls": cand.n_model_calls,
            "savings_pct": round(saved, 2),
            "build_seconds": round(cand.build_seconds, 2),
            "metrics": agg,
        }
        worst = min(worst, agg["recall_temporal"])
        if verbose:
            print(f"        calls={cand.n_model_calls} saved={saved:.1f}% "
                  f"temporal_recall={agg['recall_temporal']:.4f}")

    best = max(
        (a["metrics"]["recall_temporal"] for a in report.arms.values()), default=0.0
    )
    report.verdict = "PASS" if best >= GATE1_THRESHOLD else "FAIL"
    return report


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main() -> None:
    import argparse
    from frameroute.adapters import ClipEmbedder
    from bench.queries import QuerySet, queries_for, DEFAULT_QUERIES

    ap = argparse.ArgumentParser(
        description="Gate 1: does skipping frames preserve retrieval?"
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--queries", default="", help="path to a QuerySet JSON")
    ap.add_argument("--domain", default="generic",
                    help="generic | traffic | surveillance | lecture | action "
                         "-- picks the built-in query set. Use this or --queries; "
                         "generic queries on domain footage give a meaningless number.")
    ap.add_argument("--out", default="results/gate1_parity.json")
    ap.add_argument("--md", default="results/gate1_parity.md")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--budget-per-hour", type=float, default=240.0)
    ap.add_argument("--weights", default="default",
                    help="signal preset: default | static_cam | surveillance | ego_motion")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.queries:
        qs = QuerySet.load(args.queries)
        queries = qs.queries or queries_for(args.domain)
    else:
        queries = queries_for(args.domain) or DEFAULT_QUERIES
    print(f"[gate1] {len(queries)} queries ({args.domain}): {queries[0]!r} ...")

    embedder = ClipEmbedder(device=args.device)
    report = run_gate1(
        video_path=args.video,
        queries=queries,
        embedder=embedder,
        stride=args.stride,
        max_frames=args.max_frames,
        k=args.k,
        window_sec=args.window_sec,
        weights=args.weights,
        budget_per_hour=args.budget_per_hour,
    )

    report.to_json(args.out)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(report.to_markdown(), encoding="utf-8")

    print()
    print(report.to_markdown())
    print()
    print(f"[gate1] wrote {args.out} and {args.md}")
    print("[gate1] publish this result whatever it says.")


if __name__ == "__main__":
    main()
