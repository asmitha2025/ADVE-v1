"""
bench.selftest — verify the install without a GPU, a model, or an API key.

Run this before spending anything:

    python -m bench.selftest

It builds a synthetic video with known structure, runs the whole pipeline
against a deterministic stub embedder, and checks that the router behaves
the way the design claims. If this passes, the plumbing is sound and any
number you get from Gate 1 is about the model, not about a broken harness.

The synthetic video is deliberately adversarial in the two ways that break
naive samplers:

    0-4s    static             a uniform sampler wastes its budget here
    4-6s    object crossing    a 2-second event; 1 fps sees it twice
    6-10s   camera pan         every pixel moves, nothing happens
    10-11s  abrupt cut         a slide change; must never be missed
    11-15s  static

A router that spends evenly across those five phases has learned nothing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------

def make_test_video(path: str, w: int = 320, h: int = 180, fps: int = 30) -> Dict:
    """Deterministic 15-second video with labelled events."""
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)

    bg = np.zeros((h, w * 2, 3), np.uint8)
    for _ in range(300):
        x, y = int(rng.integers(0, w * 2)), int(rng.integers(0, h))
        cv2.circle(bg, (x, y), int(rng.integers(2, 7)),
                   tuple(int(c) for c in rng.integers(40, 200, 3)), -1)

    slide = np.full((h, w, 3), (200, 180, 60), np.uint8)
    cv2.putText(slide, "SLIDE 2", (40, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (20, 20, 20), 3)

    for i in range(15 * fps):
        t = i / fps
        if t < 10:
            off = 0 if t < 6 else int((t - 6) * 45)
            f = bg[:, off:off + w].copy()
            if 4 <= t < 6:
                cx = int((t - 4) / 2 * w)
                cv2.rectangle(f, (cx - 14, h // 2 - 22), (cx + 14, h // 2 + 22),
                              (0, 0, 255), -1)
        else:
            f = slide.copy()
        out.write(f)
    out.release()

    return {
        "path": path,
        "duration_sec": 15.0,
        "phases": {
            "static_a": (0.0, 4.0),
            "event_crossing": (4.0, 6.0),
            "camera_pan": (6.0, 10.0),
            "event_cut": (10.0, 11.0),
            "static_b": (11.0, 15.0),
        },
        "events": [(4.0, 6.0), (10.0, 11.0)],
    }


class StubEmbedder:
    """
    Deterministic, frame-content-dependent embeddings. Not semantic, but
    structured enough that retrieval is meaningful: visually similar frames
    get similar vectors, and text queries map onto colour/brightness axes.

    Exists so the harness can be validated without CLIP. Never use it for a
    real parity number.
    """

    dim = 32

    def _vec(self, seed_vals: Sequence[float]) -> np.ndarray:
        v = np.asarray(seed_vals, dtype=np.float32)
        if v.size < self.dim:
            v = np.pad(v, (0, self.dim - v.size))
        v = v[: self.dim]
        n = float(np.linalg.norm(v))
        return v / max(n, 1e-8)

    def embed_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        out = []
        for f in frames:
            small = cv2.resize(f, (8, 4), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            feats = np.concatenate([
                small.mean(axis=(0, 1)) / 255.0,
                hsv.mean(axis=(0, 1)) / 255.0,
                small.reshape(-1, 3).mean(axis=1)[:26] / 255.0,
            ])
            out.append(self._vec(feats))
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            h = abs(hash(t)) % (2 ** 31)
            rng = np.random.default_rng(h)
            out.append(self._vec(rng.normal(size=self.dim)))
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    return ok


def run(verbose: bool = True) -> int:
    from frameroute.signals import analyze_video
    from frameroute.router import FrameRouter, StreamingRouter
    from frameroute.policies import UniformFPS, UniformN, SceneCut, RouterPolicy
    from frameroute.adapters import CountingEmbedder, ExactIndex
    from bench.parity import (
        build_reference_index, build_routed_index, compare_indexes,
    )
    from bench.routing_bench import event_coverage

    tmp = Path(tempfile.mkdtemp(prefix="frameroute_selftest_"))
    vid = str(tmp / "selftest.mp4")
    print(f"\nbuilding synthetic fixture -> {vid}")
    spec = make_test_video(vid)
    events = spec["events"]
    phases = spec["phases"]

    ok = True

    # ---- signals -------------------------------------------------------
    print("\n1. signals")
    track = analyze_video(vid)
    s = track.summary()
    ok &= _check("450 frames analyzed", track.n_analyzed == 450, str(track.n_analyzed))
    ok &= _check(
        "signal cost is cheap (< 15 ms/frame)",
        s["mean_signal_cost_ms"] < 15.0,
        f"{s['mean_signal_cost_ms']:.2f} ms/frame vs 120.33 ms for YOLOv8 "
        f"in docs/GPU_PERFORMANCE_CERTIFICATE.md",
    )

    def phase_mean(attr: str, a: float, b: float) -> float:
        vals = [getattr(x, attr) for x in track.signals if a <= x.t < b]
        return float(np.mean(vals)) if vals else 0.0

    pan_struct = phase_mean("struct_delta", *phases["camera_pan"])
    pan_resid = phase_mean("motion_residual", *phases["camera_pan"])
    ok &= _check(
        "ego-motion compensation suppresses pure camera pan",
        pan_resid < pan_struct * 0.5,
        f"struct_delta={pan_struct:.4f} -> residual={pan_resid:.4f}",
    )
    ok &= _check(
        "static stretches score near zero novelty",
        phase_mean("novelty", *phases["static_b"]) < 0.01,
        f"{phase_mean('novelty', *phases['static_b']):.4f}",
    )

    # ---- router --------------------------------------------------------
    print("\n2. router")
    sel = FrameRouter().select(track, budget=12)
    times = sel.times()

    def n_in(a: float, b: float) -> int:
        return sum(1 for t in times if a <= t < b)

    static_calls = n_in(*phases["static_a"]) + n_in(*phases["static_b"])
    event_calls = n_in(*phases["event_crossing"]) + n_in(*phases["event_cut"])
    ok &= _check(
        "spends on events, not on static footage",
        event_calls > static_calls,
        f"{event_calls} calls on 3s of events vs {static_calls} on 8s of static",
    )
    ok &= _check(
        "abrupt cut fires a hard trigger",
        any(p.reason == "hard_trigger" and 9.9 <= p.t <= 11.0 for p in sel.picks),
        f"reasons={sel.reason_counts()}",
    )
    cov = event_coverage(times, events)
    ok &= _check("both labelled events covered", cov["covered"] == 2, str(cov))

    g = sel.guarantees(track)
    ok &= _check(
        "max-gap guarantee holds",
        g["max_time_gap_sec"] <= 30.0 + 1e-6,
        f"max gap {g['max_time_gap_sec']:.2f}s",
    )

    # ---- baselines at matched budget ------------------------------------
    print("\n3. baselines at matched budget")
    budget = 12
    results = {}
    for pol in (UniformN(), UniformFPS(1.0), SceneCut(), RouterPolicy()):
        picks = pol.select(track, budget=budget)
        c = event_coverage([p.t for p in picks], events)
        results[pol.name] = (len(picks), c["coverage_pct"])
        print(f"    {pol.name:<20s} calls={len(picks):3d}  event_coverage={c['coverage_pct']:.0f}%")
    ok &= _check(
        "router covers events at least as well as uniform sampling",
        results["router_coverage"][1] >= results["uniform_n"][1],
        f"router {results['router_coverage'][1]:.0f}% vs uniform {results['uniform_n'][1]:.0f}%",
    )

    # ---- parity harness --------------------------------------------------
    print("\n4. parity harness")
    emb = StubEmbedder()
    queries = ["a red object crossing", "a bright slide", "an empty textured scene",
               "something moving fast", "a static background"]

    ref = build_reference_index(vid, track, emb)
    ok &= _check("reference index spends one call per frame",
                 ref.n_model_calls == track.n_analyzed,
                 f"{ref.n_model_calls} calls")

    pick_idx = [p.idx for p in RouterPolicy().select(track, budget=24)]
    arms = {}
    for fill in ("carry_forward", "slerp"):
        cand = build_routed_index(vid, track, emb, pick_indices=pick_idx, fill=fill)
        _, agg = compare_indexes(ref, cand, emb, queries, k=10, window_sec=2.0)
        arms[fill] = (cand.n_model_calls, agg["recall_temporal"])
        print(f"    {fill:<16s} calls={cand.n_model_calls:3d}  "
              f"temporal_recall={agg['recall_temporal']:.4f}  "
              f"ndcg={agg['ndcg']:.4f}")
        ok &= _check(f"{fill} index has same row count as reference",
                     len(cand.frame_idx) == len(ref.frame_idx))

    ok &= _check(
        "routed index costs far less than full compute",
        arms["slerp"][0] < ref.n_model_calls * 0.2,
        f"{arms['slerp'][0]} vs {ref.n_model_calls} calls "
        f"({100 * (1 - arms['slerp'][0] / ref.n_model_calls):.0f}% fewer)",
    )

    # ---- accounting ------------------------------------------------------
    print("\n5. call accounting")
    counted = CountingEmbedder(emb)
    counted.embed_frames([np.zeros((8, 8, 3), np.uint8)] * 7)
    ok &= _check("CountingEmbedder counts real calls",
                 counted.ledger.frame_calls == 7,
                 str(counted.ledger.as_dict()))

    # ---- exact search ----------------------------------------------------
    print("\n6. exact search")
    V = np.eye(4, dtype=np.float32)
    idx = ExactIndex(V)
    ids, scores = idx.search(np.array([0, 1, 0, 0], np.float32), k=2)
    ok &= _check("ExactIndex returns the exact nearest neighbour",
                 int(ids[0]) == 1 and abs(float(scores[0]) - 1.0) < 1e-5,
                 f"top={int(ids[0])} score={float(scores[0]):.4f}")

    # ---- streaming -------------------------------------------------------
    print("\n7. streaming router")
    sr = StreamingRouter(budget_per_hour=240, fps=30)
    cap = cv2.VideoCapture(vid)
    i, spends = 0, 0
    while True:
        okf, fr = cap.read()
        if not okf:
            break
        if sr.should_spend(fr, i):
            spends += 1
        i += 1
    cap.release()
    rep = sr.report()
    ok &= _check("streaming router spends a bounded fraction of frames",
                 0 < spends < i * 0.5, f"{spends}/{i} frames, {rep['spend_rate_pct']}%")
    ok &= _check("streaming router caught the abrupt cut",
                 rep["hard_triggers"] >= 1, f"hard_triggers={rep['hard_triggers']}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    print(
        "\nThis validates the harness, not the product. The stub embedder is "
        "not semantic.\nRun `python -m frameroute gate1 <your video>` with real "
        "CLIP for a number that means something.\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
