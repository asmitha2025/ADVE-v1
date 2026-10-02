"""
bench.grounded_eval — retrieval scored against INDEPENDENT ground truth.

Why this exists
---------------
Every earlier quality metric compared the routed index to the full-compute
index and asked "did you return the SAME frame?". That treats full-compute's
top-1 as truth, which it is not — it is just another retrieval result. Routing
got punished for returning an equally-good *different* frame (in one run the
routed frame was 7 seconds from full's and was scored a miss).

The right question is "can the user still find what they are looking for?",
scored against ground truth that neither index produced. A lecture carries its
own: **the text on its slides**. For the query "gradient descent", the truly
relevant frames are the ones whose slides actually say it. OCR gives us that —
no VLM judge, no API key, no top-1 harshness.

And the comparison that matters commercially is not routed-vs-full (nobody
runs full compute; it is too expensive) but **routed-vs-uniform at the same
number of model calls** — uniform 1 fps is what a customer does today.

    python -m bench.grounded_eval --video LECTURE.mp4 --budget 100

Honest caveat: the queries are slide words embedded by CLIP, and CLIP reads
text in images poorly, so ABSOLUTE hit rates will be modest. That penalty is
identical for all three arms, so the COMPARISON between them is fair — which
is the number we actually need.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "are",
    "was", "were", "will", "would", "can", "could", "should", "into", "than",
    "then", "they", "them", "their", "there", "these", "those", "what", "when",
    "where", "which", "while", "about", "also", "more", "most", "some", "such",
    "only", "other", "over", "very", "your", "you", "our", "its", "but", "not",
    "all", "any", "how", "why", "who", "one", "two", "three", "using", "used",
    "use", "example", "figure", "slide", "page", "university", "lecture",
    "course", "chapter", "section", "introduction", "overview", "summary",
}


# --------------------------------------------------------------------------
# 1. OCR the analysed frames
# --------------------------------------------------------------------------

def ocr_frames(frames: Dict[int, np.ndarray], device: str = "cuda",
               max_side: int = 720, min_conf: float = 0.4,
               progress_every: int = 100) -> Dict[int, str]:
    """Slide text per analysed frame. Downscaled for speed; same EasyOCR setup
    the product's ocr_extractor uses."""
    import cv2
    import easyocr
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
    out: Dict[int, str] = {}
    for n, (idx, fr) in enumerate(sorted(frames.items())):
        if fr is None or fr.size == 0:
            out[idx] = ""
            continue
        h, w = fr.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        try:
            res = reader.readtext(fr, detail=1, paragraph=False)
            out[idx] = " ".join(t.strip() for (_b, t, c) in res
                                if c >= min_conf and t.strip()).lower()
        except Exception:
            out[idx] = ""
        if progress_every and (n + 1) % progress_every == 0:
            print(f"    ocr {n + 1}/{len(frames)} frames", flush=True)
    return out


# --------------------------------------------------------------------------
# 2. Mine queries + ground-truth labels from the slide text
# --------------------------------------------------------------------------

def mine_labels(frame_text: Dict[int, str], n_queries: int = 20,
                min_df: float = 0.02, max_df: float = 0.25,
                min_len: int = 5) -> Dict[str, Set[int]]:
    """
    Pick terms that are distinctive but recurring — a word on 2–25% of slides
    names a real topic; one on 90% is boilerplate and one on a single frame is
    noise. Ground truth for a term = every frame whose slide text contains it.
    """
    n_frames = max(len(frame_text), 1)
    df: Counter = Counter()
    tokens_per_frame: Dict[int, Set[str]] = {}
    for idx, text in frame_text.items():
        toks = {w for w in re.findall(r"[a-z]{%d,}" % min_len, text)
                if w not in STOPWORDS}
        tokens_per_frame[idx] = toks
        df.update(toks)

    lo, hi = min_df * n_frames, max_df * n_frames
    cands = [(w, c) for w, c in df.items() if lo <= c <= hi]
    # prefer terms that are frequent-but-not-ubiquitous and reasonably long
    cands.sort(key=lambda wc: (wc[1] * min(len(wc[0]), 12)), reverse=True)

    labels: Dict[str, Set[int]] = {}
    for w, _c in cands[:n_queries]:
        hits = {i for i, toks in tokens_per_frame.items() if w in toks}
        if hits:
            labels[w] = hits
    return labels


# --------------------------------------------------------------------------
# 3. Score an index against the labels
# --------------------------------------------------------------------------

@dataclass
class ArmScore:
    name: str
    model_calls: int
    hit_at_k: float          # fraction of queries whose top-k contains a labelled frame
    precision_at_k: float    # fraction of returned frames that are labelled
    n_queries: int

    def row(self, k: int) -> str:
        return (f"  {self.name:<22}{self.model_calls:>7}"
                f"{self.hit_at_k:>12.3f}{self.precision_at_k:>14.3f}")


def score_index(built, labels: Dict[str, Set[int]], embedder, k: int = 5) -> ArmScore:
    hits, precs = [], []
    for q, gt in labels.items():
        qv = embedder.embed_text([q])[0]
        ids, _ = built.index.search(qv, k=k)
        got = [built.frame_idx[int(i)] for i in ids]
        n_rel = sum(1 for g in got if g in gt)
        hits.append(1.0 if n_rel > 0 else 0.0)
        precs.append(n_rel / max(len(got), 1))
    return ArmScore(
        name=built.name, model_calls=built.n_model_calls,
        hit_at_k=float(np.mean(hits)) if hits else 0.0,
        precision_at_k=float(np.mean(precs)) if precs else 0.0,
        n_queries=len(labels),
    )


# --------------------------------------------------------------------------
# 4. Run: full vs routed vs uniform, matched budget
# --------------------------------------------------------------------------

@dataclass
class GroundedReport:
    video: str
    frames_analyzed: int
    duration_sec: float
    budget: int
    k: int
    n_queries: int
    full: dict
    routed: dict
    uniform: dict
    routed_vs_uniform_hit_delta: float
    routed_vs_uniform_rel_gain_pct: float
    routed_calls_saved_vs_full_pct: float

    def table(self) -> str:
        L = [
            "",
            f"  GROUNDED RETRIEVAL — {Path(self.video).name}",
            f"  {self.frames_analyzed} frames analysed · {self.duration_sec:.0f}s · "
            f"{self.n_queries} OCR-grounded queries · top-{self.k}",
            "  " + "-" * 60,
            f"  {'arm':<22}{'calls':>7}{'hit@k':>12}{'precision@k':>14}",
            ArmScore(**self.full).row(self.k),
            ArmScore(**self.routed).row(self.k),
            ArmScore(**self.uniform).row(self.k),
            "  " + "-" * 60,
            f"  routed vs uniform (same budget): {self.routed_vs_uniform_hit_delta:+.3f} hit@k "
            f"({self.routed_vs_uniform_rel_gain_pct:+.1f}% relative)",
            f"  routed spends {self.routed_calls_saved_vs_full_pct:.0f}% fewer calls than full compute",
            "",
        ]
        return "\n".join(L)


def run(video: str, budget: Optional[int], max_frames: int, k: int,
        n_queries: int, weights: str, device: Optional[str]) -> GroundedReport:
    import cv2
    from frameroute.signals import analyze_video
    from frameroute.policies import RouterPolicy, UniformN
    from frameroute.adapters import ClipEmbedder
    from bench.parity import build_reference_index, build_routed_index, read_frames

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    stride = max(1, total // max_frames) if total else 1
    print(f"[grounded] analysing (stride={stride}, max_frames={max_frames})", flush=True)
    track = analyze_video(video, stride=stride, max_frames=max_frames, weights=weights)

    print(f"[grounded] decoding {track.n_analyzed} frames once", flush=True)
    frames = read_frames(video, [s.idx for s in track.signals])

    print(f"[grounded] OCR for ground truth …", flush=True)
    t0 = time.perf_counter()
    frame_text = ocr_frames(frames, device=device or "cuda")
    labels = mine_labels(frame_text, n_queries=n_queries)
    print(f"[grounded] {len(labels)} grounded queries in {time.perf_counter()-t0:.0f}s: "
          f"{list(labels)[:8]}", flush=True)
    if not labels:
        raise RuntimeError("no OCR-grounded queries found — try a slide-heavy video")

    emb = ClipEmbedder(device=device)
    b = budget or max(2, track.n_analyzed // 5)

    print(f"[grounded] building indexes (budget={b})", flush=True)
    full = build_reference_index(video, track, emb, frames=frames)
    r_picks = sorted(set(p.idx for p in RouterPolicy(policy="coverage").select(track, budget=b)))
    routed = build_routed_index(video, track, emb, pick_indices=r_picks,
                                fill="slerp", name="frameroute", frames=frames)
    # Match uniform to the router's ACTUAL call count, not the requested budget:
    # the router's max-gap / hard-trigger guarantees can push it above budget,
    # and comparing 104 routed calls against 80 uniform ones would flatter us.
    n_matched = len(r_picks)
    u_picks = sorted(set(p.idx for p in UniformN(n_matched).select(track, budget=n_matched)))
    uniform = build_routed_index(video, track, emb, pick_indices=u_picks,
                                 fill="slerp", name="uniform", frames=frames)

    s_full = score_index(full, labels, emb, k=k)
    s_routed = score_index(routed, labels, emb, k=k)
    s_uniform = score_index(uniform, labels, emb, k=k)

    delta = s_routed.hit_at_k - s_uniform.hit_at_k
    rel = 100.0 * delta / s_uniform.hit_at_k if s_uniform.hit_at_k > 0 else 0.0
    saved = 100.0 * (1 - s_routed.model_calls / max(s_full.model_calls, 1))

    return GroundedReport(
        video=video, frames_analyzed=track.n_analyzed, duration_sec=track.duration_sec,
        budget=b, k=k, n_queries=len(labels),
        full=asdict(s_full), routed=asdict(s_routed), uniform=asdict(s_uniform),
        routed_vs_uniform_hit_delta=round(delta, 4),
        routed_vs_uniform_rel_gain_pct=round(rel, 2),
        routed_calls_saved_vs_full_pct=round(saved, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval vs independent OCR ground truth.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--budget", type=int, default=None, help="model calls for routed & uniform")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--weights", default="default")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rep = run(a.video, a.budget, a.max_frames, a.k, a.n_queries, a.weights, a.device)
    print(rep.table())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
        print(f"[grounded] wrote {a.out}")


if __name__ == "__main__":
    main()
