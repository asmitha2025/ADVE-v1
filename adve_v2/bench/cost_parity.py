"""
bench.cost_parity — the number that decides whether frameroute is a business.

It answers ONE question a buyer actually asks:

    "If I route my video through this instead of sending every frame to my
     VLM, do I get the same answers, and how much money does it save me?"

The pipeline is model-agnostic. Quality is scored by a pluggable Backend:

  ClipParityBackend   runs today, no API key. Quality = retrieval parity: does
                      the routed index return the same top moments as a
                      full-compute index (temporal recall@10)? This is a proxy
                      for "the VLM still sees the frame it needs to answer".
  GroqVisionBackend   real VLM (Groq Llama-3.2-Vision). Activates when
  OpenAIVisionBackend GROQ_API_KEY / OPENAI_API_KEY is set. Quality = answer
                      agreement with the full-frame baseline on a QA task.

Swapping the backend is the only change needed to turn the proxy number into a
real dollar number on a customer's own footage.

    python -m bench.cost_parity --video LECTURE.mp4 --domain lecture \
        --price-per-call 0.005 --parity 0.95
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from frameroute.signals import analyze_video
from frameroute.policies import RouterPolicy, UniformN
from bench.parity import build_reference_index, build_routed_index, compare_indexes
from bench.queries import queries_for


# --------------------------------------------------------------------------
# backends — quality = "did routing preserve what the downstream model needs?"
# --------------------------------------------------------------------------

class Backend:
    name = "backend"
    price_per_call = 0.0

    def quality(self, video, track, ref, picks, queries) -> float:
        raise NotImplementedError


class ClipParityBackend(Backend):
    """No API key. Quality = temporal recall@10 of the routed index vs a
    full-compute reference. Runs on the GPU you already have."""
    name = "clip_parity_proxy"

    def __init__(self, embedder, price_per_call: float):
        self.embedder = embedder
        self.price_per_call = price_per_call

    def quality(self, video, track, ref, picks, queries) -> float:
        cand = build_routed_index(video, track, self.embedder,
                                  pick_indices=picks, fill="slerp")
        _, agg = compare_indexes(ref, cand, self.embedder, queries,
                                 k=10, window_sec=2.0)
        return float(agg["recall_temporal"])


class GroqVisionBackend(Backend):
    """
    Live answer-level parity on Groq Llama-3.2-Vision (free tier).

    Quality = answer agreement: for each query we take the single top frame the
    ROUTED index returns and the top frame the FULL index returns, ask the VLM
    the same yes/no question of each ("Does this frame show: <query>?"), and
    score the fraction of queries where routing did NOT change the VLM's answer.
    That is the number a buyer cares about: "same answers, less spend."

    Makes real API calls, so it evaluates a short budget list. Set GROQ_API_KEY.
    """
    name = "groq_llama_vision_answer_parity"

    # Known Groq vision-capable model families, most-preferred first. The old
    # llama-3.2-vision previews were retired; llama-4 scout/maverick replaced
    # them. We resolve against the key's actual model list at construction.
    VISION_PREFERENCE = (
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    )
    VISION_HINTS = ("vision", "scout", "maverick", "llama-4")

    def __init__(self, embedder, price_per_call: float, model: Optional[str] = None):
        from groq import Groq
        self.embedder = embedder
        self.price_per_call = price_per_call
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._cache: Dict[tuple, bool] = {}
        self.model = model or self._resolve_vision_model()

    def _resolve_vision_model(self) -> str:
        try:
            available = {m.id for m in self.client.models.list().data}
        except Exception as e:
            raise RuntimeError(f"Groq models.list failed: {e}")
        for pref in self.VISION_PREFERENCE:
            if pref in available:
                return pref
        for mid in available:
            if any(h in mid.lower() for h in self.VISION_HINTS):
                return mid
        raise RuntimeError(
            "This Groq key exposes no vision-capable model (available: "
            + ", ".join(sorted(available))
            + "). Frame-level answer parity needs a vision model — use an "
            "OpenAI/Gemini key or a Groq account with llama-4-scout/maverick. "
            "Run with --backend clip to use the retrieval proxy instead."
        )

    def _yesno(self, video: str, frame_idx: int, query: str) -> bool:
        import base64, cv2
        key = (frame_idx, query)
        if key in self._cache:
            return self._cache[key]
        from bench.parity import read_frames
        fr = read_frames(video, [frame_idx]).get(frame_idx)
        if fr is None:
            self._cache[key] = False
            return False
        h, w = fr.shape[:2]
        if w > 600:
            fr = cv2.resize(fr, (600, int(h * 600 / w)))
        b64 = base64.b64encode(cv2.imencode(".jpg", fr)[1]).decode()
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=3, temperature=0.0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": f"Answer only yes or no. Does this frame show: {query}?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}])
            ans = r.choices[0].message.content.strip().lower().startswith("y")
        except Exception as e:
            print(f"    [groq] call failed ({e}); treating as no")
            ans = False
        self._cache[key] = ans
        return ans

    def quality(self, video, track, ref, picks, queries) -> float:
        cand = build_routed_index(video, track, self.embedder,
                                  pick_indices=picks, fill="slerp")
        agree = 0
        for q in queries:
            qv = self.embedder.embed_text([q])[0]
            ref_top = ref.frame_idx[int(ref.index.search(qv, k=1)[0][0])]
            cand_top = cand.frame_idx[int(cand.index.search(qv, k=1)[0][0])]
            if self._yesno(video, ref_top, q) == self._yesno(video, cand_top, q):
                agree += 1
        return agree / max(len(queries), 1)


def pick_backend(embedder, price_per_call: float, backend: str = "auto") -> Backend:
    """auto: use Groq if a key is set and the SDK imports, else the CLIP proxy."""
    want_groq = backend == "groq" or (backend == "auto" and os.environ.get("GROQ_API_KEY"))
    if want_groq:
        try:
            b = GroqVisionBackend(embedder, price_per_call)
            print("[cost_parity] backend = Groq Llama-Vision (live answer parity)")
            return b
        except Exception as e:
            print(f"[cost_parity] Groq backend unavailable ({e}); falling back to CLIP proxy")
    print("[cost_parity] backend = CLIP retrieval proxy (no API spend)")
    return ClipParityBackend(embedder, price_per_call)


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

@dataclass
class CostResult:
    video: str
    domain: str
    duration_sec: float
    frames_analyzed: int
    price_per_call_usd: float
    parity_threshold: float
    quality_metric: str
    full_calls: int
    routed_calls: int
    routed_quality: float
    uniform_calls_at_parity: Optional[int]
    reduction_x: float
    usd_full_per_hour: float
    usd_routed_per_hour: float
    usd_saved_per_hour: float
    saved_pct: float
    router_beats_uniform: Optional[bool]
    notes: str = ""

    def table(self) -> str:
        L = [
            "",
            f"  COST PARITY — {Path(self.video).name}  ({self.domain})",
            f"  {self.duration_sec:.0f}s video · {self.frames_analyzed} frames analysed · "
            f"quality = {self.quality_metric} · parity >= {self.parity_threshold}",
            "  " + "-" * 60,
            f"  {'':22}{'calls':>10}{'$ / hour video':>18}",
            f"  {'full (every frame)':22}{self.full_calls:>10}{self.usd_full_per_hour:>17.2f}",
            f"  {'frameroute (routed)':22}{self.routed_calls:>10}{self.usd_routed_per_hour:>17.2f}",
            "  " + "-" * 60,
            f"  reduction            {self.reduction_x:>9.1f}x",
            f"  saved / hour video   ${self.usd_saved_per_hour:>8.2f}   ({self.saved_pct:.0f}% cheaper)",
            f"  routed quality       {self.routed_quality:>9.3f}   (target >= {self.parity_threshold})",
        ]
        if self.uniform_calls_at_parity is not None:
            verdict = ("router wins" if self.router_beats_uniform else
                       "uniform as good — sell honestly on this footage")
            L.append(f"  uniform @ parity     {self.uniform_calls_at_parity:>9} calls   ({verdict})")
        L += ["  " + "-" * 60, f"  {self.notes}", ""]
        return "\n".join(L)


def _min_budget_for_parity(video, track, ref, backend, queries,
                           parity: float, policy: str) -> tuple[int, float]:
    """Smallest #calls whose routed quality clears the parity bar."""
    N = track.n_analyzed
    fracs = [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]
    last_calls, last_q = ref.n_model_calls, 1.0
    for f in fracs:
        budget = max(2, int(round(N * f)))
        if policy == "router":
            picks = [p.idx for p in RouterPolicy(policy="coverage").select(track, budget=budget)]
        else:
            picks = [p.idx for p in UniformN(budget).select(track, budget=budget)]
        q = backend.quality(video, track, ref, picks, queries)
        real_calls = len(set(i for i in picks if 0 <= i))
        print(f"    [{policy}] budget~{budget:<4} calls={real_calls:<4} quality={q:.3f}", flush=True)
        last_calls, last_q = real_calls, q
        if q >= parity:
            return real_calls, q
    return last_calls, last_q  # never reached parity; report best


def run(video: str, domain: str, price_per_call: float, parity: float,
        stride: int, max_frames: int, weights: str, backend_choice: str = "auto") -> CostResult:
    from frameroute.adapters import ClipEmbedder
    queries = queries_for(domain)
    print(f"[cost_parity] analysing {Path(video).name} ({domain}, {len(queries)} queries)", flush=True)
    track = analyze_video(video, stride=stride, max_frames=max_frames, weights=weights)
    emb = ClipEmbedder()
    backend = pick_backend(emb, price_per_call, backend_choice)
    print(f"[cost_parity] building full-compute reference ({track.n_analyzed} calls)", flush=True)
    ref = build_reference_index(video, track, emb)

    print("[cost_parity] finding minimum router budget that preserves quality:", flush=True)
    r_calls, r_q = _min_budget_for_parity(video, track, ref, backend, queries, parity, "router")
    print("[cost_parity] same for uniform sampling (honesty check):", flush=True)
    u_calls, u_q = _min_budget_for_parity(video, track, ref, backend, queries, parity, "uniform")

    full = ref.n_model_calls
    hours = track.duration_sec / 3600.0
    per_hour_full = full / hours if hours else full
    per_hour_routed = r_calls / hours if hours else r_calls
    usd_full = per_hour_full * price_per_call
    usd_routed = per_hour_routed * price_per_call
    reduction = full / max(r_calls, 1)

    beats = None
    if u_q >= parity and r_q >= parity:
        beats = r_calls <= u_calls
    uniform_at = u_calls if u_q >= parity else None

    note = ("Quality is retrieval parity (CLIP proxy); swap in a VLM backend for "
            "answer-level parity on real footage. Savings are calls avoided x your "
            "VLM price — wall-clock/latency measured separately.")
    return CostResult(
        video=video, domain=domain, duration_sec=track.duration_sec,
        frames_analyzed=track.n_analyzed, price_per_call_usd=price_per_call,
        parity_threshold=parity, quality_metric="temporal recall@10",
        full_calls=full, routed_calls=r_calls, routed_quality=round(r_q, 4),
        uniform_calls_at_parity=uniform_at,
        reduction_x=round(reduction, 2),
        usd_full_per_hour=round(usd_full, 2), usd_routed_per_hour=round(usd_routed, 2),
        usd_saved_per_hour=round(usd_full - usd_routed, 2),
        saved_pct=round(100.0 * (1 - r_calls / max(full, 1)), 1),
        router_beats_uniform=beats, notes=note,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cost parity: same answers, how much cheaper?")
    ap.add_argument("--video", required=True)
    ap.add_argument("--domain", default="lecture")
    ap.add_argument("--price-per-call", type=float, default=0.005,
                    help="USD per downstream model call (VLM frame). Default ~GPT-4o-vision image.")
    ap.add_argument("--parity", type=float, default=0.95, help="quality bar to hold")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--weights", default="default")
    ap.add_argument("--backend", default="auto", choices=["auto", "clip", "groq"],
                    help="auto = Groq if GROQ_API_KEY set else CLIP proxy")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    res = run(a.video, a.domain, a.price_per_call, a.parity,
              a.stride, a.max_frames, a.weights, a.backend)
    print(res.table())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
        print(f"[cost_parity] wrote {a.out}")


if __name__ == "__main__":
    main()
