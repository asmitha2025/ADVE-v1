"""
frameroute.adapters — model-agnostic downstream interfaces.

The router's value is that it does not care what the expensive model is.
Today it is CLIP; tomorrow it is a VLM captioning call at a thousand times
the cost. Keeping the downstream behind a two-method protocol is what makes
the savings claim portable, and it is why this layer exists at all.

CountingEmbedder is the honest-accounting piece. Every benchmark in bench/
routes its downstream through it, so the reported cost is a count of calls
that actually happened rather than a spreadsheet estimate. This is the
direct fix for results/traffic_benchmark_report.json, which reported 59.9%
"computing power saved" from a FLOP calculation while the measured wall
clock moved 5.3%.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------

class Embedder(Protocol):
    """Anything that turns frames and text into a shared vector space."""
    dim: int

    def embed_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray: ...
    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


class Captioner(Protocol):
    """Anything that turns a frame into text. This is the expensive one."""

    def caption(self, frames: Sequence[np.ndarray], prompt: str = "") -> List[str]: ...


# --------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------

class ClipEmbedder:
    """
    Wraps the CLIP loader already in this repo (adve.core.clip_loader) so the
    benchmark uses exactly the model the product uses.

    Batches by default. Note that batching is itself a large part of why the
    original "skip the encoder" premise did not pay off: a batched ViT-B/32
    pass amortises to a few milliseconds a frame, which is far less than the
    detector that was being run to avoid it.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None,
                 batch_size: int = 64):
        import torch
        try:
            from adve.core.clip_loader import load_clip_model
            self.model, self.preprocess = load_clip_model(model_name, device)
        except Exception:
            import clip
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()

        self.torch = torch
        self.device = next(self.model.parameters()).device
        self.batch_size = int(batch_size)
        self.dim = 512

    def embed_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        import cv2
        from PIL import Image

        if len(frames) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: List[np.ndarray] = []
        for i in range(0, len(frames), self.batch_size):
            chunk = frames[i:i + self.batch_size]
            tensors = []
            for f in chunk:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(self.preprocess(Image.fromarray(rgb)))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.no_grad():
                e = self.model.encode_image(batch)
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy().astype(np.float32))
        arr = np.concatenate(out, axis=0)
        self.dim = arr.shape[1]
        return arr

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        import clip
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        tokens = clip.tokenize(list(texts)).to(self.device)
        with self.torch.no_grad():
            e = self.model.encode_text(tokens)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

@dataclass
class CallLedger:
    """Measured cost. No estimates, no FLOPs."""
    frame_calls: int = 0
    text_calls: int = 0
    frame_seconds: float = 0.0
    text_seconds: float = 0.0
    batches: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {
            "frame_calls": self.frame_calls,
            "text_calls": self.text_calls,
            "batches": self.batches,
            "frame_seconds": round(self.frame_seconds, 4),
            "text_seconds": round(self.text_seconds, 4),
            "ms_per_frame_call": round(
                1000.0 * self.frame_seconds / self.frame_calls, 3
            ) if self.frame_calls else 0.0,
        }


class CountingEmbedder:
    """
    Decorator that meters an Embedder.

        emb = CountingEmbedder(ClipEmbedder())
        ...
        print(emb.ledger.as_dict())     # what you actually spent
    """

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.ledger = CallLedger()

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed_frames(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.inner.embed_frames(frames)
        self.ledger.frame_seconds += time.perf_counter() - t0
        self.ledger.frame_calls += len(frames)
        self.ledger.batches += 1
        return out

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.inner.embed_text(texts)
        self.ledger.text_seconds += time.perf_counter() - t0
        self.ledger.text_calls += len(texts)
        return out

    def reset(self) -> None:
        self.ledger = CallLedger()


class PricedCaptioner:
    """
    Wraps any VLM captioner with a price so a budget sweep can be reported
    in currency rather than call counts. Supply `fn(frames, prompt)->List[str]`.

    Deliberately has no vendor SDK dependency: pass a lambda that calls
    whichever API you use. Keeping vendors out of this package is what lets
    the savings claim survive a model switch.
    """

    def __init__(self, fn: Callable[[Sequence[np.ndarray], str], List[str]],
                 usd_per_call: float = 0.002, name: str = "vlm"):
        self.fn = fn
        self.usd_per_call = float(usd_per_call)
        self.name = name
        self.calls = 0
        self.seconds = 0.0

    def caption(self, frames: Sequence[np.ndarray], prompt: str = "") -> List[str]:
        t0 = time.perf_counter()
        out = self.fn(frames, prompt)
        self.seconds += time.perf_counter() - t0
        self.calls += len(frames)
        return out

    def cost_usd(self) -> float:
        return round(self.calls * self.usd_per_call, 4)

    def report(self) -> Dict:
        return {
            "model": self.name,
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
            "usd_per_call": self.usd_per_call,
            "cost_usd": self.cost_usd(),
        }


class DryRunCaptioner:
    """
    Counts and prices calls without making them. Use this to sweep budgets
    and draw the cost curve before spending a rupee on API credit.
    """

    def __init__(self, usd_per_call: float = 0.002, latency_ms: float = 0.0):
        self.usd_per_call = float(usd_per_call)
        self.latency_ms = float(latency_ms)
        self.calls = 0

    def caption(self, frames: Sequence[np.ndarray], prompt: str = "") -> List[str]:
        self.calls += len(frames)
        return [""] * len(frames)

    def cost_usd(self) -> float:
        return round(self.calls * self.usd_per_call, 4)

    def projected_seconds(self) -> float:
        return round(self.calls * self.latency_ms / 1000.0, 2)

    def report(self) -> Dict:
        return {
            "model": "dry_run",
            "calls": self.calls,
            "cost_usd": self.cost_usd(),
            "projected_seconds": self.projected_seconds(),
        }


# --------------------------------------------------------------------------
# exact search — deliberately not FAISS
# --------------------------------------------------------------------------

class ExactIndex:
    """
    Brute-force cosine search over a small embedding matrix.

    FAISS is right for production and wrong for this benchmark. An ANN index
    has its own recall loss, and mixing it into a parity measurement means
    you can no longer tell whether a missed result was the reconstructor's
    fault or the index's. Parity must be measured against exact search;
    swap in FAISS only after the parity number exists.
    """

    def __init__(self, vectors: np.ndarray, meta: Optional[List[Dict]] = None):
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2:
            raise ValueError("vectors must be 2-D (n, dim)")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.v = v / np.maximum(norms, 1e-8)
        self.meta = meta or [{} for _ in range(len(v))]

    def __len__(self) -> int:
        return int(self.v.shape[0])

    def search(self, query: np.ndarray, k: int = 10):
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        scores = self.v @ q
        k = int(min(k, len(self)))
        if k <= 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]
