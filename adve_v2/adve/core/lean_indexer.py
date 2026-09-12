"""
adve.core.lean_indexer — the clean indexing spine.

This replaces the anchor-delta reconstruction pipeline (ADVEPipeline +
reconstructor + UALW + safety_gate) for the indexing path. The audit showed
that machinery did not earn its complexity: parameter-free routing plus a real
CLIP call on the selected frames preserves retrieval better, with no
checkpoint, no drift and no failure mode (see bench/parity.py, README).

The architecture is one line:

    frameroute picks the frames that carry new information  ->  CLIP-encode
    exactly those  ->  store them in the search index.

No embedding is ever synthesised or reconstructed. Every vector in the index
is a real CLIP encoding of a real frame, so "is this frame's embedding
trustworthy?" is never a question. The cost saving comes from encoding far
fewer frames, not from faking the ones we skip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from frameroute.signals import analyze_video, SignalTrack
from frameroute.router import FrameRouter


@dataclass
class IndexStats:
    video_id: str
    frames_total: int
    frames_analyzed: int
    frames_encoded: int
    duration_sec: float
    reduction_vs_dense: float          # dense = one call per analyzed frame
    signal_cost_ms: float
    encode_seconds: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "video_id": self.video_id,
            "frames_total": self.frames_total,
            "frames_analyzed": self.frames_analyzed,
            "frames_encoded": self.frames_encoded,
            "duration_sec": round(self.duration_sec, 2),
            "reduction_vs_dense": round(self.reduction_vs_dense, 2),
            "mean_signal_cost_ms": round(self.signal_cost_ms, 3),
            "encode_seconds": round(self.encode_seconds, 3),
        }


def _read_frames(video_path: str, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    """Sequential decode of the requested frame indices (seek is inexact on
    long-GOP H.264, so we walk forward once)."""
    import cv2
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    out: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    ptr = idx = 0
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


class LeanIndexer:
    """
    Route-then-encode indexing. No reconstruction.

        indexer = LeanIndexer()                      # loads CLIP once
        records, stats = indexer.index("clip.mp4", "clip.mp4", budget_per_hour=1200)
        search_index.add_batch(records)
    """

    def __init__(
        self,
        embedder=None,
        device: Optional[str] = None,
        weights: str = "default",
        max_gap_sec: float = 30.0,
        hard_trigger: Optional[float] = 0.55,
    ):
        if embedder is None:
            from frameroute.adapters import ClipEmbedder
            embedder = ClipEmbedder(device=device)
        self.embedder = embedder
        self.weights = weights
        self.router = FrameRouter(max_gap_sec=max_gap_sec, hard_trigger=hard_trigger)

    def index(
        self,
        video_path: str,
        video_id: str,
        budget_per_hour: Optional[float] = 1200.0,
        budget: Optional[int] = None,
        max_frames: Optional[int] = None,
        stride: int = 1,
        camera_id: Optional[str] = None,
    ) -> tuple[List[Dict], IndexStats]:
        import time
        track: SignalTrack = analyze_video(
            video_path, stride=stride, max_frames=max_frames, weights=self.weights
        )
        sel = self.router.select(track, budget=budget, budget_per_hour=budget_per_hour)

        picks = sorted(set(sel.indices()))
        frames_map = _read_frames(video_path, picks)
        ordered = [i for i in picks if i in frames_map]

        t0 = time.perf_counter()
        vecs = self.embedder.embed_frames([frames_map[i] for i in ordered])
        encode_seconds = time.perf_counter() - t0

        fps = track.fps or 30.0
        cam = camera_id or video_id
        records: List[Dict] = []
        for row, idx in enumerate(ordered):
            records.append({
                "video_path": video_id,
                "camera_id": cam,
                "timestamp": idx / fps,
                "frame_idx": idx,
                "embedding": np.asarray(vecs[row], dtype=np.float32).reshape(-1),
                "is_anchor": True,          # every stored vector is a real CLIP call
                "text": "",
            })

        stats = IndexStats(
            video_id=video_id,
            frames_total=track.n_frames_total,
            frames_analyzed=track.n_analyzed,
            frames_encoded=len(records),
            duration_sec=track.duration_sec,
            reduction_vs_dense=track.n_analyzed / max(len(records), 1),
            signal_cost_ms=track.mean_signal_cost_ms(),
            encode_seconds=encode_seconds,
        )
        return records, stats
