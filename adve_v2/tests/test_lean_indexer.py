"""
LeanIndexer tests — the clean route-then-encode indexing spine.

Uses a deterministic stub embedder so the test needs no GPU, no CLIP download
and no network, and runs in well under a second. Builds a tiny synthetic video
with a static stretch and an event, and asserts the indexer:
  - encodes far fewer frames than it analyses (the whole point),
  - produces one real record per encoded frame with a unit-norm embedding,
  - stores only real frame indices (nothing synthesised / reconstructed),
  - records match the ADVESearchIndex.add_batch schema.
"""
import os
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _StubEmbedder:
    """Content-dependent unit vectors; no model, fully deterministic."""
    dim = 32

    def embed_frames(self, frames):
        out = []
        for f in frames:
            small = cv2.resize(f, (8, 4), interpolation=cv2.INTER_AREA)
            v = np.asarray(small.mean(axis=(0, 1)).tolist() + [small.std()], dtype=np.float32)
            v = np.pad(v, (0, self.dim - v.size))[: self.dim]
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)


def _make_video(path, fps=30, seconds=8, w=160, h=90):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bg = np.full((h, w, 3), 30, np.uint8)
    for i in range(seconds * fps):
        t = i / fps
        f = bg.copy()
        if 3.0 <= t < 4.0:                     # a one-second event
            cv2.rectangle(f, (60, 30), (100, 70), (0, 0, 255), -1)
        out.write(f)
    out.release()


def test_lean_indexer_routes_and_encodes():
    from adve.core.lean_indexer import LeanIndexer

    tmp = tempfile.mkdtemp()
    vid = os.path.join(tmp, "lean.mp4")
    _make_video(vid)

    indexer = LeanIndexer(embedder=_StubEmbedder(), weights="default")
    records, stats = indexer.index(vid, "lean.mp4", budget=20)

    # encoded far fewer than analysed
    assert stats.frames_analyzed > 0
    assert 0 < stats.frames_encoded <= 40
    assert stats.frames_encoded < stats.frames_analyzed
    assert stats.reduction_vs_dense > 1.0

    # one record per encoded frame, correct schema, real indices only
    assert len(records) == stats.frames_encoded
    seen_idx = set()
    for r in records:
        assert set(r) >= {"video_path", "camera_id", "timestamp", "frame_idx",
                          "embedding", "is_anchor", "text"}
        assert r["video_path"] == "lean.mp4"
        assert r["is_anchor"] is True                     # every vector is a real call
        assert r["frame_idx"] not in seen_idx             # no duplicates
        seen_idx.add(r["frame_idx"])
        emb = r["embedding"]
        assert emb.ndim == 1                              # flat, add_batch-safe
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_lean_indexer_feeds_search_index():
    """End-to-end: records from the indexer load into the real search index."""
    from adve.core.lean_indexer import LeanIndexer
    from adve.search.index import ADVESearchIndex

    tmp = tempfile.mkdtemp()
    vid = os.path.join(tmp, "lean2.mp4")
    _make_video(vid)

    records, stats = LeanIndexer(embedder=_StubEmbedder()).index(vid, "lean2.mp4", budget=15)

    idx_dir = os.path.join(tmp, "idx")
    index = ADVESearchIndex(idx_dir, dim=_StubEmbedder.dim)
    try:
        index.add_batch(records)              # must not raise (shape fix + schema)
        assert index.stats()["total_embeddings"] == len(records)
    finally:
        try:
            index.db.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_lean_indexer_routes_and_encodes()
    test_lean_indexer_feeds_search_index()
    print("[PASS] lean indexer tests")
