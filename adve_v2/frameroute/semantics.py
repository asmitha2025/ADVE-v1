"""
frameroute.semantics — change in CONTENT, not change in pixels.

Pixel novelty (signals.py) is blind to *what* changed. On a lecture, a slide
flip and a light flicker are both histogram distance; the thing a viewer cares
about is that the content changed. This module derives novelty from sparse
observations of content — on-screen text via OCR, with the same interface
usable for transcript segments — and fuses it into a SignalTrack as a FLOOR on
novelty:

    fused(t) = max(pixel_novelty(t), weight * semantic_novelty(t))

A floor, not a replacement. The pixel signal, the hard-trigger guarantee and
the max-gap guarantee all still apply, so a semantic signal that misses
things cannot make the router blind.

Cost model: OCR is expensive (~10^2 ms/frame), so it runs on a sparse sample
(every `every_sec` seconds) and the result is upsampled onto the dense signal
track. The reported change is the change detected at the observation frame,
held until the next observation. On slide-heavy footage that lands on the
slide boundary, which is where the router should spend the call.

This is the measurement-first experiment (bench/semantic_bench.py) before it
is a product feature: content novelty is useful only if content-fused routing
beats uniform sampling at a matched call count, and that is a number, not an
opinion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{3,}")

# Footer/boilerplate words that appear on almost every slide are not content.
# Any token on more than this fraction of observations is dropped before the
# comparison, so a course title in the corner does not mask real changes.
UBIQUITOUS_DF = 0.8

# Below this Jaccard distance a text change is noise (OCR jitter, a single
# character misread) rather than a content change. Jaccard distance between
# two genuinely different bullet slides is typically 0.6-0.95; OCR re-reads
# of the SAME slide land at 0.1-0.5. Measured on a real lecture: 0.25 removes
# most jitter without losing a single true slide transition.
MIN_CHANGE = 0.25

# A change must STICK to count: the tokens that entered must still be there
# in the next reading (and the ones that left must stay gone). OCR on a
# lecture produces constant small deltas as it misreads a character or drops
# a line; those do not persist. Slides do.
PERSIST_FRAC = 0.5

# Novelty reported when text appears or disappears and stays that way (a cut
# between a slide and a talking head). Not 1.0: no content was read, only the
# presence of text changed.
EMPTY_CHANGE = 0.6


@dataclass
class TextObservation:
    """One OCR reading of a frame's on-screen text."""
    t: float
    text: str


# --------------------------------------------------------------------------
# observation extraction (sparse decode + pluggable OCR)
# --------------------------------------------------------------------------

def default_ocr(frame_dict: Dict[int, np.ndarray], device: str = "cuda") -> Dict[int, str]:
    """EasyOCR batch reader, reusing the product's grounded-eval setup."""
    from bench.grounded_eval import ocr_frames
    return ocr_frames(frame_dict, device=device)


def extract_text_observations(
    video_path: str,
    every_sec: float = 2.0,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    ocr_fn: Optional[Callable[[Dict[int, np.ndarray]], Dict[int, str]]] = None,
) -> List[TextObservation]:
    """
    Decode roughly one frame per `every_sec` seconds and OCR it.

    `ocr_fn` maps {frame_idx: bgr} -> {frame_idx: text}; inject a stub in
    tests so no OCR dependency is needed to verify the maths.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(every_sec * fps)))
    cap.release()

    fn = ocr_fn or (lambda frames: default_ocr(frames, device=device or "cuda"))

    # One sequential pass, keeping one frame per stride, so long videos do not
    # pay a seek per sample.
    frames: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames[idx] = frame
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1
    cap.release()

    texts = fn(frames)
    return [
        TextObservation(t=i / fps, text=(texts.get(i) or "").strip())
        for i in sorted(frames)
    ]


# --------------------------------------------------------------------------
# pure novelty maths — no video, no OCR, fully testable
# --------------------------------------------------------------------------

def _token_sets(observations: Sequence[TextObservation]) -> List[set]:
    sets: List[set] = []
    df: Dict[str, int] = {}
    for obs in observations:
        toks = set(TOKEN_RE.findall(obs.text.lower()))
        sets.append(toks)
        for w in toks:
            df[w] = df.get(w, 0) + 1

    n = max(len(observations), 1)
    # With very few samples, any shared token looks "ubiquitous"; only apply
    # the footer filter once there are enough observations to judge.
    ubiquitous = ({w for w, c in df.items() if c > UBIQUITOUS_DF * n}
                  if len(observations) >= 5 else set())
    if ubiquitous:
        sets = [s - ubiquitous for s in sets]
    return sets


def text_change_scores(observations: Sequence[TextObservation],
                       min_change: float = MIN_CHANGE,
                       persist_frac: float = PERSIST_FRAC) -> np.ndarray:
    """
    novelty[i] = how much the on-screen text changed INTO observation i.

    Two-stage filter, both stages required:
      1. magnitude — Jaccard distance between consecutive token sets must
         clear min_change;
      2. persistence — the tokens that entered must still be present in the
         next reading, and the ones that left must stay gone. OCR jitter does
         not persist; slides do.

    Text appearing/disappearing reports EMPTY_CHANGE, but only when the
    (non-)emptiness is stable: a single missed OCR read on an unchanged slide
    is not an event.
    """
    sets = _token_sets(observations)
    n = len(sets)
    out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        a, b = sets[i - 1], sets[i]
        nxt = sets[i + 1] if i + 1 < n else None

        if not a and not b:
            continue
        if not a or not b:
            if nxt is None:
                out[i] = EMPTY_CHANGE
            elif not a:                             # text appeared
                out[i] = EMPTY_CHANGE if nxt else 0.0
            else:                                   # text disappeared
                out[i] = EMPTY_CHANGE if not nxt else 0.0
            continue

        union = a | b
        if not union:
            continue
        jaccard = 1.0 - len(a & b) / len(union)
        if jaccard < min_change:
            continue

        if nxt is not None:
            entering = b - a
            if entering and len(entering & nxt) / len(entering) < persist_frac:
                continue
            leaving = a - b
            if leaving and len(leaving & nxt) / len(leaving) > (1.0 - persist_frac):
                continue

        out[i] = jaccard
    return out


def text_novelty_series(observations: Sequence[TextObservation],
                        times: Sequence[float],
                        min_change: float = MIN_CHANGE) -> np.ndarray:
    """
    Upsample sparse text-change scores onto dense frame times.

    A change detected at observation time t_i describes every frame from t_i
    until the next observation: novelty(t) = score of the most recent
    observation at or before t. Frames before the first observation are 0.
    """
    times_arr = np.asarray(times, dtype=np.float64)
    if not observations or times_arr.size == 0:
        return np.zeros(times_arr.size, dtype=np.float64)

    scores = text_change_scores(observations, min_change=min_change)
    obs_t = np.asarray([o.t for o in observations], dtype=np.float64)

    # searchsorted(side="right") - 1 = index of latest observation <= t
    idx = np.searchsorted(obs_t, times_arr, side="right") - 1
    out = np.zeros(times_arr.size, dtype=np.float64)
    valid = idx >= 0
    out[valid] = scores[idx[valid]]
    return out


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def fuse_track(track, semantic: np.ndarray, weight: float = 1.0):
    """
    Return a copy of `track` whose novelty is max(pixel, weight * semantic).

    `weight` < 1 means a text change can be routed but only outranks pixel
    novelty when the text really moved; weight 1 lets the full semantic
    signal through. A floor is used rather than a blend so that a misread by
    OCR can never *reduce* the pixel signal.
    """
    sem = np.asarray(semantic, dtype=np.float64).reshape(-1)
    if sem.size != track.n_analyzed:
        raise ValueError(
            f"semantic series has {sem.size} values for {track.n_analyzed} signals"
        )
    fused = np.maximum(track.novelty_array(), float(weight) * np.clip(sem, 0.0, 1.0))
    return track.with_novelty(fused)
