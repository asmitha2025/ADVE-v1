"""
frameroute.router — budget-constrained frame selection.

The question every video-AI pipeline has to answer is "which frames do I
send to the expensive model?" Today the field answers it two ways, both bad:

  1 fps / uniform-N   samples in TIME. Spends the same on a static corridor
                      as on a collision. Misses anything shorter than the
                      interval.
  scene-cut           fires on shot boundaries. Blind to everything that
                      happens *within* a shot, which in surveillance,
                      lectures and dashcam footage is everything.

This module samples in CHANGE. Given a novelty signal over the video
(frameroute.signals) and a budget of N calls, it places those N calls so
that each one covers an equal amount of *accumulated change* rather than an
equal amount of time. Static stretches collapse to one call; busy stretches
get as many as they earn.

That is the whole idea, and it is small enough to state in one line:

    cut the cumulative-novelty curve into N equal-area segments,
    spend one call per segment.

On top of it sit two guarantees that make the result safe to sell:

  hard triggers   any frame whose instantaneous novelty exceeds a threshold
                  is selected regardless of budget. This is what stops a
                  0.4-second event from being averaged away.
  max gap         no two consecutive picks may be further apart than
                  max_gap_sec, regardless of how static the video is.

Both are guarantees about the SIGNAL, not about semantics. The router can
promise it never skipped a large measured change; it cannot promise the
change was meaningful. Anyone who claims the second thing without a
labelled evaluation is doing what the ADVE v3.1 docs did with "zero missed
security events" -- asserting an outcome the experiment never tested. Use
bench/routing_bench.py to find out what actually got covered.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .signals import SignalTrack, SignalExtractor, analyze_video


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class Pick:
    """One frame the router decided is worth an expensive call."""
    idx: int
    t: float
    novelty: float
    reason: str          # "first" | "coverage" | "hard_trigger" | "max_gap" | "peak"

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Selection:
    video_path: str
    fps: float
    duration_sec: float
    policy: str
    budget_requested: Optional[int]
    picks: List[Pick] = field(default_factory=list)
    frames_analyzed: int = 0
    analyze_seconds: float = 0.0
    signal_cost_ms: float = 0.0

    # -- derived -----------------------------------------------------------

    @property
    def n_calls(self) -> int:
        return len(self.picks)

    @property
    def calls_per_hour(self) -> float:
        hrs = self.duration_sec / 3600.0
        return self.n_calls / hrs if hrs > 0 else 0.0

    def indices(self) -> List[int]:
        return [p.idx for p in self.picks]

    def times(self) -> List[float]:
        return [p.t for p in self.picks]

    def reason_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self.picks:
            out[p.reason] = out.get(p.reason, 0) + 1
        return out

    def guarantees(self, track: Optional[SignalTrack] = None) -> Dict[str, float]:
        """
        What the router can honestly promise about this selection.
        Everything here is measured from the signal, not asserted.
        """
        ts = np.asarray(self.times(), dtype=np.float64)
        gaps = np.diff(ts) if ts.size > 1 else np.asarray([0.0])
        out = {
            "n_calls": self.n_calls,
            "max_time_gap_sec": round(float(gaps.max()), 3) if gaps.size else 0.0,
            "median_time_gap_sec": round(float(np.median(gaps)), 3) if gaps.size else 0.0,
        }
        if track is not None and track.n_analyzed:
            nov = track.novelty_array()
            picked = set(self.indices())
            all_idx = [s.idx for s in track.signals]
            unpicked = np.asarray(
                [n for i, n in zip(all_idx, nov) if i not in picked], dtype=np.float64
            )
            out["max_unselected_novelty"] = (
                round(float(unpicked.max()), 4) if unpicked.size else 0.0
            )
            out["novelty_covered_pct"] = round(
                100.0 * float(nov[[i in picked for i in all_idx]].sum()) /
                float(nov.sum() + 1e-9), 2
            )
        return out

    def to_json(self, path: str, track: Optional[SignalTrack] = None) -> None:
        payload = {
            "video_path": self.video_path,
            "policy": self.policy,
            "fps": self.fps,
            "duration_sec": round(self.duration_sec, 3),
            "budget_requested": self.budget_requested,
            "n_calls": self.n_calls,
            "calls_per_hour": round(self.calls_per_hour, 2),
            "frames_analyzed": self.frames_analyzed,
            "analyze_seconds": round(self.analyze_seconds, 3),
            "mean_signal_cost_ms": round(self.signal_cost_ms, 3),
            "reason_counts": self.reason_counts(),
            "guarantees": self.guarantees(track),
            "picks": [p.as_dict() for p in self.picks],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------

class FrameRouter:
    """
    Budget-constrained frame selection over a precomputed SignalTrack.

        track = analyze_video("match.mp4", weights="surveillance")
        router = FrameRouter()
        sel = router.select(track, budget_per_hour=240)

    Separating analyze() from select() is deliberate: analysis is one cheap
    decode pass, and once you have the track you can sweep dozens of budgets
    for free. bench/routing_bench.py relies on this to draw the cost-recall
    curve without re-decoding.
    """

    def __init__(
        self,
        min_gap_sec: float = 0.0,
        max_gap_sec: float = 30.0,
        hard_trigger: Optional[float] = 0.55,
        reserve_for_triggers: float = 0.25,
    ):
        """
        min_gap_sec           never place two calls closer than this
        max_gap_sec           never leave a stretch longer than this uncovered
        hard_trigger          novelty above this is always selected (None disables)
        reserve_for_triggers  fraction of budget held back for hard triggers,
                              so a single busy passage cannot eat the whole
                              budget and starve the rest of the video
        """
        self.min_gap_sec = float(min_gap_sec)
        self.max_gap_sec = float(max_gap_sec)
        self.hard_trigger = hard_trigger
        self.reserve_for_triggers = float(np.clip(reserve_for_triggers, 0.0, 0.9))

    # -- public ------------------------------------------------------------

    def analyze(self, video_path: str, **kw) -> SignalTrack:
        return analyze_video(video_path, **kw)

    def select(
        self,
        track: SignalTrack,
        budget: Optional[int] = None,
        budget_per_hour: Optional[float] = None,
        policy: str = "coverage",
    ) -> Selection:
        """
        policy:
          "coverage"  equal-accumulated-novelty segmentation (recommended)
          "peak"      top-N novelty peaks with a minimum gap
          "hybrid"    hard triggers + peaks first, coverage for the remainder
        """
        if not track.signals:
            raise ValueError("empty SignalTrack -- run analyze_video first")

        n_budget = self._resolve_budget(track, budget, budget_per_hour)

        sel = Selection(
            video_path=track.video_path,
            fps=track.fps,
            duration_sec=track.duration_sec,
            policy=policy,
            budget_requested=n_budget,
            frames_analyzed=track.n_analyzed,
            analyze_seconds=track.analyze_seconds,
            signal_cost_ms=track.mean_signal_cost_ms(),
        )

        if policy == "coverage":
            picks = self._coverage(track, n_budget)
        elif policy == "peak":
            picks = self._peaks(track, n_budget)
        elif policy == "hybrid":
            picks = self._hybrid(track, n_budget)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        picks = self._apply_max_gap(track, picks)
        picks = self._dedupe_and_sort(picks)
        sel.picks = picks
        return sel

    def route(
        self,
        video_path: str,
        budget: Optional[int] = None,
        budget_per_hour: Optional[float] = None,
        policy: str = "coverage",
        weights: str = "default",
        stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> tuple[Selection, SignalTrack]:
        """analyze + select in one call."""
        track = analyze_video(
            video_path, stride=stride, max_frames=max_frames, weights=weights
        )
        return self.select(track, budget, budget_per_hour, policy), track

    # -- policies ----------------------------------------------------------

    def _coverage(self, track: SignalTrack, n_budget: int) -> List[Pick]:
        """
        Equal-accumulated-novelty segmentation.

        Build the cumulative novelty curve C(t). Cut it at n equally spaced
        levels. Within each resulting segment, take the frame carrying the
        most novelty -- that frame is the segment's best single
        representative, because it is where the change actually happened.

        Compare with uniform sampling, which cuts C's *x-axis* into equal
        parts and is therefore optimal only when novelty is uniform, i.e.
        never.
        """
        sigs = track.signals
        nov = track.novelty_array()
        picks: List[Pick] = [Pick(sigs[0].idx, sigs[0].t, float(nov[0]), "first")]

        if n_budget <= 1 or nov.size < 2:
            return picks

        # hard triggers are honored first and consume budget
        trigger_positions = self._trigger_positions(track)
        reserved = int(round(n_budget * self.reserve_for_triggers))
        honored = trigger_positions[: max(reserved, 0)] if reserved > 0 else []
        for p in honored:
            picks.append(Pick(sigs[p].idx, sigs[p].t, float(nov[p]), "hard_trigger"))

        remaining = max(1, n_budget - len(picks))

        cum = np.cumsum(nov)
        total = float(cum[-1])
        if total <= 1e-9:
            # perfectly static video: fall back to uniform, which is correct here
            step = max(1, nov.size // remaining)
            for p in range(step, nov.size, step):
                picks.append(Pick(sigs[p].idx, sigs[p].t, float(nov[p]), "coverage"))
            return picks

        # levels at which to cut the cumulative curve
        levels = np.linspace(0.0, total, remaining + 1)[1:-1] if remaining > 1 else []
        boundaries = [0] + [int(np.searchsorted(cum, lv)) for lv in levels] + [nov.size]
        boundaries = sorted(set(int(np.clip(b, 0, nov.size)) for b in boundaries))

        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b <= a:
                continue
            seg = nov[a:b]
            local = a + int(np.argmax(seg))
            picks.append(Pick(sigs[local].idx, sigs[local].t, float(nov[local]), "coverage"))

        return self._enforce_min_gap(picks)

    def _peaks(self, track: SignalTrack, n_budget: int) -> List[Pick]:
        """Top-N local maxima with a minimum separation. Good for events, bad for coverage."""
        sigs = track.signals
        nov = track.novelty_array()
        picks: List[Pick] = [Pick(sigs[0].idx, sigs[0].t, float(nov[0]), "first")]

        order = np.argsort(-nov)
        min_gap = max(self.min_gap_sec, 1e-6)
        chosen_t: List[float] = [sigs[0].t]

        for p in order:
            if len(picks) >= n_budget:
                break
            t = sigs[p].t
            if all(abs(t - ct) >= min_gap for ct in chosen_t):
                picks.append(Pick(sigs[p].idx, t, float(nov[p]), "peak"))
                chosen_t.append(t)
        return picks

    def _hybrid(self, track: SignalTrack, n_budget: int) -> List[Pick]:
        """Hard triggers, then peaks for a slice of budget, coverage for the rest."""
        sigs = track.signals
        nov = track.novelty_array()
        picks: List[Pick] = [Pick(sigs[0].idx, sigs[0].t, float(nov[0]), "first")]

        for p in self._trigger_positions(track):
            if len(picks) >= n_budget:
                break
            picks.append(Pick(sigs[p].idx, sigs[p].t, float(nov[p]), "hard_trigger"))

        picks = self._enforce_min_gap(picks)
        remaining = max(0, n_budget - len(picks))
        if remaining > 0:
            sub = self._coverage(track, remaining)
            picks.extend(p for p in sub if p.reason != "first")
        return picks

    # -- guarantees --------------------------------------------------------

    def _trigger_positions(self, track: SignalTrack) -> List[int]:
        """Positions (not frame indices) whose novelty exceeds the hard trigger."""
        if self.hard_trigger is None:
            return []
        nov = track.novelty_array()
        hits = np.where(nov >= self.hard_trigger)[0]
        # strongest first, so that a truncated reserve keeps the biggest events
        return list(hits[np.argsort(-nov[hits])])

    def _apply_max_gap(self, track: SignalTrack, picks: List[Pick]) -> List[Pick]:
        """Insert extra picks so no stretch longer than max_gap_sec is uncovered."""
        if self.max_gap_sec <= 0 or not picks:
            return picks

        sigs = track.signals
        nov = track.novelty_array()
        times = track.time_array()
        picks = sorted(picks, key=lambda p: p.t)

        out: List[Pick] = []
        end_t = sigs[-1].t
        for i, p in enumerate(picks):
            out.append(p)
            nxt_t = picks[i + 1].t if i + 1 < len(picks) else end_t
            gap = nxt_t - p.t
            if gap > self.max_gap_sec:
                n_fill = int(math.floor(gap / self.max_gap_sec))
                for k in range(1, n_fill + 1):
                    target = p.t + k * self.max_gap_sec
                    if target >= nxt_t:
                        break
                    pos = int(np.argmin(np.abs(times - target)))
                    out.append(Pick(sigs[pos].idx, sigs[pos].t, float(nov[pos]), "max_gap"))
        return out

    def _enforce_min_gap(self, picks: List[Pick]) -> List[Pick]:
        if self.min_gap_sec <= 0:
            return picks
        picks = sorted(picks, key=lambda p: (-p.novelty))
        kept: List[Pick] = []
        for p in picks:
            if all(abs(p.t - k.t) >= self.min_gap_sec for k in kept):
                kept.append(p)
        return kept

    @staticmethod
    def _dedupe_and_sort(picks: List[Pick]) -> List[Pick]:
        seen: Dict[int, Pick] = {}
        priority = {"hard_trigger": 0, "first": 1, "peak": 2, "coverage": 3, "max_gap": 4}
        for p in picks:
            cur = seen.get(p.idx)
            if cur is None or priority.get(p.reason, 9) < priority.get(cur.reason, 9):
                seen[p.idx] = p
        return sorted(seen.values(), key=lambda p: p.idx)

    @staticmethod
    def _resolve_budget(
        track: SignalTrack, budget: Optional[int], budget_per_hour: Optional[float]
    ) -> int:
        if budget is not None:
            return max(1, int(budget))
        if budget_per_hour is not None:
            hrs = track.duration_sec / 3600.0
            return max(1, int(round(budget_per_hour * hrs)))
        # default: 1 call per 2 seconds of video, a common video-RAG setting
        return max(1, int(round(track.duration_sec / 2.0)))


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------

class StreamingRouter:
    """
    Online variant for live streams, where the cumulative curve is unknown.

    Uses a token bucket: the stream earns call-tokens at budget_per_hour and
    spends one whenever novelty clears an adaptive threshold, with the
    threshold tracking the running distribution so a busy camera does not
    permanently exhaust its budget. Hard triggers bypass the bucket.

        sr = StreamingRouter(budget_per_hour=240, fps=15)
        for idx, frame in stream:
            if sr.should_spend(frame, idx):
                embedding = expensive_model(frame)   # your call
    """

    def __init__(
        self,
        budget_per_hour: float = 240.0,
        fps: float = 15.0,
        hard_trigger: float = 0.55,
        max_gap_sec: float = 30.0,
        weights: str = "surveillance",
        target_percentile: float = 90.0,
    ):
        self.tokens_per_frame = budget_per_hour / (3600.0 * max(fps, 1e-6))
        self.fps = fps
        self.hard_trigger = hard_trigger
        self.max_gap_frames = int(max_gap_sec * fps)
        self.target_percentile = target_percentile

        self.extractor = SignalExtractor(weights=weights)
        self.tokens = 1.0
        self.max_tokens = 8.0
        self.frames_since_spend = 0
        self._recent: List[float] = []
        self._thresh = 0.15

        self.stats = {"frames": 0, "spends": 0, "hard_triggers": 0, "max_gap_fills": 0}

    def should_spend(self, frame: np.ndarray, idx: int) -> bool:
        self.stats["frames"] += 1
        self.tokens = min(self.max_tokens, self.tokens + self.tokens_per_frame)
        sig = self.extractor.step(frame, idx=idx, t=idx / self.fps)
        self.frames_since_spend += 1

        self._recent.append(sig.novelty)
        if len(self._recent) > 600:
            self._recent.pop(0)
        if len(self._recent) >= 60 and self.stats["frames"] % 30 == 0:
            self._thresh = float(np.percentile(self._recent, self.target_percentile))

        spend, reason = False, ""
        if sig.novelty >= self.hard_trigger:
            spend, reason = True, "hard_trigger"
        elif self.frames_since_spend >= self.max_gap_frames:
            spend, reason = True, "max_gap"
        elif self.tokens >= 1.0 and sig.novelty >= self._thresh:
            spend, reason = True, "coverage"

        if spend:
            self.tokens = max(0.0, self.tokens - 1.0)
            self.frames_since_spend = 0
            self.extractor.set_anchor(frame)
            self.stats["spends"] += 1
            if reason == "hard_trigger":
                self.stats["hard_triggers"] += 1
            elif reason == "max_gap":
                self.stats["max_gap_fills"] += 1
        return spend

    def report(self) -> Dict:
        f = max(self.stats["frames"], 1)
        return {
            **self.stats,
            "spend_rate_pct": round(100.0 * self.stats["spends"] / f, 2),
            "adaptive_threshold": round(self._thresh, 4),
            "signal_timing_ms": self.extractor.timing_ms(),
        }
