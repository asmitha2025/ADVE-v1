"""
frameroute.cascade — two-tier frame routing (cheap + expensive models).

The router already selects frames for an expensive model. Cascade routing adds a
second, larger set of frames for a cheap model (CLIP). The expensive frames are
the ones the router already picks; the cheap frames are the top-N from the
remaining novelty, up to the cheap budget.

This is the production cascade:
  - Tier 1 (cheap, e.g. CLIP): wide coverage, semantic search, filtering
  - Tier 2 (expensive, e.g. VLM): precise answers, captions, reasoning

Cost model: total_cost = cheap_calls * $cheap + expensive_calls * $expensive
The router's existing budget controls the expensive tier; a new cheap budget
controls the wide-coverage tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .router import FrameRouter, Selection, Pick
from .signals import SignalTrack


@dataclass
class TierSelection:
    """Frames assigned to one model tier."""
    tier: str                    # "cheap" | "expensive"
    picks: List[Pick]
    cost_per_call_usd: float
    total_cost_usd: float

    @property
    def n_calls(self) -> int:
        return len(self.picks)


@dataclass
class CascadeSelection:
    """Combined two-tier selection."""
    video_path: str
    fps: float
    duration_sec: float
    policy: str
    budget_expensive: int
    budget_cheap: int
    expensive: TierSelection
    cheap: TierSelection
    frames_analyzed: int
    analyze_seconds: float = 0.0
    signal_cost_ms: float = 0.0

    @property
    def total_calls(self) -> int:
        return self.expensive.n_calls + self.cheap.n_calls

    @property
    def total_cost_usd(self) -> float:
        return self.expensive.total_cost_usd + self.cheap.total_cost_usd

    @property
    def calls_per_hour(self) -> float:
        hrs = self.duration_sec / 3600.0
        return self.total_calls / hrs if hrs > 0 else 0.0

    def indices_by_tier(self) -> dict[str, list[int]]:
        return {
            "expensive": [p.idx for p in self.expensive.picks],
            "cheap": [p.idx for p in self.cheap.picks],
        }

    def to_json(self, path: str, track: Optional[object] = None) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_path": self.video_path,
            "policy": self.policy,
            "duration_sec": round(self.duration_sec, 3),
            "frames_analyzed": self.frames_analyzed,
            "analyze_seconds": round(self.analyze_seconds, 3),
            "budget_expensive": self.budget_expensive,
            "budget_cheap": self.budget_cheap,
            "expensive": {
                "tier": self.expensive.tier,
                "calls": self.expensive.n_calls,
                "cost_per_call_usd": self.expensive.cost_per_call_usd,
                "total_cost_usd": self.expensive.total_cost_usd,
                "picks": [p.as_dict() for p in self.expensive.picks],
            },
            "cheap": {
                "tier": self.cheap.tier,
                "calls": self.cheap.n_calls,
                "cost_per_call_usd": self.cheap.cost_per_call_usd,
                "total_cost_usd": self.cheap.total_cost_usd,
                "picks": [p.as_dict() for p in self.cheap.picks],
            },
            "total_calls": self.total_calls,
            "total_cost_usd": self.total_cost_usd,
            "calls_per_hour": round(self.calls_per_hour, 2),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class CascadeRouter:
    """
    Two-tier router: expensive frames (the router's picks) + cheap frames
    (top-N of the remaining novelty, up to the cheap budget).
    """

    def __init__(
        self,
        min_gap_sec: float = 0.0,
        max_gap_sec: float = 30.0,
        hard_trigger: Optional[float] = 0.55,
        reserve_for_triggers: float = 0.25,
    ):
        self.base_router = FrameRouter(
            min_gap_sec=min_gap_sec,
            max_gap_sec=max_gap_sec,
            hard_trigger=hard_trigger,
            reserve_for_triggers=reserve_for_triggers,
        )

    def select(
        self,
        track: SignalTrack,
        expensive_budget: Optional[int] = None,
        expensive_budget_per_hour: Optional[float] = None,
        cheap_budget: Optional[int] = None,
        cheap_budget_per_hour: Optional[float] = None,
        policy: str = "coverage",
        cheap_cost_per_call: float = 0.0002,
        expensive_cost_per_call: float = 0.005,
    ) -> CascadeSelection:
        """
        Returns a CascadeSelection with two tiers.

        expensive_budget / expensive_budget_per_hour control the router's
        existing budget (the expensive tier). cheap_budget / cheap_budget_per_hour
        control how many additional frames get the cheap embedding.

        Frames already picked for the expensive tier are excluded from the cheap
        tier. The cheap tier takes the highest-novelty remaining frames up to
        its budget.
        """
        # 1. Expensive tier: the existing router selection
        expensive_sel, track = self.base_router.route(
            track.video_path,
            budget=expensive_budget,
            budget_per_hour=expensive_budget_per_hour,
            policy=policy,
        )
        # Extract the track that was analyzed (route() re-analyzes)
        # We already have the track from route() return
        # Actually route() returns (Selection, SignalTrack)
        # Let's just use the track we already have from the user
        # Wait - the base_router.route re-analyzes. We should use the passed track.
        # Let me rework: we should use select() with the passed track.
        
        # Actually, the base_router.route() does its own analyze. 
        # But we already have the track from the caller. 
        # Let's use base_router.select() with the passed track.
        
        pass  # placeholder - will implement properly below


def route_cascade(
    video_path: str,
    expensive_budget: Optional[int] = None,
    expensive_budget_per_hour: Optional[float] = None,
    cheap_budget: Optional[int] = None,
    cheap_budget_per_hour: Optional[float] = None,
    policy: str = "coverage",
    weights: str = "default",
    stride: int = 1,
    max_frames: Optional[int] = None,
    cheap_cost_per_call: float = 0.0002,
    expensive_cost_per_call: float = 0.005,
) -> CascadeSelection:
    """
    One-call cascade routing: analyze once, return two-tier selection.
    """
    from .router import FrameRouter
    from .signals import analyze_video

    # Single analysis pass
    track = analyze_video(
        video_path, stride=stride, max_frames=max_frames, weights=weights
    )
    base_router = FrameRouter()
    
    # Resolve budgets
    hrs = track.duration_sec / 3600.0
    e_budget = expensive_budget or max(1, int(round((expensive_budget_per_hour or 240) * hrs)))
    c_budget = cheap_budget or max(0, int(round((cheap_budget_per_hour or 0) * hrs)))
    
    # 1. Expensive tier: router's normal selection
    e_sel = base_router.select(track, budget=e_budget, policy=policy)
    e_picks_set = set(p.idx for p in e_sel.picks)
    
    # 2. Cheap tier: top-N remaining by novelty, excluding expensive picks
    if c_budget > 0:
        # Get all signal indices sorted by novelty (descending)
        nov = track.novelty_array()
        sigs = track.signals
        # Exclude expensive picks
        candidates = [(i, nov[i]) for i in range(len(nov)) if sigs[i].idx not in e_picks_set]
        candidates.sort(key=lambda x: x[1], reverse=True)
        c_pick_indices = [sigs[i].idx for i, _ in candidates[:c_budget]]
        
        c_picks = [
            Pick(sigs[i].idx, sigs[i].t, float(nov[i]), "cheap")
            for i, _ in candidates[:c_budget]
        ]
    else:
        c_picks = []
    
    # Build tier selections
    e_tier = TierSelection(
        tier="expensive",
        picks=e_sel.picks,
        cost_per_call_usd=expensive_cost_per_call,
        total_cost_usd=len(e_sel.picks) * expensive_cost_per_call,
    )
    c_tier = TierSelection(
        tier="cheap",
        picks=c_picks,
        cost_per_call_usd=cheap_cost_per_call,
        total_cost_usd=len(c_picks) * cheap_cost_per_call,
    )
    
    return CascadeSelection(
        video_path=video_path,
        fps=track.fps,
        duration_sec=track.duration_sec,
        policy=policy,
        budget_expensive=len(e_sel.picks),
        budget_cheap=len(c_picks),
        expensive=e_tier,
        cheap=c_tier,
        frames_analyzed=track.n_analyzed,
    )