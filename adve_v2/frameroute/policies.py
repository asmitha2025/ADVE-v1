"""
frameroute.policies — the baselines the router has to beat.

A routing claim is meaningless without these. "We cut calls by 80%" is not
a result; "at 240 calls/hour we recover 96% of full-index retrieval while
1 fps recovers 71%" is a result. Every policy here exposes the same
interface so bench/ can hold the call budget fixed and vary only the
selection strategy.

    policy = UniformFPS(1.0)
    picks  = policy.select(track)      # -> List[Pick]

Included:
  AllFrames      the ground-truth upper bound (spend everything)
  UniformFPS     the industry default, and the thing to beat
  UniformN       fixed number of evenly spaced frames
  SceneCut       shot-boundary detection on the appearance signal
  RouterPolicy   frameroute.router, wrapped to match the interface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np

from .router import FrameRouter, Pick
from .signals import SignalTrack


class SelectionPolicy(Protocol):
    name: str

    def select(self, track: SignalTrack, budget: Optional[int] = None) -> List[Pick]:
        ...


# --------------------------------------------------------------------------

@dataclass
class AllFrames:
    """Upper bound. Every analyzed frame gets a call. Use as the oracle."""
    name: str = "all_frames"

    def select(self, track: SignalTrack, budget: Optional[int] = None) -> List[Pick]:
        return [
            Pick(s.idx, s.t, s.novelty, "coverage") for s in track.signals
        ]


@dataclass
class UniformFPS:
    """
    Sample at a fixed rate. This is what almost every video-RAG pipeline
    does today, and what the router must beat at equal cost.
    """
    fps: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"uniform_{self.fps:g}fps"

    def select(self, track: SignalTrack, budget: Optional[int] = None) -> List[Pick]:
        if not track.signals:
            return []
        interval = 1.0 / max(self.fps, 1e-9)
        picks, next_t = [], 0.0
        for s in track.signals:
            if s.t + 1e-9 >= next_t:
                picks.append(Pick(s.idx, s.t, s.novelty, "coverage"))
                next_t = s.t + interval
        return picks


@dataclass
class UniformN:
    """N evenly spaced frames. The fair time-domain comparison at a fixed budget."""
    n: int = 100
    name: str = "uniform_n"

    def select(self, track: SignalTrack, budget: Optional[int] = None) -> List[Pick]:
        n = int(budget or self.n)
        sigs = track.signals
        if not sigs:
            return []
        n = max(1, min(n, len(sigs)))
        positions = np.unique(np.linspace(0, len(sigs) - 1, n).round().astype(int))
        return [Pick(sigs[p].idx, sigs[p].t, sigs[p].novelty, "coverage") for p in positions]


@dataclass
class SceneCut:
    """
    Shot-boundary detection on the appearance signal.

    Strong on edited content, blind within a shot -- which is exactly the
    regime that matters for surveillance, lectures and dashcam footage,
    where the whole video is one shot.

    If `budget` is given, the threshold is chosen so roughly that many cuts
    fire, which keeps the comparison honest at matched cost.
    """
    threshold: float = 0.25
    name: str = "scene_cut"

    def select(self, track: SignalTrack, budget: Optional[int] = None) -> List[Pick]:
        sigs = track.signals
        if not sigs:
            return []
        app = np.asarray([s.hist_delta_prev for s in sigs], dtype=np.float64)

        thr = self.threshold
        if budget is not None and budget < len(sigs):
            # pick the threshold that yields ~budget cuts
            k = max(1, min(int(budget), app.size - 1))
            thr = float(np.partition(app, -k)[-k])

        picks = [Pick(sigs[0].idx, sigs[0].t, sigs[0].novelty, "first")]
        for i, s in enumerate(sigs[1:], start=1):
            if app[i] >= thr:
                picks.append(Pick(s.idx, s.t, s.novelty, "coverage"))
        return picks


@dataclass
class RouterPolicy:
    """frameroute.router wrapped to match the baseline interface."""
    policy: str = "coverage"
    min_gap_sec: float = 0.0
    max_gap_sec: float = 30.0
    hard_trigger: Optional[float] = 0.55
    reserve_for_triggers: float = 0.25
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"router_{self.policy}"

    def select(self, track: SignalTrack, budget: Optional[int] = None) -> List[Pick]:
        r = FrameRouter(
            min_gap_sec=self.min_gap_sec,
            max_gap_sec=self.max_gap_sec,
            hard_trigger=self.hard_trigger,
            reserve_for_triggers=self.reserve_for_triggers,
        )
        return r.select(track, budget=budget, policy=self.policy).picks


# --------------------------------------------------------------------------

def default_policy_suite(include_oracle: bool = False) -> List[SelectionPolicy]:
    """The comparison set used by bench/routing_bench.py."""
    suite: List[SelectionPolicy] = [
        UniformN(),                       # matched-budget time sampling
        UniformFPS(1.0),                  # the industry default
        SceneCut(),                       # shot boundaries
        RouterPolicy(policy="coverage"),  # ours
        RouterPolicy(policy="hybrid"),    # ours, event-weighted
    ]
    if include_oracle:
        suite.insert(0, AllFrames())
    return suite


def policy_by_name(name: str) -> SelectionPolicy:
    table: Dict[str, SelectionPolicy] = {
        "all_frames": AllFrames(),
        "uniform_n": UniformN(),
        "uniform_1fps": UniformFPS(1.0),
        "uniform_2fps": UniformFPS(2.0),
        "uniform_0.5fps": UniformFPS(0.5),
        "scene_cut": SceneCut(),
        "router": RouterPolicy(policy="coverage"),
        "router_coverage": RouterPolicy(policy="coverage"),
        "router_hybrid": RouterPolicy(policy="hybrid"),
        "router_peak": RouterPolicy(policy="peak"),
    }
    if name not in table:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(table)}")
    return table[name]
