"""
frameroute — budget-constrained frame selection for video AI pipelines.

The expensive part of a video pipeline is the model call. The cheap part is
deciding which frames deserve one. frameroute does the cheap part well so
you can do less of the expensive part.

    from frameroute import FrameRouter, analyze_video

    track = analyze_video("match.mp4", weights="surveillance")
    sel   = FrameRouter().select(track, budget_per_hour=240)

    for pick in sel.picks:
        embedding = your_expensive_model(frame_at(pick.idx))

Design rules this package holds itself to:

  1. Nothing in the hot path costs more than the call it is avoiding.
     No detector, no encoder, no tracker. Signals run on a 160x90 downscale
     and report their own measured cost via SignalExtractor.timing_ms().

  2. Savings are counted, never estimated. A saved call is a call that did
     not happen, measured by CountingEmbedder -- not a FLOP subtraction.

  3. Every claim has a baseline. frameroute.policies ships the alternatives
     the router has to beat, so results are always stated at matched cost.

  4. Guarantees are about the signal, not about meaning. The router can
     promise it never skipped a large measured change. Whether that change
     mattered is an empirical question answered by bench/, not by the
     router's own docstring.
"""

from .signals import (
    FrameSignals, SignalTrack, SignalExtractor, analyze_video,
)
from .router import FrameRouter, StreamingRouter, Pick, Selection
from .policies import (
    AllFrames, UniformFPS, UniformN, SceneCut, RouterPolicy,
    default_policy_suite, policy_by_name,
)
from .adapters import (
    ClipEmbedder, CountingEmbedder, CallLedger, ExactIndex,
    PricedCaptioner, DryRunCaptioner,
)
from .cascade import (
    TierSelection, CascadeSelection, CascadeRouter, route_cascade
)

__version__ = "0.1.0"

__all__ = [
    "FrameSignals", "SignalTrack", "SignalExtractor", "analyze_video",
    "FrameRouter", "StreamingRouter", "Pick", "Selection",
    "AllFrames", "UniformFPS", "UniformN", "SceneCut", "RouterPolicy",
    "default_policy_suite", "policy_by_name",
    "ClipEmbedder", "CountingEmbedder", "CallLedger", "ExactIndex",
    "PricedCaptioner", "DryRunCaptioner",
    "TierSelection", "CascadeSelection", "CascadeRouter", "route_cascade",
]
