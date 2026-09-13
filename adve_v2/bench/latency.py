"""
bench.latency — the wall-clock number ops teams block on.

Cost tells you the bill; latency tells you whether it keeps up. This measures
the real end-to-end frameroute indexing pipeline on a chunk of real video and
reports it the way an ops team asks the question:

    - change-signal cost      ms / frame   (the router's hot path, CPU)
    - CLIP encode cost         ms / call    (only the routed frames)
    - realtime factor          Nx           (video-seconds processed per wall-second)
    - "1 hour of video ->      M minutes"    to index end to end
    - routing speedup          vs encoding every analysed frame

Run:
    python -m bench.latency --video LECTURE.mp4 --max-frames 3000 --budget-per-hour 1200
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LatencyReport:
    video: str
    device: str
    seconds_of_video: float
    frames_analyzed: int
    frames_encoded_routed: int
    signal_ms_per_frame: float
    encode_ms_per_call: float
    decode_fps: float
    # end to end (analyze + route + decode picks + encode picks)
    routed_wall_sec: float
    routed_realtime_factor: float     # video-sec / wall-sec  (>1 = faster than realtime)
    routed_hour_minutes: float        # minutes to index 1 hour of video
    # full-compute baseline (CLIP every analysed frame)
    full_wall_sec: float
    full_realtime_factor: float
    full_hour_minutes: float
    speedup_vs_full: float

    def table(self) -> str:
        def row(label, wall, rtf, hourmin):
            return f"  {label:22}{wall:>8.1f}s{rtf:>9.1f}x{hourmin:>9.1f}m"
        return "\n".join([
            "",
            f"  LATENCY — {Path(self.video).name}   [{self.device}]",
            f"  measured on {self.seconds_of_video:.0f}s of video "
            f"({self.frames_analyzed} frames analysed)",
            "  " + "-" * 58,
            f"  change signal        {self.signal_ms_per_frame:>7.2f} ms/frame   (router hot path)",
            f"  CLIP encode          {self.encode_ms_per_call:>7.2f} ms/call    (routed frames only)",
            f"  decode               {self.decode_fps:>7.0f} fps",
            "  " + "-" * 58,
            f"  {'':22}{'wall':>9}{'realtime':>10}{'1hr video':>10}",
            row("frameroute (routed)", self.routed_wall_sec, self.routed_realtime_factor, self.routed_hour_minutes),
            row("full (every frame)", self.full_wall_sec, self.full_realtime_factor, self.full_hour_minutes),
            "  " + "-" * 58,
            f"  routed is {self.speedup_vs_full:.1f}x faster than encoding every analysed frame",
            "",
        ])


def run(video: str, max_frames: Optional[int], stride: int,
        budget_per_hour: float, weights: str, device: Optional[str]) -> LatencyReport:
    from frameroute.signals import analyze_video
    from frameroute.router import FrameRouter
    from frameroute.adapters import ClipEmbedder
    from bench.parity import read_frames

    emb = ClipEmbedder(device=device)
    dev = str(emb.device)

    # 1. change-signal analysis (one decode pass; times the router's hot path)
    t0 = time.perf_counter()
    track = analyze_video(video, stride=stride, max_frames=max_frames, weights=weights)
    t_analyze = time.perf_counter() - t0
    n = track.n_analyzed
    seconds_of_video = n * stride / (track.fps or 30.0)
    signal_ms = track.mean_signal_cost_ms()
    decode_fps = n / t_analyze if t_analyze > 0 else 0.0

    # 2. route
    t0 = time.perf_counter()
    sel = FrameRouter().select(track, budget_per_hour=budget_per_hour)
    t_route = time.perf_counter() - t0
    picks = sorted(set(sel.indices()))

    # 3. decode the picked frames + encode them (the routed downstream cost)
    t0 = time.perf_counter()
    pick_frames = read_frames(video, picks)
    t_decode_picks = time.perf_counter() - t0
    ordered = [i for i in picks if i in pick_frames]
    t0 = time.perf_counter()
    _ = emb.embed_frames([pick_frames[i] for i in ordered])
    t_encode_routed = time.perf_counter() - t0
    encode_ms = 1000.0 * t_encode_routed / max(len(ordered), 1)

    routed_wall = t_analyze + t_route + t_decode_picks + t_encode_routed

    # 4. full-compute baseline: CLIP-encode EVERY analysed frame
    #    (reuse the already-decoded signal pass cost as the shared decode; the
    #    extra cost is encoding all n frames instead of len(picks))
    all_idx = [s.idx for s in track.signals]
    t0 = time.perf_counter()
    all_frames = read_frames(video, all_idx)
    t_decode_all = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(0, len(all_idx), 256):
        chunk = [all_frames[j] for j in all_idx[i:i + 256] if j in all_frames]
        if chunk:
            _ = emb.embed_frames(chunk)
    t_encode_full = time.perf_counter() - t0
    full_wall = t_analyze + t_decode_all + t_encode_full

    def rt(wall):
        return seconds_of_video / wall if wall > 0 else 0.0
    def hour_min(wall):
        return (3600.0 / rt(wall)) / 60.0 if rt(wall) > 0 else 0.0

    return LatencyReport(
        video=video, device=dev, seconds_of_video=seconds_of_video,
        frames_analyzed=n, frames_encoded_routed=len(ordered),
        signal_ms_per_frame=round(signal_ms, 3),
        encode_ms_per_call=round(encode_ms, 3),
        decode_fps=round(decode_fps, 1),
        routed_wall_sec=round(routed_wall, 2),
        routed_realtime_factor=round(rt(routed_wall), 2),
        routed_hour_minutes=round(hour_min(routed_wall), 2),
        full_wall_sec=round(full_wall, 2),
        full_realtime_factor=round(rt(full_wall), 2),
        full_hour_minutes=round(hour_min(full_wall), 2),
        speedup_vs_full=round(full_wall / routed_wall, 2) if routed_wall > 0 else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end indexing latency.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--budget-per-hour", type=float, default=1200.0)
    ap.add_argument("--weights", default="default")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rep = run(a.video, a.max_frames, a.stride, a.budget_per_hour, a.weights, a.device)
    print(rep.table())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
        print(f"[latency] wrote {a.out}")


if __name__ == "__main__":
    main()
