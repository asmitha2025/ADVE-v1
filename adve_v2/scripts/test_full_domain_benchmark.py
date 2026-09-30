"""
Full Multi-Domain GPU Benchmark — ADVE v2 + SafetyGate
Tests across ALL available video domains:
  1. TRAFFIC (Talaimari junction CCTV)
  2. EDUCATION (Lecture / slides video)
  3. ACTION (UCF Crime walk / run / sneak)
  4. NIGHT / SPARSE (UCF Crime lying_down / stand)
"""

import os
import sys
import time
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline
from adve.core.content_detector import DomainFingerprinter

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# One representative video per domain category
DOMAIN_VIDEOS = [
    {
        "label": "TRAFFIC (Talaimari Junction CCTV)",
        "path": os.path.join(BASE, "Testing videos", "Traffic", "Talaimari", "east.mp4"),
        "expected_domain": "traffic",
    },
    {
        "label": "TRAFFIC (Vodra Junction CCTV)",
        "path": os.path.join(BASE, "Testing videos", "Traffic", "Vodra", "North.mp4"),
        "expected_domain": "traffic",
    },
    {
        "label": "EDUCATION (Phase Change Lecture)",
        "path": os.path.join(BASE, "Testing videos", "education videos",
                             "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"),
        "expected_domain": "education",
    },
    {
        "label": "ACTION / WALK (UCF Crime Walk)",
        "path": os.path.join(BASE, "Testing videos", "Videos", "walk", "UCFCRIME_Abuse007_walk_1.mp4"),
        "expected_domain": "action",
    },
    {
        "label": "ACTION / RUN (NTU Fight Run)",
        "path": os.path.join(BASE, "Testing videos", "Videos", "run", "NTU_fight0037_run_1.mp4"),
        "expected_domain": "action",
    },
    {
        "label": "SPARSE / SNEAK (UCF Burglary Sneak)",
        "path": os.path.join(BASE, "Testing videos", "Videos", "sneak", "UCFCRIME_Burglary035_sneak_1.mp4"),
        "expected_domain": "sparse",
    },
]


def run_domain_test(video_info, device="cuda"):
    label = video_info["label"]
    video_path = video_info["path"]

    print(f"\n{'='*80}")
    print(f"  DOMAIN TEST: {label}")
    print(f"{'='*80}")
    print(f"  Video: {os.path.basename(video_path)}")

    if not os.path.exists(video_path):
        print(f"  [SKIP] Video not found: {video_path}")
        return None

    # Domain fingerprint
    fp = DomainFingerprinter(sample_frames=60)
    profile = fp.fingerprint_and_route(video_path)
    detected = profile["domain"].upper()
    print(f"  Auto-Detected Domain: {detected}")

    # Config
    config = Config()
    config.DEVICE = device
    config.CLIP_DEVICE = device
    config.YOLO_DEVICE = device
    config.YOLO_HALF = False
    config.SPATIAL_THRESHOLD = profile["spatial_threshold"]
    config.APPEARANCE_THRESHOLD = profile.get("appearance_threshold", 0.10)
    config.MAX_DELTA_FRAMES = profile["max_delta_frames"]
    config.SAFETY_GATE_HARD_FLOOR = 0.88
    config.SAFETY_GATE_CONSECUTIVE_MAX = 3

    pipeline = ADVEPipeline(config)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    results = pipeline.process_video(video_path, max_frames=300)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    def _val(d, keys, default=0):
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, str):
                    v = v.replace("%", "")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    num_frames = int(_val(results, ["total_frames"], 300))
    num_anchors = int(_val(results, ["encoder_calls"]))
    num_deltas = num_frames - num_anchors
    mean_sim = _val(results, ["mean_cosine_sim", "mean_delta_cosine_sim"])
    min_sim = _val(results, ["min_delta_cosine_sim", "min_cosine_sim"])
    pass_rate = _val(results, ["pct_above_threshold"])
    gpu_savings = _val(results, ["encoder_savings_pct"])
    if gpu_savings == 0 and num_frames > 0:
        gpu_savings = ((num_frames - num_anchors) / num_frames) * 100.0

    gate_stats = pipeline.safety_gate.get_stats()
    fps = num_frames / max(elapsed, 0.001)

    vram_mb = 0
    if device == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Cleanup
    del pipeline
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    row = {
        "label": label,
        "domain_detected": detected,
        "frames": num_frames,
        "anchors": num_anchors,
        "deltas": num_deltas,
        "gpu_savings": round(gpu_savings, 1),
        "mean_sim": round(mean_sim * 100, 2),
        "min_sim": round(min_sim * 100, 2),
        "pass_rate": round(pass_rate, 1),
        "gate_interceptions": gate_stats["frames_refreshed"],
        "fps": round(fps, 1),
        "elapsed": round(elapsed, 2),
        "vram_mb": round(vram_mb),
    }

    print(f"  Result: Savings={row['gpu_savings']}% | Mean={row['mean_sim']}% | Min={row['min_sim']}% | "
          f"Pass={row['pass_rate']}% | Gate={row['gate_interceptions']} | {row['fps']} FPS")
    return row


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 90)
    print("  ADVE v2 FULL MULTI-DOMAIN BENCHMARK (SafetyGate + UALW)")
    print("=" * 90)
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Device      : {device.upper()}")
    print(f"  SafetyGate  : Hard Floor = 0.88")

    all_results = []
    for vinfo in DOMAIN_VIDEOS:
        r = run_domain_test(vinfo, device=device)
        if r:
            all_results.append(r)

    # Final master comparison table
    print("\n\n" + "=" * 120)
    print("  MASTER MULTI-DOMAIN BENCHMARK RESULTS (ADVE v2 + SafetyGate)")
    print("=" * 120)
    header = f"  {'Domain / Video':<38} | {'Detected':<10} | {'Savings':<8} | {'Mean%':<8} | {'Min%':<8} | {'Pass%':<7} | {'Gate':<5} | {'FPS':<6} | {'Time':<7} | {'VRAM':<6}"
    print(header)
    print("  " + "-" * 116)
    for r in all_results:
        line = (f"  {r['label']:<38} | {r['domain_detected']:<10} | "
                f"{r['gpu_savings']:>6.1f}% | {r['mean_sim']:>6.2f}% | {r['min_sim']:>6.2f}% | "
                f"{r['pass_rate']:>5.1f}% | {r['gate_interceptions']:>4} | "
                f"{r['fps']:>5.1f} | {r['elapsed']:>5.2f}s | {r['vram_mb']:>4} MB")
        print(line)
    print("=" * 120)

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "full_domain_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
