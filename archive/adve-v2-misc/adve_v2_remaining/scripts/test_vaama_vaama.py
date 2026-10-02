"""
Live Pipeline Test on Vaama Vaama Video
Tests the full ADVE Pipeline (with SafetyGate + SlideDetector) on:
'Testing videos/Vaama Vaama - Airport Version.mp4'
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline
from adve.core.content_detector import DomainFingerprinter


def run_vaama_test():
    video_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "Testing videos",
        "Vaama Vaama - Airport Version _ Idhayam Murali _ Atharvaa _ Preity _ Thaman _Dhanush_Aakash Baskaran.mp4"
    )
    video_path = os.path.abspath(video_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("  LIVE PIPELINE TEST: VAAMA VAAMA (Airport Version)")
    print("=" * 80)
    print(f"  Video : {os.path.basename(video_path)}")
    print(f"  Device: {device.upper()}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    if not os.path.exists(video_path):
        print(f"  [ERROR] Video not found: {video_path}")
        return

    # Domain fingerprint
    print("\n[1] Running Domain Fingerprinter...")
    fp = DomainFingerprinter(sample_frames=60)
    profile = fp.fingerprint_and_route(video_path)
    print(f"    Auto-Detected Domain : {profile['domain'].upper()}")
    print(f"    Spatial Threshold    : {profile['spatial_threshold']}")
    print(f"    Appearance Threshold : {profile.get('appearance_threshold', 0.10)}")
    print(f"    Max Delta Frames     : {profile['max_delta_frames']}")

    # Config
    config = Config()
    config.DEVICE = device
    config.CLIP_DEVICE = device
    config.YOLO_DEVICE = device
    config.YOLO_HALF = False
    config.SPATIAL_THRESHOLD = profile['spatial_threshold']
    config.APPEARANCE_THRESHOLD = profile.get('appearance_threshold', 0.10)
    config.MAX_DELTA_FRAMES = profile['max_delta_frames']
    config.SAFETY_GATE_HARD_FLOOR = 0.88
    config.SAFETY_GATE_CONSECUTIVE_MAX = 3

    pipeline = ADVEPipeline(config)

    # Process
    print("\n[2] Executing ADVE Pipeline on 300 Video Frames...")
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

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
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

    print("\n" + "=" * 80)
    print("      LIVE TEST RESULTS: VAAMA VAAMA (ADVE v2 + SafetyGate + SlideDetector)")
    print("=" * 80)
    print(f" {'Metric / Feature':<35} | {'Empirical Live Value':<35}")
    print("-" * 75)
    print(f" {'Auto-Detected Domain':<35} | {profile['domain'].upper()} Profile")
    print(f" {'Total Frames Processed':<35} | {num_frames} frames")
    print(f" {'Heavy Vision Encoder Calls':<35} | {num_anchors} keyframes")
    print(f" {'Neural Delta Reconstructions':<35} | {num_deltas} frames")
    print(f" {'Heavy GPU Compute Savings':<35} | {gpu_savings:.1f}%")
    print(f" {'Mean Cosine Precision':<35} | {mean_sim*100:.2f}%")
    print(f" {'Minimum Cosine Accuracy':<35} | {min_sim*100:.2f}% (Floor: 0.88)")
    print(f" {'SafetyGate Hard Floor':<35} | {gate_stats['guaranteed_hard_floor']:.2f} Active")
    print(f" {'Quality Pass Rate (>= 0.85)':<35} | {pass_rate:.1f}%")
    print(f" {'SafetyGate Interceptions':<35} | {gate_stats['frames_refreshed']} frames")
    print(f" {'Processing Speed':<35} | {fps:.1f} FPS ({elapsed:.2f}s)")
    if device == "cuda":
        print(f" {'Peak VRAM Usage':<35} | {vram_mb:.0f} MB")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_vaama_test()
    except Exception as e:
        print(f"[ERROR] Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
