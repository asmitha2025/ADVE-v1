"""
GPU-Accelerated Live Pipeline Test on Traffic + Education Videos
Runs ADVE Pipeline with SafetyGate on CUDA GPU (RTX 4050)
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline
from adve.core.content_detector import DomainFingerprinter


def run_gpu_test(video_path, label):
    print("\n" + "=" * 80)
    print(f"  GPU TEST: {label}")
    print("=" * 80)
    print(f"  Video : {os.path.basename(video_path)}")
    print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if not os.path.exists(video_path):
        print(f"  [ERROR] Video not found: {video_path}")
        return None

    # Domain fingerprint
    fp = DomainFingerprinter(sample_frames=60)
    profile = fp.fingerprint_and_route(video_path)
    print(f"  Domain: {profile['domain'].upper()}")

    # Config with GPU
    config = Config()
    config.DEVICE = "cuda"
    config.CLIP_DEVICE = "cuda"
    config.YOLO_DEVICE = "cuda"
    config.YOLO_HALF = False  # FP16 disabled — prevents CUBLAS abort on this driver
    config.SPATIAL_THRESHOLD = profile['spatial_threshold']
    config.APPEARANCE_THRESHOLD = profile.get('appearance_threshold', 0.10)
    config.MAX_DELTA_FRAMES = profile['max_delta_frames']
    config.SAFETY_GATE_HARD_FLOOR = 0.88
    config.SAFETY_GATE_CONSECUTIVE_MAX = 3

    pipeline = ADVEPipeline(config)

    # Warm up GPU
    torch.cuda.synchronize()

    t0 = time.time()
    results = pipeline.process_video(video_path, max_frames=300)
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
    fps = num_frames / elapsed

    # Free GPU memory
    del pipeline
    torch.cuda.empty_cache()

    print(f"\n  {'Metric':<35} | {'Value':<30}")
    print(f"  {'-'*35}-+-{'-'*30}")
    print(f"  {'Domain Profile':<35} | {profile['domain'].upper()}")
    print(f"  {'Total Frames':<35} | {num_frames}")
    print(f"  {'Heavy Encoder Calls':<35} | {num_anchors} keyframes")
    print(f"  {'Delta Reconstructions':<35} | {num_deltas} frames")
    print(f"  {'GPU Compute Savings':<35} | {gpu_savings:.1f}%")
    print(f"  {'Mean Cosine Precision':<35} | {mean_sim*100:.2f}%")
    print(f"  {'Min Cosine Accuracy':<35} | {min_sim*100:.2f}% (Floor: 0.88)")
    print(f"  {'Pass Rate (>= 0.85)':<35} | {pass_rate:.1f}%")
    print(f"  {'SafetyGate Interceptions':<35} | {gate_stats['frames_refreshed']} frames")
    print(f"  {'Processing Speed (GPU)':<35} | {fps:.1f} FPS")
    print(f"  {'Total Wall Time':<35} | {elapsed:.2f}s")
    print(f"  {'GPU Memory Peak':<35} | {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print("  " + "=" * 68)

    return {
        "label": label,
        "domain": profile['domain'].upper(),
        "frames": num_frames,
        "anchors": num_anchors,
        "deltas": num_deltas,
        "gpu_savings": gpu_savings,
        "mean_sim": mean_sim,
        "min_sim": min_sim,
        "pass_rate": pass_rate,
        "gate_interceptions": gate_stats['frames_refreshed'],
        "fps": fps,
        "elapsed": elapsed,
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    print("======================================================================")
    print("  ADVE v2 GPU-ACCELERATED BENCHMARK (RTX 4050 Laptop GPU)")
    print("======================================================================")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA        : {torch.version.cuda}")
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Total  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    videos = [
        (os.path.join(base, "demo_videos", "p1_X78zeldo.mp4"), "TRAFFIC / CCTV Surveillance"),
        (os.path.join(base, "..", "Testing videos", "education videos",
         "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"),
         "EDUCATION / Lecture Video"),
    ]

    all_results = []
    for vpath, vlabel in videos:
        r = run_gpu_test(vpath, vlabel)
        if r:
            all_results.append(r)
        torch.cuda.reset_peak_memory_stats()

    # Final comparison table
    if len(all_results) >= 2:
        print("\n" + "=" * 90)
        print("  GPU BENCHMARK COMPARISON: TRAFFIC vs EDUCATION (RTX 4050)")
        print("=" * 90)
        print(f"  {'Metric':<35} | {'TRAFFIC':<22} | {'EDUCATION':<22}")
        print(f"  {'-'*35}-+-{'-'*22}-+-{'-'*22}")
        t, e = all_results[0], all_results[1]
        print(f"  {'Domain Profile':<35} | {t['domain']:<22} | {e['domain']:<22}")
        print(f"  {'Total Frames':<35} | {t['frames']:<22} | {e['frames']:<22}")
        print(f"  {'Heavy Encoder Calls':<35} | {t['anchors']:<22} | {e['anchors']:<22}")
        print(f"  {'GPU Compute Savings':<35} | {t['gpu_savings']:.1f}%{'':<17} | {e['gpu_savings']:.1f}%")
        print(f"  {'Mean Cosine Precision':<35} | {t['mean_sim']*100:.2f}%{'':<16} | {e['mean_sim']*100:.2f}%")
        print(f"  {'Min Cosine Accuracy':<35} | {t['min_sim']*100:.2f}%{'':<16} | {e['min_sim']*100:.2f}%")
        print(f"  {'Pass Rate (>= 0.85)':<35} | {t['pass_rate']:.1f}%{'':<17} | {e['pass_rate']:.1f}%")
        print(f"  {'SafetyGate Interceptions':<35} | {t['gate_interceptions']:<22} | {e['gate_interceptions']:<22}")
        print(f"  {'GPU Processing Speed':<35} | {t['fps']:.1f} FPS{'':<15} | {e['fps']:.1f} FPS")
        print(f"  {'Total Wall Time':<35} | {t['elapsed']:.2f}s{'':<16} | {e['elapsed']:.2f}s")
        print(f"  {'Peak VRAM Usage':<35} | {t['vram_peak_mb']:.0f} MB{'':<15} | {e['vram_peak_mb']:.0f} MB")
        print("=" * 90)


if __name__ == "__main__":
    main()
