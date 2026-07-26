import os
import sys
import time
import json
import torch
import argparse
import numpy as np

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline
from adve.core.clip_loader import load_clip_model


def run_benchmark(video_path, reconstructor_path, output_json="benchmark_results.json", device="cuda"):
    print(f"=========================================================")
    print(f"   ADVE End-to-End Performance & Accuracy Benchmark")
    print(f"   Video: {os.path.basename(video_path)}")
    print(f"   Device: {device.upper()}")
    print(f"=========================================================")

    config = Config()
    config.DEVICE = device
    config.YOLO_DEVICE = device
    if reconstructor_path:
        config.MLP_MODEL_PATH = reconstructor_path

    # Check VRAM before load
    initial_vram = torch.cuda.memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    print("[Benchmark] Loading CLIP and ADVE Pipeline...")
    clip_model, clip_prep = load_clip_model(device=device)
    pipeline = ADVEPipeline(config, clip_model=clip_model, clip_preprocess=clip_prep)

    loaded_vram = torch.cuda.memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    print(f"[Benchmark] Loaded Pipeline. VRAM footprint: {loaded_vram:.2f} MB")

    if not os.path.exists(video_path):
        print(f"Error: Video path '{video_path}' not found. Generating synthetic test video for benchmark...")
        from generate_test_video import create_synthetic_video
        video_path = create_synthetic_video(output_path="benchmark_test.mp4", duration_sec=10, fps=30)

    # Run Benchmark
    t0 = time.time()
    results = pipeline.process_video(video_path, no_validation=False)
    total_time = time.time() - t0

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    summary = {
        "video": os.path.basename(video_path),
        "total_frames": results.get("total_frames", 0),
        "fps": results.get("effective_fps", 0.0),
        "encoder_savings_pct": results.get("encoder_savings_pct", 0.0),
        "mean_cosine_sim": results.get("mean_delta_cosine_sim", 0.0),
        "min_cosine_sim": results.get("min_delta_cosine_sim", 0.0),
        "peak_vram_mb": round(peak_vram, 2),
        "elapsed_sec": round(total_time, 2),
        "targets_met": {
            "fps_target_80": results.get("effective_fps", 0.0) >= 80.0,
            "savings_target_75": results.get("encoder_savings_pct", 0.0) >= 75.0,
            "cossim_target_99": results.get("mean_delta_cosine_sim", 0.0) >= 0.990,
            "vram_target_350": peak_vram <= 350.0 if device == "cuda" else True,
        }
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=========================================================")
    print("                BENCHMARK RESULTS")
    print("=========================================================")
    print(f"   Effective FPS        : {summary['fps']}  (Target: >= 80.0)")
    print(f"   Encoder Savings      : {summary['encoder_savings_pct']}%  (Target: >= 75.0%)")
    print(f"   Mean Cosine Sim      : {summary['mean_cosine_sim']}  (Target: >= 0.9900)")
    print(f"   Min Cosine Sim       : {summary['min_cosine_sim']}  (Target: >= 0.9400)")
    print(f"   Peak GPU VRAM        : {summary['peak_vram_mb']} MB  (Target: <= 350 MB)")
    print(f"   Results saved to     : {output_json}")
    print("=========================================================\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="test_video.mp4")
    parser.add_argument("--reconstructor", default="training/checkpoints/best_model.pt")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_benchmark(args.video, args.reconstructor, args.output, device=args.device)
