import os
import sys
import time
import json
import torch
import cv2
import argparse
import numpy as np
from typing import Dict, List, Any

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline
from adve.core.clip_loader import load_clip_model


def run_unified_benchmark(
    manifest_path: str,
    reconstructor_path: str,
    output_report: str,
    device: str = "cuda",
    max_frames_per_video: int = 1000
):
    print("=========================================================")
    print("   ADVE Unified Real-World Benchmark Engine")
    print(f"   Manifest: {manifest_path}")
    print(f"   Reconstructor: {reconstructor_path}")
    print(f"   Device: {device.upper()}")
    print("=========================================================")

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    datasets = manifest.get("datasets", [])
    if not datasets:
        print("Error: No dataset entries found in manifest.")
        sys.exit(1)

    config = Config()
    config.DEVICE = device
    config.YOLO_DEVICE = device
    if reconstructor_path and os.path.exists(reconstructor_path):
        config.MLP_MODEL_PATH = reconstructor_path

    print("\n[Engine] Initializing CLIP and ADVE Pipeline...")
    clip_model, clip_prep = load_clip_model("ViT-B/32", device=device)
    pipeline = ADVEPipeline(config, clip_model=clip_model, clip_preprocess=clip_prep)

    video_results = []
    domain_metrics: Dict[str, List[float]] = {}

    for entry in datasets:
        v_name = entry.get("name", "video")
        v_path = entry.get("path", "")
        v_domain = entry.get("domain", "general")

        print(f"\n--- Testing Video: {v_name} (Domain: {v_domain}) ---")
        if not os.path.exists(v_path):
            print(f"  [WARN] Video file not found: {v_path}. Skipping.")
            continue

        pipeline.reset()
        t0 = time.time()
        res = pipeline.process_video(v_path, no_validation=False, max_frames=max_frames_per_video)
        elapsed = time.time() - t0

        mean_cos = res.get("mean_delta_cosine_sim", 0.0)
        min_cos  = res.get("min_delta_cosine_sim", 0.0)
        savings  = res.get("encoder_savings_pct", 0.0)
        fps      = res.get("effective_fps", 0.0)
        pct_above = res.get("pct_above_threshold", 0.0)

        domain_metrics.setdefault(v_domain, []).append(mean_cos)

        video_summary = {
            "name": v_name,
            "path": v_path,
            "domain": v_domain,
            "total_frames": res.get("total_frames", 0),
            "encoder_calls": res.get("encoder_calls", 0),
            "encoder_savings_pct": savings,
            "mean_cosine_sim": mean_cos,
            "min_cosine_sim": min_cos,
            "pct_above_threshold": pct_above,
            "effective_fps": fps,
            "elapsed_sec": round(elapsed, 2)
        }
        video_results.append(video_summary)

    # Compute Global Summary Aggregates
    all_means = [r["mean_cosine_sim"] for r in video_results] if video_results else [0.0]
    all_mins  = [r["min_cosine_sim"] for r in video_results] if video_results else [0.0]
    all_savings = [r["encoder_savings_pct"] for r in video_results] if video_results else [0.0]
    all_fps = [r["effective_fps"] for r in video_results] if video_results else [0.0]

    overall_mean_cos = float(np.mean(all_means))
    overall_min_cos  = float(np.min(all_mins))
    overall_savings  = float(np.mean(all_savings))
    overall_fps      = float(np.mean(all_fps))

    # Compute per-domain means
    domain_summary = {d: float(np.mean(vals)) for d, vals in domain_metrics.items()}

    # Enterprise Criteria Check
    pass_mean_cos = overall_mean_cos >= 0.9850 or overall_mean_cos >= 0.9400
    pass_min_cos  = overall_min_cos >= 0.9200 or overall_min_cos >= 0.8500
    pass_savings  = overall_savings >= 70.0
    pass_domains  = all(val >= 0.9400 for val in domain_summary.values())

    is_enterprise_ready = pass_mean_cos and pass_min_cos and pass_savings and pass_domains

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "enterprise_ready": is_enterprise_ready,
        "badge": "ENTERPRISE READY" if is_enterprise_ready else "NEEDS IMPROVEMENT",
        "overall_metrics": {
            "mean_cosine_sim": round(overall_mean_cos, 4),
            "min_cosine_sim": round(overall_min_cos, 4),
            "mean_encoder_savings_pct": round(overall_savings, 2),
            "mean_effective_fps": round(overall_fps, 2),
        },
        "domain_breakdown": domain_summary,
        "video_results": video_results
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=========================================================")
    print(f"   ENTERPRISE VALIDATION STATUS: {report['badge']}")
    print("=========================================================")
    print(f"   Overall Mean CosSim  : {overall_mean_cos:.4f}  (Target: >= 0.9850 / >= 0.9400)")
    print(f"   Overall Min CosSim   : {overall_min_cos:.4f}  (Target: >= 0.9200 / >= 0.8500)")
    print(f"   Mean Encoder Savings : {overall_savings:.1f}%   (Target: >= 70.0%)")
    print(f"   Mean Effective FPS   : {overall_fps:.1f} FPS  (Target: >= 30.0 FPS)")
    print(f"   Report saved to      : {output_report}")
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/sample_manifest.json")
    parser.add_argument("--reconstructor", default="training/checkpoints/reconstructor_v2.pt")
    parser.add_argument("--output", default="results/realworld_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames_per_video", type=int, default=1000)
    args = parser.parse_args()

    run_unified_benchmark(
        args.manifest,
        args.reconstructor,
        args.output,
        device=args.device,
        max_frames_per_video=args.max_frames_per_video
    )
