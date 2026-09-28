import gc
import os
import sys
import time
import json
import torch
import cv2
import argparse
import numpy as np

from adve.core.config import Config
from adve.core.pipeline_enterprise import ADVEEnterprisePipeline
from adve.core.clip_loader import load_clip_model


def run_enterprise_audit(
    manifest_path: str,
    reconstructor_path: str,
    output_audit: str,
    device: str = "cuda",
    max_frames_per_video: int = 1000
):
    print("=========================================================")
    print("   ADVE Enterprise Commercial Audit Engine")
    print(f"   Manifest: {manifest_path}")
    print(f"   Reconstructor: {reconstructor_path}")
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

    pipeline = ADVEEnterprisePipeline(
        config=config,
        reconstructor_path=reconstructor_path,
        device=device,
        use_ego_motion=True,
        use_ema=True,
        ema_alpha=0.75
    )

    all_frame_sims = []
    all_savings = []
    all_fps = []
    critical_fails = 0
    per_video_metrics = []

    for entry in datasets:
        v_name = entry.get("name", "video")
        v_path = entry.get("path", "")
        v_domain = entry.get("domain", "general")

        if not os.path.exists(v_path):
            continue

        pipeline.reset_state()
        t0 = time.time()
        res = pipeline.process_video(v_path, no_validation=False, max_frames=max_frames_per_video)
        elapsed = time.time() - t0

        records = pipeline.validator.records
        sims = [r["cosine_sim"] for r in records] if records else [0.0]

        all_frame_sims.extend(sims)
        savings = res.get("encoder_savings_pct", 0.0)
        fps = res.get("effective_fps", 0.0)

        all_savings.append(savings)
        all_fps.append(fps)

        v_min = float(np.min(sims))
        if v_min < 0.85:
            critical_fails += 1

        per_video_metrics.append({
            "name": v_name,
            "domain": v_domain,
            "mean_sim": round(float(np.mean(sims)), 4),
            "min_sim": round(v_min, 4),
            "p5_sim": round(float(np.percentile(sims, 5)), 4),
            "savings": savings,
            "fps": fps
        })
        pipeline.reset_state()
        gc.collect()

    # Aggregates across all videos & frames
    overall_mean_cos = float(np.mean(all_frame_sims)) if all_frame_sims else 0.0
    overall_min_cos  = float(np.min(all_frame_sims)) if all_frame_sims else 0.0
    overall_p5_cos   = float(np.percentile(all_frame_sims, 5)) if all_frame_sims else 0.0
    overall_savings  = float(np.mean(all_savings)) if all_savings else 0.0
    overall_fps      = float(np.mean(all_fps)) if all_fps else 0.0
    temp_consistency = float(np.std(all_frame_sims)) if all_frame_sims else 0.0

    # Score calculation (0 to 100)
    # 1. Mean CosSim (30 pts)
    score_mean = min(30.0, max(0.0, (overall_mean_cos - 0.90) / (0.985 - 0.90) * 30.0))
    # 2. Min CosSim (20 pts)
    score_min = min(20.0, max(0.0, (overall_min_cos - 0.75) / (0.92 - 0.75) * 20.0))
    # 3. 5th Pct CosSim (15 pts)
    score_p5 = min(15.0, max(0.0, (overall_p5_cos - 0.85) / (0.95 - 0.85) * 15.0))
    # 4. Savings (15 pts)
    score_savings = min(15.0, max(0.0, (overall_savings - 50.0) / (75.0 - 50.0) * 15.0))
    # 5. FPS (10 pts)
    score_fps = min(10.0, max(0.0, (overall_fps - 15.0) / (80.0 - 15.0) * 10.0))
    # 6. Temporal Consistency (10 pts)
    score_temp = min(10.0, max(0.0, (0.05 - temp_consistency) / (0.05 - 0.01) * 10.0))

    total_score = round(score_mean + score_min + score_p5 + score_savings + score_fps + score_temp, 1)

    if total_score >= 85.0 and critical_fails == 0:
        verdict = "ENTERPRISE READY"
    elif total_score >= 75.0 and critical_fails == 0:
        verdict = "ACCEPTABLE"
    else:
        verdict = "NEEDS IMPROVEMENT"

    audit_result = {
        "overall_score": total_score,
        "verdict": verdict,
        "critical_fails": critical_fails,
        "metrics": {
            "mean_cosine_sim": round(overall_mean_cos, 4),
            "min_cosine_sim": round(overall_min_cos, 4),
            "p5_cosine_sim": round(overall_p5_cos, 4),
            "encoder_savings_pct": round(overall_savings, 2),
            "effective_fps": round(overall_fps, 1),
            "temporal_consistency_std": round(temp_consistency, 4)
        },
        "per_video_breakdown": per_video_metrics
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_audit)), exist_ok=True)
    with open(output_audit, "w") as f:
        json.dump(audit_result, f, indent=2)

    print("\n=========================================================")
    print(f"   AUDIT SCORE: {total_score} / 100")
    print(f"   COMMERCIAL VERDICT: {verdict}")
    print(f"   CRITICAL FAILS: {critical_fails}")
    print("=========================================================")
    print(f"   Mean CosSim    : {overall_mean_cos:.4f} (Target: >= 0.9850)")
    print(f"   Min CosSim     : {overall_min_cos:.4f} (Target: >= 0.9200)")
    print(f"   5th Pct CosSim : {overall_p5_cos:.4f} (Target: >= 0.9500)")
    print(f"   Savings %      : {overall_savings:.1f}%  (Target: >= 75.0%)")
    print(f"   Effective FPS  : {overall_fps:.1f} FPS (Target: >= 80.0 FPS)")
    print(f"   Temporal Std   : {temp_consistency:.4f} (Target: <= 0.0200)")
    print(f"   Audit saved to : {output_audit}")
    print("=========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/sample_manifest.json")
    parser.add_argument("--reconstructor", default="training/checkpoints/reconstructor_v2.pt")
    parser.add_argument("--output", default="results/enterprise_audit.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames_per_video", type=int, default=1000)
    args = parser.parse_args()

    run_enterprise_audit(
        args.manifest,
        args.reconstructor,
        args.output,
        device=args.device,
        max_frames_per_video=args.max_frames_per_video
    )
