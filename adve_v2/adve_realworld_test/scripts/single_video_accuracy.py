import os
import sys
import time
import json
import csv
import torch
import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline
from adve.core.clip_loader import load_clip_model


def analyze_single_video(
    video_path: str,
    reconstructor_path: str,
    output_dir: str,
    device: str = "cuda",
    max_frames: int = 1000
):
    print("=========================================================")
    print("   ADVE Single Video Per-Frame Diagnostic Tool")
    print(f"   Video: {video_path}")
    print(f"   Reconstructor: {reconstructor_path}")
    print(f"   Output Dir: {output_dir}")
    print("=========================================================")

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    config = Config()
    config.DEVICE = device
    config.YOLO_DEVICE = device
    if reconstructor_path and os.path.exists(reconstructor_path):
        config.MLP_MODEL_PATH = reconstructor_path

    clip_model, clip_prep = load_clip_model("ViT-B/32", device=device)
    pipeline = ADVEPipeline(config, clip_model=clip_model, clip_preprocess=clip_prep)

    t0 = time.time()
    res = pipeline.process_video(video_path, no_validation=False, max_frames=max_frames)
    elapsed = time.time() - t0

    records = pipeline.validator.records
    if not records:
        print("Error: No frame records collected.")
        sys.exit(1)

    frames          = [r["frame"] for r in records]
    sims            = [r["cosine_sim"] for r in records]
    delta_sims      = [r["cosine_sim"] for r in records if not r["is_anchor"]]
    delta_mags      = [r["delta_magnitude"] for r in records]
    is_anchors      = [r["is_anchor"] for r in records]

    # Calculate distance to last anchor frame
    anchor_dists = []
    dist = 0
    for r in records:
        if r["is_anchor"]:
            dist = 0
        else:
            dist += 1
        anchor_dists.append(dist)

    # Compute percentiles
    mean_cos = float(np.mean(sims))
    min_cos  = float(np.min(sims))
    p5_cos   = float(np.percentile(sims, 5))
    p50_cos  = float(np.median(sims))

    delta_mean_cos = float(np.mean(delta_sims)) if delta_sims else 1.0
    delta_min_cos  = float(np.min(delta_sims)) if delta_sims else 1.0

    # Categorize failure modes
    low_sim_count = sum(1 for s in sims if s < 0.85)
    catastrophic_failures = sum(1 for s in sims if s < 0.70)

    summary = {
        "video_path": os.path.abspath(video_path),
        "total_frames": len(records),
        "mean_cos_sim": round(mean_cos, 4),
        "min_cos_sim": round(min_cos, 4),
        "p5_cos_sim": round(p5_cos, 4),
        "p50_cos_sim": round(p50_cos, 4),
        "delta_mean_cos_sim": round(delta_mean_cos, 4),
        "delta_min_cos_sim": round(delta_min_cos, 4),
        "low_sim_frames_count": low_sim_count,
        "catastrophic_failures_count": catastrophic_failures,
        "effective_fps": round(len(records) / elapsed, 1),
        "elapsed_sec": round(elapsed, 2)
    }

    # Save summary JSON
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save per-frame CSV
    csv_path = os.path.join(output_dir, "frame_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cosine_sim", "is_anchor", "delta_magnitude", "anchor_dist"])
        for r, d in zip(records, anchor_dists):
            writer.writerow([r["frame"], r["cosine_sim"], r["is_anchor"], r["delta_magnitude"], d])

    print("\n--- Generating Diagnostic Plots ---")

    # Plot 1: CosSim Timeline
    plt.figure(figsize=(10, 4))
    plt.plot(frames, sims, color="#1E88E5", linewidth=1.2, label="CosSim")
    plt.axhline(0.94, color="orange", linestyle="--", label="Target (0.94)")
    plt.axhline(0.85, color="red", linestyle=":", label="Min Cap (0.85)")
    plt.title("CosSim Timeline", fontsize=12)
    plt.xlabel("Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.5, 1.02)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cos_sim_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Accuracy vs Motion
    plt.figure(figsize=(8, 5))
    colors = ["#E53935" if s < 0.85 else "#43A047" for s in sims]
    plt.scatter(delta_mags, sims, c=colors, alpha=0.6, edgecolors="none", s=25)
    plt.axhline(0.94, color="orange", linestyle="--", label="Target (0.94)")
    plt.title("Accuracy vs Motion Magnitude (dG)", fontsize=12)
    plt.xlabel("Spatial Delta Magnitude (dG)")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.5, 1.02)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_motion.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: CosSim Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(sims, bins=30, color="#3F51B5", edgecolor="white", alpha=0.8)
    plt.axvline(mean_cos, color="red", linestyle="-", label=f"Mean ({mean_cos:.4f})")
    plt.axvline(p5_cos, color="orange", linestyle="--", label=f"5th Pct ({p5_cos:.4f})")
    plt.title("Cosine Similarity Distribution", fontsize=12)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frame Count")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "cos_sim_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: Accuracy vs Distance from Anchor
    plt.figure(figsize=(8, 4))
    plt.scatter(anchor_dists, sims, color="#009688", alpha=0.5, s=20)
    plt.axhline(0.94, color="orange", linestyle="--", label="Target (0.94)")
    plt.title("Accuracy vs Distance to Last Anchor Frame", fontsize=12)
    plt.xlabel("Frames Since Last Anchor")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.5, 1.02)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_anchor_distance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved analysis and 4 diagnostic plots -> '{output_dir}'")
    print(f"  Mean CosSim : {mean_cos:.4f}")
    print(f"  Min CosSim  : {min_cos:.4f}")
    print(f"  5th Pct     : {p5_cos:.4f}")
    print(f"  FPS         : {summary['effective_fps']} FPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--reconstructor", default="training/checkpoints/reconstructor_v2.pt")
    parser.add_argument("--output", default="results/single_video_analysis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=1000)
    args = parser.parse_args()

    analyze_single_video(
        args.video,
        args.reconstructor,
        args.output,
        device=args.device,
        max_frames=args.max_frames
    )
