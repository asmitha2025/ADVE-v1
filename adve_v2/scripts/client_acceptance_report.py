import os
import sys
import time
import json
import argparse
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.config import Config
from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def generate_client_report(video_path: str, client_name: str, output_path: str, max_frames: int = 1000, device: str = "cuda"):
    print("=========================================================")
    print(f"   ADVE Commercial Client Acceptance Testing")
    print(f"   Client: {client_name}")
    print(f"   Video : {video_path}")
    print("=========================================================")

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    config = Config()
    config.DEVICE = device
    config.YOLO_DEVICE = device

    pipeline = ADVEEnterprisePipeline(
        config=config,
        reconstructor_path="training/checkpoints/reconstructor_v3.pt",
        device=device,
        use_ego_motion=True,
        use_ema=True,
        ema_alpha=0.75
    )

    t0 = time.time()
    res = pipeline.process_video(video_path, no_validation=False, max_frames=max_frames)
    elapsed = time.time() - t0

    records = pipeline.validator.records
    sims = [r["cosine_sim"] for r in records] if records else [0.0]

    mean_sim = float(np.mean(sims))
    min_sim  = float(np.min(sims))
    p5_sim   = float(np.percentile(sims, 5))
    savings  = res.get("encoder_savings_pct", 0.0)
    fps      = res.get("effective_fps", 0.0)

    # Client acceptance criteria
    passed_mean = mean_sim >= 0.9700
    passed_min  = min_sim >= 0.8500
    passed_sav  = savings >= 60.0

    verdict = "[PASSED] CLIENT ACCEPTANCE PASSED" if (passed_mean and passed_min and passed_sav) else "[WARN] NEEDS DOMAIN CUSTOMIZATION"

    report_data = {
        "client_name": client_name,
        "video_path": video_path,
        "verdict": verdict,
        "metrics": {
            "mean_cosine_sim": round(mean_sim, 4),
            "min_cosine_sim": round(min_sim, 4),
            "p5_cosine_sim": round(p5_sim, 4),
            "encoder_savings_pct": round(savings, 2),
            "effective_fps": round(fps, 1),
            "processing_time_sec": round(elapsed, 2)
        },
        "acceptance_checks": {
            "mean_sim_gte_0_97": passed_mean,
            "min_sim_gte_0_85": passed_min,
            "savings_gte_60pct": passed_sav
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n=========================================================")
    print(f"   CLIENT VERDICT: {verdict}")
    print("=========================================================")
    print(f"   Mean Similarity : {mean_sim:.4f}  (Threshold: >= 0.9700)")
    print(f"   Min Similarity  : {min_sim:.4f}  (Threshold: >= 0.8500)")
    print(f"   Encoder Savings : {savings:.1f}%   (Threshold: >= 60.0%)")
    print(f"   Effective Speed : {fps:.1f} FPS")
    print(f"   Report Saved To : {output_path}")
    print("=========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to client video")
    parser.add_argument("--client", default="Client Evaluation", help="Client / Organization Name")
    parser.add_argument("--output", default="results/client_report.json", help="Output JSON path")
    parser.add_argument("--max_frames", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate_client_report(args.video, args.client, args.output, args.max_frames, args.device)
