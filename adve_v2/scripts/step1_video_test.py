import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def main():
    device = "cpu"
    print(f"Loading ADVE Enterprise Pipeline on device={device}...")

    p = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device=device,
        use_ego_motion=True,
        use_ema=True
    )

    video_file = '../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4'
    res = p.process_video(
        video_file,
        max_frames=300,
        no_validation=False
    )

    recs = p.validator.records
    sims = [r['cosine_sim'] for r in recs] if recs else [0.0]

    mean_sim = float(np.mean(sims))
    min_sim  = float(np.min(sims))
    savings  = float(res.get("encoder_savings_pct", 0.0))
    fps      = float(res.get("effective_fps", 0.0))

    result = {
        "mean_cos_sim": mean_sim,
        "min_cos_sim": min_sim,
        "encoder_savings_pct": savings,
        "fps": fps
    }

    print("\n=========================================================")
    print("           === YOUR VIDEO TEST RESULT ===")
    print("=========================================================")
    print(f"Mean CosSim: {result['mean_cos_sim']:.4f}")
    print(f"Min CosSim : {result['min_cos_sim']:.4f}")
    print(f"Savings    : {result['encoder_savings_pct']:.1f}%")
    print(f"Speed      : {result['fps']:.1f} FPS")
    print("---------------------------------------------------------")

    if result['min_cos_sim'] >= 0.85 and result['mean_cos_sim'] >= 0.97:
        print("[PASSED] - Video is compatible & Enterprise Grade!")
    else:
        print("[FAILED] - Needs tuning")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
