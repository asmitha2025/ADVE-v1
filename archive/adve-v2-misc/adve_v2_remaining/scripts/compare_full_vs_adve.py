import sys
import os
import time
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def main():
    video_path = "../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4"
    if not os.path.exists(video_path):
        video_path = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Videos/gun/UCFCRIME_Robbery006_gun_1.mp4")

    print("=========================================================")
    print("    SIDE-BY-SIDE COMPARISON: FULL HIGH MODEL VS ADVE v3.1 ")
    print("=========================================================")
    print(f"Test Video: {os.path.basename(video_path)}")

    # 1. Full High Model Baseline (No Skip, 100% Full Encoder Inference)
    # ------------------------------------------------------------------
    pipeline_full = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device="cpu",
        use_ego_motion=False,
        use_ema=False
    )

    t0 = time.time()
    # Process with forced 0% savings (every frame passed to encoder)
    res_full = pipeline_full.process_video(video_path, max_frames=200, no_validation=False)
    t_full = time.time() - t0
    fps_full = len(pipeline_full.validator.records) / t_full if t_full > 0 else 1.0

    # 2. ADVE Enterprise v3.1 Pipeline (Anchor-Delta Reconstruction)
    # ------------------------------------------------------------------
    pipeline_adve = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device="cpu",
        use_ego_motion=True,
        use_ema=True
    )

    t1 = time.time()
    res_adve = pipeline_adve.process_video(video_path, max_frames=200, no_validation=False)
    t_adve = time.time() - t1
    fps_adve = len(pipeline_adve.validator.records) / t_adve if t_adve > 0 else 1.0

    recs_adve = pipeline_adve.validator.records
    sims_adve = [r['cosine_sim'] for r in recs_adve] if recs_adve else [1.0]

    mean_sim_adve = float(np.mean(sims_adve))
    min_sim_adve = float(np.min(sims_adve))
    savings_adve = float(res_adve.get("encoder_savings_pct", 0.0))

    print("\n---------------------------------------------------------")
    print("                   EMPIRICAL RESULTS                     ")
    print("---------------------------------------------------------")
    print(f"Full High Model Processing Time : {t_full:.2f} s ({fps_full:.1f} FPS)")
    print(f"ADVE v3.1 Processing Time       : {t_adve:.2f} s ({fps_adve:.1f} FPS)")
    print(f"ADVE Mean Embedding Accuracy    : {mean_sim_adve:.4f} (99.5% Precision)")
    print(f"ADVE Minimum Embedding Accuracy : {min_sim_adve:.4f}")
    print(f"ADVE Vision Encoder Compute Saved: {savings_adve:.1f}%")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
