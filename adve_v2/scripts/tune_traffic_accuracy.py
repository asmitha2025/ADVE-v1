import sys
import os
import time
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline
from adve.core.config import Config

def evaluate_config(spatial_thresh, max_delta):
    traffic_video = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Vodra/North.mp4")
    
    cfg = Config()
    cfg.SPATIAL_THRESHOLD = spatial_thresh
    cfg.MAX_DELTA_FRAMES = max_delta

    pipeline = ADVEEnterprisePipeline(
        config=cfg,
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device="cpu",
        use_ego_motion=True,
        use_ema=True
    )
    pipeline.reset_state()

    res = pipeline.process_video(traffic_video, max_frames=300, no_validation=False)
    recs = pipeline.validator.records
    sims = [r['cosine_sim'] for r in recs] if recs else [1.0]

    mean_sim = float(np.mean(sims))
    min_sim = float(np.min(sims))
    savings = float(res.get("encoder_savings_pct", 0.0))

    return {
        "spatial_thresh": spatial_thresh,
        "max_delta": max_delta,
        "mean_cos_sim": round(mean_sim, 4),
        "min_cos_sim": round(min_sim, 4),
        "savings_pct": round(savings, 1)
    }

def main():
    print("=========================================================")
    print("      ADVE TRAFFIC ACCURACY vs SAVINGS TUNING AUDIT      ")
    print("=========================================================")

    # Baseline Default Config (Balanced Mode: 0.35 threshold)
    res_default = evaluate_config(spatial_thresh=0.35, max_delta=15)
    
    # Ultra Precision Mode Config (High Accuracy Mode: 0.25 threshold)
    res_precision = evaluate_config(spatial_thresh=0.25, max_delta=10)

    # Maximum Savings Mode Config (Aggressive Mode: 0.45 threshold)
    res_savings = evaluate_config(spatial_thresh=0.45, max_delta=20)

    print("\n---------------------------------------------------------")
    print("                  CONFIGURATION PRESETS                  ")
    print("---------------------------------------------------------")
    print(f"1. ULTRA PRECISION MODE (Thresh 0.25): Mean Sim = {res_precision['mean_cos_sim']:.4f} ({res_precision['mean_cos_sim']*100:.2f}%) | Min = {res_precision['min_cos_sim']:.4f} | Savings = {res_precision['savings_pct']:.1f}%")
    print(f"2. BALANCED MODE        (Thresh 0.35): Mean Sim = {res_default['mean_cos_sim']:.4f} ({res_default['mean_cos_sim']*100:.2f}%) | Min = {res_default['min_cos_sim']:.4f} | Savings = {res_default['savings_pct']:.1f}%")
    print(f"3. HIGH SAVINGS MODE    (Thresh 0.45): Mean Sim = {res_savings['mean_cos_sim']:.4f} ({res_savings['mean_cos_sim']*100:.2f}%) | Min = {res_savings['min_cos_sim']:.4f} | Savings = {res_savings['savings_pct']:.1f}%")
    print("=========================================================\n")

    summary = {
        "precision_mode": res_precision,
        "balanced_mode": res_default,
        "savings_mode": res_savings
    }
    with open("results/traffic_tuning_modes.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
