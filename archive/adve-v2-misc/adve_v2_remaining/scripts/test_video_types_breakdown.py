import sys
import os
import glob
import time
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline
from adve.core.config import Config

def main():
    print("=========================================================")
    print("      ADVE ADAPTIVE VIDEO CONTENT AUDIT BY CATEGORY      ")
    print("=========================================================")

    traffic_vid = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Vodra/North.mp4")
    action_vid = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4")

    test_cases = [
        {"category": "High Motion Traffic", "path": traffic_vid},
        {"category": "Medium Motion CCTV Action", "path": action_vid}
    ]

    cfg = Config()
    pipeline = ADVEEnterprisePipeline(
        config=cfg,
        reconstructor_path='training/checkpoints/reconstructor_v3.1.pt',
        device="cpu",
        use_ego_motion=True,
        use_ema=True,
        ema_alpha=0.75
    )

    category_results = []

    for item in test_cases:
        vid_path = item["path"]
        cat_name = item["category"]

        if not os.path.exists(vid_path):
            continue

        pipeline.reset_state()
        t0 = time.time()
        res = pipeline.process_video(vid_path, max_frames=150, no_validation=False)
        elapsed = time.time() - t0

        recs = pipeline.validator.records if pipeline.validator else []
        sims = [r['cosine_sim'] for r in recs] if recs else [1.0]

        mean_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))
        savings = float(res.get("encoder_savings_pct", 0.0))
        fps = 150.0 / elapsed if elapsed > 0 else 0.0

        category_results.append({
            "category": cat_name,
            "video_file": os.path.basename(vid_path),
            "mean_cos_sim": round(mean_sim, 4),
            "min_cos_sim": round(min_sim, 4),
            "encoder_savings_pct": round(savings, 1),
            "fps": round(fps, 1)
        })

        print(f" • Category: {cat_name:25s} | Savings: {savings:.1f}% | Mean CosSim: {mean_sim:.4f} | Min: {min_sim:.4f}")

    print("=========================================================\n")

    out_path = "results/video_types_adaptive_report.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(category_results, f, indent=2)

if __name__ == "__main__":
    main()
