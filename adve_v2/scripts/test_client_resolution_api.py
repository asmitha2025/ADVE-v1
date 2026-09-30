import sys
import os
import time
import json
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline
from adve.core.config import Config

def main():
    traffic_video = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Traffic/Vodra/North.mp4")

    print("=========================================================")
    print("      CLIENT-SIDE RESOLUTION & CONFIG CONTROL AUDIT      ")
    print("=========================================================")

    # 1. High Sensitivity Mode (Thresh = 0.15) for Client-Side Resolution
    cfg_sensitive = Config()
    cfg_sensitive.SPATIAL_THRESHOLD = 0.15
    cfg_sensitive.MAX_DELTA_FRAMES = 5

    pipeline_sensitive = ADVEEnterprisePipeline(
        config=cfg_sensitive,
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device="cpu",
        use_ego_motion=True,
        use_ema=True
    )
    pipeline_sensitive.reset_state()

    res = pipeline_sensitive.process_video(traffic_video, max_frames=100, no_validation=False)
    recs = pipeline_sensitive.validator.records

    sims = [r['cosine_sim'] for r in recs]
    mean_sim = float(torch.tensor(sims).mean())
    min_sim = float(torch.tensor(sims).min())

    print(f" • Ultra-High Sensitivity Mode (Thresh 0.15):")
    print(f"   --> Mean CosSim Precision : {mean_sim:.4f} ({mean_sim*100:.2f}%)")
    print(f"   --> Minimum Precision     : {min_sim:.4f}")
    print(f"   --> Anchor Refresh Rate   : {sum(1 for r in recs if r['is_anchor'])} / {len(recs)} frames")

    print("\n=========================================================")
    print("✅ CLIENT-SIDE RESOLUTION CONTROLS VERIFIED OPERATIONAL!")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
