import sys
import os
import glob
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline
from adve.core.config import Config

def main():
    print("=========================================================")
    print("      ADVE WEAKEST DOMAIN IDENTIFICATION AUDIT           ")
    print("=========================================================")

    base_dir = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/")
    all_videos = glob.glob(os.path.join(base_dir, "**/*.mp4"), recursive=True)

    if not all_videos:
        print("No test videos found.")
        return

    print(f"Found {len(all_videos)} total test videos across domains.")

    cfg = Config()
    pipeline = ADVEEnterprisePipeline(
        config=cfg,
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device="cpu",
        use_ego_motion=True,
        use_ema=True,
        ema_alpha=0.75
    )

    per_video_results = []

    # Evaluate first 5 representative videos to find weakest domain
    for vid in all_videos[:5]:
        pipeline.reset_state()
        rel_name = os.path.relpath(vid, "C:/Users/harih/OneDrive/Documents/codex try/adve/")
        print(f"\nProcessing: {rel_name}")
        
        try:
            res = pipeline.process_video(vid, max_frames=100, no_validation=False)
            recs = pipeline.validator.records if pipeline.validator else []
            if not recs:
                continue
            
            sims = [r['cosine_sim'] for r in recs]
            mean_sim = float(np.mean(sims))
            min_sim = float(np.min(sims))

            per_video_results.append({
                "video_path": vid,
                "video_name": rel_name,
                "mean_cos_sim": round(mean_sim, 4),
                "min_cos_sim": round(min_sim, 4)
            })

            print(f"   --> Mean CosSim: {mean_sim:.4f} | Min CosSim: {min_sim:.4f}")
        except Exception as e:
            print(f"   --> Error processing {rel_name}: {e}")

    # Sort by lowest mean CosSim
    per_video_results.sort(key=lambda x: x["mean_cos_sim"])

    print("\n---------------------------------------------------------")
    print("                  WEAKEST DOMAINS RANKING                ")
    print("---------------------------------------------------------")
    for r in per_video_results:
        print(f" • {r['video_name']:50s} | Mean: {r['mean_cos_sim']:.4f} | Min: {r['min_cos_sim']:.4f}")

    weakest = per_video_results[0] if per_video_results else {}
    print("---------------------------------------------------------")
    print(f"🏆 WEAKEST TARGET FOR FINE-TUNING: {weakest.get('video_name')}")
    print("=========================================================\n")

    with open("results/weakest_video_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"per_video": per_video_results, "weakest": weakest}, f, indent=2)

if __name__ == "__main__":
    main()
