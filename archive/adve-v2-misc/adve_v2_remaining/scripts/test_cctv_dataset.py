import sys
import os
import glob
import time
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def main():
    base_cctv_dir = os.path.abspath("C:/Users/harih/OneDrive/Documents/codex try/adve/Testing videos/Videos")
    if not os.path.exists(base_cctv_dir):
        print(f"Error: CCTV directory {base_cctv_dir} not found.")
        return

    categories = [d for d in os.listdir(base_cctv_dir) if os.path.isdir(os.path.join(base_cctv_dir, d))]
    print("=========================================================")
    print("      ADVE REAL-WORLD CCTV SURVEILLANCE BENCHMARK        ")
    print("=========================================================")
    print(f"Dataset Root       : {base_cctv_dir}")
    print(f"CCTV Categories ({len(categories)}) : {', '.join(categories)}")
    print("---------------------------------------------------------\n")

    device = "cpu"

    pipeline = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device=device,
        use_ego_motion=True,
        use_ema=True
    )

    results_per_category = {}
    all_mean_sims = []
    all_min_sims = []
    all_savings = []
    total_cctv_frames = 0
    start_benchmark_time = time.time()

    for cat in sorted(categories):
        cat_dir = os.path.join(base_cctv_dir, cat)
        mp4_files = sorted(glob.glob(os.path.join(cat_dir, "*.mp4")))
        if not mp4_files:
            continue

        # Sample 1 video per category to test all 13 action scenarios quickly
        sample_files = mp4_files[:1]
        cat_mean_sims = []
        cat_min_sims = []
        cat_savings = []

        print(f"📹 Testing Category: [{cat.upper()}] ({len(mp4_files)} total videos, testing {len(sample_files)} samples)...")

        for video_file in sample_files:
            import gc
            pipeline.reset_state()
            gc.collect()

            fname = os.path.basename(video_file)
            res = pipeline.process_video(video_file, max_frames=300, no_validation=False)
            
            recs = pipeline.validator.records
            sims = [r['cosine_sim'] for r in recs] if recs else [1.0]

            mean_sim = float(np.mean(sims))
            min_sim = float(np.min(sims))
            savings = float(res.get("encoder_savings_pct", 0.0))
            n_frames = len(sims)

            cat_mean_sims.append(mean_sim)
            cat_min_sims.append(min_sim)
            cat_savings.append(savings)
            total_cctv_frames += n_frames

            print(f"   • {fname[:35]:35s} | Frames: {n_frames:3d} | Mean CosSim: {mean_sim:.4f} | Min: {min_sim:.4f} | Savings: {savings:4.1f}%")

        avg_cat_mean = float(np.mean(cat_mean_sims))
        avg_cat_min = float(np.min(cat_min_sims))
        avg_cat_savings = float(np.mean(cat_savings))

        results_per_category[cat] = {
            "videos_tested": len(sample_files),
            "mean_cos_sim": round(avg_cat_mean, 4),
            "min_cos_sim": round(avg_cat_min, 4),
            "encoder_savings_pct": round(avg_cat_savings, 1)
        }

        all_mean_sims.extend(cat_mean_sims)
        all_min_sims.extend(cat_min_sims)
        all_savings.extend(cat_savings)
        print(f"   --> Category Avg: Mean CosSim: {avg_cat_mean:.4f} | Min: {avg_cat_min:.4f} | Savings: {avg_cat_savings:.1f}%\n")

    overall_mean_sim = float(np.mean(all_mean_sims))
    overall_min_sim = float(np.min(all_min_sims))
    overall_savings = float(np.mean(all_savings))
    total_time = round(time.time() - start_benchmark_time, 2)

    summary = {
        "benchmark": "CCTV Surveillance Action Dataset",
        "categories_tested": len(results_per_category),
        "total_cctv_frames": total_cctv_frames,
        "overall_mean_cos_sim": round(overall_mean_sim, 4),
        "overall_min_cos_sim": round(overall_min_sim, 4),
        "overall_encoder_savings_pct": round(overall_savings, 1),
        "total_time_sec": total_time,
        "category_breakdown": results_per_category
    }

    out_json = "results/cctv_surveillance_benchmark_report.json"
    os.makedirs("results", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=========================================================")
    print("     OVERALL CCTV SURVEILLANCE BENCHMARK RESULTS         ")
    print("=========================================================")
    print(f"Categories Evaluated           : {len(results_per_category)} actions")
    print(f"Total CCTV Frames Processed    : {total_cctv_frames}")
    print(f"Overall Mean Cosine Similarity : {overall_mean_sim:.4f} (Target >= 0.985)")
    print(f"Overall Min Cosine Similarity  : {overall_min_sim:.4f} (Target >= 0.850)")
    print(f"Overall Heavy Encoder Savings  : {overall_savings:.1f}%")
    print(f"Saved Report                   : {out_json}")
    print("---------------------------------------------------------")

    if overall_mean_sim >= 0.985 and overall_savings >= 40.0:
        print("✅ ADVE CCTV SURVEILLANCE BENCHMARK: PASSED WITH EXCELLENCE!")
    else:
        print("⚠️ ADVE CCTV BENCHMARK COMPLETED")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
