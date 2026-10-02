"""
ADVE Tradeoff Curve Benchmark (Education Video)
Compares 3 Tuning Modes:
 1. High Savings Profile (Spatial 0.48, App 0.10)
 2. Balanced Profile (Spatial 0.35, App 0.07)
 3. High Precision Profile (Spatial 0.22, App 0.05)
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline

def run_tradeoff_curve():
    video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Testing videos",
        "education videos",
        "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"
    ))

    modes = [
        {"name": "High Savings Mode", "spatial": 0.48, "app": 0.10, "max_delta": 30},
        {"name": "Balanced Mode (Recommended)", "spatial": 0.35, "app": 0.07, "max_delta": 20},
        {"name": "High Precision Mode", "spatial": 0.22, "app": 0.05, "max_delta": 10},
    ]

    print("======================================================================")
    print("  ADVE TRADEOFF CURVE: SAVINGS vs PRECISION (EDUCATION VIDEO)")
    print("======================================================================")

    for m in modes:
        config = Config()
        config.SPATIAL_THRESHOLD = m["spatial"]
        config.APPEARANCE_THRESHOLD = m["app"]
        config.MAX_DELTA_FRAMES = m["max_delta"]

        pipeline = ADVEPipeline(config)
        res = pipeline.process_video(video_path, max_frames=300)

        anchors = int(res.get("encoder_calls", 0))
        savings = float(str(res.get("encoder_savings_pct", "0")).replace("%", ""))
        mean_sim = float(res.get("mean_cosine_sim", 0.0))
        min_sim = float(res.get("min_delta_cosine_sim", 0.0))
        pass_rate = float(res.get("pct_above_threshold", 0.0))

        print(f"\nMode: {m['name']}")
        print(f"  Thresholds         : Spatial={m['spatial']}, Appearance={m['app']}")
        print(f"  Anchor Keyframes   : {anchors} / 300")
        print(f"  GPU Savings        : {savings:.1f}%")
        print(f"  Mean Precision     : {mean_sim*100:.2f}%")
        print(f"  Min Precision      : {min_sim*100:.2f}%")
        print(f"  Pass Rate (>=85%)  : {pass_rate:.1f}%")

if __name__ == "__main__":
    run_tradeoff_curve()
