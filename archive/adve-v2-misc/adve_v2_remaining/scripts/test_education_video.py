"""
ADVE Education Video Testing & Benchmark Script
Runs DomainFingerprinter, ADVEPipeline, and Validator on user's education video:
'Testing videos/education videos/vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4'
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.content_detector import DomainFingerprinter
from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline

def test_education_video():
    video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Testing videos",
        "education videos",
        "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"
    ))

    print("==================================================")
    print("  ADVE EDUCATION VIDEO BENCHMARK & AUDIT")
    print("==================================================")
    print(f"Target Video Path: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Education video file not found at {video_path}")
        return

    # 1. Domain Fingerprinting
    print("\n1. Running Domain Fingerprinter & Router...")
    fingerprinter = DomainFingerprinter(sample_frames=60)
    profile = fingerprinter.fingerprint_and_route(video_path)
    
    print(f"   Routed Domain      : {profile['domain'].upper()}")
    print(f"   Reconstructor Head : {profile['reconstructor_head']}")
    print(f"   Spatial Threshold  : {profile['spatial_threshold']}")
    print(f"   Max Delta Frames   : {profile['max_delta_frames']}")
    print(f"   EMA Alpha          : {profile['ema_alpha']}")
    print(f"   Features Extracted : {profile['features']}")

    # 2. Pipeline Execution & Benchmark
    print("\n2. Initializing ADVE Pipeline...")
    config = Config()
    config.SPATIAL_THRESHOLD = profile['spatial_threshold']
    config.MAX_DELTA_FRAMES = profile['max_delta_frames']
    config.EMA_ALPHA = profile['ema_alpha']

    pipeline = ADVEPipeline(config)
    
    print("3. Processing Video Frames (Max 300 frames)...")
    start_t = time.time()
    results = pipeline.process_video(video_path, max_frames=300)
    elapsed_t = time.time() - start_t

    num_frames = results.get("num_frames", 0)
    num_anchors = results.get("num_anchors", 0)
    num_deltas = results.get("num_deltas", 0)
    mean_cossim = results.get("mean_cosine_similarity", 0.0)
    min_cossim = results.get("min_cosine_similarity", 0.0)
    savings = (num_deltas / num_frames * 100.0) if num_frames > 0 else 0.0
    fps = round(num_frames / elapsed_t, 2) if elapsed_t > 0 else 0.0

    print("\n==================================================")
    print("  ADVE EDUCATION VIDEO BENCHMARK RESULTS")
    print("==================================================")
    print(f"  Processed Frames      : {num_frames}")
    print(f"  Anchor Keyframes      : {num_anchors}")
    print(f"  Delta Reconstructions : {num_deltas}")
    print(f"  Heavy Encoder Savings : {savings:.1f}%")
    print(f"  Processing Speed      : {fps} FPS ({elapsed_t:.2f}s total)")
    print(f"  Mean Cosine Similarity: {mean_cossim:.4f}")
    print(f"  Min Cosine Similarity : {min_cossim:.4f}")
    print("==================================================")

    # Save benchmark JSON report
    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "education_video_benchmark.json"))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "video_name": os.path.basename(video_path),
            "profile": profile,
            "metrics": {
                "num_frames": num_frames,
                "num_anchors": num_anchors,
                "num_deltas": num_deltas,
                "encoder_savings_percent": round(savings, 2),
                "processing_speed_fps": fps,
                "mean_cossim": round(mean_cossim, 4),
                "min_cossim": round(min_cossim, 4)
            }
        }, f, indent=2)
    print(f"\n[OK] Benchmark report saved to: {report_path}")

if __name__ == "__main__":
    test_education_video()
