import sys
import os
import cv2
import time
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.pipeline_enterprise import ADVEEnterprisePipeline

def main():
    rel_path = "../../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4"
    video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    device = "cpu"
    print("=========================================================")
    print("      ADVE REAL-TIME STREAMING & LATENCY AUDIT          ")
    print("=========================================================")
    print(f"Video Stream  : {video_path}")
    print(f"Execution Device: {device}")
    print("---------------------------------------------------------")

    pipeline = ADVEEnterprisePipeline(
        reconstructor_path='training/checkpoints/reconstructor_v3.pt',
        device=device,
        use_ego_motion=True,
        use_ema=True
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stream_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    start_time = time.time()
    
    latencies = []
    skipped_frames = 0
    anchor_frames = 0

    print("Simulating Live Real-Time Video Stream Input...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        f_start = time.perf_counter()
        
        # Process frame through ADVE Realtime Pipeline
        res = pipeline.process_frame(frame, frame_idx, no_validation=True)
        
        f_end = time.perf_counter()
        latency_ms = (f_end - f_start) * 1000.0
        latencies.append(latency_ms)

        if res.get("is_anchor", False):
            anchor_frames += 1
        if not res.get("encoder_called", True):
            skipped_frames += 1

        if frame_idx > 0 and frame_idx % 50 == 0:
            avg_lat = np.mean(latencies[-50:])
            savings = (skipped_frames / (frame_idx + 1)) * 100
            print(f" [Stream t={(frame_idx/stream_fps):4.1f}s] Frame {frame_idx:3d}/{total_frames} | "
                  f"Latency: {avg_lat:5.2f} ms/frame | Savings: {savings:4.1f}% | "
                  f"Status: {'Anchor Keyframe' if res['is_anchor'] else 'Reconstructed (Encoder Skipped)'}")

        frame_idx += 1

    cap.release()
    total_time = time.time() - start_time

    if not latencies:
        print("Error: No frames were successfully processed from the stream.")
        return

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    overall_savings = (skipped_frames / frame_idx) * 100
    effective_throughput = frame_idx / total_time

    print("\n=========================================================")
    print("         REAL-TIME STREAMING PERFORMANCE RESULTS         ")
    print("=========================================================")
    print(f"Total Stream Frames Processed : {frame_idx}")
    print(f"Anchor Keyframes Triggered     : {anchor_frames}")
    print(f"Encoder Skips (Delta Reconstructed): {skipped_frames}")
    print(f"Overall Heavy Encoder Savings  : {overall_savings:.1f}%")
    print(f"Average Frame Latency          : {avg_latency:.2f} ms")
    print(f"95th Percentile Latency (p95)  : {p95_latency:.2f} ms")
    print(f"Effective Processing Throughput: {effective_throughput:.1f} FPS")
    print("---------------------------------------------------------")

    if overall_savings >= 50.0 and avg_latency < 50.0:
        print("✅ REAL-TIME AUDIT PASSED: Stream processing is low-latency & enterprise ready!")
    else:
        print("⚠️ REAL-TIME AUDIT NEEDS TUNING")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
