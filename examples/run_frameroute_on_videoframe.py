"""
Run frameroute on videoframe/cities_and_climate_high_res.mp4
Saves selected frames into videoframe/selected_frames/
"""
import os
import sys
import time
import cv2
import frameroute
from frameroute import FrameRouter, analyze_video

def main():
    video_path = os.path.join("videoframe", "cities_and_climate_high_res.mp4")
    output_dir = os.path.join("videoframe", "selected_frames")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Running frameroute v{frameroute.__version__} on: {video_path}")
    print(f"Installed from: {frameroute.__file__}")
    print("=" * 70)

    # 1. Analyze video
    print("\n[Step 1] Analyzing video change signals across 40,984 frames...")
    start_time = time.time()
    # Using stride=2 for 2x faster scan on long 23-minute lecture video
    track = analyze_video(
        video_path,
        stride=2,
        weights="static_cam",
        progress=True
    )
    analysis_time = time.time() - start_time

    print(f"\nAnalysis completed in {analysis_time:.2f}s ({track.n_analyzed / analysis_time:.1f} fps)!")
    print(f"  - Total video frames : {track.n_frames_total}")
    print(f"  - Frames analyzed    : {track.n_analyzed}")
    print(f"  - Video FPS          : {track.fps:.2f}")
    print(f"  - Video Duration     : {track.duration_sec:.2f}s ({track.duration_sec/60:.2f} min)")

    # 2. Route with budget
    # For a 22.7-minute lecture, target ~40-60 key slides/frames
    target_budget = 50
    print(f"\n[Step 2] Selecting keyframes with FrameRouter (Budget = {target_budget})...")
    router = FrameRouter(min_gap_sec=2.0, max_gap_sec=60.0, hard_trigger=0.45)
    selection = router.select(track, budget=target_budget)

    print(f"Router Selection Results:")
    print(f"  - Frames selected    : {selection.n_calls}")
    print(f"  - Calls per hour     : {selection.calls_per_hour:.1f}")
    print(f"  - Frame savings      : {(1.0 - selection.n_calls / track.n_frames_total) * 100:.2f}% reduction!")
    print(f"  - Selection reasons  : {selection.reason_counts()}")

    # 3. Extract and save the selected frames as images
    print(f"\n[Step 3] Extracting and saving {selection.n_calls} selected keyframes to: {output_dir}")
    cap = cv2.VideoCapture(video_path)
    picks = selection.picks
    picks_by_idx = {p.idx: p for p in picks}

    saved_count = 0
    for p in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p.idx)
        ret, frame = cap.read()
        if ret:
            mins = int(p.t // 60)
            secs = int(p.t % 60)
            out_filename = f"pick_{saved_count+1:03d}_frame_{p.idx:05d}_{mins:02d}m{secs:02d}s_{p.reason}.jpg"
            out_filepath = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_filepath, frame)
            saved_count += 1
            if saved_count <= 10 or saved_count % 10 == 0 or saved_count == len(picks):
                print(f"  Saved #{saved_count:02d}: {out_filename} (t={p.t:.1f}s, novelty={p.novelty:.3f}, reason={p.reason})")

    cap.release()
    print(f"\nSuccessfully saved all {saved_count} keyframes to {output_dir}!")
    print("=" * 70)
    print(">>> COMPLETE SUCCESS: frameroute routed and extracted keyframes! <<<")
    print("=" * 70)

if __name__ == "__main__":
    main()
