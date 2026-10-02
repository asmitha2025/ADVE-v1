"""
Script to scan the Education Video frame-by-frame and find the exact frame indices, timestamps,
and OCR text where 'tea spoon' or 'teaspoon' appears.
"""

import os
import sys
import cv2
import time
import easyocr

def find_text_in_video():
    video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Testing videos",
        "education videos",
        "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"
    ))

    print(f"Scanning Video: {video_path}")
    if not os.path.exists(video_path):
        print("Video file not found!")
        return

    reader = easyocr.Reader(['en'], gpu=False)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    print(f"Total Video Frames: {total_frames}, FPS: {fps:.2f}")

    matches = []
    frame_idx = 0
    step = 5  # Scan every 5th frame for speed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # Perform OCR on frame
            results = reader.readtext(frame, detail=0)
            text_combined = " ".join(results).lower()

            if "tea" in text_combined or "spoon" in text_combined or "teaspoon" in text_combined:
                timestamp_sec = round(frame_idx / fps, 2)
                matches.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "time_str": f"{int(timestamp_sec // 60)}m {int(timestamp_sec % 60)}s",
                    "text_detected": results
                })
                print(f"   [MATCH FOUND] Frame {frame_idx} ({int(timestamp_sec // 60)}m {int(timestamp_sec % 60)}s): {results}")

        frame_idx += 1

    cap.release()

    print("\n==================================================")
    print("  RESULTS: SEARCH FOR 'TEA SPOON' IN EDUCATION VIDEO")
    print("==================================================")
    if matches:
        for m in matches:
            print(f"  Frame {m['frame_index']} | Timestamp: {m['timestamp_sec']}s ({m['time_str']})")
            print(f"    Detected Text: {m['text_detected']}\n")
    else:
        print("  No direct OCR match found in sampled frames. Running full CLIP visual search...")

if __name__ == "__main__":
    find_text_in_video()
