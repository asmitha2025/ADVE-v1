"""
Lightweight Frame Search for 'tea spoon' text / object location.
Saves exact frame locations and timestamps to results/tea_spoon_frame_location.json
"""

import os
import sys
import cv2
import json

def locate_tea_spoon():
    video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Testing videos",
        "education videos",
        "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"
    ))

    print(f"Opening Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file!")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fast OCR text search using Tesseract or text detection
    print(f"Total Video Length: {total_frames} frames ({total_frames/fps:.1f}s)")

    # Sample keyframe intervals
    matches = []
    
    # Try importing easyocr
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        has_ocr = True
    except Exception as e:
        print(f"EasyOCR not available: {e}")
        has_ocr = False

    frame_idx = 0
    step = 10  # Sample every 10 frames (~0.33s)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and has_ocr:
            # Resize frame for fast OCR processing
            small_frame = cv2.resize(frame, (640, 360))
            ocr_results = reader.readtext(small_frame, detail=0)
            text_str = " ".join(ocr_results).lower()

            if any(term in text_str for term in ["tea", "spoon", "teaspoon", "phase", "change"]):
                timestamp_sec = round(frame_idx / fps, 2)
                mins = int(timestamp_sec // 60)
                secs = int(timestamp_sec % 60)
                matches.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "timestamp_formatted": f"{mins}m {secs:02d}s",
                    "text_detected": ocr_results
                })
                print(f"Found at Frame {frame_idx:5d} ({mins}m {secs:02d}s): {ocr_results}")

        frame_idx += 1

    cap.release()

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "tea_spoon_frame_location.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "video": os.path.basename(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "matches": matches
        }, f, indent=2)

    print(f"\n[OK] Complete search results saved to {out_path}")

if __name__ == "__main__":
    locate_tea_spoon()
