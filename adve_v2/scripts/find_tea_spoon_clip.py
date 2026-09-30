"""
Lightweight CLIP + OpenCV scanner for 'tea spoon' text and visual presence.
"""

import os
import sys
import cv2
import torch
import clip
from PIL import Image

def scan_for_tea_spoon():
    video_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Testing videos",
        "education videos",
        "vidssave.com Simulating and understanding phase change _ Guest video by Vilas Winstein 480P.mp4"
    ))

    print(f"Scanning Education Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    print(f"Total Frames: {total_frames}, FPS: {fps:.2f}, Duration: {total_frames/fps:.1f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    queries = ["a diagram showing a tea spoon", "text displaying tea spoon", "a teaspoon of liquid or phase change", "whiteboard with tea spoon written"]
    text_tokens = clip.tokenize(queries).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    top_matches = []
    frame_idx = 0
    step = 15  # Sample every 15 frames (~0.5s)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(0)
                max_sim, max_q_idx = torch.max(similarity, dim=0)

                sim_val = max_sim.item()
                if sim_val > 0.24:
                    timestamp_sec = round(frame_idx / fps, 2)
                    mins = int(timestamp_sec // 60)
                    secs = int(timestamp_sec % 60)
                    top_matches.append({
                        "frame": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "timestamp_str": f"{mins}m {secs:02d}s",
                        "similarity": round(sim_val, 4),
                        "matched_query": queries[max_q_idx.item()]
                    })
                    print(f"Frame {frame_idx:4d} ({mins}m {secs:02d}s) | Sim: {sim_val:.4f} | Query: '{queries[max_q_idx.item()]}'")

        frame_idx += 1

    cap.release()

    # Sort matches by similarity
    top_matches.sort(key=lambda x: x["similarity"], reverse=True)

    print("\n==================================================")
    print("  TOP FRAMES FOR 'TEA SPOON' IN EDUCATION VIDEO")
    print("==================================================")
    for idx, match in enumerate(top_matches[:10], 1):
        print(f"  #{idx:2d} | Frame {match['frame']:4d} | Timestamp: {match['timestamp_sec']}s ({match['timestamp_str']}) | Sim: {match['similarity']} | Query: '{match['matched_query']}'")

if __name__ == "__main__":
    scan_for_tea_spoon()
