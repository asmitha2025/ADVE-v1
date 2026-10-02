"""
Find frames where a person wearing a suit appears in the video.
Uses CLIP to compare each sampled frame against the query text "person wearing a suit".
Outputs timestamps (seconds) to stdout and writes them to a text file.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import torch
import numpy as np
from adve.core.clip_loader import load_clip_model
from tqdm import tqdm

def find_suit_frames(video_path: str, output_path: str = None, sample_rate: int = 30, similarity_threshold: float = 0.30):
    """Extract timestamps where a person wearing a suit is likely present.

    Args:
        video_path: Path to the video file.
        output_path: Optional path to write timestamps (one per line). If None, only prints.
        sample_rate: Process every Nth frame (default 30 ≈ 1 fps for 30 fps video).
        similarity_threshold: Cosine similarity threshold for a positive match.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load CLIP model once
    model, preprocess = load_clip_model("ViT-B/32", device)

    # Prepare text embedding
    import clip
    text = "person wearing a suit"
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_embed = model.encode_text(text_tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = []

    for frame_idx in tqdm(range(0, total_frames, sample_rate), desc="Scanning video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert to PIL image for CLIP preprocessing
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_embed = model.encode_image(input_tensor)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            similarity = (img_embed @ text_embed.T).item()
        if similarity >= similarity_threshold:
            sec = frame_idx / fps
            timestamps.append(sec)
            print(f"[MATCH] Frame {frame_idx} ({sec:.2f}s) similarity={similarity:.3f}")

    cap.release()

    if output_path:
        with open(output_path, "w") as f:
            for t in timestamps:
                f.write(f"{t:.2f}\n")
        print(f"[INFO] Timestamps written to {output_path}")
    else:
        print("[INFO] No output file specified; timestamps only printed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect suit appearances in a video using CLIP.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--output", help="File to write timestamps", default="suit_frames.txt")
    parser.add_argument("--rate", type=int, help="Sample every N frames", default=30)
    parser.add_argument("--threshold", type=float, help="Similarity threshold", default=0.30)
    args = parser.parse_args()
    find_suit_frames(args.video, args.output, args.rate, args.threshold)
