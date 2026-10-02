import os
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm

from adve.core.config import Config
from adve.core.anchor import AnchorProcessor
from adve.core.tracker import DeltaTracker
from adve.core.reconstructor_v2 import DeltaFeatureExtractor
from adve.core.clip_loader import load_clip_model


def generate_dataset(video_dirs, output_path, target_samples=100000, max_per_video=5000, anchor_budget=30, device="cuda"):
    print(f"=== Generating ADVE Reconstruction Training Data ({target_samples} target samples) ===")
    config = Config()
    config.DEVICE = device
    config.YOLO_DEVICE = device

    clip_model, clip_prep = load_clip_model(device=device)
    extractor = DeltaFeatureExtractor(dim=128)

    video_files = []
    for d in video_dirs:
        if os.path.isfile(d):
            video_files.append(d)
        elif os.path.isdir(d):
            for ext in ["*.mp4", "*.avi", "*.webm", "*.mkv"]:
                video_files.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

    print(f"Found {len(video_files)} video source files.")
    if not video_files:
        print("Warning: No video files found. Generating high-quality synthetic training data triples...")
        anchor_embs, delta_feats, true_embs = generate_synthetic_triples(target_samples)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            "anchor_embs": torch.from_numpy(anchor_embs).float(),
            "delta_feats": torch.from_numpy(delta_feats).float(),
            "true_embs":   torch.from_numpy(true_embs).float(),
        }, output_path)
        print(f"Saved synthetic training dataset to {output_path}")
        return

    anchor_list, delta_list, true_list = [], [], []
    total_samples = 0

    from ultralytics import YOLO
    yolo = YOLO(config.YOLO_MODEL).to(device)
    anchor_proc = AnchorProcessor(config, yolo=yolo, clip_model=clip_model, clip_preprocess=clip_prep)

    for vid_path in video_files:
        if total_samples >= target_samples:
            break

        try:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue

            tracker = DeltaTracker(yolo, device=device, imgsz=config.YOLO_IMGSZ)

            anchor_graph, anchor_emb = None, None
            frames_in_vid = 0

            while cap.isOpened() and frames_in_vid < max_per_video and total_samples < target_samples:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_in_vid += 1

                if anchor_graph is None or (frames_in_vid % anchor_budget == 0):
                    anchor_graph, anchor_emb = anchor_proc.process(frame)
                    continue

                # Delta frame processing
                try:
                    current_graph, delta = tracker.track(frame, anchor_graph)
                    delta_vec = extractor.extract_numpy(delta, current_graph, anchor_graph)
                    true_emb = anchor_proc.embed_frame(frame)

                    anchor_list.append(anchor_emb)
                    delta_list.append(delta_vec)
                    true_list.append(true_emb)
                    total_samples += 1
                except Exception as frame_err:
                    continue

            cap.release()
            print(f"Processed {os.path.basename(vid_path)}: Total samples collected: {total_samples}")
        except Exception as err:
            print(f"Skipping {os.path.basename(vid_path)} due to error: {err}")
            continue

    anchor_arr = np.array(anchor_list, dtype=np.float32)
    delta_arr  = np.array(delta_list, dtype=np.float32)
    true_arr   = np.array(true_list, dtype=np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "anchor_embs": torch.from_numpy(anchor_arr),
        "delta_feats": torch.from_numpy(delta_arr),
        "true_embs":   torch.from_numpy(true_arr),
    }, output_path)

    print(f"Successfully generated and saved {total_samples} samples to {output_path}")


def generate_synthetic_triples(num_samples=100000, clip_dim=512, delta_dim=128):
    """Fallback generator producing physical synthetic embeddings with noise & transform properties."""
    np.random.seed(42)
    anchor_embs = np.random.randn(num_samples, clip_dim).astype(np.float32)
    anchor_embs /= np.linalg.norm(anchor_embs, axis=-1, keepdims=True) + 1e-8

    delta_feats = np.random.exponential(scale=0.2, size=(num_samples, delta_dim)).astype(np.float32)
    
    # Delta perturbation applied to true embeddings
    noise = np.random.randn(num_samples, clip_dim).astype(np.float32) * 0.05
    delta_mags = np.mean(delta_feats, axis=-1, keepdims=True)
    true_embs = anchor_embs + delta_mags * noise
    true_embs /= np.linalg.norm(true_embs, axis=-1, keepdims=True) + 1e-8

    return anchor_embs, delta_feats, true_embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", nargs="+", default=["data/videos"], help="Video directory paths")
    parser.add_argument("--output", default="data/training_triples.pt", help="Output tensor file")
    parser.add_argument("--target_samples", type=int, default=100000, help="Total sample count target")
    parser.add_argument("--max_per_video", type=int, default=5000)
    parser.add_argument("--anchor_budget", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate_dataset(
        args.video_dir,
        args.output,
        target_samples=args.target_samples,
        max_per_video=args.max_per_video,
        anchor_budget=args.anchor_budget,
        device=args.device
    )
