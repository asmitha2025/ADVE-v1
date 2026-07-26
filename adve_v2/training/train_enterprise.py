import os
import cv2
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from adve.core.config import Config
from adve.core.anchor import AnchorProcessor
from adve.core.tracker import DeltaTracker
from adve.core.reconstructor_v2 import DeltaFeatureExtractor
from adve.core.reconstructor_v3 import DeltaReconstructorV3
from adve.core.ego_motion import EgoMotionEstimator
from adve.core.clip_loader import load_clip_model


class TripleDataset(Dataset):
    def __init__(self, tensor_path):
        data = torch.load(tensor_path, weights_only=False)
        self.anchor_embs = data["anchor_embs"]
        self.delta_feats = data["delta_feats"]
        self.true_embs   = data["true_embs"]

    def __len__(self):
        return len(self.anchor_embs)

    def __getitem__(self, idx):
        return self.anchor_embs[idx], self.delta_feats[idx], self.true_embs[idx]


def generate_enterprise_dataset(video_dirs, output_path, target_samples=50000, max_per_video=3000, anchor_budget=30, device="cuda"):
    print(f"=== Generating Ego-Motion Compensated Training Triples ({target_samples} samples) ===")
    config = Config()
    config.DEVICE = device
    config.YOLO_DEVICE = device

    clip_model, clip_prep = load_clip_model("ViT-B/32", device=device)
    extractor = DeltaFeatureExtractor(dim=128)
    ego_estimator = EgoMotionEstimator()

    video_files = []
    for d in video_dirs:
        if os.path.isfile(d):
            video_files.append(d)
        elif os.path.isdir(d):
            for ext in ["*.mp4", "*.webm", "*.avi", "*.mkv"]:
                video_files.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

    print(f"Found {len(video_files)} video files.")
    if not video_files:
        print("Error: No video files found for dataset generation.")
        return

    anchor_list, delta_list, true_list = [], [], []
    total_samples = 0
    anchor_proc = AnchorProcessor(config, clip_model=clip_model, clip_preprocess=clip_prep)

    for vid_path in video_files:
        if total_samples >= target_samples:
            break

        try:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue

            tracker = DeltaTracker(anchor_proc.yolo, device=device, imgsz=config.YOLO_IMGSZ)
            anchor_graph, anchor_emb = None, None
            frames_in_vid = 0

            while cap.isOpened() and frames_in_vid < max_per_video and total_samples < target_samples:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_in_vid += 1

                if anchor_graph is None or (frames_in_vid % anchor_budget == 0):
                    anchor_graph, anchor_emb = anchor_proc.process(frame)
                    ego_estimator.set_anchor_frame(frame)
                    continue

                H = ego_estimator.estimate_homography(frame)
                current_graph, delta = tracker.track(frame, anchor_graph, homography=H)
                if H is not None and current_graph is not None:
                    current_graph.objects = ego_estimator.warp_centroids(current_graph.objects, H)

                delta_vec = extractor.extract_numpy(delta, current_graph, anchor_graph)
                true_emb = anchor_proc.embed_frame(frame)

                anchor_list.append(anchor_emb)
                delta_list.append(delta_vec)
                true_list.append(true_emb)
                total_samples += 1

            cap.release()
            print(f"Processed {os.path.basename(vid_path)}: Total samples: {total_samples}")
        except Exception as err:
            print(f"Skipping {os.path.basename(vid_path)} due to error: {err}")
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "anchor_embs": torch.from_numpy(np.array(anchor_list, dtype=np.float32)),
        "delta_feats": torch.from_numpy(np.array(delta_list, dtype=np.float32)),
        "true_embs":   torch.from_numpy(np.array(true_list, dtype=np.float32)),
    }, output_path)
    print(f"Saved {total_samples} compensated triples -> {output_path}")


def train_reconstructor_v3(data_path, output_checkpoint, epochs=20, batch_size=512, lr=3e-4, device="cuda"):
    print("=== Training DeltaReconstructorV3 (Clamped GRU + Residual) ===")
    dataset = TripleDataset(data_path)
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeltaReconstructorV3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    cos_loss_fn = nn.CosineEmbeddingLoss()

    best_val_cos = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_cos_sum, n_batches = 0.0, 0.0, 0

        for a_emb, d_feat, t_emb in train_loader:
            a_emb, d_feat, t_emb = a_emb.to(device), d_feat.to(device), t_emb.to(device)
            target_ones = torch.ones(a_emb.size(0), device=device)

            optimizer.zero_grad()
            pred_emb, _ = model(a_emb, d_feat)

            c_loss = cos_loss_fn(pred_emb, t_emb, target_ones)
            mse_loss = nn.MSELoss()(pred_emb, t_emb)
            loss = c_loss + 0.2 * mse_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                cos_sim = torch.sum(pred_emb * t_emb, dim=-1).mean().item()
                train_cos_sum += cos_sim
            n_batches += 1

        scheduler.step()
        train_cos = train_cos_sum / max(n_batches, 1)

        # Validation
        model.eval()
        val_cos_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for a_emb, d_feat, t_emb in val_loader:
                a_emb, d_feat, t_emb = a_emb.to(device), d_feat.to(device), t_emb.to(device)
                pred_emb, _ = model(a_emb, d_feat)
                cos_sim = torch.sum(pred_emb * t_emb, dim=-1).mean().item()
                val_cos_sum += cos_sim
                val_batches += 1

        val_cos = val_cos_sum / max(val_batches, 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train CosSim: {train_cos:.4f} | Val CosSim: {val_cos:.4f}")

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            torch.save({
                "config": {"clip_dim": 512, "delta_dim": 128, "hidden_dim": 512},
                "model_state_dict": model.state_dict(),
                "val_cos": val_cos,
                "version": "v3"
            }, output_checkpoint)
            print(f"  --> Saved new best checkpoint (Val CosSim: {val_cos:.4f})")

    print(f"=== Training Complete. Best Val CosSim: {best_val_cos:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "train"], help="Execution mode")
    parser.add_argument("--video_dir", nargs="+", default=["demo_videos"])
    parser.add_argument("--data", default="training/data/enterprise_triples.pt")
    parser.add_argument("--output", default="training/checkpoints/reconstructor_v3.pt")
    parser.add_argument("--target_samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_enterprise_dataset(args.video_dir, args.data, target_samples=args.target_samples, device=args.device)
    elif args.mode == "train":
        train_reconstructor_v3(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
