import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from adve.core.reconstructor_v2 import DeltaReconstructor


class TripleDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=False)
        self.anchor_embs = data["anchor_embs"]
        self.delta_feats = data["delta_feats"]
        self.true_embs   = data["true_embs"]

    def __len__(self):
        return len(self.anchor_embs)

    def __getitem__(self, idx):
        return (
            self.anchor_embs[idx],
            self.delta_feats[idx],
            self.true_embs[idx]
        )


def train_reconstructor(
    data_path: str,
    output_path: str,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str = "cuda"
):
    print(f"=== Training DeltaReconstructor ({epochs} epochs, lr={lr}, device={device}) ===")
    
    dataset = TripleDataset(data_path)
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeltaReconstructor(clip_dim=512, delta_dim=128, hidden_dim=512, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_cos = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cos = 0.0, 0.0

        for anchor_b, delta_b, true_b in train_loader:
            anchor_b = anchor_b.to(device)
            delta_b = delta_b.to(device)
            true_b   = true_b.to(device)

            optimizer.zero_grad()
            rec_b, _ = model(anchor_b, delta_b)

            cos_sim = F.cosine_similarity(rec_b, true_b, dim=-1)
            loss_cos = (1.0 - cos_sim).mean()
            loss_mse = F.mse_loss(rec_b, true_b)
            loss = loss_cos + 0.01 * loss_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cos  += cos_sim.mean().item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_cos  = total_cos / len(train_loader)

        # Validation
        model.eval()
        val_cos_list = []
        with torch.no_grad():
            for anchor_b, delta_b, true_b in val_loader:
                anchor_b = anchor_b.to(device)
                delta_b = delta_b.to(device)
                true_b   = true_b.to(device)

                rec_b, _ = model(anchor_b, delta_b)
                cos_sim = F.cosine_similarity(rec_b, true_b, dim=-1)
                val_cos_list.extend(cos_sim.cpu().numpy())

        mean_val_cos = float(np.mean(val_cos_list))
        min_val_cos  = float(np.min(val_cos_list))

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Cos: {train_cos:.4f} | Val Cos: {mean_val_cos:.4f} (Min: {min_val_cos:.4f})")

        if mean_val_cos > best_val_cos:
            best_val_cos = mean_val_cos
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "config": {"clip_dim": 512, "delta_dim": 128, "hidden_dim": 512},
                "model_state_dict": model.state_dict(),
                "val_cos": mean_val_cos,
            }, output_path)
            print(f"  [SAVE] Best model saved -> {output_path} (Val Cos: {mean_val_cos:.4f})")

    print(f"\nTraining Complete! Peak Validation Cosine Similarity: {best_val_cos:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_triples.pt")
    parser.add_argument("--output", default="models/reconstructor_v2.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_reconstructor(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
