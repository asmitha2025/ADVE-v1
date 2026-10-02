"""
ADVE Enterprise — Pre-Train Domain Projection Heads
Synthesizes domain-specific feature triples and trains residual domain heads for traffic, sparse, and night profiles.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

ENTERPRISE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENTERPRISE_DIR not in sys.path:
    sys.path.insert(0, ENTERPRISE_DIR)

from core.domain_adapter import MultiDomainAdapter, WEIGHTS_PATH


def synthesize_domain_data(domain: str, num_samples: int = 5000, device: str = "cpu"):
    torch.manual_seed(42)
    # Base anchor embeddings
    anchors = torch.randn(num_samples, 512, device=device)
    anchors = torch.nn.functional.normalize(anchors, p=2, dim=-1)

    # Domain specific noise & shift
    if domain == "night":
        shift = torch.randn(num_samples, 512, device=device) * 0.08
    elif domain == "sparse":
        shift = torch.randn(num_samples, 512, device=device) * 0.03
    else:  # traffic
        shift = torch.randn(num_samples, 512, device=device) * 0.05

    targets = torch.nn.functional.normalize(anchors + shift, p=2, dim=-1)
    return anchors, targets


def train_domain_heads():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Initializing MultiDomainAdapter on {device.upper()}...")
    adapter = MultiDomainAdapter(device=device)

    domains = ["traffic", "sparse", "night"]
    criterion = nn.MSELoss()

    for domain in domains:
        print(f"[+] Training domain head: '{domain}' (5,000 triples)...")
        head = adapter.heads[domain]
        head.train()
        optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

        anchors, targets = synthesize_domain_data(domain, num_samples=5000, device=device)

        for epoch in range(1, 6):
            optimizer.zero_grad()
            preds = head(anchors)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            if epoch % 2 == 0 or epoch == 5:
                print(f"    • Domain '{domain}' | Epoch {epoch}/5 | Loss: {loss.item():.6f}")

        head.eval()

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(adapter.state_dict(), WEIGHTS_PATH)
    print(f"\n[+] Domain adapter weights saved to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    train_domain_heads()
