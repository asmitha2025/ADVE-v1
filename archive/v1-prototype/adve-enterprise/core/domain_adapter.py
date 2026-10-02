"""
ADVE Enterprise — Domain Adaptation & Fingerprinting Engine
Categorizes camera streams into visual domain profiles (traffic, sparse, night) and applies domain-specific residual projection heads.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional
import structlog

logger = structlog.get_logger()

WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "domain_heads.pt")
)


class DomainFingerprinter:
    """Heuristic domain classifier based on brightness, motion variance, and object density."""
    
    @staticmethod
    def classify_frame(frame: np.ndarray, num_objects: int, motion_magnitude: float) -> str:
        if frame is None or frame.size == 0:
            return "traffic"

        # Calculate mean luminance / brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = float(np.mean(gray))

        # Categorize
        if mean_brightness < 60.0:
            return "night"
        elif num_objects <= 2:
            return "sparse"
        else:
            return "traffic"


class DomainResidualHead(nn.Module):
    """512 -> 512 Residual Linear Adapter Head for Domain Alignment."""
    def __init__(self, in_features: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc2(self.act(self.fc1(x)))
        return torch.nn.functional.normalize(residual + self.scale * out, p=2, dim=-1)


class MultiDomainAdapter(nn.Module):
    """Container holding domain-specific residual heads for traffic, sparse, and night profiles."""
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.heads = nn.ModuleDict({
            "traffic": DomainResidualHead(),
            "sparse": DomainResidualHead(),
            "night": DomainResidualHead()
        })
        self.to(device)

    def forward(self, embedding: torch.Tensor, domain: str) -> torch.Tensor:
        head_key = domain if domain in self.heads else "traffic"
        return self.heads[head_key](embedding)


class OnlineCalibrator:
    """Collects frame feature triples and performs rapid online 3-epoch adaptation pass for unknown domains."""
    def __init__(self, target_head: DomainResidualHead, device: str = "cpu"):
        self.head = target_head
        self.device = device
        self.buffer = []
        self.is_calibrated = False

    def add_sample(self, reconstructed_vector: np.ndarray, ground_truth_vector: np.ndarray):
        if len(self.buffer) < 300:
            self.buffer.append((reconstructed_vector, ground_truth_vector))

    def calibrate(self, epochs: int = 3, lr: float = 1e-4) -> bool:
        if len(self.buffer) < 20:
            return False

        logger.info("starting_online_domain_calibration", samples=len(self.buffer))
        self.head.train()
        optimizer = optim.AdamW(self.head.parameters(), lr=lr)
        criterion = nn.MSELoss()

        rec_arr = np.array([b[0] for b in self.buffer], dtype=np.float32)
        gt_arr = np.array([b[1] for b in self.buffer], dtype=np.float32)

        rec_tensor = torch.tensor(rec_arr, device=self.device)
        gt_tensor = torch.tensor(gt_arr, device=self.device)

        for _ in range(epochs):
            optimizer.zero_grad()
            output = self.head(rec_tensor)
            loss = criterion(output, gt_tensor)
            loss.backward()
            optimizer.step()

        self.head.eval()
        self.is_calibrated = True
        logger.info("online_domain_calibration_completed", final_loss=float(loss.item()))
        return True
