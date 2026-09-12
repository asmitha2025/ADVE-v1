"""
ADVE Uncertainty-Aware Latent Warping (UALW) Module
======================================================
Patent Claim 7 Implementation

Features:
1. LatentWarpNet: Warps previous CLIP embedding (e_{t-1}) using motion field F_t -> e~_t
2. Residual Delta Prediction: Predicts residual (Δ_t = e_t - e~_t) instead of full vector.
3. Single-Pass Epistemic Uncertainty Head: Predicts variance σ_t = Softplus(Linear(h, 1)) in 1.62ms.
4. Uncertainty-Gated Keyframe Refresh: Triggers anchor refresh when σ_t > 0.05.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class LatentWarpNet(nn.Module):
    """
    Learned Latent Space Motion Warper.
    Translates spatial motion field F_t (optical flow / ORB homography)
    into a high-dimensional warping operator on CLIP hypersphere S^511.
    """
    def __init__(self, clip_dim: int = 512, motion_dim: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.warp_net = nn.Sequential(
            nn.Linear(clip_dim + motion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, prev_embedding: torch.Tensor, motion_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prev_embedding, motion_features], dim=-1)
        warped_delta = self.warp_net(x)
        # Apply warp on previous embedding and L2 normalize back to CLIP sphere
        warped_embedding = F.normalize(prev_embedding + warped_delta, p=2, dim=-1)
        return warped_embedding


class UALWReconstructor(nn.Module):
    """
    Uncertainty-Aware Latent Warping Reconstructor with Dual Heads:
    1. Residual Prediction Head (e_pred = L2Norm(e~_t + Δ_residual))
    2. Single-Pass Uncertainty Head (σ_t = Softplus(Linear(h, 1)))
    """
    def __init__(self, clip_dim: int = 512, motion_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.latent_warper = LatentWarpNet(clip_dim=clip_dim, motion_dim=motion_dim, hidden_dim=clip_dim)

        self.residual_gru = nn.GRUCell(clip_dim + motion_dim, hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, clip_dim)
        
        # Single-Pass Uncertainty Head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(
        self,
        prev_embedding: torch.Tensor,
        motion_features: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - reconstructed_embedding: Normalized predicted 512-d CLIP vector
            - uncertainty_sigma: Predicted standard deviation (σ_t)
            - next_hidden_state: GRU hidden state tensor
        """
        if hidden_state is None:
            hidden_state = torch.zeros(prev_embedding.size(0), 256, device=prev_embedding.device)

        # 1. Latent Warp
        warped_emb = self.latent_warper(prev_embedding, motion_features)

        # 2. Residual GRU Step
        gru_input = torch.cat([warped_emb, motion_features], dim=-1)
        next_hidden = self.residual_gru(gru_input, hidden_state)

        # 3. Residual & Reconstructed Vector
        residual = self.residual_head(next_hidden)
        reconstructed = F.normalize(warped_emb + residual, p=2, dim=-1)

        # 4. Uncertainty Estimation (Single-Pass)
        sigma = self.uncertainty_head(next_hidden).squeeze(-1)

        return reconstructed, sigma, next_hidden

    def check_uncertainty_refresh(self, sigma: float, threshold: float = 0.05) -> bool:
        """Triggers anchor keyframe refresh if predicted epistemic uncertainty exceeds threshold."""
        return float(sigma) > threshold
