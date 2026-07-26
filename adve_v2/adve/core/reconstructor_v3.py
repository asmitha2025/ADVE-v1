import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DeltaReconstructorV3(nn.Module):
    """
    Phase 3 Enterprise Fix: DeltaReconstructorV3.
    Features:
      - Clamped GRU hidden state [-3.0, 3.0] preventing temporal state explosion / drift.
      - Dampened update gate (max 0.50 weight) preventing single-frame over-corrections.
      - Residual skip connection (anchor + delta prediction).
      - L2 Cosine Normalization output.
    """

    def __init__(self, clip_dim: int = 512, delta_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.clip_dim = clip_dim
        self.delta_dim = delta_dim
        self.hidden_dim = hidden_dim

        # Delta Feature Encoder
        self.delta_encoder = nn.Sequential(
            nn.Linear(delta_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )

        # Anchor + Delta Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim + 256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # GRU Cell for temporal memory
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # Residual Reconstruction MLP
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, clip_dim)
        )

        # Gate predictor for blending
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        anchor_emb: torch.Tensor,
        delta_feats: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            anchor_emb: (B, 512) tensor
            delta_feats: (B, 128) tensor
            hidden_state: (B, 512) tensor or None
        Returns:
            reconstructed_emb: (B, 512) tensor (L2 normalized)
            next_hidden: (B, 512) tensor (clamped)
        """
        d_enc = self.delta_encoder(delta_feats)
        fused = self.fusion(torch.cat([anchor_emb, d_enc], dim=-1))

        if hidden_state is None:
            hidden_state = torch.zeros_like(fused)

        # GRU temporal step
        next_hidden = self.gru_cell(fused, hidden_state)

        # Enterprise Fix: Clamp GRU hidden state [-3.0, 3.0] to eliminate temporal drift
        next_hidden = torch.clamp(next_hidden, -3.0, 3.0)

        # Residual delta prediction
        res_delta = self.residual_mlp(next_hidden)

        # Enterprise Fix: Dampened gate (max 0.5 weight)
        raw_gate = self.gate_layer(next_hidden)
        gate = raw_gate * 0.50

        # Residual skip connection
        pred_emb = anchor_emb + gate * res_delta

        # L2 Cosine Normalization
        reconstructed_emb = F.normalize(pred_emb, p=2, dim=-1)

        return reconstructed_emb, next_hidden
