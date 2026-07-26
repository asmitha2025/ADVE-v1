import torch
import torch.nn as nn
import torch.nn.functional as F


class RefreshPolicy(nn.Module):
    """
    Phase 2 Defensible Differentiator: Learned Anchor Refresh Policy Network.
    Evaluates current frame state (512-d anchor embedding + 128-d delta features)
    and predicts optimal refresh probability P(refresh | state).
    """

    def __init__(self, clip_dim: int = 512, delta_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        input_dim = clip_dim + delta_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 640) tensor combining anchor embedding and delta features
        Returns:
            prob: (B, 1) scalar refresh probability in [0, 1]
        """
        return self.net(state)

    def should_refresh(self, anchor_emb: torch.Tensor, delta_features: torch.Tensor, threshold: float = 0.5) -> bool:
        if anchor_emb.dim() == 1:
            anchor_emb = anchor_emb.unsqueeze(0)
        if delta_features.dim() == 1:
            delta_features = delta_features.unsqueeze(0)

        state = torch.cat([anchor_emb, delta_features], dim=-1)
        with torch.no_grad():
            prob = self.forward(state).item()
        return prob > threshold
