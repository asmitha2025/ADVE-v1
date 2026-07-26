import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class DeltaFeatureExtractor:
    """
    Extracts a fixed 128-dimensional feature vector from SpatialGraph relation deltas
    and object state changes between anchor and current frame.
    """

    def __init__(self, dim: int = 128, max_pairs: int = 32):
        self.dim = dim
        self.max_pairs = max_pairs

    def extract_numpy(self, delta: Dict[str, Any], current_graph=None, anchor_graph=None) -> np.ndarray:
        """
        Convert delta dictionary into fixed 128-d numpy vector.
        """
        vec = np.zeros(self.dim, dtype=np.float32)

        # 1. Relation Deltas (up to 32 pairs * 3 features = 96 values)
        relation_deltas = delta.get("relation_deltas", {})
        for i, (pair, rd) in enumerate(relation_deltas.items()):
            if i >= self.max_pairs:
                break
            base = i * 3
            vec[base]     = rd.get("delta_distance", 0.0)
            vec[base + 1] = rd.get("delta_angle", 0.0)
            vec[base + 2] = rd.get("delta_size_ratio", 0.0)

        offset = self.max_pairs * 3 # 96

        # 2. Macro Delta Metrics (16 values)
        vec[offset + 0] = delta.get("total_magnitude", 0.0)
        vec[offset + 1] = delta.get("max_relation_delta", 0.0)
        vec[offset + 2] = float(len(delta.get("new_objects", [])))
        vec[offset + 3] = float(len(delta.get("lost_objects", [])))
        
        if current_graph is not None and hasattr(current_graph, "objects"):
            num_objs = len(current_graph.objects)
            vec[offset + 4] = float(num_objs)
            areas = [obj.area for obj in current_graph.objects.values()]
            if areas:
                vec[offset + 5] = float(np.mean(areas))
                vec[offset + 6] = float(np.max(areas))
                vec[offset + 7] = float(np.std(areas))

        if anchor_graph is not None and hasattr(anchor_graph, "objects"):
            vec[offset + 8] = float(len(anchor_graph.objects))

        # Remaining features (e.g., higher moments / padding) normalized
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return vec

    def __call__(self, delta: Dict[str, Any], current_graph=None, anchor_graph=None) -> torch.Tensor:
        vec_np = self.extract_numpy(delta, current_graph, anchor_graph)
        return torch.from_numpy(vec_np).float()


class DeltaReconstructor(nn.Module):
    """
    Sub-millisecond Learned Neural Reconstructor for CLIP frame embeddings.
    
    Architecture:
    Input:  512-d anchor embedding ⊕ 128-d delta features = 640-d
    Net:    Linear(640 → 512) → LayerNorm → GELU → Dropout(0.05) → Linear(512 → 512)
    Memory: GRUCell(512 → 128) for temporal continuity across consecutive delta frames
    Output: L2-normalized 512-d reconstructed frame embedding
    """

    def __init__(self, clip_dim: int = 512, delta_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        self.clip_dim = clip_dim
        self.delta_dim = delta_dim
        self.hidden_dim = hidden_dim

        input_dim = clip_dim + delta_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
        )

        self.temporal_gru = nn.GRUCell(clip_dim, 128)
        self.gru_proj = nn.Linear(128, clip_dim)
        self.gate = nn.Sequential(
            nn.Linear(clip_dim * 2, 1),
            nn.Sigmoid()
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        e_anchor: torch.Tensor,
        delta_features: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            e_anchor: (B, 512) or (512,) anchor frame embedding
            delta_features: (B, 128) or (128,) spatial delta feature vector
            h_prev: (B, 128) or (128,) previous GRU hidden state

        Returns:
            e_reconstructed: (B, 512) L2-normalized predicted frame embedding
            h_next: (B, 128) updated GRU hidden state
        """
        is_unbatched = e_anchor.dim() == 1
        if is_unbatched:
            e_anchor = e_anchor.unsqueeze(0)
            delta_features = delta_features.unsqueeze(0)
            if h_prev is not None and h_prev.dim() == 1:
                h_prev = h_prev.unsqueeze(0)

        # Concatenate anchor embedding and delta features
        x = torch.cat([e_anchor, delta_features], dim=-1)
        delta_pred = self.net(x)

        # Residual update from anchor
        e_rec = e_anchor + self.residual_weight * delta_pred

        # Temporal GRU memory integration
        batch_size = e_anchor.size(0)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, 128, device=e_anchor.device, dtype=e_anchor.dtype)

        h_next = self.temporal_gru(e_rec, h_prev)
        gru_mem = self.gru_proj(h_next)

        # Learned Gated Blend
        gate_val = self.gate(torch.cat([e_rec, gru_mem], dim=-1))
        e_final = (1.0 - gate_val * 0.2) * e_rec + (gate_val * 0.2) * gru_mem

        # L2 Normalize
        e_final = F.normalize(e_final, p=2, dim=-1)

        if is_unbatched:
            return e_final.squeeze(0), h_next.squeeze(0)

        return e_final, h_next
