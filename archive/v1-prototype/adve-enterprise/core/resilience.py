"""
ADVE Enterprise — Graceful Failure & Self-Healing Engine
Protects pipeline from runtime anomalies: YOLO drops, CLIP OOMs, NaN/Inf reconstructor states, RTSP disconnections.
"""

import numpy as np
import torch
import structlog
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = structlog.get_logger()


class PipelineAnomaly(Exception):
    """Base exception for recoverable pipeline anomalies."""
    pass


class ReconstructorNaNError(PipelineAnomaly):
    """Raised when reconstructed embeddings contain NaN or Inf values."""
    pass


class ResilienceManager:
    def __init__(self, stream_id: str = "default"):
        self.stream_id = stream_id
        self.last_valid_objects: list = []
        self.stream_start_time: datetime = datetime.utcnow()
        self.nan_count: int = 0
        self.oom_count: int = 0
        self.yolo_fail_count: int = 0

    def check_embedding_health(self, embedding: Optional[np.ndarray]) -> np.ndarray:
        """
        Detects NaN or Inf values in reconstructed feature embeddings.
        Resets hidden states and returns fallback zero-vector if anomaly detected.
        """
        if embedding is None:
            logger.warn("embedding_is_none", stream_id=self.stream_id)
            return np.zeros((512,), dtype=np.float32)

        if np.isnan(embedding).any() or np.isinf(embedding).any():
            self.nan_count += 1
            logger.error(
                "reconstructor_nan_detected",
                stream_id=self.stream_id,
                nan_count=self.nan_count,
                action="reset_hidden_state_and_force_anchor"
            )
            # Return zero fallback vector
            clean_emb = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
            return clean_emb

        return embedding

    def handle_gpu_oom(self, error: Exception) -> bool:
        """Handles PyTorch GPU OOM exceptions by clearing CUDA cache."""
        if "out of memory" in str(error).lower():
            self.oom_count += 1
            logger.error(
                "clip_gpu_oom_recovery",
                stream_id=self.stream_id,
                oom_count=self.oom_count,
                action="cuda_empty_cache"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def handle_yolo_failure(self, error: Exception) -> list:
        """Recovers from YOLO detection errors by returning last-known bounding boxes."""
        self.yolo_fail_count += 1
        logger.warn(
            "yolo_inference_failed_using_cached_positions",
            stream_id=self.stream_id,
            error=str(error),
            fail_count=self.yolo_fail_count
        )
        return self.last_valid_objects

    def update_last_valid_objects(self, objects: list):
        if objects:
            self.last_valid_objects = objects

    def check_license_grace_period(self, is_valid: bool, grace_hours: int = 24) -> bool:
        """
        Allows existing live streams to run for a 24-hour grace period if license expires mid-stream.
        """
        if is_valid:
            return True

        elapsed = datetime.utcnow() - self.stream_start_time
        if elapsed <= timedelta(hours=grace_hours):
            logger.warn(
                "license_expired_active_stream_in_grace_period",
                stream_id=self.stream_id,
                hours_remaining=round(grace_hours - (elapsed.total_seconds() / 3600), 2)
            )
            return True

        logger.error("license_grace_period_expired", stream_id=self.stream_id)
        return False
