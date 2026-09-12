"""
ADVE Enterprise — Core Pipeline Wrapper
Integrates the underlying ADVE Anchor-Delta pipeline with Enterprise safety & resilience:
- Dynamic config hot-reload support (settings override)
- Frame sanitization & multi-format input handling
- Self-healing resilience manager (YOLO drops, PyTorch CUDA OOM, NaN/Inf reconstructor state guards)
- Domain Fingerprinting & Multi-Domain Residual Adapter Head (traffic, sparse, night profiles)
- High-frequency structured logging (structlog)
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import structlog
from typing import Dict, Any, Optional

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from config import Config
from pipeline import ADVEPipeline
from api.config import settings
from core.input_sanitizer import sanitize_frame
from core.resilience import ResilienceManager
from core.domain_adapter import DomainFingerprinter, MultiDomainAdapter, WEIGHTS_PATH

logger = structlog.get_logger()


class EnterprisePipeline:
    def __init__(
        self,
        stream_id: str = "default_stream",
        device: str = "cpu",
        yolo_model: str = "yolov8n.pt",
        clip_model: str = "ViT-B/32"
    ):
        self.stream_id = stream_id
        self.device = device
        self.cfg = Config()
        self.cfg.DEVICE = device
        self.cfg.YOLO_MODEL = yolo_model
        self.cfg.CLIP_MODEL = clip_model

        # Sync initial config settings
        self._sync_settings()

        self.pipeline = ADVEPipeline(self.cfg)
        self.resilience = ResilienceManager(stream_id=stream_id)

        # Domain Adaptation Engine
        self.domain_adapter = MultiDomainAdapter(device=device)
        if os.path.exists(WEIGHTS_PATH):
            try:
                ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
                self.domain_adapter.load_state_dict(ckpt)
                logger.info("loaded_domain_adapter_weights", path=WEIGHTS_PATH)
            except Exception as e:
                logger.warn("failed_to_load_domain_weights", error=str(e))
        self.domain_adapter.eval()

        self.processed_count = 0
        self.anchor_count = 0
        self.delta_count = 0
        self.sim_history = []
        self.current_domain = "traffic"

    def _sync_settings(self):
        """Hot-reloads dynamic configuration settings."""
        self.cfg.SPATIAL_THRESHOLD = settings.spatial_threshold
        self.cfg.APPEARANCE_THRESHOLD = settings.appearance_threshold
        self.cfg.MAX_DELTA_FRAMES = settings.max_delta_frames

    def process_single_frame(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Dict[str, Any]:
        """Safely processes a single video frame with self-healing guards, domain adaptation, and structured logging."""
        t_start = time.time()
        self._sync_settings()

        if frame_idx is None:
            frame_idx = self.processed_count

        try:
            clean_frame = sanitize_frame(frame)

            # Domain Classification
            num_objects = 0
            motion_mag = 0.0
            if self.pipeline.anchor_graph:
                num_objects = len(self.pipeline.anchor_graph.objects)

            self.current_domain = DomainFingerprinter.classify_frame(clean_frame, num_objects, motion_mag)

            result = self.pipeline.process_frame(clean_frame, frame_idx, no_validation=True)

            # Check embedding health (NaN/Inf check)
            clean_embedding = self.resilience.check_embedding_health(result.get("embedding"))

            # Apply Domain Projection Head
            if clean_embedding is not None and clean_embedding.size > 0:
                with torch.no_grad():
                    emb_t = torch.tensor(clean_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
                    adapted_t = self.domain_adapter(emb_t, self.current_domain)
                    clean_embedding = adapted_t.squeeze(0).cpu().numpy()

            result["embedding"] = clean_embedding

            self.processed_count += 1
            if result["is_anchor"]:
                self.anchor_count += 1
            else:
                self.delta_count += 1

            cos_sim = float(result.get("cosine_sim", 1.0))
            if cos_sim is not None:
                self.sim_history.append(cos_sim)

            latency_ms = round((time.time() - t_start) * 1000, 2)
            encoder_saved = not result["encoder_called"]

            # Structured JSON log per frame
            logger.info(
                "frame_processed",
                stream_id=self.stream_id,
                frame_num=frame_idx,
                domain=self.current_domain,
                cos_sim=cos_sim,
                encoder_saved=encoder_saved,
                is_anchor=result["is_anchor"],
                delta_mag=round(float(result["delta_magnitude"]), 4),
                latency_ms=latency_ms
            )

            return {
                "success": True,
                "frame_idx": frame_idx,
                "domain": self.current_domain,
                "is_anchor": result["is_anchor"],
                "encoder_called": result["encoder_called"],
                "delta_magnitude": float(result["delta_magnitude"]),
                "appearance_delta": float(result["appearance_delta"]),
                "cosine_sim": cos_sim,
                "latency_ms": latency_ms,
                "embedding_shape": list(clean_embedding.shape) if clean_embedding is not None else None
            }

        except RuntimeError as oom_err:
            if self.resilience.handle_gpu_oom(oom_err):
                self.pipeline.force_refresh = True
            return {
                "success": False,
                "frame_idx": frame_idx,
                "domain": self.current_domain,
                "error": "GPU OOM - cleared cache and forced refresh",
                "is_anchor": True,
                "encoder_called": False,
                "cosine_sim": 0.85
            }
        except Exception as e:
            logger.error("frame_processing_failed", stream_id=self.stream_id, frame_idx=frame_idx, error=str(e))
            self.pipeline.force_refresh = True
            return {
                "success": False,
                "frame_idx": frame_idx,
                "domain": self.current_domain,
                "error": str(e),
                "is_anchor": False,
                "encoder_called": False,
                "cosine_sim": 0.85
            }

    def get_stats(self) -> Dict[str, Any]:
        total = self.processed_count
        savings = round((self.delta_count / total * 100), 2) if total > 0 else 0.0
        mean_sim = round(float(np.mean(self.sim_history)), 4) if self.sim_history else 1.0

        return {
            "stream_id": self.stream_id,
            "domain": self.current_domain,
            "total_frames": total,
            "anchor_frames": self.anchor_count,
            "delta_frames": self.delta_count,
            "encoder_savings_pct": savings,
            "mean_cosine_sim": mean_sim,
            "anomalies": {
                "nan_recovered": self.resilience.nan_count,
                "oom_recovered": self.resilience.oom_count,
                "yolo_fails": self.resilience.yolo_fail_count
            }
        }
