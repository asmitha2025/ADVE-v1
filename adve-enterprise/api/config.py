"""
ADVE Enterprise — Application Configuration
Manages environment variables, defaults, and runtime pipeline settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADVE_", extra="ignore")

    # --- Server Settings ---
    # --- Server Settings ---
    app_name: str = "ADVE Enterprise Neural Video Embedding Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    api_key: Optional[str] = None  # Header: X-API-Key

    # --- Pipeline Triggers ---
    anchor_budget: int = 20
    spatial_threshold: float = 0.35
    appearance_threshold: float = 0.15
    max_delta_frames: int = 30
    success_threshold: float = 0.85
    clip_model: str = "ViT-B/32"
    yolo_model: str = "yolov8n.pt"
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # --- License Key & Security ---
    license_key: Optional[str] = os.environ.get("ADVE_LICENSE_KEY", None)
    public_key_path: str = os.path.join(
        os.path.dirname(__file__), "..", "keys", "public_key.pem"
    )

    # --- Redis & Celery ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Output & Storage ---
    output_dir: str = "outputs"
    vector_db_type: str = "qdrant"  # qdrant, milvus, memory
    qdrant_url: str = "http://localhost:6333"


settings = Settings()
