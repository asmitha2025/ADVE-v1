from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for the CLIP-only indexing pipeline.

    The anchor-delta reconstruction machinery (and its thresholds, UALW,
    safety-gate and EMA settings) was removed — see adve/core/pipeline.py.
    Only the fields the live pipeline, validator and API server read remain.
    """

    # --- Models ---
    CLIP_MODEL: str = "ViT-B/32"
    YOLO_MODEL: str = "yolov8n.pt"

    # --- Hardware ---
    DEVICE: str = "cpu"
    CLIP_DEVICE: str = "cpu"
    YOLO_DEVICE: str = "cpu"
    YOLO_IMGSZ: int = 320          # YOLO input resolution
    YOLO_HALF: bool = False        # FP16 off by default (CUBLAS mismatch guard)

    # --- Motion filter (lets a static frame reuse the last embedding) ---
    MOTION_THRESHOLD: float = 0.003

    # --- Adaptive indexing rate (API server) ---
    PROCESS_FPS: float = 2.0
    MIN_PROCESS_FPS: float = 0.2
    MAX_PROCESS_FPS: float = 4.0

    # --- Validation / reporting ---
    SUCCESS_THRESHOLD: float = 0.85   # min cosine similarity counted as a pass
    SPATIAL_THRESHOLD: float = 0.35   # reference line on the validator plot

    # --- I/O ---
    OUTPUT_DIR: str = "outputs"
