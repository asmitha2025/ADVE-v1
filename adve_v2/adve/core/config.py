import torch
from dataclasses import dataclass

@dataclass
class Config:
    # --- Model ---
    CLIP_MODEL: str = "ViT-B/32"
    YOLO_MODEL: str = "yolov8n.pt"

    # --- Anchor Refresh Triggers ---
    SPATIAL_THRESHOLD: float = 0.30       # normalized ΔG magnitude
    APPEARANCE_THRESHOLD: float = 0.15    # histogram correlation drop
    MAX_DELTA_FRAMES: int = 30            # force keyframe every N frames regardless

    # --- Validation ---
    SUCCESS_THRESHOLD: float = 0.85       # min cosine similarity to pass

    # --- Hardware ---
    # --- Hardware ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_IMGSZ: int = 320  # Optimized input resolution
    
    # --- Performance Tuning ---
    PROCESS_FPS: float = 2.0       # Target FPS for indexing (downsampling from native FPS)
    MIN_PROCESS_FPS: float = 0.2   # Downsample to 0.2 FPS (1 frame every 5 seconds) in static scenes
    MAX_PROCESS_FPS: float = 4.0   # Up to 4.0 FPS for high-motion action scenes
    MOTION_THRESHOLD: float = 0.003   # Skip YOLO if motion score is below this threshold
    YOLO_HALF: bool = True         # FP16 YOLO (30% faster on GPU)

    # --- I/O ---
    OUTPUT_DIR: str = "outputs"
    MLP_MODEL_PATH: str = "training/checkpoints/best_model.pt"
