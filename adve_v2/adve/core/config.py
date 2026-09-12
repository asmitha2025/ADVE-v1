import torch
from dataclasses import dataclass

@dataclass
class Config:
    # --- Model ---
    CLIP_MODEL: str = "ViT-B/32"
    YOLO_MODEL: str = "yolov8n.pt"

    # --- Anchor Refresh Triggers ---
    SPATIAL_THRESHOLD: float = 0.35       # normalized ΔG magnitude (optimal 85+ enterprise score)
    APPEARANCE_THRESHOLD: float = 0.10    # histogram correlation drop
    MAX_DELTA_FRAMES: int = 20            # force keyframe every N frames

    # --- Validation ---
    SUCCESS_THRESHOLD: float = 0.85       # min cosine similarity to pass

    # --- Hardware ---
    # --- Hardware ---
    DEVICE: str = "cpu"
    CLIP_DEVICE: str = "cpu"
    YOLO_DEVICE: str = "cpu"
    YOLO_IMGSZ: int = 320  # Optimized input resolution
    
    # --- Performance Tuning ---
    PROCESS_FPS: float = 2.0       # Target FPS for indexing (downsampling from native FPS)
    MIN_PROCESS_FPS: float = 0.2   # Downsample to 0.2 FPS (1 frame every 5 seconds) in static scenes
    MAX_PROCESS_FPS: float = 4.0   # Up to 4.0 FPS for high-motion action scenes
    MOTION_THRESHOLD: float = 0.003   # Skip YOLO if motion score is below this threshold
    YOLO_HALF: bool = False        # Disabled FP16 to prevent CUBLAS execution errors on driver mismatches

    # --- Advanced UALW & Ego-Motion Parameters ---
    UNCERTAINTY_THRESHOLD: float = 0.05    # Epistemic uncertainty trigger σ_t threshold
    USE_UALW_GATING: bool = False          # Only enable when a trained UALW checkpoint is loaded
    UALW_CHECKPOINT: str = ""              # Path to trained UALW weights (enables gating when set)

    # --- Safety Gate Quality Floor ---
    SAFETY_GATE_HARD_FLOOR: float = 0.88   # Absolute minimum CosSim to emit (Enterprise Floor)
    SAFETY_GATE_CONSECUTIVE_MAX: int = 3   # Consecutive floor hits before forcing full CLIP refresh
    USE_EGO_MOTION: bool = True           # Enable homography-based camera shake/pan compensation
    USE_EMA_THRESHOLDS: bool = True        # Enable dynamic self-tuning thresholds
    HYSTERESIS_FRAMES: int = 3             # Consecutive frame persistence requirement for new tracks
    DYNAMIC_K_FACTOR: float = 2.0          # Multiplier for EMA threshold (μ + k*σ)

    # --- I/O ---
    OUTPUT_DIR: str = "outputs"
    MLP_MODEL_PATH: str = "training/checkpoints/reconstructor_v3.pt"

    def apply_mode_preset(self, mode: str):
        """Applies configuration tuned for specific domain scenarios."""
        mode = mode.upper()
        if mode == "SPORTS_ACTION":
            self.SPATIAL_THRESHOLD = 0.45
            self.APPEARANCE_THRESHOLD = 0.15
            self.UNCERTAINTY_THRESHOLD = 0.08
            self.MAX_DELTA_FRAMES = 15
        elif mode == "LECTURE_STATIC":
            self.SPATIAL_THRESHOLD = 0.30
            self.APPEARANCE_THRESHOLD = 0.08
            self.UNCERTAINTY_THRESHOLD = 0.04
            self.MAX_DELTA_FRAMES = 30
        elif mode == "SURVEILLANCE":
            self.SPATIAL_THRESHOLD = 0.35
            self.APPEARANCE_THRESHOLD = 0.10
            self.UNCERTAINTY_THRESHOLD = 0.05
            self.MAX_DELTA_FRAMES = 20
