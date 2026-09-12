import os
import cv2
import time
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any

from adve.core.config import Config
from adve.core.anchor import AnchorProcessor
from adve.core.tracker import DeltaTracker
from adve.core.reconstructor_v2 import DeltaFeatureExtractor
from adve.core.reconstructor_v3 import DeltaReconstructorV3
from adve.core.ego_motion import EgoMotionEstimator
from adve.core.validator import Validator
from adve.core.clip_loader import load_clip_model


class ADVEEnterprisePipeline:
    """
    Phase 3 Complete Enterprise Pipeline (ADVE Pipeline Enterprise v3).
    Includes:
      1. 9-Factor Gating System (YOLO Confidence, Blur, Object Count, Color Hist, Spatial Delta, Appearance, Max Gap, Force Refresh, Empty Scene).
      2. Ego-Motion Compensation (ORB Homography).
      3. EMA Reconstruction Smoothing (alpha = 0.65).
      4. Clamped GRU Memory Reconstructor v3.
    """

    def __init__(self, config: Optional[Config] = None, reconstructor_path: Optional[str] = None, device: str = "cuda", use_ego_motion: bool = True, use_ema: bool = True, ema_alpha: float = 0.75):
        self.config = config or Config()
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config.DEVICE = self.device
        self.config.YOLO_DEVICE = self.device

        self.use_ego_motion = use_ego_motion
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha

        # Initialize sub-modules
        clip_model, clip_prep = load_clip_model("ViT-B/32", device=self.device)
        self.anchor_proc = AnchorProcessor(self.config, clip_model=clip_model, clip_preprocess=clip_prep)
        self.delta_tracker = DeltaTracker(self.anchor_proc.yolo, device=self.device, imgsz=self.config.YOLO_IMGSZ)
        self.extractor = DeltaFeatureExtractor(dim=128)
        self.ego_estimator = EgoMotionEstimator()
        self.validator = Validator(self.config)

        # Reconstructor v3
        self.reconstructor = DeltaReconstructorV3().to(self.device)
        if reconstructor_path and os.path.exists(reconstructor_path):
            ckpt = torch.load(reconstructor_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.reconstructor.load_state_dict(state_dict)
            print(f"[ADVEEnterprisePipeline] Loaded Reconstructor v3 weights from {reconstructor_path}")

        self.reconstructor.eval()

        # Pipeline state
        self.reset_state()

    def reset_state(self):
        import gc
        self.anchor_frame = None
        self.anchor_graph = None
        self.anchor_embedding = None
        self.prev_reconstructed = None
        self.hidden_state = None
        self.frames_since_anchor = 0
        self.force_refresh = False
        self.validator.records.clear()
        if hasattr(self, "ego_estimator") and self.ego_estimator is not None:
            self.ego_estimator.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_image_blurred(self, frame: np.ndarray, blur_threshold: float = 40.0) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_NEAREST)
        var = cv2.Laplacian(small, cv2.CV_16S).var()
        return var < blur_threshold

    def process_frame(self, frame: np.ndarray, frame_idx: int, no_validation: bool = True) -> Dict[str, Any]:
        is_anchor = False
        encoder_called = False
        delta_magnitude = 0.0
        appearance_delta = 0.0
        current_graph = None
        delta = {"total_magnitude": 0.0, "new_objects": [], "lost_objects": []}

        # 1. Image Blur Check (Gating Factor 1)
        is_blurred = self._is_image_blurred(frame)

        # 2. Forced Refresh or Initial Anchor
        if self.anchor_graph is None or self.force_refresh or is_blurred or self.frames_since_anchor >= self.config.MAX_DELTA_FRAMES:
            refresh = True
            self.force_refresh = False
        else:
            # Estimate Homography if Ego-Motion Compensation is enabled
            H = self.ego_estimator.estimate_homography(self.anchor_frame) if (self.use_ego_motion and self.anchor_frame is not None) else None

            # Track objects via YOLO (tracker.py handles ego-motion warping via homography)
            current_graph, delta = self.delta_tracker.track(frame, self.anchor_graph, homography=H)

            delta_magnitude = delta["total_magnitude"]

            # 9-Factor Gating Decision
            refresh = self._evaluate_gating(current_graph, delta, delta_magnitude, frame)

        # 3. Execution Path
        if refresh:
            # ── ANCHOR FRAME ──
            self.anchor_frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            self.anchor_graph, self.anchor_embedding = self.anchor_proc.process(frame)
            if self.use_ego_motion:
                self.ego_estimator.set_anchor_frame(self.anchor_frame)

            self.frames_since_anchor = 0
            self.hidden_state = None  # Reset GRU state on anchor refresh
            is_anchor = True
            encoder_called = True
            reconstructed = self.anchor_embedding
            ground_truth = self.anchor_embedding
        else:
            # ── DELTA FRAME ──
            delta_vec = self.extractor.extract_numpy(delta, current_graph, self.anchor_graph)

            anchor_t = torch.tensor(self.anchor_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            delta_t = torch.tensor(delta_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_t, self.hidden_state = self.reconstructor(anchor_t, delta_t, self.hidden_state)
                raw_reconstructed = pred_t.squeeze(0).cpu().numpy()

            # EMA Frame Reconstruction Smoothing
            if self.use_ema and self.prev_reconstructed is not None:
                reconstructed = self.ema_alpha * raw_reconstructed + (1.0 - self.ema_alpha) * self.prev_reconstructed
                reconstructed = reconstructed / (np.linalg.norm(reconstructed) + 1e-8)
            else:
                reconstructed = raw_reconstructed

            ground_truth = None if no_validation else self.anchor_proc.embed_frame(frame)
            self.frames_since_anchor += 1

        self.prev_reconstructed = reconstructed.copy()

        # Log metrics in validator
        sim = self.validator.log(
            frame_idx=frame_idx,
            reconstructed=reconstructed,
            ground_truth=ground_truth,
            is_anchor=is_anchor,
            delta_magnitude=delta_magnitude,
            encoder_called=encoder_called
        )

        return {
            "embedding": reconstructed,
            "cosine_sim": sim,
            "is_anchor": is_anchor,
            "encoder_called": encoder_called,
            "delta_magnitude": delta_magnitude,
            "frame_idx": frame_idx
        }

    def _appearance_delta(self, f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
        """Fast histogram-based appearance change score."""
        if f1 is None or f2 is None or f1.size == 0 or f2.size == 0:
            return 0.0
        try:
            f1_small = cv2.resize(f1, (64, 64))
            f2_small = cv2.resize(f2, (64, 64))
            h1 = cv2.calcHist([f1_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            h2 = cv2.calcHist([f2_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            corr = cv2.compareHist(h1.flatten(), h2.flatten(), cv2.HISTCMP_CORREL)
            return float(1.0 - corr)
        except Exception:
            return 0.0

    def _frame_structure_delta(self, anchor_frame: Optional[np.ndarray], current_frame: Optional[np.ndarray]) -> float:
        """
        Computes structural scene change using resized grayscale MSE.
        Catches camera pan and background shifts that color histograms miss.
        """
        if anchor_frame is None or current_frame is None or anchor_frame.size == 0 or current_frame.size == 0:
            return 0.0
        try:
            a = cv2.resize(anchor_frame, (64, 64))
            c = cv2.resize(current_frame, (64, 64))
            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mse = float(np.mean((a_gray - c_gray) ** 2))
            return min(mse / 255.0, 1.0)
        except Exception:
            return 0.0

    def _evaluate_gating(self, current_graph, delta, delta_magnitude: float, frame: np.ndarray) -> bool:
        """Evaluates 9 enterprise gating conditions for anchor refresh."""
        if current_graph is None:
            return True

        # Factor A: Average YOLO Object Confidence (< 0.35)
        confs = [getattr(obj, "confidence", 0.5) for obj in current_graph.objects.values()]
        if confs and float(np.mean(confs)) < 0.35:
            return True

        # Factor B: Object Count Change (|N_curr - N_anchor| >= 2)
        n_anchor = len(self.anchor_graph.objects) if self.anchor_graph else 0
        n_curr = len(current_graph.objects)
        if abs(n_curr - n_anchor) >= 2:
            return True

        # Factor C: Spatial Delta Threshold (dG > SPATIAL_THRESHOLD)
        if delta_magnitude > self.config.SPATIAL_THRESHOLD:
            return True

        # Factor D: New / Lost Object Disruption (any new object must trigger anchor to generate its embedding)
        if len(delta.get("new_objects", [])) > 0 or len(delta.get("lost_objects", [])) > 1:
            return True

        # Factor E: Appearance / Color Histogram Change (hist correlation drop > APPEARANCE_THRESHOLD)
        if self.anchor_frame is not None:
            app_delta = self._appearance_delta(self.anchor_frame, frame)
            if app_delta > self.config.APPEARANCE_THRESHOLD:
                return True

        # Factor H (Revised): Empty/Sparse Scene Structure Drift
        # When we have 0-1 objects, spatial graph is useless. Use frame structure.
        if n_curr <= 1 and self.frames_since_anchor >= 2:
            struct_delta = self._frame_structure_delta(self.anchor_frame, frame)
            if struct_delta > 0.05:  # 5% pixel variance = scene changed significantly
                return True

        # Factor I: Hard cap for empty scenes (absolute safety net)
        if n_curr == 0 and self.frames_since_anchor >= 3:
            return True  # Never reconstruct more than 3 frames in empty scenes

        return False

    def process_video(self, video_path: str, max_frames: Optional[int] = None, no_validation: bool = False) -> Dict[str, Any]:
        self.reset_state()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        frame_idx = 0
        t0 = time.time()

        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[0] > 360:
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

            self.process_frame(frame, frame_idx, no_validation=no_validation)
            frame_idx += 1

        cap.release()
        elapsed = time.time() - t0

        summary = self.validator.summarize()
        summary["elapsed_sec"] = round(elapsed, 2)
        summary["effective_fps"] = round(frame_idx / (elapsed + 1e-8), 1)
        return summary
