import cv2
import numpy as np
import os
import time
import torch
from typing import Optional
from adve.core.frame_filter import FrameFilter

from adve.core.config import Config
from adve.core.spatial_graph import SpatialGraph
from adve.core.anchor import AnchorProcessor
from adve.core.tracker import DeltaTracker
from adve.core.reconstructor import EmbeddingReconstructor
from adve.core.reconstructor_v2 import DeltaReconstructor, DeltaFeatureExtractor
from adve.core.validator import Validator
from adve.core.ego_motion import EgoMotionEstimator
from adve.core.ualw import UALWReconstructor
from adve.core.safety_gate import SafetyGate
from adve.core.slide_detector import SlideTransitionDetector
from adve.core.transcoder import VideoTranscoder


class ADVEPipeline:
    """
    ADVE — Anchor-Delta Video Embedding

    Orchestrates the full pipeline:
      1. Decide: anchor frame or delta frame
      2. Anchor → full CLIP + YOLO + SpatialGraph build
      3. Delta  → ByteTrack only + embedding reconstruction
      4. Validate per-frame cosine similarity vs ground truth
    """

    def __init__(self, config: Config, clip_model=None, clip_preprocess=None):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        self._yolo = None
        self._yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)
        self._clip_model = clip_model
        self._clip_preprocess = clip_preprocess

        self.anchor_proc     = AnchorProcessor(config, yolo=None, clip_model=clip_model, clip_preprocess=clip_preprocess)
        yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)
        yolo_imgsz = getattr(config, "YOLO_IMGSZ", 320)
        self.delta_tracker   = DeltaTracker(self, device=yolo_device, imgsz=yolo_imgsz)
        self.reconstructor   = EmbeddingReconstructor(config.MLP_MODEL_PATH)
        self.validator       = Validator(config)
        self.hidden_state    = None
        self.safety_gate     = SafetyGate(
            hard_floor=getattr(config, 'SAFETY_GATE_HARD_FLOOR', 0.88),
            consecutive_max=getattr(config, 'SAFETY_GATE_CONSECUTIVE_MAX', 3)
        )
        self.slide_detector  = SlideTransitionDetector(
            threshold=getattr(config, 'SLIDE_STRUCT_THRESHOLD', 0.06),
            hist_threshold=getattr(config, 'SLIDE_HIST_THRESHOLD', 0.40)
        )

        # --- UALW Uncertainty-Aware Latent Warping (Patent Claim 7) ---
        self.ualw = UALWReconstructor(clip_dim=512, motion_dim=32)
        self.ualw.eval()
        self.ualw_hidden = None
        self._ualw_active = getattr(config, 'USE_UALW_GATING', False)
        # Load trained UALW checkpoint if available
        ualw_ckpt = getattr(config, 'UALW_CHECKPOINT', '')
        if ualw_ckpt and os.path.exists(ualw_ckpt):
            try:
                ckpt = torch.load(ualw_ckpt, map_location='cpu', weights_only=False)
                state_dict = ckpt.get('model_state_dict', ckpt)
                self.ualw.load_state_dict(state_dict)
                self._ualw_active = True
                print(f"[ADVEPipeline] Loaded UALW checkpoint from {ualw_ckpt} — Uncertainty Gating ACTIVE")
            except Exception as e:
                print(f"[ADVEPipeline] Warning: Could not load UALW checkpoint ({e}) — Uncertainty Gating DISABLED")
                self._ualw_active = False

        # --- Ego-Motion Compensation (ORB Homography) ---
        self.ego_estimator = EgoMotionEstimator(max_features=500) if getattr(config, 'USE_EGO_MOTION', True) else None

        # --- EMA Dynamic Threshold State ---
        self._ema_spatial_mu = 0.0
        self._ema_spatial_var = 0.0
        self._ema_appearance_mu = 0.0
        self._ema_appearance_var = 0.0
        self._ema_alpha = 0.1  # EMA smoothing factor
        self._ema_initialized = False

        # --- Track Persistence Hysteresis ---
        self._new_track_counter: dict = {}  # track_id -> consecutive frames seen
        self._hysteresis_frames = getattr(config, 'HYSTERESIS_FRAMES', 3)

        # Live state
        self.anchor_graph:     Optional[SpatialGraph] = None
        self.anchor_embedding: Optional[np.ndarray]   = None
        self.anchor_frame:     Optional[np.ndarray]   = None
        self.anchor_buffer:    list = []  # rolling anchor buffer (Improvement 3)
        self.anchor_buffer_max = 3
        self.frames_since_anchor: int = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.force_refresh:    bool = False
        motion_threshold = getattr(config, "MOTION_THRESHOLD", 0.02)
        self.frame_filter = FrameFilter(motion_threshold=motion_threshold)
        
        # Cached ORB keypoints and descriptors for fast homography
        self.anchor_kp = None
        self.anchor_des = None
        self.anchor_orb_scale = 1.0

    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            import torch
            device = self._yolo_device
            if not torch.cuda.is_available() and device == "cuda":
                device = "cpu"
            print(f"[ADVEPipeline] Loading YOLO model '{self.config.YOLO_MODEL}' on {device}...")
            self._yolo = YOLO(self.config.YOLO_MODEL)
            self._yolo.to(device)
            
            yolo_half = getattr(self.config, "YOLO_HALF", True)
            if yolo_half and device == "cuda":
                try:
                    self._yolo.model.half()
                    print("[ADVEPipeline] Enabled FP16 (Half Precision) for YOLOv8 tracking on GPU.")
                except Exception as e:
                    print(f"[ADVEPipeline] Warning: Failed to convert YOLO model to FP16: {e}")
            
            self.anchor_proc.yolo = self._yolo
        return self._yolo

    def reset(self) -> None:
        """Resets the pipeline state to start processing a new video."""
        self.anchor_graph = None
        self.anchor_embedding = None
        self.anchor_frame = None
        self.anchor_buffer = []
        self.frames_since_anchor = 0
        self.prev_frame = None
        self.force_refresh = False
        self.frame_filter.prev_gray = None
        self.anchor_kp = None
        self.anchor_des = None
        self.anchor_orb_scale = 1.0
        self.hidden_state = None
        # Reset UALW state
        self.ualw_hidden = None
        # Reset Ego-Motion state
        if self.ego_estimator is not None:
            self.ego_estimator.reset()
        # Reset EMA thresholds
        self._ema_spatial_mu = 0.0
        self._ema_spatial_var = 0.0
        self._ema_appearance_mu = 0.0
        self._ema_appearance_var = 0.0
        self._ema_initialized = False
        # Reset track hysteresis
        self._new_track_counter = {}


    # ------------------------------------------------------------------
    # EMA Dynamic Threshold Update
    # ------------------------------------------------------------------

    def _update_ema_thresholds(self, spatial_mag: float, appearance_delta: float):
        """Update running EMA mean/variance for adaptive thresholding."""
        alpha = self._ema_alpha
        if not self._ema_initialized:
            self._ema_spatial_mu = spatial_mag
            self._ema_spatial_var = 0.0
            self._ema_appearance_mu = appearance_delta
            self._ema_appearance_var = 0.0
            self._ema_initialized = True
        else:
            # Spatial EMA
            self._ema_spatial_mu = (1 - alpha) * self._ema_spatial_mu + alpha * spatial_mag
            diff_s = spatial_mag - self._ema_spatial_mu
            self._ema_spatial_var = (1 - alpha) * self._ema_spatial_var + alpha * (diff_s ** 2)
            # Appearance EMA
            self._ema_appearance_mu = (1 - alpha) * self._ema_appearance_mu + alpha * appearance_delta
            diff_a = appearance_delta - self._ema_appearance_mu
            self._ema_appearance_var = (1 - alpha) * self._ema_appearance_var + alpha * (diff_a ** 2)

    def _get_dynamic_thresholds(self):
        """Returns adaptive spatial and appearance thresholds based on EMA statistics."""
        k = getattr(self.config, 'DYNAMIC_K_FACTOR', 2.0)
        use_ema = getattr(self.config, 'USE_EMA_THRESHOLDS', True)

        if use_ema and self._ema_initialized:
            spatial_thresh = max(
                self._ema_spatial_mu + k * (self._ema_spatial_var ** 0.5),
                self.config.SPATIAL_THRESHOLD * 0.5  # floor at 50% of static threshold
            )
            appearance_thresh = max(
                self._ema_appearance_mu + k * (self._ema_appearance_var ** 0.5),
                self.config.APPEARANCE_THRESHOLD * 0.5
            )
            return spatial_thresh, appearance_thresh
        else:
            return self.config.SPATIAL_THRESHOLD, self.config.APPEARANCE_THRESHOLD

    # ------------------------------------------------------------------
    # Track Persistence Hysteresis
    # ------------------------------------------------------------------

    def _filter_new_objects_hysteresis(self, new_objects: list) -> list:
        """Require new track IDs to persist for N consecutive frames before triggering anchor."""
        confirmed = []
        current_ids = set()
        for obj_id in new_objects:
            current_ids.add(obj_id)
            self._new_track_counter[obj_id] = self._new_track_counter.get(obj_id, 0) + 1
            if self._new_track_counter[obj_id] >= self._hysteresis_frames:
                confirmed.append(obj_id)

        # Decay counters for tracks no longer seen
        stale_ids = [k for k in self._new_track_counter if k not in current_ids]
        for sid in stale_ids:
            del self._new_track_counter[sid]

        return confirmed

    # ------------------------------------------------------------------
    # UALW Uncertainty Check
    # ------------------------------------------------------------------

    def _check_ualw_uncertainty(self, prev_embedding: np.ndarray) -> tuple:
        """
        Run UALW single-pass uncertainty estimation.
        Returns (should_refresh: bool, sigma_t: float).
        """
        try:
            prev_tensor = torch.tensor(prev_embedding, dtype=torch.float32).unsqueeze(0)
            # Use spatial delta magnitude as a proxy motion feature
            motion_feat = torch.randn(1, 32)  # Placeholder; real integration uses ΔG vector
            with torch.no_grad():
                _, sigma, self.ualw_hidden = self.ualw(
                    prev_tensor, motion_feat, self.ualw_hidden
                )
            sigma_val = float(sigma.item())
            threshold = getattr(self.config, 'UNCERTAINTY_THRESHOLD', 0.05)
            return self.ualw.check_uncertainty_refresh(sigma_val, threshold), sigma_val
        except Exception:
            return False, 0.0

    # ------------------------------------------------------------------
    # Anchor decision logic (Enhanced)
    # ------------------------------------------------------------------

    def _needs_anchor(self, delta: dict, appearance_delta: float) -> bool:
        # Get dynamic or static thresholds
        spatial_thresh, appearance_thresh = self._get_dynamic_thresholds()

        # Update EMA with current observations
        self._update_ema_thresholds(delta["total_magnitude"], appearance_delta)

        # Filter new objects through hysteresis
        confirmed_new = self._filter_new_objects_hysteresis(delta["new_objects"])

        # Standard gating with dynamic thresholds
        standard_trigger = (
            delta["total_magnitude"]    > spatial_thresh  or
            appearance_delta            > appearance_thresh or
            self.frames_since_anchor   >= self.config.MAX_DELTA_FRAMES    or
            len(confirmed_new)          > 0
        )

        # UALW Uncertainty Gating (Patent Claim 7) — only when trained checkpoint is loaded
        ualw_trigger = False
        if self._ualw_active and self.anchor_embedding is not None:
            ualw_trigger, sigma_t = self._check_ualw_uncertainty(self.anchor_embedding)

        return standard_trigger or ualw_trigger

    def _appearance_delta(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Fast structural appearance change score (0 = identical, 1 = max change)."""
        g1 = cv2.cvtColor(cv2.resize(f1, (160, 90)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, (160, 90)), cv2.COLOR_BGR2GRAY)
        diff = float(np.mean(cv2.absdiff(g1, g2))) / 255.0
        return diff

    def _estimate_homography(self, img2: np.ndarray) -> Optional[np.ndarray]:
        try:
            if self.anchor_kp is None or self.anchor_des is None:
                return None

            h, w = img2.shape[:2]
            scale = self.anchor_orb_scale
            if scale < 1.0:
                img2_small = cv2.resize(img2, (0, 0), fx=scale, fy=scale)
            else:
                img2_small = img2

            gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=300)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            if des2 is None or len(kp2) < 10:
                return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.anchor_des, des2)
            if len(matches) < 8:
                return None

            src_pts = np.float32([self.anchor_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Scale coordinates back to original size before estimating homography
            if scale < 1.0:
                src_pts = src_pts / scale
                dst_pts = dst_pts / scale
                
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_idx: int, no_validation: bool = True) -> dict:
        # --- Motion filter ---
        if self.anchor_graph is not None:
            has_motion, score = self.frame_filter.has_motion(frame)
            if not has_motion:
                self.prev_frame = frame.copy()
                self.frames_since_anchor += 1
                
                # Log in validator to keep tracking continuous
                sim = self.validator.log(
                    frame_idx       = frame_idx,
                    reconstructed   = self.anchor_embedding,
                    ground_truth    = None if no_validation else self.anchor_proc.embed_frame(frame),
                    is_anchor       = False,
                    delta_magnitude = 0.0,
                    encoder_called  = False,
                )
                
                objects = []
                for obj_id, obj in self.anchor_graph.objects.items():
                    if obj.embedding is not None:
                        objects.append({
                            "obj_id":     obj_id,
                            "class_name": obj.class_name,
                            "bbox":       obj.bbox,
                            "embedding":  obj.embedding,
                        })
                
                return {
                    "embedding":       self.anchor_embedding,
                    "is_anchor":       False,
                    "encoder_called":  False,
                    "delta_magnitude": 0.0,
                    "appearance_delta": 0.0,
                    "cosine_sim":      sim,
                    "frame_idx":       frame_idx,
                    "yolo_skipped":    True,
                    "objects":         objects,
                }

        is_anchor      = False
        encoder_called = False
        delta_magnitude = 0.0
        appearance_delta = 0.0
        current_graph  = None
        delta          = {"total_magnitude": 0.0, "new_objects": [], "lost_objects": []}

        # --- Appearance delta ---
        if self.prev_frame is not None:
            appearance_delta = self._appearance_delta(self.prev_frame, frame)

        # --- Decide frame type ---
        if self.anchor_graph is None or self.force_refresh:
            refresh = True
            self.force_refresh = False
        else:
            # Skip homography estimation if motion is low, or on CPU / Hugging Face Spaces to prioritize speed
            if score < 0.03 or os.environ.get("SPACE_ID") or os.environ.get("LOW_MEMORY") or not torch.cuda.is_available():
                homography = None
            else:
                homography = self._estimate_homography(frame)

            current_graph, delta = self.delta_tracker.track(
                frame, self.anchor_graph, homography=homography
            )
            delta_magnitude = delta["total_magnitude"]
            refresh = self._needs_anchor(delta, appearance_delta)

            # Fix #2: Confidence-Gated Anchor Refresh to protect worst-case frames
            if current_graph is not None and current_graph.objects:
                confs = [getattr(obj, "confidence", 0.5) for obj in current_graph.objects.values()]
                avg_conf = float(np.mean(confs)) if confs else 1.0
                if avg_conf < 0.35:
                    refresh = True

            # Slide Transition Detector: Compare current frame vs ANCHOR frame
            # Catches education slide changes that prev_frame delta misses
            if not refresh and self.anchor_frame is not None:
                is_slide, struct_diff, hist_corr = self.slide_detector.check(frame)
                if is_slide:
                    refresh = True

        # --- Process ---
        if refresh:
            # ── ANCHOR FRAME ──────────────────────────────────────
            self.anchor_frame = frame.copy()
            self.anchor_graph, self.anchor_embedding = self.anchor_proc.process(frame)
            self.slide_detector.set_anchor(frame)  # Cache anchor for slide transition detection
            
            # Cache anchor ORB keypoints and descriptors
            try:
                h, w = self.anchor_frame.shape[:2]
                scale = 480.0 / max(h, w)
                if scale < 1.0:
                    img_small = cv2.resize(self.anchor_frame, (0, 0), fx=scale, fy=scale)
                else:
                    img_small = self.anchor_frame
                gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
                orb = cv2.ORB_create(nfeatures=300)
                self.anchor_kp, self.anchor_des = orb.detectAndCompute(gray, None)
                self.anchor_orb_scale = scale
            except Exception:
                self.anchor_kp, self.anchor_des = None, None
                self.anchor_orb_scale = 1.0

            # Manage rolling anchor buffer (Improvement 3)
            self.anchor_buffer.append(self.anchor_embedding)
            if len(self.anchor_buffer) > self.anchor_buffer_max:
                self.anchor_buffer.pop(0)

            self.frames_since_anchor = 0
            self.hidden_state = None  # Reset GRU state on anchor refresh
            self.ualw_hidden = None   # Reset UALW hidden state on anchor refresh
            self._new_track_counter = {}  # Reset hysteresis counters
            # Set ego-motion anchor frame
            if self.ego_estimator is not None:
                self.ego_estimator.set_anchor_frame(frame)
            # Reset reconstructor v3 internal state on anchor refresh
            if hasattr(self.reconstructor, 'hidden_state'):
                self.reconstructor.hidden_state = None
            if hasattr(self.reconstructor, '_prev_reconstructed'):
                self.reconstructor._prev_reconstructed = None
            is_anchor      = True
            encoder_called = True

            reconstructed     = self.anchor_embedding
            ground_truth      = self.anchor_embedding    # trivially 1.0

        else:
            # ── DELTA FRAME ───────────────────────────────────────
            # Improvement 6: Appearance Delta Per Object
            appearance_deltas = self.delta_tracker.compute_appearance_delta_per_object(
                frame, self.anchor_graph, current_graph
            )
            objs_to_reembed = []
            rois_to_reembed = []
            for oid, app_delta in appearance_deltas.items():
                if app_delta > 0.15: # threshold
                    obj = current_graph.objects[oid]
                    x1, y1, x2, y2 = obj.bbox
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        objs_to_reembed.append(obj)
                        rois_to_reembed.append(roi)
            
            if rois_to_reembed:
                embeddings = self.anchor_proc._embed_batch(rois_to_reembed)
                for obj, emb, roi in zip(objs_to_reembed, embeddings, rois_to_reembed):
                    obj.embedding = emb
                    # Cache new histogram
                    h = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8],
                                     [0, 256, 0, 256, 0, 256])
                    obj.appearance_hist = cv2.normalize(h, h).flatten()

            reconstructed = self.reconstructor.reconstruct(
                self.anchor_graph,
                current_graph,
                delta,
                self.anchor_buffer, # Pass the rolling buffer instead of a single anchor
            )

            # Ground truth: full CLIP (only for validation — normally skipped)
            ground_truth = None if no_validation else self.anchor_proc.embed_frame(frame)
            self.frames_since_anchor += 1

            # SAFETY GATE: Hard Floor Quality Assurance Gate (Guaranteed 0.88 Floor)
            safe_embedding, was_forced, actual_sim = self.safety_gate.check(
                predicted_embedding=reconstructed,
                anchor_embedding=self.anchor_embedding,
                true_clip_embedding=ground_truth
            )
            reconstructed = safe_embedding
            if self.safety_gate.should_force_full_clip():
                self.force_refresh = True

            # Check if current frame was empty, schedule refresh if anchor was not empty
            if len(current_graph.objects) == 0 and len(self.anchor_graph.objects) > 0:
                self.force_refresh = True

        self.prev_frame = frame.copy()

        # --- Validate ---
        sim = self.validator.log(
            frame_idx       = frame_idx,
            reconstructed   = reconstructed,
            ground_truth    = ground_truth,
            is_anchor       = is_anchor,
            delta_magnitude = delta_magnitude,
            encoder_called  = encoder_called,
        )

        # Extract object metadata and embeddings from active graph
        objects = []
        active_graph = self.anchor_graph if refresh else current_graph
        if active_graph is not None:
            for obj_id, obj in active_graph.objects.items():
                if obj.embedding is not None:
                    objects.append({
                        "obj_id":     obj_id,
                        "class_name": obj.class_name,
                        "bbox":       obj.bbox,
                        "embedding":  obj.embedding,
                    })

        return {
            "embedding":       reconstructed,
            "is_anchor":       is_anchor,
            "encoder_called":  encoder_called,
            "delta_magnitude": delta_magnitude,
            "appearance_delta": appearance_delta,
            "cosine_sim":      sim,
            "frame_idx":       frame_idx,
            "objects":         objects,
        }

    def process_video(self, video_path: str, no_validation: bool = False, max_frames: Optional[int] = None) -> dict:
        if getattr(self.config, 'ENABLE_TRANSCODER', True):
            transcoder = VideoTranscoder(
                max_width=getattr(self.config, 'MAX_VIDEO_WIDTH', 1280),
                max_height=getattr(self.config, 'MAX_VIDEO_HEIGHT', 720),
                target_fps=getattr(self.config, 'TARGET_VIDEO_FPS', 30)
            )
            video_path, was_transcoded = transcoder.transcode_if_needed(video_path)
            if was_transcoded:
                print(f"[VideoTranscoder] Input video normalized & transcoded to: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n{'='*55}")
        print(f"  ADVE Pipeline  |  Device: {self.config.DEVICE.upper()}")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Frames: {total_vid_frames}  |  FPS: {total_fps:.1f}")
        print(f"{'='*55}")

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.validator.records = []
        frame_idx = 0
        t_start   = time.time()

        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break

            try:
                ret, frame = cap.read()
            except Exception as read_err:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(0.05)
                try:
                    ret, frame = cap.read()
                except Exception:
                    break

            if not ret or frame is None:
                break

            try:
                res = self.process_frame(frame, frame_idx, no_validation=no_validation)
            except Exception as frame_err:
                import traceback
                print(f"[ADVEPipeline] Frame {frame_idx} error: {frame_err}")
                traceback.print_exc()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

            if frame_idx % 15 == 0:
                tag = "ANCHOR" if res["is_anchor"] else "DELTA "
                print(
                    f"  [{frame_idx:>5}] {tag}  "
                    f"sim={res['cosine_sim']:.4f}  dG={res['delta_magnitude']:.4f}  "
                    f"dapp={res['appearance_delta']:.4f}"
                )

            # Periodically free memory to prevent Windows OOM crashes
            if frame_idx % 30 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            frame_idx += 1

        cap.release()
        elapsed = time.time() - t_start

        # --- Results ---
        summary = self.validator.summarize()
        summary["elapsed_sec"] = round(elapsed, 2)
        summary["effective_fps"] = round(frame_idx / elapsed, 1)

        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        try:
            self.validator.plot(
                os.path.join(self.config.OUTPUT_DIR, "adve_results.png")
            )
            self.validator.save_json(
                os.path.join(self.config.OUTPUT_DIR, "adve_results.json")
            )
        except Exception as e:
            print(f"[ADVEPipeline] Notice: Could not save validation plot/json ({e})")

        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    def _print_summary(self, s: dict) -> None:
        verdict = "[PASS] HYPOTHESIS VALIDATED" if s["hypothesis_validated"] else "[WARN] NEEDS REFINEMENT"
        print(f"\n{'='*55}")
        print(f"  {verdict}")
        print(f"{'='*55}")
        print(f"  Total frames         : {s['total_frames']}")
        print(f"  Encoder calls        : {s['encoder_calls']}")
        print(f"  Delta frames         : {s['delta_frames']}")
        print(f"  Encoder savings      : {s['encoder_savings_pct']}%")
        print(f"  Mean cosine sim (d)  : {s['mean_delta_cosine_sim']}")
        print(f"  Min  cosine sim (d)  : {s['min_delta_cosine_sim']}")
        print(f"  Frames >= threshold   : {s['pct_above_threshold']}%")
        print(f"  Effective FPS        : {s['effective_fps']}")
        print(f"{'='*55}\n")
