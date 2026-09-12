"""
adve.core.pipeline — CLIP-only frame processing.

History: earlier versions of this pipeline synthesised the embeddings of
"delta" frames from YOLO object-track deltas via a trained reconstructor
(reconstructor*/UALW) guarded by a safety gate. An audit (see bench/parity.py
and the README) showed that machinery did not earn its complexity — parameter
-free routing plus a real CLIP call on the frames that matter preserves
retrieval better, with no checkpoint, drift or failure mode. The reconstruction
stack has been removed.

What remains is deliberately simple: every frame this pipeline processes gets a
real CLIP embedding (full frame + detected-object crops). No embedding is ever
synthesised, so every vector in the index is trustworthy by construction. The
cost saving belongs upstream, in choosing *which* frames to process — that is
frameroute's job (see frameroute/ and adve.core.lean_indexer).
"""

import os
import time
from typing import Optional

import cv2
import numpy as np

from adve.core.config import Config
from adve.core.spatial_graph import SpatialGraph
from adve.core.anchor import AnchorProcessor
from adve.core.validator import Validator
from adve.core.frame_filter import FrameFilter
from adve.core.transcoder import VideoTranscoder


class ADVEPipeline:
    """
    CLIP-only anchor pipeline.

    Every processed frame is encoded with real CLIP (whole frame + object
    crops via AnchorProcessor). A cheap motion filter lets a static frame reuse
    the previous embedding (carry-forward) to save redundant encodes, but
    nothing is ever reconstructed.
    """

    def __init__(self, config: Config, clip_model=None, clip_preprocess=None):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        self._yolo = None
        self._yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)

        self.anchor_proc = AnchorProcessor(
            config, yolo=None, clip_model=clip_model, clip_preprocess=clip_preprocess
        )
        self.validator = Validator(config)

        motion_threshold = getattr(config, "MOTION_THRESHOLD", 0.02)
        self.frame_filter = FrameFilter(motion_threshold=motion_threshold)

        # Live state
        self.anchor_graph: Optional[SpatialGraph] = None
        self.anchor_embedding: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.frames_since_anchor: int = 0

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
            self.anchor_proc.yolo = self._yolo
        return self._yolo

    def reset(self) -> None:
        """Reset state to start a new video."""
        self.anchor_graph = None
        self.anchor_embedding = None
        self.prev_frame = None
        self.frames_since_anchor = 0
        self.frame_filter.prev_gray = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _appearance_delta(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Fast structural appearance change score (0 = identical, 1 = max change)."""
        g1 = cv2.cvtColor(cv2.resize(f1, (160, 90)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(f2, (160, 90)), cv2.COLOR_BGR2GRAY)
        return float(np.mean(cv2.absdiff(g1, g2))) / 255.0

    @staticmethod
    def _objects_from_graph(graph: Optional[SpatialGraph]) -> list:
        objects = []
        if graph is not None:
            for obj_id, obj in graph.objects.items():
                if obj.embedding is not None:
                    objects.append({
                        "obj_id": obj_id,
                        "class_name": obj.class_name,
                        "bbox": obj.bbox,
                        "embedding": obj.embedding,
                    })
        return objects

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_idx: int, no_validation: bool = True) -> dict:
        # --- Motion filter: a static frame reuses the last real CLIP embedding ---
        if self.anchor_graph is not None:
            has_motion, _ = self.frame_filter.has_motion(frame)
            if not has_motion:
                self.prev_frame = frame.copy()
                self.frames_since_anchor += 1
                sim = self.validator.log(
                    frame_idx=frame_idx,
                    reconstructed=self.anchor_embedding,
                    ground_truth=None if no_validation else self.anchor_embedding,
                    is_anchor=False,
                    delta_magnitude=0.0,
                    encoder_called=False,
                )
                return {
                    "embedding": self.anchor_embedding,
                    "is_anchor": False,
                    "encoder_called": False,
                    "delta_magnitude": 0.0,
                    "appearance_delta": 0.0,
                    "cosine_sim": sim,
                    "frame_idx": frame_idx,
                    "yolo_skipped": True,
                    "objects": self._objects_from_graph(self.anchor_graph),
                }

        appearance_delta = (
            self._appearance_delta(self.prev_frame, frame)
            if self.prev_frame is not None else 0.0
        )

        # --- Real CLIP encode (full frame + object crops) ---
        graph, embedding = self.anchor_proc.process(frame)

        # Delta magnitude vs previous graph — kept for downstream analytics only
        delta_magnitude = 0.0
        if self.anchor_graph is not None:
            try:
                delta_magnitude = float(self.anchor_graph.compute_delta(graph).get("total_magnitude", 0.0))
            except Exception:
                delta_magnitude = 0.0

        self.anchor_graph = graph
        self.anchor_embedding = embedding
        self.prev_frame = frame.copy()
        self.frames_since_anchor = 0

        sim = self.validator.log(
            frame_idx=frame_idx,
            reconstructed=embedding,
            ground_truth=None if no_validation else embedding,
            is_anchor=True,
            delta_magnitude=delta_magnitude,
            encoder_called=True,
        )

        return {
            "embedding": embedding,
            "is_anchor": True,
            "encoder_called": True,
            "delta_magnitude": delta_magnitude,
            "appearance_delta": appearance_delta,
            "cosine_sim": sim,
            "frame_idx": frame_idx,
            "objects": self._objects_from_graph(graph),
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
        print(f"  ADVE Pipeline (CLIP-only)  |  Device: {self.config.DEVICE.upper()}")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Frames: {total_vid_frames}  |  FPS: {total_fps:.1f}")
        print(f"{'='*55}")

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.validator.records = []
        self.reset()
        frame_idx = 0
        t_start = time.time()

        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                break
            try:
                ret, frame = cap.read()
            except Exception:
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
                tag = "ENCODE" if res["is_anchor"] else "REUSE "
                print(
                    f"  [{frame_idx:>5}] {tag}  "
                    f"sim={res['cosine_sim']:.4f}  dG={res['delta_magnitude']:.4f}  "
                    f"dapp={res['appearance_delta']:.4f}"
                )

            if frame_idx % 30 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            frame_idx += 1

        cap.release()
        elapsed = time.time() - t_start

        summary = self.validator.summarize()
        summary["elapsed_sec"] = round(elapsed, 2)
        summary["effective_fps"] = round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0

        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        try:
            self.validator.plot(os.path.join(self.config.OUTPUT_DIR, "adve_results.png"))
            self.validator.save_json(os.path.join(self.config.OUTPUT_DIR, "adve_results.json"))
        except Exception as e:
            print(f"[ADVEPipeline] Notice: Could not save validation plot/json ({e})")

        self._print_summary(summary)
        return summary

    def _print_summary(self, s: dict) -> None:
        print(f"\n{'='*55}")
        print(f"  ADVE Pipeline — CLIP-only run")
        print(f"{'='*55}")
        print(f"  Total frames         : {s.get('total_frames')}")
        print(f"  Encoder calls        : {s.get('encoder_calls')}")
        print(f"  Reused (static)      : {s.get('delta_frames')}")
        print(f"  Encoder savings      : {s.get('encoder_savings_pct')}%")
        print(f"  Effective FPS        : {s.get('effective_fps')}")
        print(f"{'='*55}\n")
