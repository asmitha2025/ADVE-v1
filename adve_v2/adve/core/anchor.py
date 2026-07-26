import numpy as np
import cv2
import torch
import clip
from PIL import Image
from ultralytics import YOLO
from typing import Tuple

from adve.core.config import Config
from adve.core.spatial_graph import SpatialGraph, ObjectState


class AnchorProcessor:
    """
    Processes anchor (keyframe) frames.
    Runs full CLIP on the whole frame AND on each detected object's RoI.
    Builds a SpatialGraph with per-object embeddings.
    """

    def __init__(self, config: Config, yolo: YOLO = None, clip_model=None, clip_preprocess=None):
        self.config = config
        self.device = getattr(config, "CLIP_DEVICE", config.DEVICE)
        self.yolo_device = getattr(config, "YOLO_DEVICE", config.DEVICE)
        self._yolo   = yolo  # Shared YOLO instance (preserves ByteTrack ID state)
        self._clip_model = clip_model
        self._clip_preprocess = clip_preprocess
        self._clip_dim = 512

    @property
    def clip_model(self):
        if self._clip_model is None:
            import torch
            device = self.device
            if not torch.cuda.is_available() and device == "cuda":
                device = "cpu"
            from adve.core.clip_loader import load_clip_cached
            print(f"[AnchorProcessor] Loading CLIP model '{self.config.CLIP_MODEL}' on {device}...")
            self._clip_model, self._clip_preprocess = load_clip_cached(
                self.config.CLIP_MODEL, device=device
            )
            self._clip_dim = self._clip_model.visual.output_dim
        return self._clip_model

    @property
    def clip_preprocess(self):
        if self._clip_preprocess is None:
            # Trigger loading clip_model
            _ = self.clip_model
        return self._clip_preprocess

    @property
    def clip_dim(self):
        if self._clip_model is None:
            return 512
        return self._clip_model.visual.output_dim

    @clip_dim.setter
    def clip_dim(self, val):
        self._clip_dim = val

    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            import torch
            device = self.yolo_device
            if not torch.cuda.is_available() and device == "cuda":
                device = "cpu"
            print(f"[AnchorProcessor] Loading YOLO model '{self.config.YOLO_MODEL}' on {device}...")
            self._yolo = YOLO(self.config.YOLO_MODEL)
            self._yolo.to(device)
            if device != "cpu":
                self._yolo.model.half()
        return self._yolo

    @yolo.setter
    def yolo(self, val):
        self._yolo = val

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Tuple[SpatialGraph, np.ndarray]:
        """
        Parameters
        ----------
        frame : BGR numpy array

        Returns
        -------
        graph             : SpatialGraph with embeddings on each ObjectState
        frame_embedding   : clip_dim-d CLIP embedding of full frame (normalised)
        """
        # Run YOLO first so we can batch the full frame + object crops
        imgsz = getattr(self.config, "YOLO_IMGSZ", 320)
        results = self.yolo.track(frame, imgsz=imgsz, persist=True, verbose=False, device=self.yolo_device)[0]
        
        # Collect candidate boxes to embed (max 6 to prevent CPU spikes / large batch sizes)
        candidate_boxes = []
        if results.boxes is not None and len(results.boxes):
            # Sort boxes by area to prioritize larger objects if there are many
            sorted_boxes = []
            for box in results.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                sorted_boxes.append((area, box))
            
            sorted_boxes.sort(key=lambda x: x[0], reverse=True)
            candidate_boxes = [box for _, box in sorted_boxes[:6]]

        # Prepare batch list: index 0 is full frame, followed by ROIs
        frames_to_embed = [frame]
        roi_infos = [] # list of dicts: (obj_id, class_name, bbox, roi_small_for_hist)

        for box in candidate_boxes:
            obj_id     = int(box.id[0])
            class_name = self.yolo.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Clamp to frame bounds
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(frame.shape[1], x2);  y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            frames_to_embed.append(roi)
            
            # Prepare hist BGR image for quick calcHist later
            roi_small = roi
            rh, rw = roi.shape[:2]
            if rh > 64 or rw > 64:
                roi_small = cv2.resize(roi, (64, 64))
            
            roi_infos.append({
                "obj_id": obj_id,
                "class_name": class_name,
                "bbox": (x1, y1, x2, y2),
                "roi_small": roi_small,
                "area": float((x2 - x1) * (y2 - y1)),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            })

        # Run a single batch CLIP forward pass!
        embeddings = self._embed_batch(frames_to_embed)
        
        # Split results
        frame_embedding = embeddings[0]
        crop_embeddings = embeddings[1:]

        graph = SpatialGraph()
        for info, obj_embedding in zip(roi_infos, crop_embeddings):
            # Compute histogram for appearance check (Improvement 6)
            hist = cv2.calcHist([info["roi_small"]], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            graph.add_object(ObjectState(
                obj_id=info["obj_id"],
                class_name=info["class_name"],
                bbox=info["bbox"],
                center=info["center"],
                area=info["area"],
                embedding=obj_embedding,
                appearance_hist=hist,
            ))

        graph.build_relations(frame.shape[1], frame.shape[0])
        return graph, frame_embedding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def embed_frame(self, frame: np.ndarray) -> np.ndarray:
        """Public method used by Validator for ground-truth computation."""
        return self._embed(frame)

    def _embed_object(self, roi: np.ndarray) -> np.ndarray:
        """Embed an object crop using the standard CLIP visual encoder."""
        return self._embed(roi)

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros(self.clip_dim, dtype=np.float32)

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():
            tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
            emb    = self.clip_model.encode_image(tensor)
            emb    = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().astype(np.float32)

    def _embed_batch(self, frames: list) -> list:
        if not frames:
            return []

        preprocess = self.clip_preprocess
        tensors = []
        for f in frames:
            if f is None or f.size == 0:
                tensors.append(torch.zeros(3, 224, 224))
            else:
                rgb     = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensors.append(preprocess(pil_img))

        with torch.no_grad():
            model_device = next(self.clip_model.parameters()).device
            batch_tensor = torch.stack(tensors).to(model_device)
            embs    = self.clip_model.encode_image(batch_tensor)
            embs    = embs / embs.norm(dim=-1, keepdim=True)

        return [emb.cpu().numpy().flatten().astype(np.float32) for emb in embs]
