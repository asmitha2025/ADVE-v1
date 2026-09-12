"""
ADVE Enterprise — Input Sanitizer & Cloud Storage Gateway
Validates and parses video sources (.mp4, .avi, .mkv, .mov, RTSP, S3, GCS).
"""

import os
import re
import cv2
import numpy as np
import structlog
from typing import Tuple, Dict, Any, Optional

logger = structlog.get_logger()


def parse_video_source(source_path: str) -> Dict[str, Any]:
    """
    Parses source protocol and returns normalized path or temporary local cache target.
    Supports local files, RTSP streams, http(s) URLs, s3:// and gs:// paths.
    """
    source_type = "local"
    if source_path.startswith("rtsp://") or source_path.startswith("rtsps://"):
        source_type = "rtsp"
    elif source_path.startswith("s3://"):
        source_type = "s3"
    elif source_path.startswith("gs://"):
        source_type = "gcs"
    elif source_path.startswith("http://") or source_path.startswith("https://"):
        source_type = "http"

    valid_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".m4v")
    is_supported_extension = any(source_path.lower().endswith(ext) for ext in valid_extensions)

    return {
        "source": source_path,
        "type": source_type,
        "is_stream": source_type in ("rtsp", "http"),
        "is_cloud_storage": source_type in ("s3", "gcs"),
        "is_valid_format": is_supported_extension or source_type in ("rtsp", "http", "s3", "gcs")
    }


def sanitize_frame(frame: np.ndarray, target_size: Optional[Tuple[int, int]] = (1280, 720), force_resize: bool = False) -> np.ndarray:
    """
    Sanitizes, checks, and resizes input frames to pipeline-friendly standard resolution.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty or invalid frame input.")

    # 4-channel BGRA -> 3-channel BGR
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Grayscale -> 3-channel BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Resize if target_size specified and dimensions differ
    if target_size is not None:
        h, w = frame.shape[:2]
        if (w, h) != target_size or force_resize:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    return frame
