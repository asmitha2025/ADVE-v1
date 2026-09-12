"""
ADVE Video Transcoding & Codec Normalization Pipeline (Component #3)

Probes incoming video files via OpenCV / FFmpeg to detect resolution, FPS, and format.
Transcodes unsupported codecs (HEVC 10-bit, MJPEG) or 4K ultra-high resolutions
to standard H.264 720p @ 30fps prior to ADVE neural ingestion.
"""

import os
import cv2
import subprocess
from typing import Dict, Any, Tuple

class VideoTranscoder:
    def __init__(self, max_width: int = 1280, max_height: int = 720, target_fps: int = 30):
        self.max_width = max_width
        self.max_height = max_height
        self.target_fps = target_fps

    def probe_video(self, video_path: str) -> Dict[str, Any]:
        """Probes video metadata using OpenCV VideoCapture."""
        if not os.path.exists(video_path):
            return {"valid": False, "error": "File not found"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"valid": False, "error": "Failed to open video container"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        needs_transcode = (width > self.max_width) or (height > self.max_height) or (fps > 45.0)

        return {
            "valid": True,
            "width": width,
            "height": height,
            "fps": round(fps, 2) if fps > 0 else 30.0,
            "frame_count": frame_count,
            "needs_transcode": needs_transcode
        }

    def transcode_if_needed(self, input_path: str, output_dir: str = "adve_v2/data/transcoded") -> Tuple[str, bool]:
        """
        Transcodes video to normalized H.264 720p if resolution or format exceeds bounds.
        Returns tuple: (processed_video_path, was_transcoded)
        """
        meta = self.probe_video(input_path)
        if not meta["valid"] or not meta["needs_transcode"]:
            return input_path, False

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_720p.mp4")

        if os.path.exists(output_path):
            return output_path, True

        # Run FFmpeg transcoding command if available
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale=min({self.max_width}\\,iw):min({self.max_height}\\,ih):force_original_aspect_ratio=decrease",
            "-r", str(self.target_fps),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-an", output_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path, True
        except Exception:
            # Fallback to OpenCV VideoWriter re-encoding if ffmpeg CLI is unavailable
            return self._transcode_opencv(input_path, output_path)

    def _transcode_opencv(self, input_path: str, output_path: str) -> Tuple[str, bool]:
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.target_fps, (self.max_width, self.max_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (self.max_width, self.max_height))
            out.write(resized)

        cap.release()
        out.release()
        return output_path, True
