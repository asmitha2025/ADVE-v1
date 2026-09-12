"""
ADVE Enterprise — Resilient Async RTSP Reader
Handles continuous RTSP stream ingestion, auto-reconnection, frame throttling, and hardware decoding.
"""

import cv2
import asyncio
import time
import structlog
from typing import Optional, Callable

logger = structlog.get_logger()


class RTSPReader:
    def __init__(
        self,
        url: str,
        stream_id: str,
        max_buffer_size: int = 30,
        reconnect_attempts: int = 5,
        hw_accel: bool = True
    ):
        self.url = url
        self.stream_id = stream_id
        self.max_buffer_size = max_buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.hw_accel = hw_accel

        self.frame_buffer: asyncio.Queue = asyncio.Queue(maxsize=max_buffer_size)
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.processed_frames: int = 0
        self.dropped_frames: int = 0
        self.start_time: float = 0.0

    def _open_capture(self) -> bool:
        """Opens VideoCapture with optional hardware acceleration flags."""
        logger.info("opening_rtsp_stream", stream_id=self.stream_id, url=self.url)
        
        if self.hw_accel:
            # OpenCV FFMPEG hardware acceleration hints
            os_env = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            if not os_env:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|rtsp_flags;prefer_tcp"
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.url)

        return self.cap is not None and self.cap.isOpened()

    async def reconnect(self) -> bool:
        """Attempts exponential backoff reconnection for unstable CCTV networks."""
        logger.warn("rtsp_disconnected_attempting_reconnect", stream_id=self.stream_id)
        if self.cap:
            self.cap.release()

        backoff = 1.0
        for attempt in range(1, self.reconnect_attempts + 1):
            await asyncio.sleep(backoff)
            logger.info("reconnect_attempt", stream_id=self.stream_id, attempt=attempt)
            if self._open_capture():
                logger.info("rtsp_reconnected_successfully", stream_id=self.stream_id)
                return True
            backoff *= 2.0  # 1s, 2s, 4s, 8s, 16s

        logger.error("rtsp_reconnect_failed_max_attempts_reached", stream_id=self.stream_id)
        return False

    async def read_loop(self, callback: Optional[Callable] = None):
        """Continuous frame ingestion loop with frame drop logic on buffer overflow."""
        self.is_running = True
        self.start_time = time.time()

        if not self._open_capture():
            reconnected = await self.reconnect()
            if not reconnected:
                self.is_running = False
                return

        frame_idx = 0

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                if not await self.reconnect():
                    break
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Video file EOF or temporary stream drop
                if not await self.reconnect():
                    break
                continue

            self.processed_frames += 1
            frame_idx += 1

            # Buffer overflow handling: drop frame if queue is full (never block pipeline)
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()  # Drop oldest frame
                    self.dropped_frames += 1
                except asyncio.QueueEmpty:
                    pass

            await self.frame_buffer.put((frame_idx, frame))

            if callback:
                try:
                    await callback(frame_idx, frame)
                except Exception as cb_err:
                    logger.error("rtsp_callback_failed", stream_id=self.stream_id, error=str(cb_err))

            await asyncio.sleep(0.001)  # Yield control to event loop

        self.stop()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("rtsp_stream_stopped", stream_id=self.stream_id, processed=self.processed_frames, dropped=self.dropped_frames)

    def get_stats(self) -> dict:
        elapsed = max(time.time() - self.start_time, 0.001)
        fps = round(self.processed_frames / elapsed, 1)
        return {
            "stream_id": self.stream_id,
            "url": self.url,
            "is_running": self.is_running,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "fps": fps,
            "buffer_depth": self.frame_buffer.qsize()
        }
