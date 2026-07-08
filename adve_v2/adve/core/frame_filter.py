import cv2
import numpy as np

class FrameFilter:
    """
    Cheap motion pre-filter using frame differencing.
    Skips YOLO entirely on frames where nothing moved.
    Cost: ~1ms per frame (vs 8-280ms for YOLO)
    """

    def __init__(self, motion_threshold: float = 0.02):
        self.threshold  = motion_threshold
        self.prev_gray  = None

    def has_motion(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Returns (has_motion, motion_score).
        If has_motion is False, skip YOLO — nothing changed.
        """
        h, w = frame.shape[:2]
        scale = 128.0 / max(h, w)
        if scale < 1.0:
            frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            frame_small = frame

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True, 1.0

        diff  = cv2.absdiff(self.prev_gray, gray)
        score = float(diff.mean()) / 255.0

        self.prev_gray = gray
        return score > self.threshold, score
