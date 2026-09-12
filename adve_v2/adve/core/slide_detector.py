"""
ADVE Slide Transition Detector
===============================
Detects rapid full-frame content changes (slide transitions, scene cuts)
by comparing current frame against the ANCHOR frame (not just prev_frame).

This catches education video slide changes that the standard appearance_delta
(prev_frame vs current_frame) misses because gradual drift accumulates.
"""

import cv2
import numpy as np


class SlideTransitionDetector:
    """
    Detects slide transitions by computing structural similarity
    between the current frame and the anchor frame.

    When a slide transition is detected, forces immediate anchor refresh
    BEFORE any delta reconstruction attempt — preventing garbage embeddings.
    """

    def __init__(self, threshold: float = 0.06, hist_threshold: float = 0.40):
        self.threshold = threshold
        self.hist_threshold = hist_threshold
        self._anchor_gray = None
        self._anchor_hist = None

    def set_anchor(self, anchor_frame: np.ndarray):
        """Cache grayscale + histogram of anchor frame for fast comparison."""
        try:
            if anchor_frame is None or anchor_frame.size == 0:
                return
            small = cv2.resize(anchor_frame, (160, 90))
            if len(small.shape) == 3 and small.shape[2] == 3:
                self._anchor_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            else:
                self._anchor_gray = small.copy()
                hsv = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
            
            self._anchor_hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(self._anchor_hist, self._anchor_hist)
        except Exception:
            self._anchor_gray = None
            self._anchor_hist = None

    def check(self, current_frame: np.ndarray) -> tuple:
        """
        Compare current frame against the anchor frame.

        Returns:
            (is_slide_transition: bool, structural_diff: float, hist_corr: float)
        """
        if self._anchor_gray is None or current_frame is None or current_frame.size == 0:
            return False, 0.0, 1.0

        try:
            small = cv2.resize(current_frame, (160, 90))
            if len(small.shape) == 3 and small.shape[2] == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            else:
                gray = small.copy()
                hsv = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

            # 1. Structural grayscale difference (anchor vs current)
            structural_diff = float(np.mean(cv2.absdiff(self._anchor_gray, gray))) / 255.0

            # 2. Histogram correlation (anchor vs current)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            hist_corr = float(cv2.compareHist(self._anchor_hist, hist, cv2.HISTCMP_CORREL))

            # Slide transition = high structural diff AND low histogram correlation
            is_transition = (
                structural_diff > self.threshold and
                hist_corr < self.hist_threshold
            )

            return is_transition, structural_diff, hist_corr
        except Exception:
            return False, 0.0, 1.0
