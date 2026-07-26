import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class EgoMotionEstimator:
    """
    Phase 3 Enterprise Fix: Ego-Motion Compensation.
    Estimates 3x3 Homography matrix between anchor frame and current frame using ORB feature matching.
    Warp object centroids back to anchor coordinate frame to eliminate camera pan/tilt artifacts.
    """

    def __init__(self, max_features: int = 500, match_threshold: int = 10):
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.anchor_kp = None
        self.anchor_des = None

    def set_anchor_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self.anchor_kp, self.anchor_des = self.orb.detectAndCompute(gray, None)

    def estimate_homography(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        if self.anchor_des is None or len(self.anchor_kp) < 10:
            return None

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if current_frame.ndim == 3 else current_frame
        kp2, des2 = self.orb.detectAndCompute(gray, None)

        if des2 is None or len(kp2) < 10:
            return None

        matches = self.bf.match(self.anchor_des, des2)
        if len(matches) < self.match_threshold:
            return None

        src_pts = np.float32([self.anchor_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return H

    def warp_centroids(self, objects: Dict[str, Any], H: Optional[np.ndarray]) -> Dict[str, Any]:
        """Apply inverse homography warp to bounding box centroids to cancel ego-motion."""
        if H is None or not objects:
            return objects

        for obj_id, obj in objects.items():
            bbox = getattr(obj, "bbox", None)
            if bbox is None:
                continue

            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            pt = np.array([cx, cy, 1.0], dtype=np.float32).reshape(3, 1)
            warped_pt = H @ pt

            if abs(warped_pt[2, 0]) > 1e-6:
                wcx = warped_pt[0, 0] / warped_pt[2, 0]
                wcy = warped_pt[1, 0] / warped_pt[2, 0]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                # Update object bbox to ego-motion compensated coordinates
                obj.bbox = [int(wcx - w / 2), int(wcy - h / 2), int(wcx + w / 2), int(wcy + h / 2)]

        return objects
