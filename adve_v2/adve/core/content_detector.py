"""
ADVE Domain Fingerprinter & Router Module (Critical Item #1)

Analyzes initial video frames to extract heuristic visual & motion features:
 - Brightness Histogram: Detects low-light / IR night vision
 - Frame Difference Variance: Distinguishes static CCTV vs steady Traffic vs high-motion Action
 - Edge Density: Measures spatial complexity (sparse indoor vs dense urban traffic)

Routes streams to 5 domain profiles:
 1. 'night'     - Night vision / low-light surveillance
 2. 'traffic'   - Urban traffic / vehicles / highway surveillance
 3. 'sparse'    - Indoor office / warehouse / retail showroom
 4. 'action'    - High-speed sports / dynamic action
 5. 'education' - Lectures / presentations / slides with sharp transitions
"""

import cv2
import numpy as np
from typing import Dict, Any

class DomainFingerprinter:
    def __init__(self, sample_frames: int = 60):
        self.sample_frames = sample_frames

    def fingerprint_and_route(self, video_path: str) -> Dict[str, Any]:
        """
        Analyzes initial frames and returns the routed domain profile along with configuration thresholds.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Fallback profile
            return self._get_profile("sparse", 0.0, 0.0, 0.0)

        brightness_list = []
        motion_diffs = []
        edge_densities = []
        prev_gray = None
        count = 0

        # Skip intro/title frames which are often dark/blank
        skip_intro = 300
        total_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_available > skip_intro + self.sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_intro)

        while cap.isOpened() and count < self.sample_frames:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # 1. Brightness feature
            mean_bright = float(np.mean(gray))
            brightness_list.append(mean_bright)

            # 2. Edge density feature (Canny edge count ratio)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(edges) / (320 * 180))
            edge_densities.append(edge_density)

            # 3. Motion variance feature
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_diffs.append(float(np.mean(diff)))

            prev_gray = gray
            count += 1

        cap.release()

        avg_brightness = float(np.mean(brightness_list)) if brightness_list else 100.0
        avg_motion = float(np.mean(motion_diffs)) if motion_diffs else 5.0
        avg_edge_density = float(np.mean(edge_densities)) if edge_densities else 0.05

        # Routing Logic
        if avg_brightness < 45.0:
            # Low light or Night vision
            domain = "night"
        elif avg_motion > 15.0:
            # High speed sports or dynamic panning
            domain = "action"
        elif avg_edge_density > 0.08 and avg_motion > 3.0:
            # High spatial complexity + moderate motion = Traffic / Highway
            domain = "traffic"
        elif avg_motion < 3.0 and avg_edge_density < 0.06 and avg_brightness > 80.0:
            # Low motion + low edges + well-lit = Education / Lecture / Presentation
            domain = "education"
        else:
            # Low motion / clean background = Sparse indoor / retail / office
            domain = "sparse"

        return self._get_profile(domain, avg_brightness, avg_motion, avg_edge_density)

    def _get_profile(self, domain: str, brightness: float, motion: float, edge_density: float) -> Dict[str, Any]:
        profiles = {
            "night": {
                "domain": "night",
                "reconstructor_head": "night_head",
                "spatial_threshold": 0.40,
                "max_delta_frames": 22,
                "ema_alpha": 0.80,
                "description": "Low-light / IR Night Vision Profile"
            },
            "traffic": {
                "domain": "traffic",
                "reconstructor_head": "traffic_head",
                "spatial_threshold": 0.38,
                "max_delta_frames": 20,
                "ema_alpha": 0.75,
                "description": "Urban Traffic & Highway CCTV Profile"
            },
            "sparse": {
                "domain": "sparse",
                "reconstructor_head": "sparse_head",
                "spatial_threshold": 0.48,
                "max_delta_frames": 30,
                "ema_alpha": 0.85,
                "description": "Indoor Retail / Warehouse / Office Profile"
            },
            "action": {
                "domain": "action",
                "reconstructor_head": "action_head",
                "spatial_threshold": 0.22,
                "max_delta_frames": 12,
                "ema_alpha": 0.65,
                "description": "High-Motion Sports & Dynamic Action Profile"
            },
            "education": {
                "domain": "education",
                "reconstructor_head": "education_head",
                "spatial_threshold": 0.22,
                "appearance_threshold": 0.05,
                "max_delta_frames": 10,
                "ema_alpha": 0.75,
                "description": "Lecture / Presentation / Slide-Transition Profile"
            }
        }

        profile = profiles.get(domain, profiles["sparse"])
        profile["features"] = {
            "avg_brightness": round(brightness, 2),
            "avg_motion": round(motion, 2),
            "avg_edge_density": round(edge_density, 4)
        }
        return profile


# Backward compatibility wrapper for ContentDetector
class ContentDetector:
    def __init__(self, sample_frames: int = 60):
        self.fingerprinter = DomainFingerprinter(sample_frames)

    def detect_and_configure(self, video_path: str) -> Dict[str, Any]:
        return self.fingerprinter.fingerprint_and_route(video_path)
