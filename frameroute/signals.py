"""
frameroute.signals — cheap per-frame change signals.

Design constraint that defines this module: every signal here must be
computable in single-digit milliseconds on CPU. No YOLO. No CLIP. No
detector of any kind in the hot path.

This is the correction to ADVE v1-v3. That pipeline spent ~120 ms of
YOLOv8 + ByteTrack on every frame in order to skip a CLIP pass costing a
few milliseconds (see docs/GPU_PERFORMANCE_CERTIFICATE.md: YOLO 120.33 ms,
reconstructor 1.43 ms). The shortcut cost more than the road, which is why
the measured wall-clock gain in results/traffic_benchmark_report.json was
5.3% against a claimed 60%.

Everything below runs on a 160x90 downscale. Measured cost is reported by
SignalExtractor.timing_ms() so the claim is never taken on faith.

Signals produced per frame
--------------------------
hist_delta_prev      HSV histogram distance vs previous frame
hist_delta_anchor    HSV histogram distance vs current anchor frame
edge_delta           relative change in Sobel edge density
struct_delta         mean abs difference of downscaled luma
global_motion_mag    magnitude of estimated camera motion (px, on small frame)
motion_residual      content change AFTER cancelling estimated camera motion
novelty              scalar fusion of the above, in [0, 1]

motion_residual is the important one. A panning camera moves every pixel
without changing what the scene contains; a person walking into a static
frame changes little globally but matters enormously. Subtracting the
estimated global affine before differencing separates those two cases,
which is the one idea from ADVE's ego-motion module worth keeping -- but
done with a partial affine on a 160x90 image instead of full ORB
homography on 720p.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class FrameSignals:
    """Cheap descriptors for one frame."""
    idx: int
    t: float

    hist_delta_prev: float = 0.0
    hist_delta_anchor: float = 0.0
    edge_delta: float = 0.0
    struct_delta: float = 0.0
    global_motion_mag: float = 0.0
    motion_residual: float = 0.0
    region_residual: float = 0.0  # largest motion-compensated change in ANY tile

    novelty: float = 0.0          # fused, in [0, 1]
    cost_ms: float = 0.0          # measured, per frame

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SignalTrack:
    """Per-frame signals for a whole video, plus the metadata to interpret them."""
    video_path: str
    fps: float
    n_frames_total: int
    stride: int
    signals: List[FrameSignals] = field(default_factory=list)
    analyze_seconds: float = 0.0

    @property
    def duration_sec(self) -> float:
        return self.n_frames_total / self.fps if self.fps > 0 else 0.0

    @property
    def n_analyzed(self) -> int:
        return len(self.signals)

    def novelty_array(self) -> np.ndarray:
        return np.asarray([s.novelty for s in self.signals], dtype=np.float64)

    def with_novelty(self, novelty) -> "SignalTrack":
        """
        Copy of this track with a replaced novelty array.

        Used by frameroute.semantics to fuse content-space novelty as a floor
        on top of the pixel signals without touching any other field.
        """
        nov = np.asarray(novelty, dtype=np.float64).reshape(-1)
        if nov.size != len(self.signals):
            raise ValueError(
                f"novelty has {nov.size} values for {len(self.signals)} signals"
            )
        return SignalTrack(
            video_path=self.video_path,
            fps=self.fps,
            n_frames_total=self.n_frames_total,
            stride=self.stride,
            signals=[replace(s, novelty=float(v)) for s, v in zip(self.signals, nov)],
            analyze_seconds=self.analyze_seconds,
        )

    def time_array(self) -> np.ndarray:
        return np.asarray([s.t for s in self.signals], dtype=np.float64)

    def mean_signal_cost_ms(self) -> float:
        if not self.signals:
            return 0.0
        return float(np.mean([s.cost_ms for s in self.signals]))

    def summary(self) -> Dict[str, float]:
        nov = self.novelty_array()
        return {
            "frames_analyzed": self.n_analyzed,
            "duration_sec": round(self.duration_sec, 2),
            "fps": round(self.fps, 3),
            "stride": self.stride,
            "analyze_seconds": round(self.analyze_seconds, 3),
            "analyze_fps": round(self.n_analyzed / self.analyze_seconds, 1)
            if self.analyze_seconds > 0 else 0.0,
            "mean_signal_cost_ms": round(self.mean_signal_cost_ms(), 3),
            "novelty_mean": round(float(nov.mean()), 4) if nov.size else 0.0,
            "novelty_p95": round(float(np.percentile(nov, 95)), 4) if nov.size else 0.0,
            "novelty_total": round(float(nov.sum()), 3),
        }


# --------------------------------------------------------------------------
# extractor
# --------------------------------------------------------------------------

class SignalExtractor:
    """
    Stateful per-frame signal extractor.

    Usage:
        ex = SignalExtractor()
        for idx, frame in enumerate(frames):
            sig = ex.step(frame, idx=idx, t=idx / fps)
        # ex.set_anchor(frame) whenever the downstream pipeline spends a call

    Weights are deliberately exposed. The right fusion differs by domain --
    a lecture cares about struct_delta (slide changes), a traffic camera
    cares about motion_residual. DomainFingerprinter in adve/core can pick
    a preset; see WEIGHT_PRESETS below.
    """

    WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
        # general purpose
        "default":     {"hist": 0.30, "edge": 0.15, "struct": 0.20, "residual": 0.35},
        # slides, screen recordings, talking-head lectures: content changes
        # are full-frame and abrupt; there is rarely any real camera motion
        "static_cam":  {"hist": 0.35, "edge": 0.25, "struct": 0.35, "residual": 0.05},
        # traffic / surveillance: fixed camera, objects move through
        "surveillance": {"hist": 0.20, "edge": 0.10, "struct": 0.15, "residual": 0.55},
        # handheld, drone, dashcam: heavy camera motion that must be cancelled
        "ego_motion":  {"hist": 0.15, "edge": 0.10, "struct": 0.10, "residual": 0.65},
    }

    def __init__(
        self,
        proc_width: int = 160,
        proc_height: int = 90,
        weights: str | Dict[str, float] = "default",
        estimate_camera_motion: bool = True,
        max_track_points: int = 120,
        ema_alpha: float = 0.05,
        region_grid: int = 4,
        region_gain: float = 0.6,
    ):
        self.pw, self.ph = int(proc_width), int(proc_height)
        self.estimate_camera_motion = bool(estimate_camera_motion)
        self.max_track_points = int(max_track_points)
        self.ema_alpha = float(ema_alpha)
        # Region-aware detection: split each frame into region_grid x region_grid
        # tiles and track the change in the BUSIEST tile, not the frame average.
        # A small object appearing in one corner barely moves the whole-frame
        # signal but spikes its own tile. region_gain scales how strongly a
        # full single-tile change lifts novelty. Set region_grid=1 to disable.
        self.region_grid = max(1, int(region_grid))
        self.region_gain = float(region_gain)
        self._region_mu, self._region_sd = 0.0, 1e-6
        self._n_region = 0
        # TODO(region-adaptive-gain): region-aware detection helps at TIGHT
        # budgets (measured: +0.125 recall at 15-25 calls on MOT17) but slightly
        # hurts at LOOSE budgets (-0.037 at 100 calls) where the extra
        # sensitivity picks up noise. A budget-aware gain would raise it when
        # skipping hard and lower it when not. Needs the region signal blended
        # at ROUTE time (budget known) rather than baked in here at analysis
        # time — store region_residual separately and combine in the router.
        # Low priority: the fixed default wins in the aggressive regime we
        # deploy in; only worth it when tuning to a specific customer budget.

        if isinstance(weights, str):
            if weights not in self.WEIGHT_PRESETS:
                raise ValueError(
                    f"unknown weight preset {weights!r}; "
                    f"choose from {sorted(self.WEIGHT_PRESETS)}"
                )
            self.weights = dict(self.WEIGHT_PRESETS[weights])
            self.preset_name = weights
        else:
            self.weights = dict(weights)
            self.preset_name = "custom"

        # running state
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_edge: float = 0.0
        self._anchor_hist: Optional[np.ndarray] = None
        self._anchor_gray: Optional[np.ndarray] = None

        # running scale estimates so novelty is comparable across videos
        # without a second pass; robust to outliers via EMA of mean+std
        self._resid_mu, self._resid_sd = 0.0, 1e-6
        self._n_seen = 0

        self._cost_samples: List[float] = []

    # -- public ------------------------------------------------------------

    def set_anchor(self, frame: np.ndarray) -> None:
        """Call whenever the downstream pipeline actually spends a model call."""
        small = self._small(frame)
        self._anchor_gray = self._gray(small)
        self._anchor_hist = self._hist(small)

    def step(self, frame: np.ndarray, idx: int, t: float) -> FrameSignals:
        t0 = time.perf_counter()
        sig = FrameSignals(idx=idx, t=t)

        if frame is None or frame.size == 0:
            sig.cost_ms = (time.perf_counter() - t0) * 1000.0
            return sig

        small = self._small(frame)
        gray = self._gray(small)
        hist = self._hist(small)
        edge = self._edge_density(gray)

        # --- appearance ---
        if self._prev_hist is not None:
            sig.hist_delta_prev = self._hist_distance(hist, self._prev_hist)
        if self._anchor_hist is not None:
            sig.hist_delta_anchor = self._hist_distance(hist, self._anchor_hist)

        # --- structure ---
        if self._prev_gray is not None:
            sig.struct_delta = float(
                np.mean(np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)))
            ) / 255.0

        # --- edges ---
        if self._prev_edge > 1e-8:
            sig.edge_delta = min(1.0, abs(edge - self._prev_edge) / (self._prev_edge + 1e-8))

        # --- camera motion and residual content change ---
        if self._prev_gray is not None:
            gm, resid, region_resid = self._motion(self._prev_gray, gray)
            sig.global_motion_mag = gm
            sig.motion_residual = resid
            sig.region_residual = region_resid
        elif self._prev_gray is None:
            sig.motion_residual = 0.0
            sig.region_residual = 0.0

        # --- fuse ---
        sig.novelty = self._fuse(sig)

        # --- roll state ---
        self._prev_gray = gray
        self._prev_hist = hist
        self._prev_edge = edge
        if self._anchor_hist is None:
            self._anchor_hist = hist
            self._anchor_gray = gray

        sig.cost_ms = (time.perf_counter() - t0) * 1000.0
        self._cost_samples.append(sig.cost_ms)
        return sig

    def timing_ms(self) -> Dict[str, float]:
        """Measured cost of this module. Report this, never a FLOP estimate."""
        if not self._cost_samples:
            return {"n": 0}
        a = np.asarray(self._cost_samples)
        return {
            "n": int(a.size),
            "mean_ms": round(float(a.mean()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "max_ms": round(float(a.max()), 3),
        }

    # -- internals ---------------------------------------------------------

    def _small(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self.pw, self.ph), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _gray(small: np.ndarray) -> np.ndarray:
        if small.ndim == 3 and small.shape[2] == 3:
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return small if small.ndim == 2 else small[:, :, 0]

    @staticmethod
    def _hist(small: np.ndarray) -> np.ndarray:
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h

    @staticmethod
    def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
        # correlation in [-1, 1]; convert to a distance in [0, 1]
        corr = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
        return float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(gx * gx + gy * gy))) / 255.0

    def _region_max(self, core: np.ndarray) -> float:
        """
        Largest mean absolute difference in any grid tile of the (already
        motion-compensated, border-trimmed) diff, in [0,1]. This is what makes
        a small localized change visible: the frame average may be tiny while
        one tile is large. Cheap -- G*G means on a ~150x86 array.
        """
        g = self.region_grid
        if g <= 1 or core.size == 0:
            return float(np.mean(core)) / 255.0
        h, w = core.shape[:2]
        rs = np.linspace(0, h, g + 1, dtype=int)
        cs = np.linspace(0, w, g + 1, dtype=int)
        best = 0.0
        for i in range(g):
            for j in range(g):
                tile = core[rs[i]:rs[i + 1], cs[j]:cs[j + 1]]
                if tile.size:
                    m = float(np.mean(tile))
                    if m > best:
                        best = m
        return best / 255.0

    def _motion(self, prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (global_motion_magnitude_px, motion_residual_0_1, region_residual_0_1).

        Estimates a partial affine (translation + rotation + uniform scale)
        between the two small frames using sparse LK flow, warps the previous
        frame by it, and measures what is left over. What is left over is
        content change that camera motion does not explain. region_residual is
        the same leftover, but measured in the single busiest tile rather than
        averaged over the whole frame.
        """
        if not self.estimate_camera_motion:
            diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
            resid = float(np.mean(diff)) / 255.0
            return 0.0, resid, self._region_max(diff)

        global_mag = 0.0
        warped = None

        try:
            p0 = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=self.max_track_points,
                qualityLevel=0.01, minDistance=4, blockSize=5,
            )
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, p0, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
                )
                if p1 is not None and st is not None:
                    good0 = p0[st.ravel() == 1].reshape(-1, 2)
                    good1 = p1[st.ravel() == 1].reshape(-1, 2)
                    if len(good0) >= 6:
                        disp = np.linalg.norm(good1 - good0, axis=1)
                        global_mag = float(np.median(disp))
                        M, inliers = cv2.estimateAffinePartial2D(
                            good0, good1, method=cv2.RANSAC,
                            ransacReprojThreshold=2.0, maxIters=200,
                        )
                        if M is not None:
                            warped = cv2.warpAffine(
                                prev_gray, M, (gray.shape[1], gray.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE,
                            )
        except cv2.error:
            warped = None

        base = warped if warped is not None else prev_gray
        diff = np.abs(gray.astype(np.float32) - base.astype(np.float32))

        # trim the border: warping introduces edge artifacts that are not
        # content change and would otherwise inflate the residual on every pan
        m = max(2, min(gray.shape) // 12)
        core = diff[m:-m, m:-m] if diff.shape[0] > 2 * m and diff.shape[1] > 2 * m else diff
        resid = float(np.mean(core)) / 255.0
        region_resid = self._region_max(core)

        return global_mag, resid, region_resid

    def _fuse(self, sig: FrameSignals) -> float:
        w = self.weights
        raw = (
            w.get("hist", 0.0) * sig.hist_delta_prev
            + w.get("edge", 0.0) * sig.edge_delta
            + w.get("struct", 0.0) * sig.struct_delta
            + w.get("residual", 0.0) * self._standardize_residual(sig.motion_residual)
        )
        base = float(np.clip(raw, 0.0, 1.0))

        # Region-aware floor: a strong change confined to one tile lifts
        # novelty even when the whole-frame fusion stays low. Uses max(), so
        # it can only INCREASE sensitivity -- we never skip a frame we would
        # have caught before, we only catch additional localized events.
        if self.region_grid > 1:
            region = self.region_gain * self._standardize_region(sig.region_residual)
            return float(np.clip(max(base, region), 0.0, 1.0))
        return base

    # A residual below this is indistinguishable from sensor noise, whatever
    # the running statistics say. Expressed as mean absolute luma difference
    # in [0,1]; 0.01 is roughly 2.5 grey levels out of 255.
    ABS_NOISE_FLOOR = 0.01

    def _standardize_residual(self, r: float) -> float:
        """
        Residual magnitude is scene-dependent (a noisy night camera has a
        high floor, a locked-off studio camera has almost none). Rescale
        against a running estimate so novelty means 'unusual for this
        stream' rather than 'large in absolute pixels'.

        The absolute gate below is not optional. On a perfectly static
        scene the running std collapses toward zero, so the first bit of
        sensor noise produces an enormous z-score and the router reads it
        as a major event. Requiring the change to be BOTH statistically
        unusual AND above an absolute noise floor is what stops an empty
        corridor at 3 a.m. from burning the whole call budget.
        """
        self._n_seen += 1
        a = self.ema_alpha if self._n_seen > 30 else 0.2
        self._resid_mu = (1 - a) * self._resid_mu + a * r
        dev = abs(r - self._resid_mu)
        self._resid_sd = (1 - a) * self._resid_sd + a * dev

        # relative: how unusual is this for the stream? 0 at the mean, 1 at +3σ
        sd = max(self._resid_sd, 1e-4)
        z = (r - self._resid_mu) / sd
        relative = float(np.clip(z / 3.0, 0.0, 1.0)) if z > 0 else 0.0

        # absolute: is the change big enough to be real at all?
        absolute = float(np.clip(r / self.ABS_NOISE_FLOOR, 0.0, 1.0))

        return relative * absolute

    def _standardize_region(self, r: float) -> float:
        """
        Same relative-times-absolute standardization as the global residual,
        but with the tile signal's own running statistics. A single tile is
        noisier than the frame average, so the absolute noise gate matters
        even more here: it is what stops one flickering tile of a static
        night scene from firing a false event.
        """
        self._n_region += 1
        a = self.ema_alpha if self._n_region > 30 else 0.2
        self._region_mu = (1 - a) * self._region_mu + a * r
        dev = abs(r - self._region_mu)
        self._region_sd = (1 - a) * self._region_sd + a * dev

        sd = max(self._region_sd, 1e-4)
        z = (r - self._region_mu) / sd
        relative = float(np.clip(z / 3.0, 0.0, 1.0)) if z > 0 else 0.0
        absolute = float(np.clip(r / self.ABS_NOISE_FLOOR, 0.0, 1.0))
        return relative * absolute


# --------------------------------------------------------------------------
# whole-video convenience
# --------------------------------------------------------------------------

def analyze_video(
    video_path: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
    weights: str | Dict[str, float] = "default",
    proc_size: Tuple[int, int] = (160, 90),
    progress: bool = False,
    region_grid: int = 4,
    region_gain: float = 0.6,
) -> SignalTrack:
    """
    One decode pass over the video producing a SignalTrack.

    stride > 1 subsamples the analysis itself. Note that this is a different
    knob from the routing budget: stride controls how finely we *look*,
    budget controls how often we *spend*. Looking is cheap; spending is not.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ex = SignalExtractor(
        proc_width=proc_size[0], proc_height=proc_size[1], weights=weights,
        region_grid=region_grid, region_gain=region_gain,
    )
    track = SignalTrack(
        video_path=video_path, fps=fps, n_frames_total=n_total, stride=stride
    )

    t_start = time.perf_counter()
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            track.signals.append(ex.step(frame, idx=idx, t=idx / fps))
            kept += 1
            if progress and kept % 500 == 0:
                print(f"  signals: {kept} frames analyzed")
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1
    cap.release()

    track.analyze_seconds = time.perf_counter() - t_start
    if track.n_frames_total <= 0:
        track.n_frames_total = idx
    return track
