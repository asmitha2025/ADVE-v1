from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SportEvent:
    timestamp:   float
    event_type:  str     # GOAL, FOUL, CORNER, FREEKICK, CARD, SUB
    confidence:  float
    description: str
    delta_mag:   float


class SportsEventDetector:
    """
    Detects match events from ADVE's ΔG timeline.
    No extra model needed. Pure spatial geometry + thresholds.

    The key insight:
    Every significant match event causes a significant scene change.
    Scene change = ΔG spike.
    The TYPE of spike tells you WHAT happened.
    """

    THRESHOLDS = {
        "GOAL":         0.65,  # massive formation collapse
        "RED_CARD":     0.55,  # player leaves + shape rebalances
        "FOUL":         0.40,  # local collision + stop
        "CORNER":       0.35,  # ball reset + players cluster
        "SUBSTITUTION": 0.30,  # player swap at edge of pitch
        "FORMATION":    0.25,  # tactical shape change
    }

    def __init__(self, audio_segments=None):
        self.audio = audio_segments or []

    def detect_from_timeline(
        self,
        delta_timeline: List[dict],
        player_counts:  List[dict] = None,
    ) -> List[SportEvent]:
        """
        delta_timeline: [{"timestamp": float, "magnitude": float}, ...]
        player_counts:  [{"timestamp": float, "team1": int, "team2": int}, ...]
        """
        events = []
        if not delta_timeline:
            return events

        # Smooth the delta signal (5-frame rolling average)
        mags  = [d["magnitude"] for d in delta_timeline]
        times = [d["timestamp"] for d in delta_timeline]
        smoothed = self._smooth(mags, window=5)

        # Find peaks (local maxima above any threshold)
        peaks = self._find_peaks(smoothed, min_height=0.25, min_distance_sec=15.0, times=times)

        for peak_idx, peak_time, peak_mag in peaks:
            # Classify by magnitude + context
            event_type, confidence, desc = self._classify_peak(
                peak_mag, peak_time, peak_idx, smoothed, times, player_counts
            )

            if event_type:
                events.append(SportEvent(
                    timestamp   = peak_time,
                    event_type  = event_type,
                    confidence  = confidence,
                    description = desc,
                    delta_mag   = peak_mag,
                ))

        return sorted(events, key=lambda e: e.timestamp)

    def _classify_peak(self, mag, time, idx, smoothed, times, player_counts):
        # Check audio at this timestamp
        audio_at = self._audio_near(time, window=3.0)
        crowd_spike = "crowd" in audio_at.lower() or "goal" in audio_at.lower()

        # Check if player count changed (red card / substitution)
        count_changed = False
        if player_counts:
            counts_near = [p for p in player_counts
                          if abs(p["timestamp"] - time) < 30.0]
            if counts_near:
                before = [p for p in counts_near if p["timestamp"] < time]
                after  = [p for p in counts_near if p["timestamp"] > time]
                if before and after:
                    count_changed = (
                        before[-1]["team1"] + before[-1]["team2"] !=
                        after[0]["team1"] + after[0]["team2"]
                    )

        # Decision tree
        if mag >= self.THRESHOLDS["GOAL"] and crowd_spike:
            return "GOAL", 0.92, f"Goal scored at {self._fmt(time)}"

        if mag >= self.THRESHOLDS["GOAL"] and not crowd_spike:
            return "GOAL", 0.75, f"Possible goal at {self._fmt(time)}"

        if mag >= self.THRESHOLDS["RED_CARD"] and count_changed:
            return "RED_CARD", 0.88, f"Player sent off at {self._fmt(time)}"

        if mag >= self.THRESHOLDS["FOUL"]:
            # Check if play resumed quickly (short stop = foul, long stop = injury)
            post_mag = smoothed[idx+1] if idx+1 < len(smoothed) else 0
            if post_mag < 0.15:
                return "FOUL", 0.70, f"Foul or set piece at {self._fmt(time)}"

        if mag >= self.THRESHOLDS["FORMATION"]:
            return "FORMATION_CHANGE", 0.65, f"Tactical shape changed at {self._fmt(time)}"

        return None, 0, ""

    def _smooth(self, values, window=5):
        result = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end   = min(len(values), i + window // 2 + 1)
            result.append(float(np.mean(values[start:end])))
        return result

    def _find_peaks(self, signal, min_height, min_distance_sec, times):
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # Check min distance from last peak
                if not peaks or (times[i] - peaks[-1][1]) > min_distance_sec:
                    peaks.append((i, times[i], signal[i]))
        return peaks

    def _audio_near(self, timestamp, window=3.0):
        for seg in self.audio:
            if abs(seg.get("start", 0) - timestamp) < window:
                return seg.get("text", "")
        return ""

    def _fmt(self, sec):
        return f"{int(sec//60):02d}:{int(sec%60):02d}"
