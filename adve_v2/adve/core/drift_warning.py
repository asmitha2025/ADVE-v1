"""
ADVE Semantic Drift Early Warning Module
Tracks cosine similarity and feature residual trends across recent delta frames.
Triggers an early keyframe anchor refresh before structural degradation or latent drift occurs.
"""

import numpy as np

class DriftEarlyWarning:
    def __init__(self, history_size=8, slope_threshold=-0.015):
        self.history_size = history_size
        self.slope_threshold = slope_threshold
        self.history = []

    def reset(self):
        self.history = []

    def update_and_check(self, current_cos_sim):
        self.history.append(current_cos_sim)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        if len(self.history) < 4:
            return False

        # Fit a 1D linear regression line to cosine similarity history
        x = np.arange(len(self.history))
        y = np.array(self.history)
        slope, _ = np.polyfit(x, y, 1)

        # If similarity is decreasing faster than slope_threshold, trigger early refresh
        if slope < self.slope_threshold or current_cos_sim < 0.8800:
            return True

        return False
