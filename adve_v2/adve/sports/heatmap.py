import cv2
import numpy as np
from typing import List, Tuple


class PlayerHeatmap:
    """
    Generates player position heatmaps from ADVE tracking data.
    Shows where a player or team spent time on the pitch.
    """

    def __init__(self, pitch_w: int = 105, pitch_h: int = 68):
        self.pitch_w = pitch_w  # standard football pitch metres
        self.pitch_h = pitch_h

    def generate(
        self,
        positions:    List[Tuple[float, float]],  # normalised (0-1) x, y
        output_size:  Tuple[int, int] = (800, 520),
        player_name:  str = "Player/Team",
        colormap:     int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        positions: list of (x, y) coordinates normalised to [0, 1]
        Returns: BGR numpy array of the heatmap overlaid on pitch diagram
        """
        w_out, h_out = output_size[0], output_size[1]

        # Create accumulator
        accumulator = np.zeros((h_out, w_out), dtype=np.float32)

        if positions:
            for (x_norm, y_norm) in positions:
                px = int(x_norm * w_out)
                py = int(y_norm * h_out)
                px = np.clip(px, 0, w_out - 1)
                py = np.clip(py, 0, h_out - 1)
                accumulator[py, px] += 1

            # Gaussian blur for smooth heatmap representation
            # Use kernel size depending on output dimensions, ensuring it is odd
            ksize = 51
            blurred = cv2.GaussianBlur(accumulator, (ksize, ksize), 0)

            # Normalise to 0-255 range
            if blurred.max() > 0:
                blurred = (blurred / blurred.max() * 255).astype(np.uint8)
            else:
                blurred = blurred.astype(np.uint8)

            # Apply colormap
            heatmap = cv2.applyColorMap(blurred, colormap)
        else:
            # Empty heatmap
            heatmap = np.zeros((h_out, w_out, 3), dtype=np.uint8)

        # Overlay on pitch diagram
        pitch = self._draw_pitch(w_out, h_out)
        
        # Merge diagram with heatmap
        if positions:
            overlay = cv2.addWeighted(pitch, 0.4, heatmap, 0.6, 0)
        else:
            overlay = pitch

        # Add label
        cv2.putText(
            overlay, player_name,
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2, cv2.LINE_AA
        )

        return overlay

    def _draw_pitch(self, w: int, h: int) -> np.ndarray:
        """Draw a standard football pitch diagram (dark green with white lines)."""
        # Dark green background
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = [34, 112, 34]  # BGR green

        # Margins/Padding for outer border
        pad = 20

        # Pitch outline
        cv2.rectangle(img, (pad, pad), (w - pad, h - pad), (255, 255, 255), 2)

        # Centre line
        cv2.line(img, (w // 2, pad), (w // 2, h - pad), (255, 255, 255), 2)

        # Centre circle
        cv2.circle(img, (w // 2, h // 2), int(h * 0.15), (255, 255, 255), 2)
        cv2.circle(img, (w // 2, h // 2), 4, (255, 255, 255), -1)  # center spot

        # Penalty areas
        pa_w = int(w * 0.16)
        pa_h = int(h * 0.45)
        pa_top = (h - pa_h) // 2

        cv2.rectangle(img, (pad, pa_top), (pad + pa_w, pa_top + pa_h), (255, 255, 255), 2)
        cv2.rectangle(img, (w - pad - pa_w, pa_top), (w - pad, pa_top + pa_h), (255, 255, 255), 2)

        # Goal areas (6-yard box)
        ga_w = int(w * 0.05)
        ga_h = int(h * 0.20)
        ga_top = (h - ga_h) // 2
        cv2.rectangle(img, (pad, ga_top), (pad + ga_w, ga_top + ga_h), (255, 255, 255), 2)
        cv2.rectangle(img, (w - pad - ga_w, ga_top), (w - pad, ga_top + ga_h), (255, 255, 255), 2)

        # Corner arcs
        arc_r = 10
        cv2.ellipse(img, (pad, pad), (arc_r, arc_r), 0, 0, 90, (255, 255, 255), 1)
        cv2.ellipse(img, (pad, h - pad), (arc_r, arc_r), 0, 270, 360, (255, 255, 255), 1)
        cv2.ellipse(img, (w - pad, pad), (arc_r, arc_r), 0, 90, 180, (255, 255, 255), 1)
        cv2.ellipse(img, (w - pad, h - pad), (arc_r, arc_r), 0, 180, 270, (255, 255, 255), 1)

        return img
