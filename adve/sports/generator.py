import cv2
import shutil
import subprocess
import numpy as np
from typing import List
from pathlib import Path


class HighlightGenerator:
    """
    Generates highlight reels automatically from ADVE's ΔG timeline.

    The insight: the most exciting moments in a match are the ones
    where the scene changed most dramatically.
    ΔG measures scene change.
    Top N ΔG peaks = Top N exciting moments.

    No labelled data needed. No extra model.
    This is a free output of the ADVE indexing process.
    """

    def generate(
        self,
        video_path:     str,
        delta_timeline: List[dict],
        events:         List,              # SportEvent or dict list
        output_path:    str = "highlights.mp4",
        top_n:          int = 10,
        clip_before:    float = 3.0,       # seconds before the peak
        clip_after:     float = 8.0,       # seconds after the peak
        min_gap:        float = 30.0,      # min seconds between selected moments
    ) -> str:
        """
        Selects top N moments from match and assembles them into a highlight reel.
        """
        # Ensure static_ffmpeg paths are added if available
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
        except ImportError:
            pass

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            print("[HighlightGenerator] Warning: ffmpeg not found in path. Cannot generate highlights.")
            return ""

        if not delta_timeline:
            return ""

        # Combine ΔG peaks with detected events for best selection
        candidates = []

        # Add all ΔG peaks
        for d in delta_timeline:
            candidates.append({
                "timestamp":  d["timestamp"],
                "score":      d["magnitude"],
                "source":     "delta",
                "label":      f"High action at {self._fmt(d['timestamp'])}",
            })

        # Boost score for confirmed events (handle both objects and dicts)
        event_timestamps = {}
        for e in events:
            if hasattr(e, "timestamp"):
                event_timestamps[e.timestamp] = getattr(e, "description", "")
            elif isinstance(e, dict) and "timestamp" in e:
                event_timestamps[e["timestamp"]] = e.get("description", "")

        if event_timestamps:
            for c in candidates:
                nearest_event = min(
                    event_timestamps.keys(),
                    key=lambda t: abs(t - c["timestamp"]),
                    default=None,
                )
                if nearest_event is not None and abs(nearest_event - c["timestamp"]) < 5.0:
                    c["score"] *= 1.5  # boost if near a detected event
                    c["label"] = event_timestamps[nearest_event]

        # Sort by score, deduplicate by time
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        for c in candidates:
            too_close = any(
                abs(c["timestamp"] - s["timestamp"]) < min_gap
                for s in selected
            )
            if not too_close:
                selected.append(c)
            if len(selected) >= top_n:
                break

        selected.sort(key=lambda x: x["timestamp"])  # chronological order

        # Extract clips
        clips = []
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        temp_dir = Path("temp_clips")
        temp_dir.mkdir(exist_ok=True)

        # Clean old temp clips to avoid contamination
        for p in temp_dir.glob("*.mp4"):
            try:
                p.unlink()
            except Exception:
                pass

        for i, moment in enumerate(selected):
            start    = max(0.0, moment["timestamp"] - clip_before)
            duration = clip_before + clip_after
            clip_out = str(temp_dir / f"clip_{i:03d}.mp4")

            # Extract video clip segment
            subprocess.run([
                ffmpeg_bin, "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                clip_out,
            ], capture_output=True)

            if Path(clip_out).exists() and Path(clip_out).stat().st_size > 1000:
                clips.append(clip_out)

        if not clips:
            return ""

        # Concatenate clips
        concat_list = temp_dir / "concat_list.txt"
        with open(concat_list, "w") as f:
            for clip in clips:
                # Use absolute path to ensure safety with concat filter
                abs_path = Path(clip).resolve().as_posix()
                f.write(f"file '{abs_path}'\n")

        subprocess.run([
            ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            output_path,
        ], capture_output=True)

        # Cleanup temporary files
        try:
            for clip in clips:
                Path(clip).unlink()
            concat_list.unlink()
            temp_dir.rmdir()
        except Exception:
            pass

        return output_path if Path(output_path).exists() else ""

    def _fmt(self, sec):
        return f"{int(sec//60):02d}:{int(sec%60):02d}"
