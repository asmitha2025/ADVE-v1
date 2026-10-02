import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class FormationSnapshot:
    timestamp:  float
    formation:  str     # "4-3-3", "4-4-2", etc.
    team_id:    int     # 1 or 2
    confidence: float


class FormationTracker:
    """
    Derives football formation from player positions using ADVE's spatial graph.

    The spatial graph already knows where every player is.
    K-means clustering into defensive / midfield / attacking lines
    gives the formation for free.
    """

    STANDARD_FORMATIONS = {
        (4, 4, 2): "4-4-2",
        (4, 3, 3): "4-3-3",
        (4, 5, 1): "4-5-1",
        (4, 2, 4): "4-2-4",
        (3, 5, 2): "3-5-2",
        (3, 4, 3): "3-4-3",
        (5, 3, 2): "5-3-2",
        (5, 4, 1): "5-4-1",
        (4, 1, 4): "4-1-4-1",
    }

    def classify(
        self,
        player_positions: List[Tuple[float, float]],
        n_outfield: int = 10,
    ) -> FormationSnapshot:
        """
        player_positions: list of (x, y) normalised to [0,1] x [0,1]
        Returns the most likely formation.
        """
        if len(player_positions) < 3:
            return FormationSnapshot(0, "unknown", 0, 0.0)

        positions = np.array(player_positions[:n_outfield])
        y_vals = positions[:, 1]

        # Handle edge cases with insufficient unique Y coordinates to form 3 clusters
        unique_y = np.unique(y_vals)
        if len(unique_y) < 3:
            # Fall back to a default representation if we can't cluster into 3 distinct bands
            return FormationSnapshot(
                timestamp  = 0,
                formation  = f"{len(positions)}-0-0",
                team_id    = 1,
                confidence = 0.1
            )

        # K-means into 3 clusters
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        labels = km.fit_predict(y_vals.reshape(-1, 1))

        # Sort clusters by y position (defence=lowest y, attack=highest)
        cluster_centers = km.cluster_centers_.flatten()
        sorted_clusters = np.argsort(cluster_centers)

        def_count  = int((labels == sorted_clusters[0]).sum())
        mid_count  = int((labels == sorted_clusters[1]).sum())
        att_count  = int((labels == sorted_clusters[2]).sum())

        formation_key = (def_count, mid_count, att_count)
        formation_str = self.STANDARD_FORMATIONS.get(
            formation_key,
            f"{def_count}-{mid_count}-{att_count}"
        )

        # Confidence based on cluster separation
        cluster_spread = float(np.std(cluster_centers))
        confidence = min(0.95, cluster_spread * 3)

        return FormationSnapshot(
            timestamp  = 0,
            formation  = formation_str,
            team_id    = 1,
            confidence = confidence,
        )

    def track_over_match(
        self,
        position_timeline: List[dict],
        min_duration_sec: float = 60.0,
    ) -> List[dict]:
        """
        Tracks formation changes throughout a match.
        Only reports changes that last at least min_duration_sec.
        """
        history = []
        current = None
        current_start = 0

        for frame in position_timeline:
            ts       = frame["timestamp"]
            players  = frame["positions"]
            snapshot = self.classify(players)

            if snapshot.formation != current:
                if current and (ts - current_start) >= min_duration_sec:
                    history.append({
                        "formation":  current,
                        "start_time": current_start,
                        "end_time":   ts,
                        "duration":   ts - current_start,
                    })
                current       = snapshot.formation
                current_start = ts

        # Append last formation block if valid
        if current and position_timeline:
            last_ts = position_timeline[-1]["timestamp"]
            if (last_ts - current_start) >= min_duration_sec:
                history.append({
                    "formation":  current,
                    "start_time": current_start,
                    "end_time":   last_ts,
                    "duration":   last_ts - current_start,
                })

        return history
