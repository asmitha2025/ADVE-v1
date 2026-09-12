"""
ADVE Model Registry & Versioning Engine (Component #5)

Maintains neural reconstructor version pinning (`v3.1`, `v3.2`).
Allows clients to pass `?model_version=3.1` in API calls for exact reproducibility and instant rollback.
"""

import os
from typing import Dict, Any, Optional, Tuple
from adve.core.reconstructor import EmbeddingReconstructor

class ModelRegistry:
    def __init__(self, models_dir: str = "training/checkpoints"):
        self.models_dir = models_dir
        self.registry: Dict[str, str] = {
            "v3.1": os.path.join(models_dir, "reconstructor_v3.1.pt"),
            "traffic": os.path.join(models_dir, "domain_traffic.pt"),
            "sparse": os.path.join(models_dir, "domain_sparse.pt"),
            "night": os.path.join(models_dir, "domain_night.pt"),
            "v3.0": os.path.join(models_dir, "best_model.pt"),
            "default": os.path.join(models_dir, "reconstructor_v3.1.pt")
        }
        self.loaded_instances: Dict[str, EmbeddingReconstructor] = {}

    def get_model(self, version: Optional[str] = None) -> Tuple[EmbeddingReconstructor, str]:
        """
        Retrieves a loaded model instance by version tag (e.g. 'v3.1').
        Falls back to default model if requested version is missing.
        """
        ver_tag = version if version in self.registry else "default"
        weights_path = self.registry[ver_tag]

        if not os.path.exists(weights_path):
            weights_path = self.registry["default"]
            ver_tag = "default"

        if ver_tag not in self.loaded_instances:
            self.loaded_instances[ver_tag] = EmbeddingReconstructor(weights_path)

        return self.loaded_instances[ver_tag], ver_tag

    def register_version(self, version_name: str, file_path: str):
        """Registers a new model checkpoint version."""
        self.registry[version_name] = file_path
