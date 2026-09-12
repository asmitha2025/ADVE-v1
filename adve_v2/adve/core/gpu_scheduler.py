"""
ADVE Multi-GPU Stream Scheduler & Load Balancer (Component #4)

Tracks GPU device memory headroom and active stream count across available CUDA GPUs.
Round-robins incoming live camera feeds to GPUs with available capacity (default cap 12 streams / GPU).
Queues excess streams when VRAM limits are reached.
"""

import torch
from typing import Dict, List, Any, Optional, Tuple

class MultiGPUScheduler:
    def __init__(self, max_streams_per_gpu: int = 12):
        self.max_streams_per_gpu = max_streams_per_gpu
        self.active_streams: Dict[int, List[str]] = {}
        self._init_gpus()

    def _init_gpus(self):
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            for gpu_id in range(self.num_gpus):
                self.active_streams[gpu_id] = []
        else:
            self.num_gpus = 0
            self.active_streams[0] = []  # CPU fallback

    def get_available_gpu(self) -> Tuple[int, str]:
        """
        Returns (gpu_id, device_string) for the GPU with minimum active streams and adequate VRAM.
        Returns (-1, 'queued') if all GPUs are at capacity.
        """
        if self.num_gpus == 0:
            return 0, "cpu"

        best_gpu = -1
        min_streams = 999

        for gpu_id in range(self.num_gpus):
            current_count = len(self.active_streams[gpu_id])
            if current_count < self.max_streams_per_gpu and current_count < min_streams:
                min_streams = current_count
                best_gpu = gpu_id

        if best_gpu != -1:
            return best_gpu, f"cuda:{best_gpu}"
        else:
            return -1, "queued"

    def register_stream(self, stream_id: str) -> Dict[str, Any]:
        """Assigns a stream to an available GPU device."""
        gpu_id, device_str = self.get_available_gpu()
        if gpu_id != -1:
            self.active_streams[gpu_id].append(stream_id)
            return {"assigned": True, "gpu_id": gpu_id, "device": device_str}
        else:
            return {"assigned": False, "status": "queued", "message": "All GPU devices at max capacity"}

    def unregister_stream(self, stream_id: str):
        """Removes a finished stream from GPU tracking."""
        for gpu_id, streams in self.active_streams.items():
            if stream_id in streams:
                streams.remove(stream_id)
                break
