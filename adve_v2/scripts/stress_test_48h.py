"""
ADVE 48-Hour Continuous RTSP Memory & VRAM Stability Daemon (Component #10)

Runs a continuous 48-hour stability monitoring loop.
Logs System RAM (MB), CUDA VRAM Allocated (MB), and CUDA VRAM Reserved (MB)
every 5 minutes (300s) to results/stability_48h.log.
"""

import os
import sys
import time
import torch
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_48h_stability_daemon(duration_hours: float = 48.0, interval_seconds: int = 300):
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "stability_48h.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    process = psutil.Process(os.getpid())
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n==================================================\n")
        f.write(f"=== ADVE 48-HOUR CONTINUOUS STRESS TEST LAUNCHED AT {time.ctime(start_time)} ===\n")
        f.write("Timestamp, Elapsed (s), System RAM (MB), VRAM Allocated (MB), VRAM Reserved (MB)\n")
        f.write(f"==================================================\n")
        f.flush()

        while time.time() < end_time:
            now = time.time()
            elapsed = round(now - start_time, 2)
            ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)

            if torch.cuda.is_available():
                vram_alloc = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                vram_res = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
            else:
                vram_alloc = 0.0
                vram_res = 0.0

            log_line = f"{time.ctime(now)}, {elapsed}, {ram_mb}, {vram_alloc}, {vram_res}\n"
            f.write(log_line)
            f.flush()
            time.sleep(interval_seconds)

        f.write(f"=== 48-HOUR STABILITY TEST COMPLETED AT {time.ctime(time.time())} ===\n")

if __name__ == "__main__":
    run_48h_stability_daemon(duration_hours=48.0, interval_seconds=300)
