import sys
import time
import torch
import cv2
import numpy as np
import argparse

def run_sanity_check(reconstructor_path: str = "training/checkpoints/reconstructor_v2.pt", device: str = "cuda"):
    print("=========================================================")
    print("   ADVE Real-World Testing Framework — Sanity Check")
    print("=========================================================")

    # 1. Check CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == "cuda" else "CPU"
    print(f"[OK] Device Status: {device.upper()} ({device_name})")

    # 2. Check CLIP Loader
    try:
        from PIL import Image
        from adve.core.clip_loader import load_clip_model
        clip_model, clip_prep = load_clip_model("ViT-B/32", device=device)
        dummy_pil = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        dummy_tensor = clip_prep(dummy_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(dummy_tensor)
        print(f"[OK] CLIP Model: Loaded successfully. Output shape: {emb.shape}")
    except Exception as e:
        print(f"[ERROR] CLIP Error: {e}")
        sys.exit(1)

    # 3. Check YOLO
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt").to(device)
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        results = yolo.track(dummy_frame, device=device, verbose=False)
        print(f"[OK] YOLOv8 Tracking: Loaded successfully on {device}.")
    except Exception as e:
        print(f"[ERROR] YOLO Error: {e}")
        sys.exit(1)

    # 4. Check Reconstructor
    try:
        from adve.core.reconstructor import EmbeddingReconstructor
        reconstructor = EmbeddingReconstructor(reconstructor_path)
        dummy_anchor = torch.randn(1, 512, device=device)
        dummy_delta = torch.randn(1, 128, device=device)
        
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            if hasattr(reconstructor, "model") and reconstructor.model is not None:
                if getattr(reconstructor, "is_v2", False):
                    _ = reconstructor.model(dummy_anchor, dummy_delta)
                else:
                    dummy_pool = torch.randn(1, 512, device=device)
                    _ = reconstructor.model(dummy_anchor, dummy_pool, dummy_delta)
        t1 = time.perf_counter()
        latency_ms = ((t1 - t0) / iters) * 1000.0
        fps = 1000.0 / (latency_ms + 1e-8)
        print(f"[OK] Reconstructor ({reconstructor_path}): Latency {latency_ms:.3f} ms ({fps:.0f} FPS)")
    except Exception as e:
        print(f"[ERROR] Reconstructor Error: {e}")
        sys.exit(1)

    print("\n=========================================================")
    print("  [PASS] ALL CHECKS PASSED — System ready for Real-World Testing!")
    print("=========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructor", default="training/checkpoints/reconstructor_v2.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_sanity_check(args.reconstructor, args.device)
