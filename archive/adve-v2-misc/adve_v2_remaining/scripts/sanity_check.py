import torch
import cv2
import sys
import os
from pathlib import Path

# Add project root and adve subdirectory to sys.path for robust importing
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "adve"))

print("=" * 50)
print("ADVE PRE-FLIGHT SANITY CHECK")
print("=" * 50)

# 1. CUDA
if torch.cuda.is_available():
    print(f"OK: CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("FAIL: CUDA NOT AVAILABLE - Tests will run on CPU (slow)")
    sys.exit(1)

# 2. CLIP
try:
    from adve.core.clip_loader import load_clip_model
    clip_model, clip_prep = load_clip_model("ViT-B/32", device="cuda")
    print("OK: CLIP loaded")
except Exception as e:
    print(f"FAIL: CLIP failed: {e}")
    sys.exit(1)

# 3. YOLO
try:
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    print("OK: YOLO loaded")
except Exception as e:
    print(f"FAIL: YOLO failed: {e}")
    sys.exit(1)

# 4. Reconstructor
try:
    from adve.core.reconstructor_v3 import DeltaReconstructorV3
    rec = DeltaReconstructorV3().to("cuda")
    rec.eval()
    dummy = torch.randn(1, 512).to("cuda"), torch.randn(1, 128).to("cuda")
    with torch.no_grad():
        out, _ = rec(*dummy)
    assert out.shape == (1, 512)
    print("OK: Reconstructor v3 forward pass OK")
except Exception as e:
    print(f"FAIL: Reconstructor failed: {e}")
    sys.exit(1)

# 5. Checkpoint exists
ckpt = Path("training/checkpoints/reconstructor_v3.pt")
if ckpt.exists():
    print(f"OK: Checkpoint found: {ckpt}")
else:
    print(f"FAIL: Checkpoint NOT FOUND: {ckpt}")
    print("   Cannot test without trained model.")
    sys.exit(1)

# 6. Test video exists
test_videos = (
    list(Path("demo_videos").glob("*.mp4")) + 
    list(Path("datasets").glob("*.mp4")) + 
    list(Path("adve_realworld_test/datasets").glob("**/*.mp4")) +
    list(Path(".").glob("*.mp4"))
)
if test_videos:
    print(f"OK: Test videos found: {len(test_videos)}")
else:
    print("WARN: No test videos found in demo_videos/, datasets/, or adve_realworld_test/")

print("=" * 50)
print("PASSED: ALL CHECKS PASSED - Ready to test")
print("=" * 50)
