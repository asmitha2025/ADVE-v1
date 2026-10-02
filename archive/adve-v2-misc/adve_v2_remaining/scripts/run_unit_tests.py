import sys
import os
import time
import torch
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_spatial_graph():
    print("[1/3] Testing Spatial Graph Construction & Delta...")
    from adve.core.spatial_graph import SpatialGraph, ObjectState

    obj1 = ObjectState(obj_id=1, bbox=[100, 100, 200, 200], class_name="person", center=(150.0, 150.0), area=10000.0)
    obj2 = ObjectState(obj_id=2, bbox=[300, 300, 400, 400], class_name="car", center=(350.0, 350.0), area=10000.0)
    
    g1 = SpatialGraph()
    g1.add_object(obj1)
    g1.add_object(obj2)
    g1.build_relations(640, 480)

    obj1_moved = ObjectState(obj_id=1, bbox=[120, 110, 220, 210], class_name="person", center=(170.0, 160.0), area=10000.0)
    obj2_moved = ObjectState(obj_id=2, bbox=[300, 300, 400, 400], class_name="car", center=(350.0, 350.0), area=10000.0)
    
    g2 = SpatialGraph()
    g2.add_object(obj1_moved)
    g2.add_object(obj2_moved)
    g2.build_relations(640, 480)

    delta = g1.compute_delta(g2)
    mag = delta["total_magnitude"]
    print(f"   Delta magnitude: {mag:.4f}")
    assert mag > 0, "Delta magnitude should be positive when an object moves"
    print("   [PASSED] SpatialGraph Unit Test")

def test_ego_motion():
    print("[2/3] Testing Ego-Motion Estimator...")
    from adve.core.ego_motion import EgoMotionEstimator

    est = EgoMotionEstimator()
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame1, "ADVE TEST OBJECT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    est.set_anchor_frame(frame1)

    # Shift frame to simulate pan
    M = np.float32([[1, 0, 30], [0, 1, 10]])
    frame2 = cv2.warpAffine(frame1, M, (640, 480))

    H = est.estimate_homography(frame2)
    print(f"   Estimated Homography: {'Found' if H is not None else 'None'}")
    assert H is not None, "EgoMotion Estimator should compute homography matrix for panned frame"
    print("   [PASSED] EgoMotion Unit Test")

def test_reconstructor_v3():
    print("[3/3] Testing Reconstructor v3...")
    from adve.core.reconstructor_v3 import DeltaReconstructorV3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeltaReconstructorV3().to(device)
    model.eval()

    # Forward check
    anchor = torch.randn(4, 512).to(device)
    delta = torch.randn(4, 128).to(device)
    with torch.no_grad():
        out, h = model(anchor, delta)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"

    # Latency check (100 passes)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(anchor[:1], delta[:1])
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 100.0 * 1000.0
    print(f"   Avg Reconstructor Latency: {ms:.3f} ms / frame (Device: {device})")
    assert ms < 5.0, f"Reconstructor latency too high ({ms:.2f} ms)"
    print("   [PASSED] Reconstructor v3 Unit Test")

def main():
    print("=========================================================")
    print("   ADVE Enterprise Unit Test Suite")
    print("=========================================================")
    test_spatial_graph()
    test_ego_motion()
    test_reconstructor_v3()
    print("\n[PASSED] ALL UNIT TESTS PASSED!\n")

if __name__ == "__main__":
    main()
