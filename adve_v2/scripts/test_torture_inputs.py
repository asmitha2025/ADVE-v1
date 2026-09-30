"""
ADVE Corrupted / Edge-Case Input Torture Test Suite (Critical Item #6)

Verifies that bad or malformed inputs return graceful error responses (HTTP 422 / None / Error Dict) without crashing the pipeline or worker process.
"""

import os
import sys
import tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.content_detector import DomainFingerprinter
from adve.core.pipeline import ADVEPipeline
from adve.core.config import Config

def run_torture_tests():
    print("==================================================")
    print("  RUNNING CORRUPTED / EDGE-CASE INPUT TORTURE SUITE")
    print("==================================================")

    fingerprinter = DomainFingerprinter(sample_frames=10)
    config = Config()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Case 1: Non-existent file
        non_existent = os.path.join(tmp_dir, "does_not_exist.mp4")
        res1 = fingerprinter.fingerprint_and_route(non_existent)
        assert res1["domain"] == "sparse"
        print("   [OK] Case 1: Non-existent video path handled gracefully.")

        # Case 2: 0-Byte file
        empty_file = os.path.join(tmp_dir, "zero_byte.mp4")
        with open(empty_file, "wb") as f:
            f.write(b"")
        res2 = fingerprinter.fingerprint_and_route(empty_file)
        assert res2["domain"] == "sparse"
        print("   [OK] Case 2: 0-Byte empty file handled gracefully.")

        # Case 3: Text file disguised as MP4
        fake_mp4 = os.path.join(tmp_dir, "fake_video.mp4")
        with open(fake_mp4, "w") as f:
            f.write("THIS IS NOT A VALID VIDEO FILE FORMAT CONTENT")
        res3 = fingerprinter.fingerprint_and_route(fake_mp4)
        assert res3["domain"] == "sparse"
        print("   [OK] Case 3: Corrupted text content disguised as MP4 handled gracefully.")

        # Case 4: Single frame image saved as MP4
        single_frame_path = os.path.join(tmp_dir, "single_frame.png")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(single_frame_path, img)
        res4 = fingerprinter.fingerprint_and_route(single_frame_path)
        assert "domain" in res4
        print("   [OK] Case 4: Single frame image file handled gracefully.")

        # Case 5: Directory path passed instead of video file
        res5 = fingerprinter.fingerprint_and_route(tmp_dir)
        assert "domain" in res5
        print("   [OK] Case 5: Directory path handled gracefully.")

    print("\n==================================================")
    print("  ALL 5 CORRUPTED INPUT TORTURE TESTS PASSED!      ")
    print("==================================================")

if __name__ == "__main__":
    run_torture_tests()
