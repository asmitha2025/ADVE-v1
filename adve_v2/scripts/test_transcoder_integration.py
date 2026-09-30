"""
Test Suite for VideoTranscoder Probing & Normalization Integration
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adve.core.transcoder import VideoTranscoder
from adve.core.config import Config
from adve.core.pipeline import ADVEPipeline

def run_transcoder_tests():
    print("=" * 70)
    print("  TEST SUITE: ADVE VideoTranscoder & Pipeline Ingestion")
    print("=" * 70)

    # 1. Probe real test video
    test_video = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "Testing videos",
        "Vaama Vaama - Airport Version _ Idhayam Murali _ Atharvaa _ Preity _ Thaman _Dhanush_Aakash Baskaran.mp4"
    ))

    transcoder = VideoTranscoder(max_width=1280, max_height=720, target_fps=30)
    
    print("\n[Test 1] Probing standard test video...")
    meta = transcoder.probe_video(test_video)
    print(f"  Valid           : {meta['valid']}")
    print(f"  Resolution      : {meta['width']}x{meta['height']}")
    print(f"  FPS             : {meta['fps']}")
    print(f"  Needs Transcode : {meta['needs_transcode']}")
    assert meta["valid"] == True, "Failed to probe video!"
    print("  [PASS] Video metadata probing working correctly.")

    # 2. Synthetic 4K Video Test (Forces Transcoding)
    print("\n[Test 2] Creating synthetic 4K test video to test transcoding trigger...")
    synth_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_tmp")
    os.makedirs(synth_dir, exist_ok=True)
    synth_4k_path = os.path.join(synth_dir, "test_4k.mp4")

    # Create short 4K video (3840x2160, 10 frames)
    out = cv2.VideoWriter(synth_4k_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (3840, 2160))
    for i in range(10):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        cv2.putText(frame, f"4K Frame {i}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        out.write(frame)
    out.release()

    meta_4k = transcoder.probe_video(synth_4k_path)
    print(f"  4K Video Resolution : {meta_4k['width']}x{meta_4k['height']}")
    print(f"  Needs Transcode     : {meta_4k['needs_transcode']}")
    assert meta_4k["needs_transcode"] == True, "4K video should trigger needs_transcode!"

    # Perform Transcode
    print("  Executing transcode_if_needed on 4K input...")
    output_path, was_transcoded = transcoder.transcode_if_needed(synth_4k_path)
    print(f"  Was Transcoded : {was_transcoded}")
    print(f"  Output Path    : {output_path}")
    assert was_transcoded == True, "Video should have been transcoded!"
    
    # Check normalized dimensions
    norm_meta = transcoder.probe_video(output_path)
    print(f"  Normalized Resolution : {norm_meta['width']}x{norm_meta['height']}")
    assert norm_meta["width"] <= 1280 and norm_meta["height"] <= 720, "Transcoded video exceeds 720p bounds!"
    print("  [PASS] 4K Video successfully transcoded & normalized to 720p.")

    # Clean up synthetic test files
    try:
        os.remove(synth_4k_path)
        os.remove(output_path)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("  ALL TRANSCODER INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_transcoder_tests()
