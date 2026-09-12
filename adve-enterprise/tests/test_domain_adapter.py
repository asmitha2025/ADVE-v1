"""
ADVE Enterprise — Domain Adaptation & Fingerprinting Test Suite (v3.2)
"""

import sys
import os
import pytest
import numpy as np
import torch
from unittest.mock import patch

ENTERPRISE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENTERPRISE_DIR not in sys.path:
    sys.path.insert(0, ENTERPRISE_DIR)

from core.domain_adapter import DomainFingerprinter, MultiDomainAdapter, OnlineCalibrator
from core.pipeline_wrapper import EnterprisePipeline


def test_domain_fingerprinter_classification():
    # Bright traffic frame (> 60 brightness, > 2 objects)
    bright_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    domain_traffic = DomainFingerprinter.classify_frame(bright_frame, num_objects=5, motion_magnitude=0.5)
    assert domain_traffic == "traffic"

    # Sparse frame (bright, <= 2 objects)
    domain_sparse = DomainFingerprinter.classify_frame(bright_frame, num_objects=1, motion_magnitude=0.1)
    assert domain_sparse == "sparse"

    # Dark / Night frame (< 60 brightness)
    dark_frame = np.full((480, 640, 3), 20, dtype=np.uint8)
    domain_night = DomainFingerprinter.classify_frame(dark_frame, num_objects=4, motion_magnitude=0.2)
    assert domain_night == "night"


def test_multi_domain_adapter_forward():
    adapter = MultiDomainAdapter(device="cpu")
    dummy_emb = torch.randn(1, 512)

    for domain in ["traffic", "sparse", "night"]:
        out = adapter(dummy_emb, domain)
        assert out.shape == (1, 512)
        norm = torch.linalg.norm(out, dim=-1).item()
        assert np.isclose(norm, 1.0, atol=1e-3)


def test_online_calibrator_adaptation():
    adapter = MultiDomainAdapter(device="cpu")
    head = adapter.heads["sparse"]
    calibrator = OnlineCalibrator(target_head=head, device="cpu")

    # Add 30 dummy feature triples
    for _ in range(30):
        rec = np.random.randn(512).astype(np.float32)
        gt = rec + np.random.randn(512).astype(np.float32) * 0.05
        rec /= np.linalg.norm(rec)
        gt /= np.linalg.norm(gt)
        calibrator.add_sample(rec, gt)

    calibrated = calibrator.calibrate(epochs=3, lr=1e-3)
    assert calibrated is True
    assert calibrator.is_calibrated is True


def test_pipeline_domain_tag_propagation():
    with patch("core.pipeline_wrapper.ADVEPipeline") as MockPipeline:
        mock_inst = MockPipeline.return_value
        mock_inst.anchor_graph = None
        mock_inst.process_frame.return_value = {
            "embedding": np.ones((512,), dtype=np.float32),
            "is_anchor": True,
            "encoder_called": True,
            "delta_magnitude": 0.0,
            "appearance_delta": 0.0,
            "cosine_sim": 1.0,
            "frame_idx": 0
        }

        engine = EnterprisePipeline(stream_id="domain_test_stream", device="cpu")
        dark_frame = np.full((480, 640, 3), 15, dtype=np.uint8)

        res = engine.process_single_frame(dark_frame)
        assert res["success"] is True
        assert res["domain"] == "night"
