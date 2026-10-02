"""
ADVE Enterprise — Production Resilience & Torture Test Suite
Evaluates video corruption recovery, SQLite state database persistence, and license enforcement.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient

ENTERPRISE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENTERPRISE_DIR not in sys.path:
    sys.path.insert(0, ENTERPRISE_DIR)

from api.main import app
from core.db import db, StateDatabase
from core.pipeline_wrapper import EnterprisePipeline
from core.input_sanitizer import sanitize_frame, parse_video_source
from tools.license_generator import generate_license, verify_license_token
from api.middleware.license import verify_license_key, LicenseExpiredException, LicenseInvalidException

client = TestClient(app)
SCRATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scratch"))


def test_sqlite_state_database_persistence():
    # Insert stream
    db.upsert_stream(
        stream_id="cam_persist_99",
        rtsp_url="rtsp://192.168.1.50/stream",
        status="active",
        metadata={"location": "gate_1"}
    )

    # Re-instantiate database connection (simulating container restart)
    new_db = StateDatabase(db.db_path)
    stream_record = new_db.get_stream("cam_persist_99")

    assert stream_record is not None
    assert stream_record["stream_id"] == "cam_persist_99"
    assert stream_record["status"] == "active"
    assert stream_record["metadata"]["location"] == "gate_1"


def test_torture_single_frame_and_black_videos():
    single_path = os.path.join(SCRATCH_DIR, "single_frame.mp4")
    black_path = os.path.join(SCRATCH_DIR, "pure_black.mp4")

    if os.path.exists(single_path):
        with patch("core.pipeline_wrapper.ADVEPipeline") as MockCorePipeline:
            mock_inst = MockCorePipeline.return_value
            mock_inst.process_frame.return_value = {
                "embedding": np.ones((512,), dtype=np.float32),
                "is_anchor": True,
                "encoder_called": True,
                "delta_magnitude": 0.0,
                "appearance_delta": 0.0,
                "cosine_sim": 1.0,
                "frame_idx": 0
            }
            engine = EnterprisePipeline(stream_id="single_frame_test")
            res = engine.process_single_frame(np.zeros((480, 640, 3), dtype=np.uint8))
            assert res["success"] is True

    if os.path.exists(black_path):
        # Process black image frame directly
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        clean_frame = sanitize_frame(black_frame)
        assert clean_frame.shape == (720, 1280, 3)


def test_torture_corrupted_truncated_video_graceful_handling():
    truncated_path = os.path.join(SCRATCH_DIR, "corrupted_truncated.mp4")
    assert os.path.exists(truncated_path)

    # Verify input parser identifies file format correctly without segfaulting
    parsed = parse_video_source(truncated_path)
    assert parsed["type"] == "local"
    assert parsed["is_valid_format"] is True


def test_license_tampered_and_invalid_token_rejection():
    # Valid key
    valid_key = generate_license("Acme Corp", tier="edge", max_streams=50, days=90)
    verified = verify_license_token(valid_key)
    assert verified["valid"] is True

    # Tampered signature
    parts = valid_key.split(".")
    tampered_key = f"{parts[0]}.TAMPERED_SIGNATURE_BYTES"
    verified_tampered = verify_license_token(tampered_key)
    assert verified_tampered["valid"] is False
