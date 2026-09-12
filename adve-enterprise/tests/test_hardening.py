"""
ADVE Enterprise — Phase 2 Enterprise Hardening Test Suite
"""

import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

ENTERPRISE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENTERPRISE_DIR not in sys.path:
    sys.path.insert(0, ENTERPRISE_DIR)

from api.main import app
from api.config import settings
from core.resilience import ResilienceManager
from core.input_sanitizer import parse_video_source, sanitize_frame
from plugins.vector_db import InMemoryVectorPlugin, QdrantPlugin, get_vector_db_adapter

client = TestClient(app)


def test_config_hot_reload():
    # Get initial config
    res_get = client.get("/v1/config")
    assert res_get.status_code == 200
    initial_budget = res_get.json()["anchor_budget"]

    # Post hot-reload update
    res_post = client.post(
        "/v1/config",
        json={"anchor_budget": 25, "spatial_threshold": 0.40}
    )
    assert res_post.status_code == 200
    data = res_post.json()
    assert data["status"] == "hot_reloaded"
    assert data["updated_parameters"]["anchor_budget"] == 25
    assert settings.anchor_budget == 25
    assert settings.spatial_threshold == 0.40

    # Reset back to original
    settings.anchor_budget = initial_budget


def test_input_sanitizer_and_protocol_parsing():
    # RTSP
    rtsp_info = parse_video_source("rtsp://admin:pass@192.168.1.100:554/stream1")
    assert rtsp_info["type"] == "rtsp"
    assert rtsp_info["is_stream"] is True

    # S3
    s3_info = parse_video_source("s3://cctv-archive-bucket/2026/video_001.mp4")
    assert s3_info["type"] == "s3"
    assert s3_info["is_cloud_storage"] is True

    # Frame resize and BGRA conversion
    dummy_bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
    clean_bgr = sanitize_frame(dummy_bgra, target_size=(1280, 720))
    assert clean_bgr.shape == (720, 1280, 3)


def test_resilience_manager_nan_check():
    res = ResilienceManager(stream_id="test_stream")
    
    # Valid embedding
    valid_emb = np.ones((512,), dtype=np.float32)
    cleaned_valid = res.check_embedding_health(valid_emb)
    assert np.allclose(cleaned_valid, valid_emb)
    assert res.nan_count == 0

    # Corrupted NaN embedding
    corrupt_emb = np.array([1.0, np.nan, np.inf, 2.0], dtype=np.float32)
    cleaned_corrupt = res.check_embedding_health(corrupt_emb)
    assert not np.isnan(cleaned_corrupt).any()
    assert not np.isinf(cleaned_corrupt).any()
    assert res.nan_count == 1


def test_resilience_license_grace_period():
    res = ResilienceManager(stream_id="grace_stream")
    # License invalid but stream started recently -> inside 24h grace period
    in_grace = res.check_license_grace_period(is_valid=False, grace_hours=24)
    assert in_grace is True


def test_vector_db_plugins():
    adapter = get_vector_db_adapter("inmemory")
    test_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    stored = adapter.store(test_vec, stream_id="cam_01", frame_idx=100)
    assert stored is True

    search_res = adapter.search(test_vec, top_k=1)
    assert len(search_res) == 1
    assert search_res[0]["stream_id"] == "cam_01"
    assert search_res[0]["score"] > 0.99
