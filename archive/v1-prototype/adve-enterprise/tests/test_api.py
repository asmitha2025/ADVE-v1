"""
ADVE Enterprise — Comprehensive API & License Test Suite
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure adve-enterprise directory is on path
ENTERPRISE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENTERPRISE_DIR not in sys.path:
    sys.path.insert(0, ENTERPRISE_DIR)

from api.main import app
from tools.license_generator import generate_license, verify_license_token
from api.middleware.license import StreamLimitExceededException, check_stream_limit

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "license_valid" in data
    assert data["license_valid"] is True


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "adve_frames_total" in response.text
    assert "adve_encoder_savings_ratio" in response.text
    assert "adve_license_days_remaining" in response.text


def test_license_generation_and_verification():
    # 1. Evaluation Tier
    eval_token = generate_license("Test Corp", tier="evaluation", max_streams=5, days=90)
    eval_data = verify_license_token(eval_token)
    assert eval_data["valid"] is True
    assert eval_data["customer"] == "Test Corp"
    assert eval_data["max_streams"] == 5

    # 2. Expired Tier
    expired_token = generate_license("Expired Corp", tier="edge", max_streams=50, days=-1)
    expired_data = verify_license_token(expired_token)
    assert expired_data["valid"] is False
    assert "expired" in expired_data["reason"].lower()


def test_stream_limit_enforcement():
    license_data = {"max_streams": 5, "tier": "evaluation"}
    # Allowed
    check_stream_limit(4, license_data)

    # Exceeded
    with pytest.raises(StreamLimitExceededException):
        check_stream_limit(5, license_data)


from unittest.mock import patch

def test_batch_job_submission():
    with patch("api.routes.batch.EnterprisePipeline") as MockPipeline:
        mock_instance = MockPipeline.return_value
        mock_instance.pipeline.process_video.return_value = {
            "total_frames": 10, "encoder_savings_pct": 80.0, "mean_cosine_sim": 0.95
        }
        response = client.post(
            "/v1/batch",
            json={"video_path": "test_video.mp4", "job_id": "test_batch_001"}
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
        assert data["job_id"] == "test_batch_001"

        status_resp = client.get("/v1/batch/test_batch_001")
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == "test_batch_001"


def test_embeddings_store_and_search():
    # Store test vector
    store_resp = client.post(
        "/v1/embeddings",
        json={
            "stream_id": "cam_test",
            "frame_idx": 10,
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"location": "entrance"}
        }
    )
    assert store_resp.status_code == 201

    # Search query
    search_resp = client.post(
        "/v1/embeddings/search",
        json={
            "query_vector": [0.1, 0.2, 0.3, 0.4],
            "top_k": 5
        }
    )
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert len(results) > 0
    assert results[0]["stream_id"] == "cam_test"
    assert results[0]["score"] > 0.99
