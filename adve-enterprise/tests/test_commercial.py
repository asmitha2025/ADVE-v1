"""
ADVE Enterprise — Phase 3 Commercial Weapons Test Suite
"""

import sys
import os
import tempfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

ENTERPRISE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ENTERPRISE_DIR not in sys.path:
    sys.path.insert(0, ENTERPRISE_DIR)

from api.main import app
from tools.report_generator import generate_compatibility_pdf

client = TestClient(app)


def test_sales_document_templates_exist():
    docs_dir = os.path.join(ENTERPRISE_DIR, "docs")
    nda_path = os.path.join(docs_dir, "nda_template.md")
    poc_path = os.path.join(docs_dir, "poc_proposal_template.md")
    demo_path = os.path.join(docs_dir, "demo_video_script.md")

    assert os.path.exists(nda_path)
    assert os.path.exists(poc_path)
    assert os.path.exists(demo_path)


def test_pdf_report_generator():
    temp_dir = tempfile.mkdtemp()
    out_pdf = os.path.join(temp_dir, "Tata_Elxsi_Report.pdf")

    summary_stats = {
        "mean_cosine_sim": 0.9782,
        "encoder_savings_pct": 72.4,
        "total_frames": 100,
        "anchor_frames": 28,
        "delta_frames": 72,
        "effective_fps": 88.5
    }
    sim_history = [0.98, 0.97, 0.99, 0.96, 0.98, 0.97]

    generated_path = generate_compatibility_pdf(
        customer_name="Tata Elxsi",
        video_name="test_feed.mp4",
        summary_stats=summary_stats,
        sim_history=sim_history,
        output_pdf_path=out_pdf
    )

    assert os.path.exists(generated_path)
    assert os.path.getsize(generated_path) > 1000  # Non-empty PDF file


def test_compatibility_route_endpoint():
    temp_video = os.path.join(tempfile.gettempdir(), "mock_cctv.mp4")
    with open(temp_video, "wb") as f:
        f.write(b"dummy_video_bytes")

    with patch("api.routes.compatibility.EnterprisePipeline") as MockPipeline:
        mock_instance = MockPipeline.return_value
        mock_instance.pipeline.process_video.return_value = {
            "mean_cosine_sim": 0.965,
            "encoder_savings_pct": 68.5,
            "total_frames": 50,
            "anchor_frames": 15,
            "delta_frames": 35,
            "effective_fps": 92.0
        }
        mock_instance.sim_history = [0.97, 0.96, 0.98]

        with open(temp_video, "rb") as video_file:
            response = client.post(
                "/v1/compatibility",
                files={"video": ("mock_cctv.mp4", video_file, "video/mp4")},
                data={"customer": "Hikvision_India"}
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
