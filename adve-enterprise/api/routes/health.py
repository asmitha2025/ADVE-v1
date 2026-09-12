"""
ADVE Enterprise — Health & Prometheus Metrics Endpoints
Provides system status, GPU diagnostics, license validity, and Prometheus metrics export.
"""

import torch
from fastapi import APIRouter, Response
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from api.middleware.license import verify_license_key
from api.config import settings

router = APIRouter(tags=["Health & Observability"])

# Prometheus Metrics Definitions
METRIC_FRAMES_TOTAL = Counter(
    "adve_frames_total",
    "Total video frames processed by ADVE engine",
    ["stream_id"]
)

METRIC_ENCODER_SAVINGS = Gauge(
    "adve_encoder_savings_ratio",
    "Percentage of transformer encoder calls saved (0.0 to 1.0)",
    ["stream_id"]
)

METRIC_COSINE_SIMILARITY = Gauge(
    "adve_cosine_similarity_mean",
    "Mean cosine similarity of reconstructed frame embeddings",
    ["stream_id"]
)

METRIC_ACTIVE_STREAMS = Gauge(
    "adve_active_streams",
    "Number of active ingest RTSP streams"
)

METRIC_LICENSE_DAYS = Gauge(
    "adve_license_days_remaining",
    "Number of days remaining on current ADVE enterprise license"
)


@router.get("/health", summary="System Health & License Check")
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (Fallback)"

    license_info = verify_license_key()
    METRIC_LICENSE_DAYS.set(license_info.get("days_remaining", 9999))

    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "gpu_available": gpu_available,
        "device": gpu_name,
        "license_valid": license_info.get("valid", True),
        "license_tier": license_info.get("tier", "evaluation"),
        "customer": license_info.get("customer", "Evaluation"),
        "max_streams": license_info.get("max_streams", 5),
        "days_remaining": license_info.get("days_remaining", 9999)
    }


@router.get("/metrics", summary="Prometheus Operational Metrics")
async def get_metrics():
    license_info = verify_license_key()
    METRIC_LICENSE_DAYS.set(license_info.get("days_remaining", 9999))
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
