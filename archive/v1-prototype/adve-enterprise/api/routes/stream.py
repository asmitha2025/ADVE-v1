"""
ADVE Enterprise — RTSP Stream Processing Endpoints
Manages live RTSP stream ingestion, stream metrics, and stream termination.
"""

import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from api.middleware.license import verify_license_key, check_stream_limit
from api.routes.health import METRIC_ACTIVE_STREAMS, METRIC_FRAMES_TOTAL, METRIC_ENCODER_SAVINGS, METRIC_COSINE_SIMILARITY
from worker.rtsp_reader import RTSPReader
from core.pipeline_wrapper import EnterprisePipeline

router = APIRouter(prefix="/v1/streams", tags=["RTSP Live Streams"])

# Global stream registry: stream_id -> {"reader": RTSPReader, "engine": EnterprisePipeline, "task": asyncio.Task}
active_streams: Dict[str, Dict[str, Any]] = {}


class StreamCreateRequest(BaseModel):
    rtsp_url: str
    stream_id: str
    callback_url: Optional[str] = None
    hw_accel: bool = True


@router.post("", status_code=status.HTTP_201_CREATED, summary="Start RTSP Stream Ingestion")
async def start_stream(payload: StreamCreateRequest):
    # Enforce license limits
    license_info = verify_license_key()
    check_stream_limit(len(active_streams), license_info)

    if payload.stream_id in active_streams:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stream ID '{payload.stream_id}' is already running."
        )

    reader = RTSPReader(
        url=payload.rtsp_url,
        stream_id=payload.stream_id,
        hw_accel=payload.hw_accel
    )
    engine = EnterprisePipeline()

    async def frame_callback(frame_idx: int, frame):
        res = engine.process_single_frame(frame, frame_idx=frame_idx)
        METRIC_FRAMES_TOTAL.labels(stream_id=payload.stream_id).inc()

        stats = engine.get_stats()
        METRIC_ENCODER_SAVINGS.labels(stream_id=payload.stream_id).set(stats["encoder_savings_pct"] / 100.0)
        METRIC_COSINE_SIMILARITY.labels(stream_id=payload.stream_id).set(stats["mean_cosine_sim"])

    # Launch background ingest loop
    task = asyncio.create_task(reader.read_loop(callback=frame_callback))

    active_streams[payload.stream_id] = {
        "reader": reader,
        "engine": engine,
        "task": task,
        "callback_url": payload.callback_url
    }

    # Persist in SQLite
    from core.db import db
    db.upsert_stream(
        stream_id=payload.stream_id,
        rtsp_url=payload.rtsp_url,
        status="active",
        metadata={"callback_url": payload.callback_url}
    )

    METRIC_ACTIVE_STREAMS.set(len(active_streams))

    return {
        "status": "started",
        "stream_id": payload.stream_id,
        "rtsp_url": payload.rtsp_url,
        "license_tier": license_info.get("tier"),
        "max_streams": license_info.get("max_streams")
    }


@router.get("/{stream_id}", summary="Get Stream Status & Compute Savings")
async def get_stream_status(stream_id: str):
    if stream_id not in active_streams:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream ID '{stream_id}' not found."
        )

    entry = active_streams[stream_id]
    reader: RTSPReader = entry["reader"]
    engine: EnterprisePipeline = entry["engine"]

    reader_stats = reader.get_stats()
    pipeline_stats = engine.get_stats()

    return {
        "stream_id": stream_id,
        "status": "active" if reader_stats["is_running"] else "stopped",
        "reader_stats": reader_stats,
        "pipeline_stats": pipeline_stats
    }


@router.delete("/{stream_id}", summary="Stop RTSP Stream")
async def stop_stream(stream_id: str):
    if stream_id not in active_streams:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream ID '{stream_id}' not found."
        )

    entry = active_streams.pop(stream_id)
    reader: RTSPReader = entry["reader"]
    task: asyncio.Task = entry["task"]

    reader.stop()
    task.cancel()

    from core.db import db
    db.upsert_stream(stream_id=stream_id, rtsp_url=reader.url, status="stopped")

    METRIC_ACTIVE_STREAMS.set(len(active_streams))

    return {
        "status": "stopped",
        "stream_id": stream_id
    }


@router.get("", summary="List All Active Streams")
async def list_streams():
    summary_list = []
    for sid, entry in active_streams.items():
        reader_stats = entry["reader"].get_stats()
        pipeline_stats = entry["engine"].get_stats()
        summary_list.append({
            "stream_id": sid,
            "running": reader_stats["is_running"],
            "processed_frames": pipeline_stats["total_frames"],
            "encoder_savings_pct": pipeline_stats["encoder_savings_pct"],
            "mean_cosine_sim": pipeline_stats["mean_cosine_sim"]
        })
    return {"active_streams_count": len(summary_list), "streams": summary_list}
