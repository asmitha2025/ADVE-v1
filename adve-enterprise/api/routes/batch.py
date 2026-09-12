"""
ADVE Enterprise — Batch Processing Endpoints
Submits and monitors offline file or archive video embedding tasks.
"""

import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status, BackgroundTasks

from worker.celery_app import process_batch_video
from core.pipeline_wrapper import EnterprisePipeline
from api.config import settings

router = APIRouter(prefix="/v1/batch", tags=["Batch Video Archive Ingestion"])

# In-memory batch job store fallback
batch_jobs: Dict[str, Dict[str, Any]] = {}


class BatchSubmitRequest(BaseModel):
    video_path: str
    job_id: Optional[str] = None
    no_validation: bool = True


def _run_local_batch_task(job_id: str, video_path: str, no_validation: bool):
    from core.db import db
    try:
        batch_jobs[job_id]["status"] = "processing"
        db.upsert_batch_job(job_id=job_id, video_path=video_path, status="processing")
        engine = EnterprisePipeline(device=settings.device)
        summary = engine.pipeline.process_video(video_path, no_validation=no_validation)
        batch_jobs[job_id]["status"] = "completed"
        batch_jobs[job_id]["results"] = summary
        db.upsert_batch_job(job_id=job_id, video_path=video_path, status="completed", results=summary)
    except Exception as e:
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error"] = str(e)
        db.upsert_batch_job(job_id=job_id, video_path=video_path, status="failed", error=str(e))


@router.post("", status_code=status.HTTP_202_ACCEPTED, summary="Submit Batch Video Job")
async def submit_batch_job(payload: BatchSubmitRequest, background_tasks: BackgroundTasks):
    from core.db import db
    job_id = payload.job_id or f"job_{uuid.uuid4().hex[:8]}"

    batch_jobs[job_id] = {
        "job_id": job_id,
        "video_path": payload.video_path,
        "status": "queued",
        "results": None,
        "error": None
    }
    db.upsert_batch_job(job_id=job_id, video_path=payload.video_path, status="queued")

    # Try dispatching to Celery worker if available; fallback to local background task
    try:
        celery_task = process_batch_video.delay(
            job_id=job_id,
            video_path=payload.video_path,
            no_validation=payload.no_validation
        )
        batch_jobs[job_id]["celery_task_id"] = celery_task.id
        mode = "celery"
    except Exception:
        background_tasks.add_task(
            _run_local_batch_task,
            job_id=job_id,
            video_path=payload.video_path,
            no_validation=payload.no_validation
        )
        mode = "local_async"

    return {
        "status": "accepted",
        "job_id": job_id,
        "execution_mode": mode,
        "video_path": payload.video_path
    }


@router.get("/{job_id}", summary="Get Batch Job Status & Summary")
async def get_batch_job_status(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found."
        )

    return batch_jobs[job_id]
