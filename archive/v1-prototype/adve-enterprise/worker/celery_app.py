"""
ADVE Enterprise — Redis-Backed Celery Worker Engine
Handles background batch video processing tasks and offline archive embeddings.
"""

import os
import sys
import time
import structlog
from celery import Celery

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from api.config import settings

logger = structlog.get_logger()

celery_app = Celery(
    "adve_worker",
    broker=settings.redis_url,
    backend=settings.redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(bind=True, name="tasks.process_batch_video")
def process_batch_video(self, job_id: str, video_path: str, no_validation: bool = True):
    """Celery background task for batch file / archive embedding generation."""
    logger.info("celery_batch_job_started", job_id=job_id, video_path=video_path)

    try:
        from core.pipeline_wrapper import EnterprisePipeline
        engine = EnterprisePipeline(device=settings.device)

        summary = engine.pipeline.process_video(video_path, no_validation=no_validation)
        logger.info("celery_batch_job_completed", job_id=job_id, summary=summary)
        return {
            "status": "completed",
            "job_id": job_id,
            "video_path": video_path,
            "summary": summary
        }
    except Exception as e:
        logger.error("celery_batch_job_failed", job_id=job_id, error=str(e))
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise e
