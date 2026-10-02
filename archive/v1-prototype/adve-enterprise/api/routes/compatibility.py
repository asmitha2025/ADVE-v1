"""
ADVE Enterprise — Compatibility Audit API Route
Generates downloadable PDF compatibility reports for customer video uploads.
"""

import os
import shutil
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse

from tools.report_generator import generate_compatibility_pdf
from core.pipeline_wrapper import EnterprisePipeline
from api.config import settings

router = APIRouter(prefix="/v1/compatibility", tags=["Compatibility Audit & PDF Reports"])


@router.post("", summary="Generate PDF Compatibility Audit Report")
async def generate_compatibility_report(
    video: UploadFile = File(...),
    customer: str = Form("Hikvision_India"),
):
    if not video.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video file provided.")

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video.filename)

    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        # Run pipeline evaluation
        engine = EnterprisePipeline(device=settings.device)
        summary = engine.pipeline.process_video(temp_video_path, no_validation=False)
        sim_history = engine.sim_history

        output_pdf = os.path.join(temp_dir, f"{customer}_Compatibility_Report.pdf")
        generate_compatibility_pdf(
            customer_name=customer,
            video_name=video.filename,
            summary_stats=summary,
            sim_history=sim_history,
            output_pdf_path=output_pdf
        )

        return FileResponse(
            path=output_pdf,
            filename=f"{customer}_Compatibility_Report.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate compatibility report: {str(e)}"
        )
