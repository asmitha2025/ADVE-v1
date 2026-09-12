"""
ADVE Enterprise — Runtime Configuration & Hot-Reload Endpoints
Allows live updates of pipeline parameters (anchor_budget, thresholds) without container restart.
"""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
from api.config import settings

router = APIRouter(prefix="/v1/config", tags=["Configuration Hot-Reload"])


class ConfigUpdateRequest(BaseModel):
    anchor_budget: Optional[int] = Field(None, ge=1, le=100, description="Max anchor frames per 100 frames")
    spatial_threshold: Optional[float] = Field(None, ge=0.01, le=1.0, description="Spatial delta trigger threshold")
    appearance_threshold: Optional[float] = Field(None, ge=0.01, le=1.0, description="Histogram correlation drop threshold")
    max_delta_frames: Optional[int] = Field(None, ge=1, le=300, description="Force keyframe interval")


@router.get("", summary="Get Current Configuration")
async def get_config():
    return {
        "anchor_budget": settings.anchor_budget,
        "spatial_threshold": settings.spatial_threshold,
        "appearance_threshold": settings.appearance_threshold,
        "max_delta_frames": settings.max_delta_frames,
        "success_threshold": settings.success_threshold,
        "device": settings.device,
        "vector_db_type": settings.vector_db_type
    }


@router.post("", summary="Hot-Reload Configuration")
async def update_config(payload: ConfigUpdateRequest):
    updates = {}
    if payload.anchor_budget is not None:
        settings.anchor_budget = payload.anchor_budget
        updates["anchor_budget"] = payload.anchor_budget

    if payload.spatial_threshold is not None:
        settings.spatial_threshold = payload.spatial_threshold
        updates["spatial_threshold"] = payload.spatial_threshold

    if payload.appearance_threshold is not None:
        settings.appearance_threshold = payload.appearance_threshold
        updates["appearance_threshold"] = payload.appearance_threshold

    if payload.max_delta_frames is not None:
        settings.max_delta_frames = payload.max_delta_frames
        updates["max_delta_frames"] = payload.max_delta_frames

    return {
        "status": "hot_reloaded",
        "updated_parameters": updates,
        "current_config": {
            "anchor_budget": settings.anchor_budget,
            "spatial_threshold": settings.spatial_threshold,
            "appearance_threshold": settings.appearance_threshold,
            "max_delta_frames": settings.max_delta_frames,
        }
    }
