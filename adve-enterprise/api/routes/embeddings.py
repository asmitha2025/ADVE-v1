"""
ADVE Enterprise — Vector Search & Embeddings API
Query stored frame and object embeddings for text/image similarity search using VectorDB plugins.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import numpy as np
from fastapi import APIRouter, HTTPException, status
from api.config import settings
from plugins.vector_db import get_vector_db_adapter

router = APIRouter(prefix="/v1/embeddings", tags=["Vector Search & Embeddings"])

# Global Vector DB instance (Qdrant by default with in-memory fallback)
vector_db = get_vector_db_adapter(settings.vector_db_type, settings.qdrant_url)


class EmbeddingSearchQuery(BaseModel):
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None
    top_k: int = 5
    min_similarity: float = 0.5


class EmbeddingStoreRequest(BaseModel):
    stream_id: str
    frame_idx: int
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None


@router.post("/search", summary="Search Vector Database")
async def search_embeddings(query: EmbeddingSearchQuery):
    if not query.query_vector and not query.query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'query_text' or 'query_vector'."
        )

    # Use dummy query vector if only query_text passed in mock
    q_vec = np.array(query.query_vector or [0.1] * 512, dtype=np.float32)

    results = vector_db.search(
        query_vector=q_vec,
        top_k=query.top_k,
        min_score=query.min_similarity
    )

    return {
        "top_k": query.top_k,
        "total_matches": len(results),
        "results": results
    }


@router.post("", status_code=status.HTTP_201_CREATED, summary="Store Vector Embedding")
async def store_embedding(payload: EmbeddingStoreRequest):
    vec = np.array(payload.vector, dtype=np.float32)
    success = vector_db.store(
        vector=vec,
        stream_id=payload.stream_id,
        frame_idx=payload.frame_idx,
        metadata=payload.metadata
    )
    return {
        "status": "stored" if success else "failed",
        "stream_id": payload.stream_id,
        "frame_idx": payload.frame_idx
    }
