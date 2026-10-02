"""
ADVE Enterprise — Vector Database Plugin System
Provides a pluggable adapter pattern for Qdrant, Milvus, and high-speed in-memory vector stores.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import structlog

logger = structlog.get_logger()


class VectorDB(ABC):
    @abstractmethod
    def store(self, vector: np.ndarray, stream_id: str, frame_idx: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5, min_score: float = 0.5) -> List[Dict[str, Any]]:
        pass


class InMemoryVectorPlugin(VectorDB):
    def __init__(self):
        self.store_records: List[Dict[str, Any]] = []

    def store(self, vector: np.ndarray, stream_id: str, frame_idx: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        self.store_records.append({
            "vector": np.array(vec_list, dtype=np.float32),
            "stream_id": stream_id,
            "frame_idx": frame_idx,
            "metadata": metadata or {}
        })
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 5, min_score: float = 0.5) -> List[Dict[str, Any]]:
        if not self.store_records:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-8

        results = []
        for record in self.store_records:
            db_vec = record["vector"]
            db_norm = np.linalg.norm(db_vec) + 1e-8
            score = float(np.dot(q_vec, db_vec) / (q_norm * db_norm))

            if score >= min_score:
                results.append({
                    "score": round(score, 4),
                    "stream_id": record["stream_id"],
                    "frame_idx": record["frame_idx"],
                    "metadata": record["metadata"]
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class QdrantPlugin(VectorDB):
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "adve_embeddings"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.fallback = InMemoryVectorPlugin()
        self.client_available = False

        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=qdrant_url, timeout=2.0)
            self.client_available = True
            logger.info("qdrant_plugin_connected", url=qdrant_url)
        except Exception as e:
            logger.warn("qdrant_client_not_available_using_inmemory_fallback", error=str(e))
            self.client_available = False

    def store(self, vector: np.ndarray, stream_id: str, frame_idx: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if self.client_available:
            try:
                from qdrant_client.models import PointStruct
                point_id = hash(f"{stream_id}_{frame_idx}") & 0xFFFFFFFF
                payload = {"stream_id": stream_id, "frame_idx": frame_idx, **(metadata or {})}
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)]
                )
                return True
            except Exception as e:
                logger.error("qdrant_upsert_failed_falling_back", error=str(e))

        return self.fallback.store(vector, stream_id, frame_idx, metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5, min_score: float = 0.5) -> List[Dict[str, Any]]:
        if self.client_available:
            try:
                search_res = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=top_k,
                    score_threshold=min_score
                )
                return [
                    {
                        "score": round(hit.score, 4),
                        "stream_id": hit.payload.get("stream_id"),
                        "frame_idx": hit.payload.get("frame_idx"),
                        "metadata": hit.payload
                    }
                    for hit in search_res
                ]
            except Exception as e:
                logger.error("qdrant_search_failed_falling_back", error=str(e))

        return self.fallback.search(query_vector, top_k, min_score)


def get_vector_db_adapter(adapter_type: str = "qdrant", qdrant_url: str = "http://localhost:6333") -> VectorDB:
    if adapter_type.lower() == "qdrant":
        return QdrantPlugin(qdrant_url=qdrant_url)
    return InMemoryVectorPlugin()
