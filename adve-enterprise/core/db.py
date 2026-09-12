"""
ADVE Enterprise — SQLite Persistent State Database
Stores active streams, batch job states, and health logs durably across restarts.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger()

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "adve_state.db")


class StateDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS streams (
                    stream_id TEXT PRIMARY KEY,
                    rtsp_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    results_json TEXT,
                    error TEXT
                )
            """)
            conn.commit()

    def upsert_stream(self, stream_id: str, rtsp_url: str, status: str, metadata: Optional[Dict[str, Any]] = None):
        now = datetime.utcnow().isoformat()
        meta_str = json.dumps(metadata or {})
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO streams (stream_id, rtsp_url, status, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(stream_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    metadata_json=excluded.metadata_json
            """, (stream_id, rtsp_url, status, now, now, meta_str))
            conn.commit()

    def get_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM streams WHERE stream_id = ?", (stream_id,)).fetchone()
            if row:
                return {
                    "stream_id": row["stream_id"],
                    "rtsp_url": row["rtsp_url"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "metadata": json.loads(row["metadata_json"] or "{}")
                }
            return None

    def list_streams(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM streams").fetchall()
            return [
                {
                    "stream_id": r["stream_id"],
                    "rtsp_url": r["rtsp_url"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "metadata": json.loads(r["metadata_json"] or "{}")
                }
                for r in rows
            ]

    def delete_stream(self, stream_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM streams WHERE stream_id = ?", (stream_id,))
            conn.commit()

    def upsert_batch_job(self, job_id: str, video_path: str, status: str, results: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        now = datetime.utcnow().isoformat()
        res_str = json.dumps(results) if results else None
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO batch_jobs (job_id, video_path, status, created_at, results_json, error)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    results_json=excluded.results_json,
                    error=excluded.error
            """, (job_id, video_path, status, now, res_str, error))
            conn.commit()

    def get_batch_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row:
                return {
                    "job_id": row["job_id"],
                    "video_path": row["video_path"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "results": json.loads(row["results_json"]) if row["results_json"] else None,
                    "error": row["error"]
                }
            return None


db = StateDatabase()
