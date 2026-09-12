"""
ADVE Webhook Dispatcher & Retry Queue (Critical Item #9)

Dispatches drift alert notifications and event callbacks to client endpoints.
Uses exponential backoff retries (5s, 25s, 125s, 625s) with a durable SQLite queue.
"""

import os
import time
import json
import sqlite3
import urllib.request
import urllib.error
import threading
from typing import Dict, Any, Optional

BACKOFF_DELAYS = [5, 25, 125, 625]  # Exponential backoff in seconds

class WebhookDispatcher:
    def __init__(self, db_path: str = "adve_v2/data/webhooks.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()
        self._worker_thread = None
        self._running = False

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_url TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    next_retry_at REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def dispatch(self, target_url: str, event_type: str, data: Dict[str, Any]):
        """Queues a webhook payload for delivery."""
        payload = json.dumps({
            "event": event_type,
            "timestamp": time.time(),
            "data": data
        })
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO webhook_queue (target_url, payload, attempts, next_retry_at, created_at) VALUES (?, ?, 0, ?, ?)",
                (target_url, payload, now, now)
            )
            conn.commit()
        
        # Trigger immediate background send attempt
        threading.Thread(target=self.process_queue, daemon=True).start()

    def process_queue(self):
        """Processes pending webhooks due for retry."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, target_url, payload, attempts FROM webhook_queue WHERE status = 'pending' AND next_retry_at <= ? LIMIT 10",
                (now,)
            )
            rows = cursor.fetchall()

            for row_id, target_url, payload_str, attempts in rows:
                success = self._send_http(target_url, payload_str)
                if success:
                    cursor.execute("UPDATE webhook_queue SET status = 'delivered' WHERE id = ?", (row_id,))
                else:
                    new_attempts = attempts + 1
                    if new_attempts >= len(BACKOFF_DELAYS):
                        cursor.execute("UPDATE webhook_queue SET status = 'failed', attempts = ? WHERE id = ?", (new_attempts, row_id))
                    else:
                        delay = BACKOFF_DELAYS[new_attempts]
                        next_retry = time.time() + delay
                        cursor.execute(
                            "UPDATE webhook_queue SET attempts = ?, next_retry_at = ? WHERE id = ?",
                            (new_attempts, next_retry, row_id)
                        )
            conn.commit()

    def _send_http(self, target_url: str, payload_str: str) -> bool:
        try:
            req = urllib.request.Request(
                target_url,
                data=payload_str.encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "ADVE-Webhook/3.1"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status in (200, 201, 202, 204)
        except Exception:
            return False
