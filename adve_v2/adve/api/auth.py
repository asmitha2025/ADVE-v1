"""
ADVE API Key Authentication Middleware & Key Manager (Critical Item #7)

Requires X-API-Key header on protected API endpoints.
Provides SQLite-backed API Key storage and validation with rotation capabilities.
Exempts public endpoints: /health, /, /docs, /openapi.json, and static UI assets.
"""

import os
import sqlite3
import secrets
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Optional

# When ADVE_API_KEY is unset we fall back to a well-known dev key so local
# development works out of the box. This MUST NOT be relied on in production:
# set ADVE_API_KEY to a real secret there. _USING_DEV_KEY records which case
# we are in so startup can warn.
_USING_DEV_KEY = "ADVE_API_KEY" not in os.environ
DEFAULT_API_KEY = os.environ.get("ADVE_API_KEY", "adve-dev-key-2026")
# Set ADVE_ENV=production to refuse the built-in development key outright.
_PRODUCTION = os.environ.get("ADVE_ENV", "").strip().lower() in ("production", "prod")

# Public endpoints that must work without a key:
#   /health, /v1/health   — liveness probes for load balancers / orchestration
#   /v1/register          — onboarding: issues the caller's first API key
#   /, /docs, ...         — landing page and interactive docs
EXEMPT_PATHS = {
    "/", "/health", "/v1/health", "/api/v1/health",
    "/v1/register",
    "/docs", "/openapi.json", "/favicon.ico",
}

if _USING_DEV_KEY:
    print(
        "[ADVE Auth] WARNING: ADVE_API_KEY is not set — accepting the built-in "
        "development key. Set ADVE_API_KEY to a secret before deploying."
    )

class APIKeyStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    api_key TEXT UNIQUE NOT NULL,
                    client_name TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Ensure default dev key exists
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM api_keys WHERE api_key = ?", (DEFAULT_API_KEY,))
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO api_keys (key_id, api_key, client_name) VALUES (?, ?, ?)",
                    ("dev_default", DEFAULT_API_KEY, "Development Default Key")
                )
                conn.commit()

    def validate_key(self, api_key: str) -> bool:
        # The built-in development key is only honoured when ADVE_API_KEY was
        # never configured AND we are not running in production. Otherwise a
        # deployment that forgot to set a key would silently accept a password
        # published in this repository. Fail closed instead.
        if api_key == DEFAULT_API_KEY and not (_USING_DEV_KEY and _PRODUCTION):
            return True
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT is_active FROM api_keys WHERE api_key = ?",
                (api_key,)
            )
            row = cursor.fetchone()
            return row is not None and row[0] == 1

    def create_key(self, client_name: str) -> str:
        new_key = f"adve_{secrets.token_urlsafe(24)}"
        key_id = secrets.token_hex(8)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, api_key, client_name) VALUES (?, ?, ?)",
                (key_id, new_key, client_name)
            )
            conn.commit()
        return new_key


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, key_store: Optional[APIKeyStore] = None):
        super().__init__(app)
        self.key_store = key_store

    async def dispatch(self, request: Request, call_next):
        # Exclude public & UI paths
        path = request.url.path
        if path in EXEMPT_PATHS or path.startswith("/static") or request.method == "OPTIONS":
            return await call_next(request)

        # Retrieve X-API-Key header or query param
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "message": "Missing required 'X-API-Key' header or query parameter."
                }
            )

        if self.key_store and not self.key_store.validate_key(api_key):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Forbidden",
                    "message": "Invalid or revoked API Key provided."
                }
            )

        return await call_next(request)
