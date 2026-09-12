"""
ADVE Enterprise — High-Performance Structlog Middleware
Formats logs in structured JSON format for DevOps / Grafana Loki parsing.
"""

import time
import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


class StructlogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=process_time_ms,
            client_ip=request.client.host if request.client else "unknown"
        )
        return response
