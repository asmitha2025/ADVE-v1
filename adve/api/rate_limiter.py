"""
ADVE Rate Limiting Middleware (Component #1)

Sliding-window rate limiter per API key / IP address.
Enforces request rate limits (default 100 requests per minute).
Returns HTTP 429 Too Many Requests with Retry-After header upon threshold breach.
"""

import time
from collections import defaultdict
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, List, Tuple

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 100, window_size: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.history: Dict[str, List[float]] = defaultdict(list)
        # The identifier is caller-controlled (an API-key header), so an
        # attacker can mint unlimited distinct keys and grow this table without
        # bound. Prune expired identifiers periodically and cap the table size.
        self.max_identifiers = 10000
        self._requests_since_sweep = 0
        self._sweep_every = 500

    def _sweep(self, window_start: float) -> None:
        stale = [k for k, ts in self.history.items()
                 if not ts or ts[-1] <= window_start]
        for k in stale:
            del self.history[k]
        if len(self.history) > self.max_identifiers:
            # Still oversized after pruning: drop the least-recently-seen.
            for k, _ in sorted(self.history.items(), key=lambda kv: kv[1][-1])[
                    : len(self.history) - self.max_identifiers]:
                del self.history[k]

    async def dispatch(self, request: Request, call_next):
        # Exclude public / static paths from rate limiting
        path = request.url.path
        if path in {"/health", "/", "/docs", "/openapi.json"} or path.startswith("/static"):
            return await call_next(request)

        # Identify caller by X-API-Key header or client IP
        identifier = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
        now = time.time()
        window_start = now - self.window_size

        self._requests_since_sweep += 1
        if self._requests_since_sweep >= self._sweep_every:
            self._requests_since_sweep = 0
            self._sweep(window_start)

        # Clean old requests outside window
        timestamps = [ts for ts in self.history[identifier] if ts > window_start]
        self.history[identifier] = timestamps

        if len(timestamps) >= self.requests_per_minute:
            retry_after = int(self.window_size - (now - timestamps[0]))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(max(1, retry_after))},
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded ({self.requests_per_minute} req/min). Try again in {retry_after} seconds.",
                    "retry_after": retry_after
                }
            )

        self.history[identifier].append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.requests_per_minute - len(timestamps) - 1))
        return response
