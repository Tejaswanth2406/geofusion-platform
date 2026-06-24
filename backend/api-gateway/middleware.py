"""
GeoFusion API Gateway — Middleware
====================================
Request logging and lightweight auth middleware.
"""

import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        logger.info(f"[{request_id}] -> {request.method} {request.url.path}")

        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(f"[{request_id}] <- {response.status_code} ({elapsed:.1f}ms)")
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Token-based auth middleware.
    Skips /health and /docs in development.
    """

    SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        token = request.headers.get("Authorization", "")
        if not token.startswith("Bearer "):
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"detail": "Missing or invalid Authorization header"},
                status_code=401,
            )

        # In production: validate JWT here
        return await call_next(request)
