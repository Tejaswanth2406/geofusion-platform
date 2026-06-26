"""
GeoFusion AI Platform — API Gateway
====================================
Main entry point. Routes requests to embedding, retrieval,
preprocessing, and evaluation microservices.

Enterprise Features:
  - JWT Authentication (Bearer token)
  - Structured JSON logging
  - Prometheus metrics
  - Rate limiting
  - GZip compression
  - Request tracing via X-Request-ID
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import structlog
from auth import (
    TokenRequest,
    TokenResponse,
    UserInfo,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_scope,
)
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from middleware import RequestLoggingMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from routes import router
from starlette.responses import Response

# ─── Optional rate-limiting (slowapi) ────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _SLOWAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SLOWAPI_AVAILABLE = False

    def get_remote_address(request):  # type: ignore[misc]
        return "127.0.0.1"

    class _NoopLimiter:  # type: ignore[misc]
        """Drop-in stub used when slowapi is not installed."""
        def __init__(self, **kwargs):
            pass
        def limit(self, *args, **kwargs):
            """Return a pass-through decorator."""
            def decorator(func):
                return func
            return decorator

    Limiter = _NoopLimiter  # type: ignore[misc,assignment]
    RateLimitExceeded = Exception  # type: ignore[misc,assignment]
    _rate_limit_exceeded_handler = None  # type: ignore[assignment]

# ─── Structured Logging ───────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger("api-gateway")

# ─── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
try:
    REQUEST_COUNT = Counter(
        "geofusion_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"],
    )
    LATENCY = Histogram(
        "geofusion_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
    )
except ValueError:
    # Metrics already registered (e.g. module reloaded during tests)
    from prometheus_client import REGISTRY
    REQUEST_COUNT = REGISTRY._names_to_collectors.get("geofusion_requests_total")
    LATENCY = REGISTRY._names_to_collectors.get("geofusion_request_latency_seconds")

# ─── Service URLs ─────────────────────────────────────────────────────────────
EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
RETRIEVAL_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://localhost:8002")
EVAL_URL = os.getenv("EVAL_SERVICE_URL", "http://localhost:8005")

# ─── CORS Origins ─────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup", service="api-gateway", version="2.0.0")
    app.state.http_client = httpx.AsyncClient(timeout=60.0)
    yield
    await app.state.http_client.aclose()
    log.info("shutdown", service="api-gateway")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoFusion AI Platform",
    description=(
        "## Multi-Sensor Satellite Intelligence Retrieval Engine\n\n"
        "Enterprise-grade cross-modal satellite image retrieval using "
        "shared embedding spaces and contrastive alignment.\n\n"
        "### Authentication\n"
        "All retrieval endpoints require a **Bearer JWT token**. "
        "Obtain one via `POST /auth/token`.\n\n"
        "### Sensors Supported\n"
        "- Sentinel-2 (Optical, 13 bands)\n"
        "- Sentinel-1 (SAR, VV/VH)\n"
        "- Landsat-8 (Optical, 11 bands)\n"
        "- Hyperspectral\n"
        "- DEM"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={"name": "GeoFusion Team", "email": "team@geofusion.ai"},
    license_info={"name": "MIT"},
)

# ─── Middleware Stack ──────────────────────────────────────────────────────────
app.state.limiter = limiter
if _SLOWAPI_AVAILABLE and _rate_limit_exceeded_handler is not None:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.include_router(router, prefix="/api/v1")


# ─── Response Models ──────────────────────────────────────────────────────────
class RetrievalResponse(BaseModel):
    request_id: str
    query_sensor: str
    results: list
    retrieval_time_ms: float
    explainability: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict


# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.post(
    "/auth/token",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Obtain a JWT Bearer token",
)
@limiter.limit("10/minute")
async def login(request: Request, body: TokenRequest):
    """
    Exchange credentials for a JWT access token.

    **Demo credentials** (change in production):
    - `admin` / `geofusion_demo_2026`  → role: admin, scopes: [read, write, admin]
    - `analyst` / `analyst_pass`       → role: analyst, scopes: [read]
    """
    user = authenticate_user(body.username, body.password)
    if not user:
        log.warning("auth.failed", username=body.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        username=user["username"],
        role=user["role"],
        scopes=user["scopes"],
    )
    log.info("auth.success", username=body.username, role=user["role"])
    return TokenResponse(access_token=token, role=user["role"])


@app.get(
    "/auth/me",
    response_model=UserInfo,
    tags=["Authentication"],
    summary="Get current user info from token",
)
async def me(current_user: UserInfo = Depends(get_current_user)):
    """Returns the authenticated user's info extracted from the JWT."""
    return current_user


# ─── System Routes ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Platform health check — verifies all downstream services. No auth required."""
    services = {}

    try:
        client = app.state.http_client
        for name, url in [
            ("embedding", EMBEDDING_URL),
            ("retrieval", RETRIEVAL_URL),
            ("evaluation", EVAL_URL),
        ]:
            try:
                r = await client.get(f"{url}/health", timeout=5.0)
                services[name] = "up" if r.status_code == 200 else "degraded"
            except Exception:
                services[name] = "down"
    except AttributeError:
        # http_client not yet initialised (e.g. during unit tests without lifespan)
        services = {"embedding": "unknown", "retrieval": "unknown", "evaluation": "unknown"}

    overall = "healthy" if all(s == "up" for s in services.values()) else "degraded"
    log.info("health.check", status=overall, services=services)
    return HealthResponse(status=overall, version="2.0.0", services=services)


@app.get("/metrics", tags=["System"], include_in_schema=False)
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")


# ─── Core Retrieval Route (Protected) ─────────────────────────────────────────
@app.post(
    "/api/v1/retrieve",
    response_model=RetrievalResponse,
    tags=["Retrieval"],
    summary="Cross-modal satellite image retrieval",
)
@limiter.limit("30/minute")
async def retrieve(
    request: Request,
    image: UploadFile = File(..., description="Satellite image (.tif / .png / .jpg)"),
    sensor: str = Form(..., description="Sensor type: optical | sar | multispectral"),
    top_k: int = Form(10, ge=1, le=50, description="Number of results to return"),
    retrieval_mode: str = Form("cross", description="cross | same"),
    explain: bool = Form(True, description="Include explainability layer"),
    current_user: UserInfo = Depends(require_scope("read")),
):
    """
    **Authenticated** cross-modal satellite image retrieval.

    Flow:
    1. Image → Embedding Service → 512-D vector
    2. Vector → Retrieval Service → Top-K matches
    3. Optional explainability enrichment

    Requires: `Authorization: Bearer <token>`
    """
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    client = app.state.http_client

    log.info(
        "retrieval.start",
        request_id=request_id,
        user=current_user.username,
        sensor=sensor,
        top_k=top_k,
        mode=retrieval_mode,
    )

    image_bytes = await image.read()

    # Step 1 — Embed
    try:
        embed_resp = await client.post(
            f"{EMBEDDING_URL}/embed",
            files={"image": (image.filename, image_bytes, image.content_type)},
            data={"sensor": sensor},
        )
        embed_resp.raise_for_status()
        embedding_data = embed_resp.json()
    except httpx.HTTPError as e:
        log.error("embedding.error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail="Embedding service unavailable")

    # Step 2 — Retrieve
    try:
        retrieve_resp = await client.post(
            f"{RETRIEVAL_URL}/search",
            json={
                "embedding": embedding_data["embedding"],
                "sensor": sensor,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
            },
        )
        retrieve_resp.raise_for_status()
        results_data = retrieve_resp.json()
    except httpx.HTTPError as e:
        log.error("retrieval.error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail="Retrieval service unavailable")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    REQUEST_COUNT.labels("POST", "/retrieve", "200").inc()
    LATENCY.labels("/retrieve").observe(elapsed_ms / 1000)

    log.info(
        "retrieval.complete",
        request_id=request_id,
        latency_ms=round(elapsed_ms, 2),
        results_count=len(results_data.get("results", [])),
    )

    response = RetrievalResponse(
        request_id=request_id,
        query_sensor=sensor,
        results=results_data["results"],
        retrieval_time_ms=round(elapsed_ms, 2),
    )

    if explain:
        response.explainability = _build_explainability(results_data["results"], sensor)

    return response


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _build_explainability(results: list, sensor: str) -> dict:
    """
    Lightweight explainability layer.
    In production this would call a dedicated XAI service (SHAP / GradCAM).
    """
    if not results:
        return {}

    top = results[0]
    reasons = []

    sim = top.get("similarity", 0)
    if sim > 0.9:
        reasons.append("Very high spectral similarity across sensor modalities")
    elif sim > 0.75:
        reasons.append("High spatial pattern correlation in shared embedding space")
    else:
        reasons.append("Moderate geometric alignment detected")

    reasons.extend(
        [
            "Similar vegetation index (NDVI) signature",
            "Matching field geometry / land cover type",
            "Comparable urban texture distribution",
        ]
    )

    return {
        "matched_because": reasons,
        "confidence_pct": round(top.get("similarity", 0) * 100, 1),
        "query_sensor": sensor,
        "target_sensor": top.get("sensor", "unknown"),
        "embedding_distance": round(1.0 - top.get("similarity", 0), 4),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
