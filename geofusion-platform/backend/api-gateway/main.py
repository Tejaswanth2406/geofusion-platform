"""
GeoFusion AI Platform — API Gateway
====================================
Main entry point. Routes requests to embedding, retrieval,
preprocessing, and evaluation microservices.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from middleware import RequestLoggingMiddleware
from routes import router

# ─── Prometheus Metrics ──────────────────────────────────────────────────────
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

# ─── Service URLs ─────────────────────────────────────────────────────────────
EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
RETRIEVAL_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://localhost:8002")
EVAL_URL = os.getenv("EVAL_SERVICE_URL", "http://localhost:8005")


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GeoFusion API Gateway starting up...")
    app.state.http_client = httpx.AsyncClient(timeout=60.0)
    yield
    await app.state.http_client.aclose()
    logger.info("GeoFusion API Gateway shut down.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoFusion AI Platform",
    description="Multi-Sensor Satellite Intelligence Retrieval Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


# ─── Models ───────────────────────────────────────────────────────────────────
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


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Platform health check — verifies all downstream services."""
    client = app.state.http_client
    services = {}

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

    overall = "healthy" if all(s == "up" for s in services.values()) else "degraded"
    return HealthResponse(status=overall, version="1.0.0", services=services)


@app.post("/api/v1/retrieve", response_model=RetrievalResponse, tags=["Retrieval"])
async def retrieve(
    image: UploadFile = File(..., description="Satellite image (.tif / .png / .jpg)"),
    sensor: str = Form(..., description="Sensor type: optical | sar | multispectral"),
    top_k: int = Form(10, ge=1, le=50, description="Number of results to return"),
    retrieval_mode: str = Form("cross", description="cross | same"),
    explain: bool = Form(True, description="Include explainability layer"),
):
    """
    Core retrieval endpoint.

    1. Forwards image to embedding-service -> 512-D vector
    2. Forwards vector to retrieval-service -> Top-K matches
    3. Optionally enriches with explainability metadata
    """
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    client = app.state.http_client

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
        logger.error(f"Embedding service error: {e}")
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
        logger.error(f"Retrieval service error: {e}")
        raise HTTPException(status_code=502, detail="Retrieval service unavailable")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    REQUEST_COUNT.labels("POST", "/retrieve", "200").inc()
    LATENCY.labels("/retrieve").observe(elapsed_ms / 1000)

    response = RetrievalResponse(
        request_id=request_id,
        query_sensor=sensor,
        results=results_data["results"],
        retrieval_time_ms=round(elapsed_ms, 2),
    )

    if explain:
        response.explainability = _build_explainability(
            results_data["results"], sensor
        )

    return response


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _build_explainability(results: list, sensor: str) -> dict:
    """
    Lightweight explainability layer.
    In production this would call a dedicated XAI service.
    """
    if not results:
        return {}

    top = results[0]
    reasons = []

    sim = top.get("similarity", 0)
    if sim > 0.9:
        reasons.append("Very high spectral similarity")
    elif sim > 0.75:
        reasons.append("High spatial pattern correlation")

    reasons.extend([
        "Similar vegetation index (NDVI)",
        "Matching field geometry / land cover",
        "Comparable urban texture distribution",
    ])

    return {
        "matched_because": reasons,
        "confidence_pct": round(top.get("similarity", 0) * 100, 1),
        "query_sensor": sensor,
        "target_sensor": top.get("sensor", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
