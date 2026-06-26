"""
GeoFusion AI Platform — Embedding Service (Enterprise)
======================================================
Converts satellite images into 512-D embedding vectors.
Enterprise features: structured JSON logging, Prometheus metrics,
health probes, model info endpoint.
"""

import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
import torch
from encoder import SatelliteEncoder
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

# ─── Structured Logging ───────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger("embedding-service")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_TYPE = os.getenv("MODEL_TYPE", "vit")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
CHECKPOINT = os.getenv("CHECKPOINT_PATH", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
try:
    EMBED_COUNT = Counter(
        "geofusion_embed_total", "Total embedding requests", ["sensor", "status"]
    )
    EMBED_LATENCY = Histogram(
        "geofusion_embed_latency_seconds", "Embedding inference latency", ["sensor"]
    )
except ValueError:
    # Already registered (module reload in tests)
    from prometheus_client import REGISTRY
    EMBED_COUNT = REGISTRY._names_to_collectors.get("geofusion_embed_total")
    EMBED_LATENCY = REGISTRY._names_to_collectors.get("geofusion_embed_latency_seconds")

encoder: Optional[SatelliteEncoder] = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder
    log.info("startup", service="embedding-service", model=MODEL_TYPE, device=DEVICE)
    encoder = SatelliteEncoder(
        model_type=MODEL_TYPE,
        embedding_dim=EMBEDDING_DIM,
        checkpoint_path=CHECKPOINT if CHECKPOINT else None,
        device=DEVICE,
    )
    log.info("encoder.ready", model=MODEL_TYPE, embedding_dim=EMBEDDING_DIM)
    yield
    log.info("shutdown", service="embedding-service")


app = FastAPI(
    title="GeoFusion Embedding Service",
    version="2.0.0",
    description="Satellite image → 512-D embedding vector encoder.",
    lifespan=lifespan,
)


# ─── Models ───────────────────────────────────────────────────────────────────
class EmbedResponse(BaseModel):
    id: str
    sensor: str
    embedding: List[float]
    location: dict
    inference_time_ms: float
    embedding_dim: int
    model_type: str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "up",
        "service": "embedding-service",
        "model": MODEL_TYPE,
        "device": DEVICE,
        "encoder_loaded": encoder is not None,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/embed", response_model=EmbedResponse)
async def embed_image(
    image: UploadFile = File(...),
    sensor: str = Form("optical"),
    location_lat: float = Form(0.0),
    location_lon: float = Form(0.0),
):
    """Encode a satellite image into a 512-D embedding vector."""
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not initialised yet")

    request_id = str(uuid.uuid4())
    log.info(
        "embed.start", request_id=request_id, sensor=sensor, filename=image.filename
    )

    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        log.error("embed.decode_error", request_id=request_id, error=str(e))
        EMBED_COUNT.labels(sensor, "error").inc()
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    t0 = time.perf_counter()
    embedding = encoder.encode(pil_image, sensor=sensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    EMBED_COUNT.labels(sensor, "success").inc()
    EMBED_LATENCY.labels(sensor).observe(elapsed_ms / 1000)

    log.info(
        "embed.complete",
        request_id=request_id,
        sensor=sensor,
        latency_ms=round(elapsed_ms, 2),
        embedding_dim=len(embedding),
    )

    return EmbedResponse(
        id=image.filename or request_id,
        sensor=sensor,
        embedding=embedding.tolist(),
        location={"lat": location_lat, "lon": location_lon},
        inference_time_ms=round(elapsed_ms, 2),
        embedding_dim=len(embedding),
        model_type=MODEL_TYPE,
    )


@app.post("/embed_batch")
async def embed_batch(
    images: List[UploadFile] = File(...),
    sensor: str = Form("optical"),
):
    """Batch encode multiple images."""
    results = []
    for img_file in images:
        raw = await img_file.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        embedding = encoder.encode(pil_image, sensor=sensor)
        results.append(
            {"id": img_file.filename, "sensor": sensor, "embedding": embedding.tolist()}
        )
    log.info("embed_batch.complete", count=len(results), sensor=sensor)
    return {"embeddings": results, "count": len(results)}


@app.get("/model_info")
async def model_info():
    return {
        "model_type": MODEL_TYPE,
        "embedding_dim": EMBEDDING_DIM,
        "device": DEVICE,
        "supports_sensors": ["optical", "sar", "multispectral", "landsat", "dem"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
