"""
GeoFusion AI Platform — Retrieval Service (Enterprise)
======================================================
FAISS-backed vector search with structured logging and Prometheus metrics.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from retrieval import GeoFusionRetriever
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
log = structlog.get_logger("retrieval-service")

# ─── Config ───────────────────────────────────────────────────────────────────
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../../vector-database/faiss.index")
METADATA_PATH = os.getenv("METADATA_DB_PATH", "../../vector-database/metadata.json")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# ─── Prometheus Metrics ─────────────────────────────────────────────────
try:
    SEARCH_COUNT = Counter(
        "geofusion_search_total", "Total search requests", ["sensor", "mode"]
    )
    SEARCH_LATENCY = Histogram(
        "geofusion_search_latency_seconds",
        "FAISS search latency",
        ["mode"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
except ValueError:
    # Metrics already registered (e.g. module reloaded during tests)
    from prometheus_client import REGISTRY
    SEARCH_COUNT = REGISTRY._names_to_collectors.get("geofusion_search_total")
    SEARCH_LATENCY = REGISTRY._names_to_collectors.get("geofusion_search_latency_seconds")

retriever: Optional[GeoFusionRetriever] = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    log.info("startup", service="retrieval-service", index_path=INDEX_PATH)
    retriever = GeoFusionRetriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_dim=EMBEDDING_DIM,
    )
    log.info("retriever.ready", index_size=retriever.size, embedding_dim=EMBEDDING_DIM)
    yield
    log.info("shutdown", service="retrieval-service")


app = FastAPI(
    title="GeoFusion Retrieval Service",
    version="2.0.0",
    description="FAISS vector search + cross-modal similarity ranking.",
    lifespan=lifespan,
)


# ─── Models ───────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    embedding: List[float]
    sensor: Optional[str] = None
    top_k: int = 10
    retrieval_mode: Literal["cross", "same"] = "cross"


class IndexAddRequest(BaseModel):
    embeddings: List[List[float]]
    records: List[dict]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "up",
        "service": "retrieval-service",
        "index_size": retriever.size if retriever else 0,
        "embedding_dim": EMBEDDING_DIM,
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/search")
async def search(req: SearchRequest):
    """Search the FAISS index for top-k nearest neighbours."""
    request_id = str(uuid.uuid4())

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    if len(req.embedding) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding must have dimension {EMBEDDING_DIM}, got {len(req.embedding)}",
        )

    log.info(
        "search.start",
        request_id=request_id,
        sensor=req.sensor,
        top_k=req.top_k,
        mode=req.retrieval_mode,
    )

    t0 = time.perf_counter()
    results = retriever.search(
        query_embedding=req.embedding,
        sensor=req.sensor,
        top_k=req.top_k,
        retrieval_mode=req.retrieval_mode,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    SEARCH_COUNT.labels(req.sensor or "any", req.retrieval_mode).inc()
    SEARCH_LATENCY.labels(req.retrieval_mode).observe(elapsed_ms / 1000)

    log.info(
        "search.complete",
        request_id=request_id,
        results_count=len(results),
        latency_ms=round(elapsed_ms, 2),
        query_sensor=req.sensor,
    )

    return {
        "results": results,
        "count": len(results),
        "search_time_ms": round(elapsed_ms, 2),
        "index_size": retriever.size,
        "request_id": request_id,
    }


@app.post("/index/add")
async def add_to_index(req: IndexAddRequest):
    """Add new embeddings + metadata records to the live FAISS index."""
    import numpy as np

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    embeddings = np.array(req.embeddings, dtype="float32")
    retriever.add(embeddings, req.records)
    retriever.save()

    log.info("index.add", added=len(req.records), new_size=retriever.size)
    return {"status": "ok", "added": len(req.records), "index_size": retriever.size}


@app.get("/index/stats")
async def index_stats():
    return {
        "index_size": retriever.size if retriever else 0,
        "embedding_dim": EMBEDDING_DIM,
        "index_path": INDEX_PATH,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
