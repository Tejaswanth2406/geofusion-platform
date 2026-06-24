"""
GeoFusion AI Platform — Retrieval Service
============================================
Exposes the FAISS-backed GeoFusionRetriever over HTTP.
"""

import os
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from retrieval import GeoFusionRetriever

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../../vector-database/faiss.index")
METADATA_PATH = os.getenv("METADATA_DB_PATH", "../../vector-database/metadata.json")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

app = FastAPI(title="GeoFusion Retrieval Service", version="1.0.0")

retriever: Optional[GeoFusionRetriever] = None


@app.on_event("startup")
async def startup():
    global retriever
    logger.info(f"Loading FAISS index from {INDEX_PATH}")
    retriever = GeoFusionRetriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_dim=EMBEDDING_DIM,
    )
    logger.info(f"Retriever ready. {retriever.size} vectors indexed.")


class SearchRequest(BaseModel):
    embedding: List[float]
    sensor: Optional[str] = None
    top_k: int = 10
    retrieval_mode: Literal["cross", "same"] = "cross"


class IndexAddRequest(BaseModel):
    embeddings: List[List[float]]
    records: List[dict]


@app.get("/health")
async def health():
    return {
        "status": "up",
        "index_size": retriever.size if retriever else 0,
    }


@app.post("/search")
async def search(req: SearchRequest):
    """Search the vector index for the top-k nearest neighbours."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    if len(req.embedding) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding must have dimension {EMBEDDING_DIM}, got {len(req.embedding)}",
        )

    t0 = time.perf_counter()
    results = retriever.search(
        query_embedding=req.embedding,
        sensor=req.sensor,
        top_k=req.top_k,
        retrieval_mode=req.retrieval_mode,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "results": results,
        "count": len(results),
        "search_time_ms": round(elapsed_ms, 2),
        "index_size": retriever.size,
    }


@app.post("/index/add")
async def add_to_index(req: IndexAddRequest):
    """Add new embeddings + metadata records to the live index."""
    import numpy as np

    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")

    embeddings = np.array(req.embeddings, dtype="float32")
    retriever.add(embeddings, req.records)
    retriever.save()

    return {
        "status": "ok",
        "added": len(req.records),
        "index_size": retriever.size,
    }


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
