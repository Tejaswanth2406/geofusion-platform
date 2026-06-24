"""
GeoFusion AI Platform — Embedding Service
==========================================
Converts satellite images into 512-D embedding vectors
using the configured encoder (ResNet50 or ViT).
"""

import io
import os
import time
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger
from PIL import Image

from encoder import SatelliteEncoder

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_TYPE = os.getenv("MODEL_TYPE", "vit")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
CHECKPOINT = os.getenv("CHECKPOINT_PATH", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="GeoFusion Embedding Service", version="1.0.0")

# ─── Load Model at Startup ───────────────────────────────────────────────────
encoder: SatelliteEncoder = None


@app.on_event("startup")
async def startup():
    global encoder
    logger.info(f"Loading encoder: {MODEL_TYPE} on {DEVICE}")
    encoder = SatelliteEncoder(
        model_type=MODEL_TYPE,
        embedding_dim=EMBEDDING_DIM,
        checkpoint_path=CHECKPOINT if CHECKPOINT else None,
        device=DEVICE,
    )
    logger.info("Encoder ready.")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "up", "model": MODEL_TYPE, "device": DEVICE}


@app.post("/embed")
async def embed_image(
    image: UploadFile = File(...),
    sensor: str = Form("optical"),
    location_lat: float = Form(0.0),
    location_lon: float = Form(0.0),
):
    """
    Encode a satellite image into a 512-D embedding vector.

    Returns:
        {
          "id": "...",
          "sensor": "optical",
          "embedding": [...512 floats...],
          "location": {"lat": ..., "lon": ...},
          "inference_time_ms": ...
        }
    """
    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    t0 = time.perf_counter()
    embedding: np.ndarray = encoder.encode(pil_image, sensor=sensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "id": image.filename or "query",
        "sensor": sensor,
        "embedding": embedding.tolist(),
        "location": {"lat": location_lat, "lon": location_lon},
        "inference_time_ms": round(elapsed_ms, 2),
        "embedding_dim": len(embedding),
    }


@app.post("/embed_batch")
async def embed_batch(
    images: List[UploadFile] = File(...),
    sensor: str = Form("optical"),
):
    """Batch encode multiple images. Returns list of embeddings."""
    results = []
    for img_file in images:
        raw = await img_file.read()
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
        embedding = encoder.encode(pil_image, sensor=sensor)
        results.append({
            "id": img_file.filename,
            "sensor": sensor,
            "embedding": embedding.tolist(),
        })
    return {"embeddings": results, "count": len(results)}


@app.get("/model_info")
async def model_info():
    return {
        "model_type": MODEL_TYPE,
        "embedding_dim": EMBEDDING_DIM,
        "device": DEVICE,
        "supports_sensors": ["optical", "sar", "multispectral", "landsat"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
