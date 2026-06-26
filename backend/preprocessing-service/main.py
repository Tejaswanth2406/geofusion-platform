"""
GeoFusion AI Platform — Preprocessing Service
=================================================
Exposes the ETL pipeline (metadata extraction, cloud masking,
radiometric correction, normalization, tiling) over HTTP.
"""

import os
import shutil
import tempfile

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pipeline import (
    cloud_mask,
    extract_metadata,
    normalize,
    radiometric_correction,
    register_dataset_entry,
    tile_image,
)
from prometheus_client import generate_latest
from pydantic import BaseModel
from starlette.responses import Response

app = FastAPI(title="GeoFusion Preprocessing Service", version="1.0.0")

REGISTRY_PATH = os.getenv("REGISTRY_PATH", "data/dataset_registry.json")


class TileResponse(BaseModel):
    tile_id: str
    sensor: str
    num_tiles: int
    metadata: dict


@app.get("/health")
async def health():
    return {"status": "up", "service": "preprocessing-service", "version": "1.0.0"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/process", response_model=TileResponse)
async def process_image(
    image: UploadFile = File(...),
    sensor: str = Form("optical"),
    tile_id: str = Form("tile001"),
    tile_size: int = Form(224),
):
    """
    Run the full ETL pipeline on an uploaded satellite image:
    metadata extraction -> cloud mask -> radiometric correction
    -> normalization -> tiling -> registry update.
    """
    suffix = os.path.splitext(image.filename or "")[-1] or ".tif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        metadata = extract_metadata(tmp_path)

        pil_img = Image.open(tmp_path).convert("RGB")
        array = np.array(pil_img)

        mask = cloud_mask(array)
        corrected = radiometric_correction(array)
        normalized = normalize(corrected)

        tiles = tile_image(normalized, tile_size=tile_size, stride=tile_size)

        register_dataset_entry(
            REGISTRY_PATH,
            tile_id=tile_id,
            sensor=sensor,
            metadata={
                **metadata,
                "num_tiles": len(tiles),
                "cloud_coverage_pct": round(100 * (1 - mask.mean()), 2),
            },
        )

        return TileResponse(
            tile_id=tile_id,
            sensor=sensor,
            num_tiles=len(tiles),
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)


@app.get("/registry")
async def get_registry():
    """Return the current dataset registry."""
    import json

    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)
