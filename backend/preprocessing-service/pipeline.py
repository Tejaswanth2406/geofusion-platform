"""
GeoFusion — Satellite ETL / Preprocessing Pipeline
=======================================================
Raw Image -> Metadata Extraction -> Cloud Masking -> Radiometric
Correction -> Normalization -> Tiling -> Dataset Registry
"""

import json
import os
from typing import List, Tuple

import numpy as np
from loguru import logger
from PIL import Image

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available — falling back to PIL for raster I/O.")


def extract_metadata(image_path: str) -> dict:
    """Extract basic geospatial metadata from a raster file."""
    if RASTERIO_AVAILABLE and image_path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(image_path) as src:
            bounds = src.bounds
            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs),
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
            }
    img = Image.open(image_path)
    return {"width": img.width, "height": img.height, "bands": len(img.getbands())}


def cloud_mask(array: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """
    Simple brightness-based cloud mask placeholder.
    Production systems should use Fmask / s2cloudless instead.
    """
    if array.ndim == 3:
        brightness = array.mean(axis=-1)
    else:
        brightness = array
    mask = brightness < (threshold * brightness.max() + 1e-6)
    return mask


def radiometric_correction(array: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 1] per band."""
    array = array.astype("float32")
    for b in range(array.shape[-1] if array.ndim == 3 else 1):
        band = array[..., b] if array.ndim == 3 else array
        lo, hi = np.percentile(band, [2, 98])
        band[:] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    return array


def normalize(array: np.ndarray, mean=None, std=None) -> np.ndarray:
    """Standardise array using provided or computed mean/std."""
    mean = mean if mean is not None else array.mean()
    std = std if std is not None else array.std() + 1e-6
    return (array - mean) / std


def tile_image(
    array: np.ndarray, tile_size: int = 224, stride: int = 224
) -> List[Tuple[np.ndarray, int, int]]:
    """Split a large raster into fixed-size tiles for model input."""
    h, w = array.shape[:2]
    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = array[y : y + tile_size, x : x + tile_size]
            tiles.append((tile, x, y))
    return tiles


def register_dataset_entry(
    registry_path: str, tile_id: str, sensor: str, metadata: dict
) -> None:
    """Append a processed tile's metadata to the dataset registry JSON."""
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)

    registry[tile_id] = {"sensor": sensor, **metadata}

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
