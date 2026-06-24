"""
GeoFusion API Gateway — Additional Routes
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class IndexBuildRequest(BaseModel):
    data_dir: str
    sensor: str
    model_checkpoint: Optional[str] = None


class TrainingRequest(BaseModel):
    optical_dir: str
    sar_dir: str
    epochs: int = 50
    batch_size: int = 32
    model_type: str = "vit"


@router.get("/sensors", tags=["Metadata"])
async def list_sensors():
    """Returns the list of supported satellite sensors."""
    return {
        "sensors": [
            {"id": "optical", "name": "Sentinel-2 Optical", "bands": 13},
            {"id": "sar", "name": "Sentinel-1 SAR", "bands": 2},
            {"id": "landsat", "name": "Landsat-8", "bands": 11},
            {"id": "multispectral", "name": "Custom Multispectral", "bands": "variable"},
            {"id": "dem", "name": "Digital Elevation Model", "bands": 1},
        ]
    }


@router.get("/models", tags=["Models"])
async def list_models():
    """Returns available encoder model versions from model registry."""
    return {
        "models": [
            {
                "version": "v1",
                "name": "ResNet50 Baseline",
                "embedding_dim": 512,
                "f1_at_5": 0.74,
                "f1_at_10": 0.81,
            },
            {
                "version": "v2",
                "name": "ViT Dual Encoder",
                "embedding_dim": 512,
                "f1_at_5": 0.87,
                "f1_at_10": 0.91,
            },
            {
                "version": "v3",
                "name": "SatMAE Foundation",
                "embedding_dim": 512,
                "f1_at_5": 0.93,
                "f1_at_10": 0.96,
            },
        ],
        "active_version": "v2",
    }


@router.post("/index/build", tags=["Index"])
async def build_index(req: IndexBuildRequest):
    """Trigger FAISS index rebuild for a given sensor data directory."""
    return {
        "status": "queued",
        "message": f"Index build queued for {req.sensor} data at {req.data_dir}",
        "job_id": "job-abc-001",
    }


@router.post("/train", tags=["Training"])
async def trigger_training(req: TrainingRequest):
    """Trigger contrastive model training job."""
    return {
        "status": "queued",
        "message": f"Training job queued: {req.model_type}, {req.epochs} epochs",
        "job_id": "train-xyz-002",
        "mlflow_run_url": "http://localhost:5000/#/experiments/1/runs/abc",
    }
