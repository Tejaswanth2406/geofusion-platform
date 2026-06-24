"""
GeoFusion — Satellite Image Encoder
=====================================
Wraps optical and SAR encoders with a unified interface.
Supports ResNet50, ViT-B/16, and SatMAE-compatible checkpoints.
"""

import os
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from loguru import logger
from PIL import Image

# Try to import timm (ViT support)
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available — ViT encoder disabled. Falling back to ResNet.")


# ─── Projection Head ─────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """Maps backbone features to the shared 512-D embedding space."""

    def __init__(self, in_features: int, embedding_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── ResNet50 Encoder ────────────────────────────────────────────────────────
class ResNetEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC
        self.projection = ProjectionHead(2048, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x).flatten(1)
        return self.projection(features)


# ─── ViT Encoder ─────────────────────────────────────────────────────────────
class ViTEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for ViT encoder.")
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        in_features = self.backbone.embed_dim  # 768 for ViT-B
        self.projection = ProjectionHead(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


# ─── Sensor-Specific Preprocessing ───────────────────────────────────────────
SENSOR_TRANSFORMS = {
    "optical": T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet
        ]
    ),
    "sar": T.Compose(
        [
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "multispectral": T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "landsat": T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


# ─── Unified Encoder Interface ────────────────────────────────────────────────
class SatelliteEncoder:
    """
    Unified encoder that wraps optical and SAR-specific models.

    Usage:
        encoder = SatelliteEncoder(model_type="vit", embedding_dim=512)
        vec = encoder.encode(pil_image, sensor="optical")
    """

    def __init__(
        self,
        model_type: Literal["resnet50", "vit"] = "vit",
        embedding_dim: int = 512,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

        # Build model
        if model_type == "vit" and TIMM_AVAILABLE:
            self.model = ViTEncoder(embedding_dim=embedding_dim)
        else:
            if model_type == "vit":
                logger.warning("timm unavailable, using ResNet50 instead.")
            self.model = ResNetEncoder(embedding_dim=embedding_dim)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state.get("model_state", state))
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image: Image.Image, sensor: str = "optical") -> np.ndarray:
        """
        Encode a PIL image into a 512-D L2-normalised embedding vector.

        Args:
            image: PIL Image
            sensor: Sensor type for preprocessing selection

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        transform = SENSOR_TRANSFORMS.get(sensor, SENSOR_TRANSFORMS["optical"])
        tensor = transform(image).unsqueeze(0).to(self.device)

        embedding = self.model(tensor)

        # L2 normalise so cosine similarity == dot product
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_batch(
        self, images: List[Image.Image], sensor: str = "optical"
    ) -> np.ndarray:
        """Batch encode. Returns (N, embedding_dim) array."""
        transform = SENSOR_TRANSFORMS.get(sensor, SENSOR_TRANSFORMS["optical"])
        tensors = torch.stack([transform(img) for img in images]).to(self.device)
        embeddings = self.model(tensors)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
