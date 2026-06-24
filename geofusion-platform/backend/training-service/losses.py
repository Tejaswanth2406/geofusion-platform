"""
GeoFusion — Multi-Modal Contrastive Loss
===========================================
Implements a symmetric InfoNCE-style contrastive loss for aligning
optical and SAR (or any two modality) embeddings in a shared space.

Positive pair  : optical crop  <-> SAR crop of the SAME geography
Negative pairs : all other crops in the batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalContrastiveLoss(nn.Module):
    """
    Symmetric cross-modal contrastive loss (CLIP-style).

    Args:
        temperature: softmax temperature; lower = sharper distribution
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings_a: (B, D) optical embeddings, L2-normalised
            embeddings_b: (B, D) SAR embeddings, L2-normalised (paired by index)

        Returns:
            scalar loss
        """
        batch_size = embeddings_a.shape[0]
        device = embeddings_a.device

        # Cosine similarity matrix (B, B)
        logits = (embeddings_a @ embeddings_b.T) / self.temperature

        labels = torch.arange(batch_size, device=device)

        loss_a_to_b = self.cross_entropy(logits, labels)
        loss_b_to_a = self.cross_entropy(logits.T, labels)

        return (loss_a_to_b + loss_b_to_a) / 2.0


class PrototypeAlignmentLoss(nn.Module):
    """
    Optional auxiliary loss that pulls each modality's embeddings
    toward shared land-cover-class prototypes, improving alignment
    when explicit pairs are scarce but class labels are available.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=1)
        embeddings = F.normalize(embeddings, dim=1)
        logits = embeddings @ prototypes.T
        return F.cross_entropy(logits, labels)
