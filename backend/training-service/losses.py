"""
GeoFusion — Multi-Modal Contrastive Losses
============================================
Research-grade loss functions for aligning optical and SAR satellite
embeddings in a shared latent space.

Mathematical Foundation
-----------------------
The core hypothesis:
  Images from different sensors observing the same geographic region
  should have nearby representations in a common latent space.

Let:
  x_o ∈ R^(H×W×C_o)  — optical image
  x_s ∈ R^(H×W×C_s)  — SAR image

Modality-specific encoders map both into a shared d-dimensional space:
  z_o = f_o(x_o; θ_o),   z_s = f_s(x_s; θ_s),   z_o, z_s ∈ R^d

Cross-modal alignment objective (GeoFusion Theoretical Statement):
  min_{θ_o, θ_s} Σ_i  D( f_o(x_o^i), f_s(x_s^i) )
  subject to:          D( f_o(x_o^i), f_s(x_s^j) ) > m,  ∀ i ≠ j

This file implements:
  1. MultiModalContrastiveLoss  — Symmetric InfoNCE (CLIP-style)
  2. TripletContrastiveLoss     — Margin-based triplet loss
  3. PrototypeAlignmentLoss     — Class-prototype auxiliary loss
  4. HybridGeoLoss              — Weighted combination (recommended)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Symmetric InfoNCE Loss ────────────────────────────────────────────────
class MultiModalContrastiveLoss(nn.Module):
    """
    Symmetric cross-modal InfoNCE loss for optical ↔ SAR alignment.

    For a batch of N matched (optical, SAR) pairs, the InfoNCE loss is:

        L = -log [ exp(Sim(z_o, z_s) / τ) ]
                 [ ──────────────────────── ]
                 [ Σ_{k=1}^{N} exp(Sim(z_o, z_k) / τ) ]

    where:
        Sim(z_i, z_j) = (z_i · z_j) / (|z_i| |z_j|)  — cosine similarity
        τ (tau)       = temperature hyperparameter
        N             = batch size

    The symmetric form averages loss in both directions:
        L_total = (L_{o→s} + L_{s→o}) / 2

    Goal:
        Sim(z_o^i, z_s^i) ↑  (positive pair: same geography)
        Sim(z_o^i, z_s^j) ↓  (negative pair: different geography, i ≠ j)

    Args:
        temperature: τ in the InfoNCE formula. Lower = sharper probability
                     distribution, harder negative mining. Default: 0.07
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(temperature), requires_grad=False
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings_o: torch.Tensor,
        embeddings_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings_o: (B, D) — optical embeddings, L2-normalised
            embeddings_s: (B, D) — SAR embeddings, L2-normalised, paired by index

        Returns:
            scalar InfoNCE loss
        """
        B = embeddings_o.shape[0]
        device = embeddings_o.device

        # Ensure L2 normalisation: Sim(z_i, z_j) = z_i · z_j when ||z|| = 1
        embeddings_o = F.normalize(embeddings_o, p=2, dim=1)
        embeddings_s = F.normalize(embeddings_s, p=2, dim=1)

        # Cosine similarity matrix S ∈ R^(B×B), scaled by 1/τ
        # S[i, j] = Sim(z_o^i, z_s^j) / τ
        logits = (embeddings_o @ embeddings_s.T) / self.temperature

        # Diagonal entries are positive pairs (same geography index)
        labels = torch.arange(B, device=device)

        # L_{o→s}: for each optical query, find matching SAR
        loss_o_to_s = self.cross_entropy(logits, labels)
        # L_{s→o}: for each SAR query, find matching optical
        loss_s_to_o = self.cross_entropy(logits.T, labels)

        return (loss_o_to_s + loss_s_to_o) / 2.0

    def extra_repr(self) -> str:
        return f"temperature={self.temperature.item():.4f}"


# ─── 2. Triplet Contrastive Loss ──────────────────────────────────────────────
class TripletContrastiveLoss(nn.Module):
    """
    Margin-based triplet loss for cross-modal satellite embedding alignment.

    Given a triplet (anchor a, positive p, negative n):

        L = max( 0,  D(a, p) - D(a, n) + m )

    where:
        D(u, v) = 1 - Sim(u, v)  — cosine distance
        m       = margin (minimum separation between pos/neg distances)

    The loss enforces:
        D(a, p) < D(a, n) - m
    i.e., the positive pair is always closer than the negative by margin m.

    Here, the anchor is the optical image, positive is the matched SAR image
    from the same geography, and negatives are all other SAR images in the batch.

    Args:
        margin: m in the triplet formula. Default: 0.3
        hard_negative: If True, mine the hardest (closest) negative per anchor
                       instead of averaging over all negatives.
    """

    def __init__(self, margin: float = 0.3, hard_negative: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchors:   (B, D) optical embeddings — L2-normalised
            positives: (B, D) SAR embeddings of same geography — L2-normalised

        Returns:
            scalar triplet loss averaged over valid (non-zero) triplets
        """
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)

        # Cosine similarity matrix: Sim[i, j] = cos(anchor_i, positive_j)
        sim_matrix = anchors @ positives.T  # (B, B)

        # Positive distances: D(a_i, p_i) = 1 - Sim[i, i]
        pos_sim = sim_matrix.diagonal()  # (B,)
        pos_dist = 1.0 - pos_sim         # (B,)

        if self.hard_negative:
            # Hard negative mining: find closest non-matching positive
            # Mask the diagonal (true positives) before taking max
            B = anchors.shape[0]
            mask = torch.eye(B, dtype=torch.bool, device=anchors.device)
            sim_matrix_masked = sim_matrix.masked_fill(mask, float("-inf"))
            neg_sim, _ = sim_matrix_masked.max(dim=1)      # hardest negative
            neg_dist = 1.0 - neg_sim
        else:
            # Average over all negatives
            B = anchors.shape[0]
            mask = torch.eye(B, dtype=torch.bool, device=anchors.device)
            neg_sim = sim_matrix.masked_fill(mask, 0).sum(dim=1) / (B - 1)
            neg_dist = 1.0 - neg_sim

        # L = max(0,  D(a, p) - D(a, n) + m)
        losses = F.relu(pos_dist - neg_dist + self.margin)

        # Only count non-trivially satisfied triplets
        valid = losses > 0
        return losses[valid].mean() if valid.any() else losses.mean()

    def extra_repr(self) -> str:
        return f"margin={self.margin}, hard_negative={self.hard_negative}"


# ─── 3. Prototype Alignment Loss ─────────────────────────────────────────────
class PrototypeAlignmentLoss(nn.Module):
    """
    Auxiliary loss that pulls embeddings toward shared land-cover prototypes.

    Improves cross-modal alignment when explicit sensor pairs are scarce
    but semantic land-cover class labels are available.

    Each modality's embeddings are aligned to shared class centroids
    (prototypes) in the embedding space — enforcing that:
        z_o^{class_k} ≈ z_s^{class_k} ≈ prototype_k

    Args:
        num_classes:   Number of land-cover classes (e.g. 10 for EuroSAT)
        embedding_dim: Shared embedding space dimensionality d
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()
        # Learnable prototype matrix P ∈ R^(K × d)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalised embeddings (any modality)
            labels:     (B,) integer class labels

        Returns:
            scalar cross-entropy loss against prototype similarities
        """
        # Normalise prototypes onto the unit hypersphere
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        emb_norm = F.normalize(embeddings, p=2, dim=1)

        # Similarity to each prototype: Sim(z, prototype_k) = z · prototype_k
        logits = emb_norm @ proto_norm.T  # (B, K)
        return F.cross_entropy(logits, labels)


# ─── 4. Hybrid GeoFusion Loss ─────────────────────────────────────────────────
class HybridGeoLoss(nn.Module):
    """
    Production-recommended loss combining InfoNCE + Triplet for robust training.

        L_total = λ_nce · L_InfoNCE + λ_tri · L_Triplet

    Using both:
    - InfoNCE provides global alignment across the entire batch
    - Triplet enforces local margin constraints between specific pairs
    This combination has been shown to produce more generalizable embeddings
    for cross-modal remote sensing retrieval tasks.

    Args:
        temperature:     τ for InfoNCE (default: 0.07)
        margin:          m for Triplet (default: 0.3)
        lambda_nce:      Weight on InfoNCE loss (default: 1.0)
        lambda_triplet:  Weight on Triplet loss (default: 0.5)
        hard_negative:   Use hard negative mining in Triplet (default: True)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.3,
        lambda_nce: float = 1.0,
        lambda_triplet: float = 0.5,
        hard_negative: bool = True,
    ):
        super().__init__()
        self.infonce = MultiModalContrastiveLoss(temperature=temperature)
        self.triplet = TripletContrastiveLoss(margin=margin, hard_negative=hard_negative)
        self.lambda_nce = lambda_nce
        self.lambda_triplet = lambda_triplet

    def forward(
        self,
        embeddings_o: torch.Tensor,
        embeddings_s: torch.Tensor,
    ) -> dict:
        """
        Args:
            embeddings_o: (B, D) optical embeddings (will be normalised internally)
            embeddings_s: (B, D) SAR embeddings (will be normalised internally)

        Returns:
            dict with keys: total, infonce, triplet
        """
        l_nce = self.infonce(embeddings_o, embeddings_s)
        l_tri = self.triplet(embeddings_o, embeddings_s)
        total = self.lambda_nce * l_nce + self.lambda_triplet * l_tri
        return {"total": total, "infonce": l_nce, "triplet": l_tri}

    def extra_repr(self) -> str:
        return (
            f"λ_nce={self.lambda_nce}, λ_triplet={self.lambda_triplet}, "
            f"τ={self.infonce.temperature.item():.4f}"
        )
