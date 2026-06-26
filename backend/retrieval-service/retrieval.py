"""
GeoFusion — Retrieval Engine
==============================
Wraps a FAISS index + metadata store to provide cross-modal
nearest-neighbour search over satellite image embeddings.
"""

import json
import os
from typing import List, Literal, Optional

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class GeoFusionRetriever:
    """
    Loads a FAISS index + metadata file and performs similarity search.

    Index file: vector-database/faiss.index   (IndexFlatIP, L2-normalised vectors)
    Metadata file: vector-database/metadata.json
        {
          "0": {"id": "optical_001", "sensor": "optical", "lat": .., "lon": ..},
          "1": {"id": "sar_001", "sensor": "sar", "lat": .., "lon": ..},
          ...
        }
    """

    def __init__(
        self,
        index_path: str = "vector-database/faiss.index",
        metadata_path: Optional[str] = None,
        embedding_dim: int = 512,
    ):
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not installed. `pip install faiss-cpu`.")

        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = metadata_path or os.path.join(
            os.path.dirname(index_path) or ".", "metadata.json"
        )

        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()

    # ── Index management ──────────────────────────────────────────────────
    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        # Inner product on L2-normalised vectors == cosine similarity
        return faiss.IndexFlatIP(self.embedding_dim)

    def _load_metadata(self) -> dict:
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                data = json.load(f)
                if "tiles" in data:
                    return {str(i): tile for i, tile in enumerate(data["tiles"])}
                return data
        return {}

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    # ── Index building ────────────────────────────────────────────────────
    def add(self, embeddings: np.ndarray, records: List[dict]):
        """
        Add a batch of embeddings + metadata records to the index.

        Args:
            embeddings: (N, embedding_dim) float32 array, L2-normalised
            records: list of dicts with keys {id, sensor, lat, lon, ...}
        """
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        for offset, record in enumerate(records):
            self.metadata[str(start_idx + offset)] = record

    # ── Search ────────────────────────────────────────────────────────────
    def search(  # noqa: C901
        self,
        query_embedding: List[float],
        sensor: Optional[str] = None,
        top_k: int = 10,
        retrieval_mode: Literal["cross", "same"] = "cross",
    ) -> List[dict]:
        """
        Search for the top-k most similar embeddings.

        retrieval_mode:
            "cross" -> only return results from a *different* sensor than query
            "same"  -> only return results from the *same* sensor as query
        """
        if self.index.ntotal == 0:
            # Fallback for demo when no vectors are added yet
            mock_results = []
            for idx, record in self.metadata.items():
                if (
                    retrieval_mode == "cross"
                    and sensor
                    and record.get("sensor") == sensor
                ):
                    continue
                if (
                    retrieval_mode == "same"
                    and sensor
                    and record.get("sensor") != sensor
                ):
                    continue
                mock_results.append(
                    {
                        "id": record.get("id", record.get("tile_id", str(idx))),
                        "sensor": record.get("sensor", "unknown"),
                        "similarity": max(0.4, 0.95 - (len(mock_results) * 0.05)),
                        "location": {
                            "lat": record.get("lat"),
                            "lon": record.get("lon"),
                        },
                        "metadata": record,
                    }
                )
                if len(mock_results) >= top_k:
                    break
            return mock_results

        query = np.array([query_embedding], dtype="float32")
        # Over-fetch to allow for post-filtering by sensor
        fetch_k = min(self.index.ntotal, max(top_k * 5, top_k))
        scores, indices = self.index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self.metadata.get(str(idx))
            if record is None:
                continue

            if retrieval_mode == "cross" and sensor and record.get("sensor") == sensor:
                continue
            if retrieval_mode == "same" and sensor and record.get("sensor") != sensor:
                continue

            results.append(
                {
                    "id": record.get("id", record.get("tile_id", str(idx))),
                    "sensor": record.get("sensor", "unknown"),
                    "similarity": float(score),
                    "location": {
                        "lat": record.get("lat"),
                        "lon": record.get("lon"),
                    },
                    "metadata": record,
                }
            )
            if len(results) >= top_k:
                break

        return results

    @property
    def size(self) -> int:
        return self.index.ntotal
