"""
Basic smoke tests for the GeoFusion retrieval engine.
Run with: pytest tests/
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "backend", "retrieval-service")
)

faiss = pytest.importorskip("faiss")
from retrieval import GeoFusionRetriever  # noqa: E402


@pytest.fixture
def tmp_retriever(tmp_path):
    index_path = str(tmp_path / "test.index")
    metadata_path = str(tmp_path / "test_metadata.json")
    return GeoFusionRetriever(
        index_path=index_path, metadata_path=metadata_path, embedding_dim=8
    )


def test_add_and_search(tmp_retriever):
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(5, 8)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    records = [
        {"id": f"img_{i}", "sensor": "optical" if i % 2 == 0 else "sar", "lat": 0, "lon": 0}
        for i in range(5)
    ]

    tmp_retriever.add(embeddings, records)
    assert tmp_retriever.size == 5

    results = tmp_retriever.search(
        query_embedding=embeddings[0].tolist(), sensor="optical", top_k=3, retrieval_mode="same"
    )
    assert len(results) <= 3
    assert all(r["sensor"] == "optical" for r in results)


def test_empty_index_returns_no_results(tmp_retriever):
    results = tmp_retriever.search(query_embedding=[0.0] * 8, top_k=5)
    assert results == []
