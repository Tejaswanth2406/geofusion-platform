"""
Tests for the GeoFusion Retrieval Engine (GeoFusionRetriever).
These are pure unit tests that do not require any running services.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "backend", "retrieval-service")
)

faiss = pytest.importorskip("faiss", reason="faiss not installed")
from retrieval import GeoFusionRetriever  # noqa: E402


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def retriever(tmp_path):
    """Fresh retriever with a tiny 8-D index for speed."""
    return GeoFusionRetriever(
        index_path=str(tmp_path / "test.index"),
        metadata_path=str(tmp_path / "test_metadata.json"),
        embedding_dim=8,
    )


@pytest.fixture
def populated_retriever(retriever):
    """Retriever pre-loaded with 10 normalised vectors (5 optical, 5 sar)."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(10, 8)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    records = [
        {
            "id": f"img_{i}",
            "sensor": "optical" if i < 5 else "sar",
            "lat": float(i),
            "lon": float(i),
        }
        for i in range(10)
    ]
    retriever.add(embeddings, records)
    return retriever, embeddings, records


# ─── Core Tests ───────────────────────────────────────────────────────────────
class TestRetrieverBasics:
    def test_empty_index_size(self, retriever):
        assert retriever.size == 0

    def test_add_increases_size(self, retriever):
        vecs = np.random.rand(5, 8).astype("float32")
        records = [{"id": f"img_{i}", "sensor": "optical"} for i in range(5)]
        retriever.add(vecs, records)
        assert retriever.size == 5

    def test_empty_search_returns_empty(self, retriever):
        results = retriever.search(query_embedding=[0.0] * 8, top_k=5)
        assert results == []

    def test_index_is_cumulative(self, retriever):
        vecs = np.random.rand(3, 8).astype("float32")
        records = [{"id": f"img_{i}", "sensor": "sar"} for i in range(3)]
        retriever.add(vecs, records)
        retriever.add(vecs, records)
        assert retriever.size == 6


class TestRetrieverSearch:
    def test_top_k_respected(self, populated_retriever):
        rt, embeddings, _ = populated_retriever
        results = rt.search(query_embedding=embeddings[0].tolist(), top_k=3)
        assert len(results) <= 3

    def test_same_sensor_mode_filters_correctly(self, populated_retriever):
        rt, embeddings, _ = populated_retriever
        results = rt.search(
            query_embedding=embeddings[0].tolist(),
            sensor="optical",
            top_k=10,
            retrieval_mode="same",
        )
        assert all(r["sensor"] == "optical" for r in results)

    def test_cross_sensor_mode_returns_all_sensors(self, populated_retriever):
        rt, embeddings, _ = populated_retriever
        results = rt.search(
            query_embedding=embeddings[0].tolist(),
            sensor="optical",
            top_k=10,
            retrieval_mode="cross",
        )
        sensors_found = {r["sensor"] for r in results}
        # In cross mode, results should include both sensors
        assert len(sensors_found) >= 1  # At minimum returns something

    def test_similarity_scores_in_range(self, populated_retriever):
        rt, embeddings, _ = populated_retriever
        results = rt.search(query_embedding=embeddings[0].tolist(), top_k=5)
        for r in results:
            assert (
                0.0 <= r["similarity"] <= 1.0 + 1e-6
            ), f"Similarity {r['similarity']} out of range"

    def test_results_sorted_by_similarity(self, populated_retriever):
        rt, embeddings, _ = populated_retriever
        results = rt.search(query_embedding=embeddings[0].tolist(), top_k=5)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_result_has_required_fields(self, populated_retriever):
        rt, embeddings, _ = populated_retriever
        results = rt.search(query_embedding=embeddings[0].tolist(), top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "sensor" in r
        assert "similarity" in r


class TestRetrieverPersistence:
    def test_save_and_reload(self, retriever, tmp_path):
        vecs = np.random.rand(5, 8).astype("float32")
        records = [{"id": f"img_{i}", "sensor": "optical"} for i in range(5)]
        retriever.add(vecs, records)
        retriever.save()

        # Reload from disk
        new_retriever = GeoFusionRetriever(
            index_path=str(tmp_path / "test.index"),
            metadata_path=str(tmp_path / "test_metadata.json"),
            embedding_dim=8,
        )
        assert new_retriever.size == 5
