"""
Tests for the GeoFusion Retrieval Service API endpoints.
"""

import importlib.util
import os
import sys

import pytest
from fastapi.testclient import TestClient

retrieval_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "retrieval-service")
)


@pytest.fixture(scope="module")
def client():
    # Ensure retrieval-service is first in sys.path so `from retrieval import ...` works.
    if retrieval_path not in sys.path:
        sys.path.insert(0, retrieval_path)
    else:
        sys.path.remove(retrieval_path)
        sys.path.insert(0, retrieval_path)

    # Flush any stale cached modules before loading.
    for mod_name in ("retrieval", "main"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(
        "retrieval_service_main",
        os.path.join(retrieval_path, "main.py"),
    )
    retrieval_service_main = importlib.util.module_from_spec(spec)
    sys.modules["retrieval_service_main"] = retrieval_service_main

    old_main = sys.modules.get("main")
    sys.modules["main"] = retrieval_service_main
    try:
        spec.loader.exec_module(retrieval_service_main)
    finally:
        if old_main is not None:
            sys.modules["main"] = old_main
        else:
            sys.modules.pop("main", None)

    # Use TestClient as a context manager so the lifespan runs and
    # the retriever is initialised before tests execute.
    with TestClient(retrieval_service_main.app, raise_server_exceptions=False) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "up"
    assert data["service"] == "retrieval-service"
    assert "index_size" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "geofusion_search_total" in response.text


def test_search_endpoint_missing_body(client):
    response = client.post("/search")
    assert response.status_code == 422  # Unprocessable Entity
