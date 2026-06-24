"""
Tests for the GeoFusion Retrieval Service API endpoints.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add retrieval-service to path
retrieval_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "retrieval-service")
)
sys.path.insert(0, retrieval_path)

if "main" in sys.modules:
    del sys.modules["main"]

from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Use TestClient with lifespan to trigger the retriever initialization
    with TestClient(app, raise_server_exceptions=False) as c:
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
