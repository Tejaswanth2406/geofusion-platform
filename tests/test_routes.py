"""
Tests for the GeoFusion API Gateway Routes endpoints.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add api-gateway to path
gateway_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "backend", "api-gateway")
)
sys.path.insert(0, gateway_path)

if "main" in sys.modules:
    del sys.modules["main"]

from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_list_sensors(client):
    response = client.get("/sensors")
    assert response.status_code == 200
    data = response.json()
    assert "sensors" in data
    assert len(data["sensors"]) > 0


def test_list_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "active_version" in data


def test_build_index(client):
    payload = {"data_dir": "/path/to/data", "sensor": "optical"}
    response = client.post("/index/build", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data


def test_trigger_training(client):
    payload = {
        "optical_dir": "/path/optical",
        "sar_dir": "/path/sar",
        "epochs": 10,
        "batch_size": 16,
        "model_type": "vit",
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "mlflow_run_url" in data
