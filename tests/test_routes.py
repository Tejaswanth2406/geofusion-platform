"""
Tests for the GeoFusion API Gateway Routes endpoints.
"""

import os
import sys
import importlib.util

import pytest
from fastapi.testclient import TestClient

gateway_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "api-gateway"))


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://localhost:8001")
    os.environ.setdefault("RETRIEVAL_SERVICE_URL", "http://localhost:8002")
    os.environ.setdefault("EVAL_SERVICE_URL", "http://localhost:8005")
    os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_testing_only")

    # Ensure the gateway path is first so sub-module imports resolve correctly.
    if gateway_path not in sys.path:
        sys.path.insert(0, gateway_path)
    else:
        sys.path.remove(gateway_path)
        sys.path.insert(0, gateway_path)

    # Flush any stale sub-module cache before loading.
    for mod_name in ("auth", "middleware", "routes", "main"):
        sys.modules.pop(mod_name, None)

    # Pre-load each gateway sub-module under its bare name so that
    # `from auth import ...` etc. inside main.py resolve to the right files.
    for mod_name in ("auth", "middleware", "routes"):
        sub_spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(gateway_path, f"{mod_name}.py")
        )
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[mod_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)

    spec = importlib.util.spec_from_file_location(
        "api_gateway_main_routes", os.path.join(gateway_path, "main.py")
    )
    api_gateway_main = importlib.util.module_from_spec(spec)
    sys.modules["api_gateway_main_routes"] = api_gateway_main

    old_main = sys.modules.get("main")
    sys.modules["main"] = api_gateway_main
    try:
        spec.loader.exec_module(api_gateway_main)
    finally:
        if old_main is not None:
            sys.modules["main"] = old_main
        else:
            sys.modules.pop("main", None)

    with TestClient(api_gateway_main.app, raise_server_exceptions=False) as c:
        yield c


def test_list_sensors(client):
    response = client.get("/api/v1/sensors")
    assert response.status_code == 200
    data = response.json()
    assert "sensors" in data
    assert len(data["sensors"]) > 0


def test_list_models(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "active_version" in data


def test_build_index(client):
    payload = {"data_dir": "/path/to/data", "sensor": "optical"}
    response = client.post("/api/v1/index/build", json=payload)
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
    response = client.post("/api/v1/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "mlflow_run_url" in data
