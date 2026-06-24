"""
Tests for the GeoFusion API Gateway (integration-style unit tests).
Uses FastAPI's TestClient — no running external services required.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "backend", "api-gateway")
)

httpx = pytest.importorskip("httpx", reason="httpx not installed")
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the API Gateway app."""
    # Patch slow external calls at import time by setting env vars
    os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://localhost:8001")
    os.environ.setdefault("RETRIEVAL_SERVICE_URL", "http://localhost:8002")
    os.environ.setdefault("EVAL_SERVICE_URL", "http://localhost:8005")
    os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_testing_only")

    gateway_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend", "api-gateway")
    )
    # Ensure the gateway path is first in sys.path so its modules take precedence
    if gateway_path not in sys.path:
        sys.path.insert(0, gateway_path)
    else:
        sys.path.remove(gateway_path)
        sys.path.insert(0, gateway_path)

    import importlib.util

    # Remove any cached gateway sub-modules that could have been loaded from
    # another service's path (e.g. auth, middleware, routes) before we exec.
    for mod_name in ("auth", "middleware", "routes", "main"):
        sys.modules.pop(mod_name, None)

    spec = importlib.util.spec_from_file_location(
        "api_gateway_main", os.path.join(gateway_path, "main.py")
    )
    api_gateway_main = importlib.util.module_from_spec(spec)
    sys.modules["api_gateway_main"] = api_gateway_main

    # Register the gateway sub-modules under their bare names so that
    # relative imports inside main.py (from auth import …) resolve correctly.
    old_modules = {}
    for mod_name in ("auth", "middleware", "routes"):
        old_modules[mod_name] = sys.modules.get(mod_name)
        sub_spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(gateway_path, f"{mod_name}.py")
        )
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[mod_name] = sub_mod
        sub_spec.loader.exec_module(sub_mod)

    old_main = sys.modules.get("main")
    sys.modules["main"] = api_gateway_main
    try:
        spec.loader.exec_module(api_gateway_main)
    finally:
        if old_main is not None:
            sys.modules["main"] = old_main
        else:
            sys.modules.pop("main", None)

    # Use TestClient as a context manager so the lifespan runs and
    # app.state.http_client gets initialised before tests hit /health.
    with TestClient(api_gateway_main.app, raise_server_exceptions=False) as c:
        yield c


# ─── Health Endpoint ──────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data

    def test_health_has_services_field(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "services" in data
        assert isinstance(data["services"], dict)

    def test_health_has_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


# ─── Auth Endpoints ───────────────────────────────────────────────────────────
class TestAuthEndpoints:
    def test_token_endpoint_exists(self, client):
        resp = client.post(
            "/auth/token",
            json={"username": "admin", "password": "geofusion_demo_2026"},
        )
        # Either 200 (success) or 401 (bcrypt not matching in test env)
        assert resp.status_code in (200, 401)

    def test_wrong_credentials_return_401(self, client):
        resp = client.post(
            "/auth/token",
            json={"username": "admin", "password": "completely_wrong_pass"},
        )
        assert resp.status_code == 401

    def test_me_without_token_returns_403(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code in (401, 403)

    def test_retrieve_without_token_returns_401(self, client):
        resp = client.post(
            "/api/v1/retrieve",
            data={"sensor": "optical", "top_k": 5},
            files={"image": ("test.png", b"fake_image_bytes", "image/png")},
        )
        assert resp.status_code in (401, 403)


# ─── Docs ─────────────────────────────────────────────────────────────────────
class TestDocs:
    def test_openapi_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_accessible(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_json_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "GeoFusion AI Platform"
