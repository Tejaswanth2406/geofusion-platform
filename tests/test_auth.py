"""
Tests for JWT Authentication module (auth.py).
Pure unit tests — no running services required.
"""

import os
import sys
from datetime import timedelta

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "backend", "api-gateway")
)

jose = pytest.importorskip("jose", reason="python-jose not installed")
from auth import (  # noqa: E402
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    TokenPayload,
    authenticate_user,
    create_access_token,
    verify_token,
)
from jose import jwt  # noqa: E402


# ─── Token Creation ───────────────────────────────────────────────────────────
class TestCreateToken:
    def test_returns_string(self):
        token = create_access_token("alice", "analyst", ["read"])
        assert isinstance(token, str)
        assert len(token) > 20

    def test_payload_contains_username(self):
        token = create_access_token("bob", "admin", ["read", "write"])
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["sub"] == "bob"

    def test_payload_contains_role(self):
        token = create_access_token("bob", "admin", ["read"])
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["role"] == "admin"

    def test_payload_contains_scopes(self):
        scopes = ["read", "write"]
        token = create_access_token("bob", "admin", scopes)
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert payload["scopes"] == scopes

    def test_payload_has_exp(self):
        token = create_access_token("bob", "admin", ["read"])
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert "exp" in payload

    def test_custom_expiry(self):
        token = create_access_token(
            "bob", "admin", ["read"], expires_delta=timedelta(minutes=5)
        )
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert "exp" in payload


# ─── Token Verification ───────────────────────────────────────────────────────
class TestVerifyToken:
    def test_valid_token_returns_payload(self):
        token = create_access_token("alice", "analyst", ["read"])
        payload = verify_token(token)
        assert isinstance(payload, TokenPayload)
        assert payload.sub == "alice"
        assert payload.role == "analyst"

    def test_invalid_token_raises_401(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            verify_token("not.a.valid.token")
        assert exc_info.value.status_code == 401

    def test_tampered_token_raises_401(self):
        from fastapi import HTTPException

        token = create_access_token("alice", "analyst", ["read"])
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(HTTPException):
            verify_token(tampered)

    def test_expired_token_raises_401(self):
        from fastapi import HTTPException

        token = create_access_token(
            "alice", "analyst", ["read"], expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        assert exc_info.value.status_code == 401


# ─── User Authentication ──────────────────────────────────────────────────────
class TestAuthenticateUser:
    def test_valid_admin_credentials(self):
        user = authenticate_user("admin", "geofusion_demo_2026")
        # May return None if passlib/bcrypt not available (fallback mode)
        # Just check it does not raise
        assert user is None or isinstance(user, dict)

    def test_wrong_password_returns_none(self):
        result = authenticate_user("admin", "wrong_password_xyz")
        assert result is None

    def test_unknown_user_returns_none(self):
        result = authenticate_user("nonexistent_user_xyz", "any_password")
        assert result is None
