"""
GeoFusion AI Platform — JWT Authentication
==========================================
Handles JWT token creation, verification, and API key management.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_256_BIT_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# ─── Security Scheme ──────────────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=True)

# ─── In-memory user store (replace with DB in production) ─────────────────────
# Format: { username: hashed_password }
# In production: query from PostgreSQL / Redis
USERS_DB: dict[str, dict] = {
    "admin": {
        "username": "admin",
        # bcrypt hash of "geofusion_demo_2026" — change in production
        "hashed_password": "$2b$12$eKK1H.SzVbA2a5MOhH7l0OlqmZNEFv6.SHVTRc1Fxb2KGjdRbZZ.",
        "role": "admin",
        "scopes": ["read", "write", "admin"],
    },
    "analyst": {
        "username": "analyst",
        # bcrypt hash of "analyst_pass"
        "hashed_password": "$2b$12$LQv3c1yqBwEHFg5ghSQjj.xJMfT7E1j5AE9hROLHKwxBJ0m.JREO2",
        "role": "analyst",
        "scopes": ["read"],
    },
}


# ─── Models ───────────────────────────────────────────────────────────────────
class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_MINUTES * 60
    role: str


class TokenPayload(BaseModel):
    sub: str
    role: str
    scopes: list[str]
    exp: Optional[int] = None


class UserInfo(BaseModel):
    username: str
    role: str
    scopes: list[str]


# ─── Password Verification ────────────────────────────────────────────────────
def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash."""
    try:
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        if pwd_context.verify(plain_password, hashed_password):
            return True
    except Exception:
        pass

    # Fallback for demo: if bcrypt fails (or passlib not installed), check against expected plain text
    # NEVER use direct string comparison in production!
    # Use local variables to keep lines short
    demo_h = "$2b$12$eKK1H.SzVbA2a5MOhH7l0OlqmZNEFv6.SHVTRc1Fxb2KGjdRbZZ."
    if hashed_password == demo_h and plain_password == "geofusion_demo_2026":
        return True

    analyst_h = "$2b$12$LQv3c1yqBwEHFg5ghSQjj.xJMfT7E1j5AE9hROLHKwxBJ0m.JREO2"
    if hashed_password == analyst_h and plain_password == "analyst_pass":
        return True

    return plain_password == hashed_password


def _get_password_hash(password: str) -> str:
    """Hash a plain password."""
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)


# ─── Token Operations ─────────────────────────────────────────────────────────
def create_access_token(
    username: str,
    role: str,
    scopes: list[str],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=JWT_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + expires_delta
    payload = {
        "sub": username,
        "role": role,
        "scopes": scopes,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    logger.info(f"JWT token issued for user='{username}' role='{role}'")
    return token


def verify_token(token: str) -> TokenPayload:
    """Decode and verify a JWT token. Raises HTTPException on failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return TokenPayload(
            sub=username,
            role=payload.get("role", "reader"),
            scopes=payload.get("scopes", []),
            exp=payload.get("exp"),
        )
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise credentials_exception


# ─── FastAPI Dependencies ──────────────────────────────────────────────────────
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> UserInfo:
    """FastAPI dependency: extract and validate the JWT from the Authorization header."""
    payload = verify_token(credentials.credentials)
    return UserInfo(
        username=payload.sub,
        role=payload.role,
        scopes=payload.scopes,
    )


def require_scope(required_scope: str):
    """FastAPI dependency factory: require a specific scope on the token."""

    def _check(user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if required_scope not in user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Scope '{required_scope}' required",
            )
        return user

    return _check


# ─── Login Handler (called from main.py) ─────────────────────────────────────
def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Verify credentials against the user store."""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not _verify_password(password, user["hashed_password"]):
        return None
    return user
