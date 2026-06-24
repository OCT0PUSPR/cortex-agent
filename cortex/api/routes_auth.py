"""Authentication routes: register, login (JWT), and API-key issuance."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings, get_settings
from ..db.engine import get_session
from ..db.repository import UserRepository
from ..security import create_access_token, generate_api_key
from .deps import Principal, get_principal
from .schemas import (
    ApiKeyRequest,
    ApiKeyResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Create a new user account and return an access token."""
    repo = UserRepository(session)
    if await repo.get_by_email(body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = await repo.create_user(body.email, body.password)
    await session.commit()
    token = create_access_token(
        user.id,
        settings.jwt_secret,
        settings.jwt_algorithm,
        settings.jwt_expire_minutes,
        {"is_admin": user.is_admin},
    )
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Authenticate and return a JWT access token."""
    repo = UserRepository(session)
    user = await repo.authenticate(body.email, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = create_access_token(
        user.id,
        settings.jwt_secret,
        settings.jwt_algorithm,
        settings.jwt_expire_minutes,
        {"is_admin": user.is_admin},
    )
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyRequest,
    principal: Principal = Depends(get_principal),
    session: AsyncSession = Depends(get_session),
) -> ApiKeyResponse:
    """Issue a new API key for the authenticated user (key shown once)."""
    if principal.kind != "user" or not principal.user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "A logged-in user (JWT) is required to mint API keys.")
    raw = generate_api_key()
    repo = UserRepository(session)
    api_key = await repo.create_api_key(
        principal.user_id,
        raw,
        name=body.name,
        allowed_tools=body.allowed_tools,
        rate_limit_per_minute=body.rate_limit_per_minute,
    )
    await session.commit()
    return ApiKeyResponse(id=api_key.id, name=api_key.name, api_key=raw, allowed_tools=api_key.allowed_tools)
