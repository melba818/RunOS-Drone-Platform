"""Standalone local sign-in endpoints for RunOS.

Response shapes:

- ``POST /api/auth/login``  -> ``{access_token, token_type,
  expires_in, account}``
- ``GET  /api/auth/me``     -> ``{admin_id, username}``

Included by ``main.py`` as the only sign-in backend.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.local_auth import (
    SESSION_TTL_SECONDS,
    authenticate_admin,
    issue_session,
    token_from_request,
    verify_session,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


@router.post("/api/auth/login")
async def login(payload: LoginRequest) -> dict[str, Any]:
    account = authenticate_admin(payload.username, payload.password)
    if account is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {
        "access_token": issue_session(account),
        "token_type": "bearer",
        "expires_in": SESSION_TTL_SECONDS,
        "account": {"admin_id": account["admin_id"], "username": account["username"]},
    }


@router.get("/api/auth/me")
async def current_account(request: Request) -> dict[str, Any]:
    try:
        claims = verify_session(token_from_request(request))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="RunOS session is missing or expired") from exc
    return {"admin_id": int(claims["sub"]), "username": claims.get("username", "")}
