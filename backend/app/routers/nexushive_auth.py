"""RunHive single-sign-on endpoints for RunOS."""

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.nexushive import issue_session, redeem_ticket, token_from_request, verify_session, RunHiveServiceError

router = APIRouter()


class TicketExchange(BaseModel):
    ticket: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")


@router.post("/api/auth/exchange")
async def exchange_ticket(payload: TicketExchange) -> dict[str, Any]:
    try:
        account = redeem_ticket(payload.ticket)
    except RunHiveServiceError as exc:
        detail = "RunHive login ticket is invalid or expired" if exc.status_code == 401 else "RunHive account is unavailable" if exc.status_code == 403 else "RunHive sign-in service is unavailable"
        raise HTTPException(status_code=exc.status_code, detail=detail) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail="RunHive login ticket is invalid or expired") from exc

    owner_admin_id = int(account.get("admin_id", 0))
    if owner_admin_id <= 0:
        raise HTTPException(status_code=403, detail="RunHive administrator account is unavailable")
    if (
        os.environ.get("RUNOS_ALLOW_CROSS_TENANT_EXCHANGE") != "1"
        and owner_admin_id != int(os.environ.get("RUNOS_OWNER_ADMIN_ID", "1"))
    ):
        raise HTTPException(status_code=403, detail="This RunOS worker belongs to another RunHive account")

    return {
        "access_token": issue_session(account),
        "token_type": "bearer",
        "expires_in": 8 * 60 * 60,
        "account": {"admin_id": owner_admin_id, "username": account.get("username", "")},
    }


@router.get("/api/auth/me")
async def current_account(request: Request) -> dict[str, Any]:
    try:
        claims = verify_session(token_from_request(request))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="RunOS session is missing or expired") from exc
    return {"admin_id": int(claims["sub"]), "username": claims.get("username", "")}
