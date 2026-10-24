"""Small stdlib-only bridge to RunHive SSO and account settings."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _runtime_file(env_name: str, filename: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser()
    workspace = Path(__file__).resolve().parents[3]
    return workspace / "runtime" / filename


def _read_secret(env_name: str, filename: str) -> bytes:
    path = _runtime_file(env_name, filename)
    secret = path.read_text(encoding="utf-8").strip().encode("utf-8")
    if len(secret) < 64:
        raise RuntimeError(f"RunHive secret is missing or too short: {path}")
    return secret


def _api_base() -> str:
    return os.environ.get("NEXUSHIVE_API_URL", "http://127.0.0.1:8080").rstrip("/")


class RunHiveServiceError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__("RunHive service request failed")
        self.status_code = status_code


def _response_json(request: Request) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RunHiveServiceError(exc.code if exc.code in (401, 403) else 503) from None
    except json.JSONDecodeError:
        raise RunHiveServiceError(502) from None
    except (URLError, TimeoutError, OSError):
        raise RunHiveServiceError(503) from None
    if not isinstance(data, dict) or data.get("code") != 1 or not isinstance(data.get("data"), dict):
        raise RunHiveServiceError(502)
    return data["data"]


def redeem_ticket(ticket: str) -> dict[str, Any]:
    body = urlencode({"ticket": ticket}).encode("ascii")
    request = Request(
        f"{_api_base()}/api/Runos/exchange",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    return _response_json(request)


def read_account_settings(admin_id: int) -> dict[str, Any]:
    request = Request(
        f"{_api_base()}/api/Runos/settings?admin_id={int(admin_id)}",
        headers={"X-RunOS-Internal-Key": _read_secret("NEXUSHIVE_INTERNAL_SECRET_FILE", "runos-internal-secret.txt").decode("ascii")},
        method="GET",
    )
    return _response_json(request).get("settings", {})


def write_account_settings(admin_id: int, settings: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps({"admin_id": int(admin_id), "settings": settings}).encode("utf-8")
    request = Request(
        f"{_api_base()}/api/Runos/settings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-RunOS-Internal-Key": _read_secret("NEXUSHIVE_INTERNAL_SECRET_FILE", "runos-internal-secret.txt").decode("ascii"),
        },
        method="POST",
    )
    return _response_json(request).get("settings", {})


def issue_session(account: dict[str, Any]) -> str:
    now = int(time.time())
    claims = {
        "iss": "nexushive",
        "aud": "runos",
        "sub": int(account["admin_id"]),
        "username": str(account.get("username", "")),
        "iat": now,
        "exp": now + 8 * 60 * 60,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    signature = hmac.new(
        _read_secret("RUNOS_SESSION_SECRET_FILE", "runos-session-secret.txt"),
        encoded,
        hashlib.sha256,
    ).digest()
    return encoded.decode("ascii") + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def verify_session(token: str) -> dict[str, Any]:
    try:
        payload_part, signature_part = token.split(".", 1)
        payload_bytes = payload_part.encode("ascii")
        supplied = base64.urlsafe_b64decode(signature_part + "=" * (-len(signature_part) % 4))
        expected = hmac.new(
            _read_secret("RUNOS_SESSION_SECRET_FILE", "runos-session-secret.txt"),
            payload_bytes,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("Invalid session signature")
        claims = json.loads(base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4)))
        if claims.get("iss") != "nexushive" or claims.get("aud") != "runos":
            raise ValueError("Invalid session issuer")
        if int(claims.get("exp", 0)) <= int(time.time()) or int(claims.get("sub", 0)) <= 0:
            raise ValueError("Expired session")
        return claims
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or expired RunOS session") from exc


def token_from_request(request: Any) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.cookies.get("runos_session", "")
