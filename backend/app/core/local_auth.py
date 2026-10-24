"""Standalone local authentication for RunOS.

This module is the standalone counterpart of ``core.nexushive``. It
provides the same function names (``issue_session``,
``verify_session``, ``token_from_request``) so ``main.py`` can select
the auth backend with the ``RUNOS_AUTH_MODE`` environment variable:

- ``RUNOS_AUTH_MODE=local`` (default in the standalone deployment):
  username/password login against a single admin account configured
  through ``RUNOS_ADMIN_USER`` / ``RUNOS_ADMIN_PASSWORD``.
- ``RUNOS_AUTH_MODE=nexushive``: the original RunHive portal ticket
  flow (``core.nexushive``). Only useful when RunOS runs embedded
  inside a RunHive deployment.

Sessions are HMAC-SHA256 signed tokens, valid for 8 hours. The
signing secret is persisted to a file (default
``<RUNOS_DATA_ROOT>/.session_secret``, mode 0600) so sessions survive
restarts; if the file does not exist it is created with a fresh
random secret on first use. No credential is ever stored in the
repository: the demo admin password below is only a fallback default
and MUST be overridden in production deployments.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

SESSION_TTL_SECONDS = 8 * 60 * 60


class LocalAuthError(ValueError):
    """Raised for missing, malformed or expired sessions."""


def _secret_file_path() -> Path:
    override = os.environ.get("RUNOS_SESSION_SECRET_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    data_root = os.environ.get("RUNOS_DATA_ROOT", "").strip()
    if data_root:
        return Path(data_root).expanduser() / ".session_secret"
    return Path(__file__).parent.parent / "workspace" / ".session_secret"


def _signing_secret() -> bytes:
    env_secret = os.environ.get("RUNOS_SESSION_SECRET", "").strip()
    if env_secret:
        return env_secret.encode("utf-8")
    path = _secret_file_path()
    try:
        if path.exists():
            return path.read_bytes().strip()
    except OSError:
        pass
    secret = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secret)
    except OSError:
        # Fall back to an ephemeral secret: sessions will not survive
        # a restart, but authentication still works.
        pass
    return secret.encode("utf-8")


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def issue_session(account: dict[str, Any]) -> str:
    payload = {
        "sub": str(account.get("admin_id", 1)),
        "username": str(account.get("username", "")),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_signing_secret(), body.encode("ascii"), hashlib.sha256).digest()
    return f"v1.{body}.{_b64encode(sig)}"


def verify_session(token: str) -> dict[str, Any]:
    if not token:
        raise LocalAuthError("session is missing")
    try:
        version, body, sig = token.split(".")
        if version != "v1":
            raise LocalAuthError("unsupported session version")
        expected = hmac.new(_signing_secret(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64decode(sig), expected):
            raise LocalAuthError("session signature mismatch")
        claims = json.loads(_b64decode(body))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LocalAuthError("session is invalid") from exc
    if int(claims.get("exp", 0)) < int(time.time()):
        raise LocalAuthError("session is expired")
    return claims


def token_from_request(request: Any) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _password_matches(password: str) -> bool:
    """Constant-time check against the configured admin password.

    The hash is derived with PBKDF2-HMAC-SHA256 and a fixed salt so
    no password hash needs to be stored anywhere. This is a
    single-admin deployment convenience, not a multi-user identity
    system; deployments needing real user management should replace
    this module.
    """
    expected = os.environ.get("RUNOS_ADMIN_PASSWORD", "runos-demo")
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), b"runos-local-admin", 100_000)
    reference = hashlib.pbkdf2_hmac("sha256", expected.encode("utf-8"), b"runos-local-admin", 100_000)
    return hmac.compare_digest(candidate, reference)


def authenticate_admin(username: str, password: str) -> dict[str, Any] | None:
    expected_user = os.environ.get("RUNOS_ADMIN_USER", "admin")
    if username != expected_user or not _password_matches(password):
        return None
    return {"admin_id": 1, "username": expected_user}
