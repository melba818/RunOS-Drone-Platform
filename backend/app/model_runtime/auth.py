"""auth_ref → 真实 secret 的解析门面（P2B）。

★这一层只做"门面"，不重复实现读文件★
    真正的只读解析在 `secrets.LegacySecretResolver`；这里补上 Deployment 级别的规则：
        - auth_type 为空 / "none"      → 返回 None（允许无凭据）
        - auth_ref 为空                → 按 auth_type 决定是否报错
        - 解析失败                     → 抛 AuthError，**消息里绝不带 secret 本身**

★红线★
    1. 只读，绝不写 lmm_config.json / vlm_providers.json
    2. 绝不把 secret 复制进 Deployment / Model / 日志 / 异常消息
    3. 返回值只活在调用栈里，不缓存、不落盘
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .secrets import LegacySecretResolver, SecretNotAvailable, parse_auth_ref

__all__ = ["AuthError", "resolve_auth", "redact", "default_secret_resolver"]

_AUTH_NONE = ("", "none", "no", "no_auth", "public")

# 疑似密钥/令牌：连续 24 位以上 base62（含 - _ .）→ 一律打码
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{24,}")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}")


class AuthError(RuntimeError):
    """鉴权解析失败。消息不含 secret。"""


def redact(text: Optional[str]) -> str:
    """把文本里疑似 token / api_key 的片段打成 ******。

    用于 health_error / 异常消息 / 日志。宁可多打码，不可漏一个。
    """
    s = "" if text is None else str(text)
    if not s:
        return ""
    s = _SK_RE.sub("******", s)
    s = _TOKEN_RE.sub("******", s)
    return s


def default_secret_resolver() -> LegacySecretResolver:
    """默认指向 workspace/models 下的旧配置（只读）。"""
    try:
        from config import settings
        base = settings.PLATFORM_MODELS_DIR
    except Exception:  # pragma: no cover - settings 不可用时退化为空解析
        return LegacySecretResolver()
    return LegacySecretResolver(
        lmm_config_path=base / "lmm_config.json",
        vlm_providers_path=base / "vlm_providers.json",
    )


def resolve_auth(deployment: Dict[str, Any], resolver: Optional[LegacySecretResolver] = None
                 ) -> Optional[str]:
    """返回真实 secret；不需要凭据时返回 None；无法解析时抛 AuthError。"""
    dep = deployment or {}
    auth_type = str(dep.get("auth_type") or "").strip().lower()
    ref = str(dep.get("auth_ref") or "").strip()

    if auth_type in _AUTH_NONE:
        return None

    if not ref:
        # 声明了需要凭据却没给引用 → 明确报错，不允许静默无凭据调用
        raise AuthError(f"deployment 声明 auth_type={auth_type} 但未配置 auth_ref")

    kind = parse_auth_ref(ref)["kind"]
    if kind == "unknown":
        raise AuthError(f"无法识别的 auth_ref 前缀: {redact(ref)}")

    res = resolver if resolver is not None else default_secret_resolver()
    try:
        return res.get_secret(ref)
    except SecretNotAvailable as exc:
        raise AuthError(f"无法解析 auth_ref（{ref}）：{exc}") from None
