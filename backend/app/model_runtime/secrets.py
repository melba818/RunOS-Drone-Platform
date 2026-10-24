"""auth_ref 前缀解析（SecretStore 之前的过渡层）。

★P0/P1 范围★
    * Deployment 里**只保存引用**（auth_ref），真实 key/token 继续留在旧配置里。
    * LegacySecretResolver **只读**旧配置文件，**绝不写**它们。
    * 本轮不参与推理，因此解析器可以只被调用来确认前缀格式；是否真去取值按需。

支持的 ref 前缀：
    "lmm_config:vlx_seek_token"   → 读 lmm_config.json 的 vlx_seek_token
    "vlm_providers:<pid>"         → 读 vlm_providers.json 中该 provider 的 api_key
    "secret:<id>"                 → 未来的 SecretStore（DPAPI），**本轮未实现**
    "" / None                     → 无凭据（auth_type = none）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["LegacySecretResolver", "SecretNotAvailable", "parse_auth_ref"]

PREFIX_LMM_CONFIG = "lmm_config:"
PREFIX_VLM_PROVIDERS = "vlm_providers:"
PREFIX_SECRET = "secret:"

KNOWN_PREFIXES = (PREFIX_LMM_CONFIG, PREFIX_VLM_PROVIDERS, PREFIX_SECRET)


class SecretNotAvailable(RuntimeError):
    """ref 格式不认识，或目标配置缺失。"""


def parse_auth_ref(ref: Optional[str]) -> Dict[str, str]:
    """把 auth_ref 拆成 {kind, target}。空值返回 kind='none'。"""
    ref = (ref or "").strip()
    if not ref:
        return {"kind": "none", "target": "", "raw": ""}
    for prefix, kind in (
        (PREFIX_LMM_CONFIG, "lmm_config"),
        (PREFIX_VLM_PROVIDERS, "vlm_providers"),
        (PREFIX_SECRET, "secret"),
    ):
        if ref.startswith(prefix):
            return {"kind": kind, "target": ref[len(prefix):], "raw": ref}
    return {"kind": "unknown", "target": ref, "raw": ref}


class LegacySecretResolver:
    """只读解析 auth_ref。文件路径由外部注入，不 import 全局 settings。"""

    def __init__(self, lmm_config_path=None, vlm_providers_path=None):
        self._lmm_config_path = Path(lmm_config_path) if lmm_config_path else None
        self._vlm_providers_path = Path(vlm_providers_path) if vlm_providers_path else None

    # ── 对外 ────────────────────────────────────────────────────────────────
    def supports(self, ref: Optional[str]) -> bool:
        return parse_auth_ref(ref)["kind"] in ("lmm_config", "vlm_providers", "none")

    def has_value(self, ref: Optional[str]) -> bool:
        """只回答"有没有"，**不返回值本身**（供 API 层生成 auth_configured 布尔）。"""
        info = parse_auth_ref(ref)
        try:
            value = self._read(info)
        except SecretNotAvailable:
            return False
        return bool(value)

    def get_secret(self, ref: Optional[str]) -> Optional[str]:
        """返回真实 secret。**调用方禁止打印 / 落盘 / 回给前端。**"""
        info = parse_auth_ref(ref)
        if info["kind"] == "none":
            return None
        if info["kind"] == "secret":
            raise SecretNotAvailable("SecretStore not implemented yet (phase P7)")
        if info["kind"] == "unknown":
            raise SecretNotAvailable(f"unrecognized auth_ref: {info['raw']}")
        return self._read(info)

    # ── 内部（只读）──────────────────────────────────────────────────────────
    def _read(self, info: Dict[str, str]) -> Optional[str]:
        if info["kind"] == "lmm_config":
            data = self._load_json(self._lmm_config_path, "lmm_config.json")
            return data.get(info["target"]) or None
        if info["kind"] == "vlm_providers":
            data = self._load_json(self._vlm_providers_path, "vlm_providers.json")
            provider = data.get(info["target"])
            if not isinstance(provider, dict):
                raise SecretNotAvailable(f"provider not found: {info['target']}")
            return provider.get("api_key") or None
        raise SecretNotAvailable(f"unsupported ref kind: {info['kind']}")

    @staticmethod
    def _load_json(path: Optional[Path], label: str) -> Dict[str, Any]:
        if path is None or not path.exists():
            raise SecretNotAvailable(f"{label} not configured/not found")
        try:
            return json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            raise SecretNotAvailable(f"cannot read {label}: {exc}") from exc
