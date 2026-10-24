"""Legacy Bridge —— 把旧 VLX / VLM 配置**只读**映射为 Model + Deployment。

★绝对约束★
    * 只读 `lmm_config.json` 与 `vlm_providers.json`，**一个字节都不写**。
    * **禁止**把 api_key / token 复制到 Model / Deployment —— 只放 `auth_ref` 引用。
    * VLX 的逻辑 ID 恒为 `vlx-seek`（★禁止 `vlx-seek-1.5-10b`★），
      版本信息只进 `model_name` / `model_version` / `metadata`。
    * VLM 的 model_id **沿用原 provider.id，不改 ID**（历史 job 依赖它）。
    * 本阶段映射结果**不得参与推理**。

不直接 import `lmm.config` / `vlm_annotator`，避免拉起重量级依赖与全局单例；
路径由外部注入，默认指向 workspace/models 下的同名文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "LegacyBridge",
    "read_lmm_config",
    "read_vlm_providers",
    "LEGACY_VLX_DEPLOYMENT_ID",
]

LEGACY_VLX_DEPLOYMENT_ID = "legacy-vlx-default"
LEGACY_VLM_PREFIX = "legacy-"

VLX_MODEL_ID = "vlx-seek"          # ★稳定逻辑 ID，永不改★
VLX_MODEL_NAME = "VLX-Seek-1.5-10B"
VLX_MODEL_VERSION = "1.5"

# provider.coord_space → 统一坐标空间枚举
_COORD_SPACE_MAP = {
    "0-1000": "norm_0_1000",
    "1000": "norm_0_1000",
    "pixel": "pixel_xyxy",
    "0-1": "norm_0_1",
    "1": "norm_0_1",
}

# ── 只读读取 ────────────────────────────────────────────────────────────────
def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return {}


def read_lmm_config(path) -> Dict[str, Any]:
    return _read_json(Path(path))


def read_vlm_providers(path) -> Dict[str, Any]:
    data = _read_json(Path(path))
    return data if isinstance(data, dict) else {}


class LegacyBridge:
    """只读合成 Model / Deployment。文件缺失时返回空集合（不报错、不创建）。"""

    def __init__(self, *, lmm_config_path=None, vlm_providers_path=None):
        self._lmm_config_path = Path(lmm_config_path) if lmm_config_path else None
        self._vlm_providers_path = Path(vlm_providers_path) if vlm_providers_path else None

    # ── Model ───────────────────────────────────────────────────────────────
    def legacy_models(self) -> Dict[str, Dict[str, Any]]:
        """只包含 **VLM** Model（VLX 由 ModelCatalog 内置提供，不在这里重复定义）。"""
        out: Dict[str, Dict[str, Any]] = {}
        for pid, provider in read_vlm_providers(self._vlm_providers_path).items():
            if not isinstance(provider, dict):
                continue
            out[str(pid)] = self._vlm_model(str(pid), provider)
        return out

    @staticmethod
    def _vlm_model(pid: str, provider: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_id": pid,                       # ★沿用旧 provider.id，不改★
            "name": str(provider.get("name") or pid),
            "display_name": str(provider.get("name") or pid),
            "kind": "vlm",
            "protocol": "openai_compatible",
            "provider": "",                        # 由 vision_registry.kind_of 推断，此处不重复实现
            "model_name": str(provider.get("model") or ""),
            "model_version": "",
            "task_types": ["detect", "vqa"],
            "default_coord_space": _COORD_SPACE_MAP.get(
                str(provider.get("coord_space") or "0-1000"), "norm_0_1000"),
            "default_timeout": provider.get("timeout", 120),
            "default_max_tokens": None,
            "default_task": None,
            "default_lang": None,
            "enabled": True,
            "source": "legacy_mapped",
            "metadata": {"extra_instruction": provider.get("extra_instruction") or ""},
            "created_at": "",
            "updated_at": "",
        }

    # ── Deployment ──────────────────────────────────────────────────────────
    def legacy_deployments(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        vlx = self._vlx_deployment()
        if vlx:
            rows.append(vlx)
        rows.extend(self._vlm_deployments())
        return rows

    def _vlx_deployment(self) -> Optional[Dict[str, Any]]:
        cfg = read_lmm_config(self._lmm_config_path)
        if not cfg:
            return None
        node_id = cfg.get("vlx_seek_node_id") or None
        transport = str(cfg.get("vlx_seek_transport") or "ssh").strip().lower()
        target_configured = (bool(node_id and cfg.get("vlx_seek_port")) if transport == "ssh"
                             else bool(cfg.get("vlx_seek_base_url")) if transport == "direct" else False)
        return {
            "deployment_id": LEGACY_VLX_DEPLOYMENT_ID,
            "model_id": VLX_MODEL_ID,
            "deployment_type": "external_service",
            "node_id": node_id,
            "gpu_id": None,
            "host": "127.0.0.1",
            "port": cfg.get("vlx_seek_port", 8021),
            "base_url": cfg.get("vlx_seek_base_url") or "",
            "base_path": "",
            "protocol": "vlx_detection",
            "coord_space_override": None,
            "auth_type": "header_token",
            # ★只存引用，绝不存真实 token★
            "auth_ref": "lmm_config:vlx_seek_token",
            "timeout_override": cfg.get("vlx_seek_timeout"),
            "max_tokens_override": cfg.get("vlx_seek_max_new_tokens"),
            # A saved external service can be selected manually while automatic VLX remains off.
            "enabled": bool(target_configured and cfg.get("vlx_seek_token")),
            # ★priority=1000：正式 Deployment 默认 100 → 排序上**正式永远优先**，
            #   legacy 只当"用户还没建正式实例"时的兜底。
            #  （若将来 legacy 的 priority 被调小，Resolver 就可能永远选 legacy。）
            "priority": 1000,
            "weight": 100,
            "metadata": {
                "transport": transport,
                "automatic_enabled": bool(cfg.get("vlx_seek_enabled", False)),
                "task": cfg.get("vlx_seek_task") or "detection",
                "lang": cfg.get("vlx_seek_lang") or "zh",
            },
            "legacy_ref": "lmm_config:vlx_seek_*",
            "created_at": "",
            "updated_at": "",
        }

    def _vlm_deployments(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for pid, provider in read_vlm_providers(self._vlm_providers_path).items():
            if not isinstance(provider, dict):
                continue
            rows.append({
                "deployment_id": f"{LEGACY_VLM_PREFIX}{pid}",
                "model_id": str(pid),
                "deployment_type": "remote_api",
                "node_id": None,
                "gpu_id": None,
                "host": None,
                "port": None,
                "base_url": str(provider.get("base_url") or ""),
                "base_path": "",
                "protocol": "openai_compatible",
                "coord_space_override": None,
                "auth_type": "bearer",
                # ★只存引用，绝不存真实 api_key★
                "auth_ref": f"vlm_providers:{pid}",
                "timeout_override": provider.get("timeout"),
                "max_tokens_override": None,
                "enabled": True,
                # 同 VLX：legacy 兜底优先级放最后
                "priority": 1000,
                "weight": 100,
                "metadata": {"unmatched": provider.get("unmatched") or "drop"},
                "legacy_ref": f"vlm_providers:{pid}",
                "created_at": "",
                "updated_at": "",
            })
        return rows
