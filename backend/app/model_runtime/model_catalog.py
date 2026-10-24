"""ModelCatalog —— 只读 Model 注册表。

职责仅三个：get / list / exists。

数据来源（P0/P1）：
    1. 内置 Model（`vlx-seek`）
    2. Legacy Bridge 只读映射出的 **VLM** Model（model_id = 原 provider.id，不改 ID）
    3. 未来 models.json（**本轮不创建、不读取**）

★硬约束★
    VLX 的稳定逻辑 ID 永远是 `vlx-seek`。
    **禁止**创建 `vlx-seek-1.5-10b` 作为 model_id —— 版本信息只能进
    `model_name` / `model_version` / `metadata`。

Registry / Resolver 校验 model_id、取 Model.enabled / task_types / default_*，
统一走这里；不允许自己读旧配置。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

__all__ = ["ModelCatalog", "BUILTIN_MODELS"]

# ── 内置 Model ──────────────────────────────────────────────────────────────
# 只描述"这是什么模型"，不包含任何运行位置信息（那属于 Deployment）。
BUILTIN_MODELS: Dict[str, Dict[str, Any]] = {
    "vlx-seek": {
        "model_id": "vlx-seek",
        "name": "vlx-seek",
        "display_name": "VLX-Seek 检测服务",
        "kind": "vlx",
        "protocol": "vlx_detection",
        "provider": "omlab",
        "model_name": "VLX-Seek-1.5-10B",
        "model_version": "1.5",
        "task_types": ["detect", "reasoning_detection"],
        "default_coord_space": "pixel_xyxy",
        "default_timeout": 300.0,
        "default_max_tokens": 256,
        "default_task": "detection",
        "default_lang": "zh",
        "enabled": True,
        "source": "builtin",
        "metadata": {"version_from": "server /model-info"},
        "created_at": "",
        "updated_at": "",
    },
}


class ModelCatalog:
    """只读 Model 注册表。构造时可注入 provider（Legacy Bridge 或未来的 models.json）。"""

    def __init__(self, providers: Optional[List] = None, *, enable_legacy: bool = True):
        """providers: 产出 Dict[model_id, model_dict] 的 callable 列表（惰性求值）。"""
        self._providers = list(providers or [])
        self._enable_legacy = bool(enable_legacy)
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None

    # ── 加载 ────────────────────────────────────────────────────────────────
    def _load(self) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = deepcopy(BUILTIN_MODELS)
        if self._enable_legacy:
            for provider in self._providers:
                try:
                    part = provider() or {}
                except Exception:
                    # 某个来源坏了不能拖垮整个 catalog
                    continue
                for mid, model in part.items():
                    if not isinstance(model, dict):
                        continue
                    model = dict(model)
                    model.setdefault("model_id", mid)
                    if mid in merged:
                        # 内置优先：外部来源只能补空字段，不能改写内置定义
                        merged[mid] = {**model, **merged[mid]}
                    else:
                        merged[mid] = model
        return merged

    def all(self) -> Dict[str, Dict[str, Any]]:
        if self._cache is None:
            self._cache = self._load()
        return self._cache

    def refresh(self) -> None:
        self._cache = None

    # ── 三个公开方法 ────────────────────────────────────────────────────────
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        if not model_id:
            return None
        m = self.all().get(str(model_id))
        return deepcopy(m) if m else None

    def list(self) -> List[Dict[str, Any]]:
        return [deepcopy(m) for m in self.all().values()]

    def exists(self, model_id: str) -> bool:
        return bool(model_id) and str(model_id) in self.all()
