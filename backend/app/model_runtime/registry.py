"""DeploymentRegistry —— 只负责 Deployment **静态配置**的 CRUD。

★职责边界★
    本类**不能**：
        set_status / touch_health / incr_requests / 改 RuntimeState
    那些一律属于 `runtime_registry.DeploymentRuntimeRegistry`。

★依赖注入★
    store  —— DeploymentStore（不 import 全局 settings）
    models —— ModelCatalog（校验 model_id、取 Model 定义）
    两者都由外部注入，本模块不 import 任何全局单例、不读旧配置。
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .deployment_store import DeploymentStore

__all__ = ["DeploymentRegistry", "DEPLOYMENT_FIELDS", "IMMUTABLE_FIELDS"]

# Deployment 静态配置字段集合（★刻意不含任何运行时字段★）
DEPLOYMENT_FIELDS = frozenset({
    "deployment_id",
    "model_id",
    "deployment_type",
    "node_id",
    "gpu_id",
    "host",
    "port",
    "base_url",
    "base_path",
    "protocol",
    "coord_space_override",
    "auth_type",
    "auth_ref",
    "timeout_override",
    "max_tokens_override",
    "enabled",
    "priority",
    "weight",
    "metadata",
    "legacy_ref",
    "created_at",
    "updated_at",
})

# 以下字段禁止出现在 Deployment 配置里（它们属于 RuntimeState）
FORBIDDEN_FIELDS = frozenset({
    "status", "runtime_host", "runtime_port", "latency_ms",
    "active_requests", "queue_length", "last_health_check",
    "health_error", "health_detail", "started_at", "last_used_at",
    "pid", "remote_root",
})

IMMUTABLE_FIELDS = frozenset({"deployment_id", "created_at"})

DEPLOYMENT_TYPES = ("external_service", "managed_worker", "remote_api")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class DeploymentRegistry:
    def __init__(self, store: DeploymentStore, models, *,
                 on_delete: Optional[Callable[[Dict[str, Any]], None]] = None,
                 id_factory: Optional[Callable[[Dict[str, Any]], str]] = None,
                 legacy_providers: Optional[List[Callable[[], List[Dict[str, Any]]]]] = None):
        """models: 提供 exists(model_id) 的对象（ModelCatalog）。

        legacy_providers: 产出 Deployment 列表的 callable（**只读**，来自 Legacy Bridge）。
            P2B 之前 deployments.json 根本不存在，若没有它 Resolver 将看不到任何实例。
            同名 deployment_id 以 **正式配置为准**，legacy 只补空缺。
        """
        self._store = store
        self._models = models
        self._on_delete = on_delete          # managed_worker 停进程的回调（可选注入）
        self._id_factory = id_factory
        self._legacy_providers = list(legacy_providers or [])
        self._lock = threading.RLock()

    # ── 写 ──────────────────────────────────────────────────────────────────
    def register(self, dep: Dict[str, Any]) -> str:
        data = dict(dep or {})
        with self._lock:
            did = str(data.get("deployment_id") or "").strip()
            if not did:
                did = self._make_id(data)
                data["deployment_id"] = did

            model_id = str(data.get("model_id") or "").strip()
            if not model_id:
                raise ValueError("model_id is required")
            if self._models is not None and not self._models.exists(model_id):
                raise ValueError(f"unknown model_id: {model_id}")
            data["model_id"] = model_id

            dtype = data.get("deployment_type") or "external_service"
            if dtype not in DEPLOYMENT_TYPES:
                raise ValueError(f"unknown deployment_type: {dtype}")
            data["deployment_type"] = dtype

            data.setdefault("enabled", True)
            data.setdefault("priority", 100)
            data.setdefault("weight", 100)
            if "metadata" not in data or data["metadata"] is None:
                data["metadata"] = {}
            data.setdefault("created_at", _now_iso())
            data["updated_at"] = _now_iso()

            row = {k: v for k, v in data.items() if k in DEPLOYMENT_FIELDS}
            self._store.upsert(row)
            return did

    def update(self, deployment_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            cur = self._store.get(deployment_id)
            if cur is None:
                raise KeyError(deployment_id)
            for bad in IMMUTABLE_FIELDS:
                if bad in patch and patch[bad] != cur.get(bad):
                    raise ValueError(f"{bad} is immutable")
            if "model_id" in patch and patch["model_id"] != cur.get("model_id"):
                mid = str(patch["model_id"] or "")
                if self._models is not None and not self._models.exists(mid):
                    raise ValueError(f"unknown model_id: {mid}")
            if "deployment_type" in patch and patch["deployment_type"] not in DEPLOYMENT_TYPES:
                raise ValueError(f"unknown deployment_type: {patch['deployment_type']}")

            merged = dict(cur)
            for k, v in (patch or {}).items():
                if k in DEPLOYMENT_FIELDS and k not in IMMUTABLE_FIELDS:
                    merged[k] = v
            merged["updated_at"] = _now_iso()
            row = {k: v for k, v in merged.items() if k in DEPLOYMENT_FIELDS}
            self._store.upsert(row)
            return row

    def delete(self, deployment_id: str) -> bool:
        with self._lock:
            cur = self._store.get(deployment_id)
            if cur is None:
                return False
            if self._on_delete is not None:
                try:
                    self._on_delete(cur)
                except Exception:
                    # 回调失败不阻断配置删除，但必须让调用方看见
                    pass
            return self._store.delete(deployment_id)

    def set_enabled(self, deployment_id: str, enabled: bool) -> Dict[str, Any]:
        return self.update(deployment_id, {"enabled": bool(enabled)})

    # ── 读 ──────────────────────────────────────────────────────────────────
    def _all_rows(self) -> List[Dict[str, Any]]:
        """正式配置（store）+ 只读合成的 legacy 部署。

        ★同名时正式配置胜出★：legacy 只是"没建过正式实例"时的兜底，
          一旦用户建了正式实例，就不该再被 legacy 顶掉。
        ★legacy 一律 priority=1000★（正式默认 100）→ 排序上正式永远在前。
        """
        rows: List[Dict[str, Any]] = list(self._store.list())
        seen = {str(r.get("deployment_id")) for r in rows if r.get("deployment_id")}
        for provider in self._legacy_providers:
            try:
                extra = provider() or []
            except Exception:
                # 某个来源坏了不能拖垮整个 Registry
                continue
            for row in extra:
                if not isinstance(row, dict):
                    continue
                did = str(row.get("deployment_id") or "")
                if not did or did in seen:
                    continue
                seen.add(did)
                rows.append(dict(row))
        return rows

    def get(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        row = next((r for r in self._all_rows()
                    if str(r.get("deployment_id")) == str(deployment_id)), None)
        return copy.deepcopy(row) if row else None

    def list(self, *, enabled_only: bool = False) -> List[Dict[str, Any]]:
        rows = self._all_rows()
        if enabled_only:
            rows = [r for r in rows if r.get("enabled", True)]
        return self._sorted(rows)

    def list_by_model(self, model_id: str, *, enabled_only: bool = True) -> List[Dict[str, Any]]:
        rows = [r for r in self._all_rows() if str(r.get("model_id")) == str(model_id)]
        if enabled_only:
            rows = [r for r in rows if r.get("enabled", True)]
        return self._sorted(rows)

    def list_by_node(self, node_id: str) -> List[Dict[str, Any]]:
        rows = [r for r in self._all_rows() if str(r.get("node_id")) == str(node_id)]
        return self._sorted(rows)

    # ── 内部 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _sorted(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(rows, key=lambda r: (int(r.get("priority", 100) or 100), str(r.get("deployment_id"))))

    def _make_id(self, data: Dict[str, Any]) -> str:
        if self._id_factory is not None:
            return str(self._id_factory(data))
        import uuid
        return "dep-" + uuid.uuid4().hex[:8]
