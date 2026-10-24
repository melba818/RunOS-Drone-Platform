"""DeploymentResolver —— **纯选择**，只回答"按什么顺序试哪几个实例"。

★严格职责边界（V1.1 修订）★
    本类绝对不能：
        调用模型 / 调用 Adapter / 修改 RuntimeState / 执行 failover / set_status
        / 记录推理成功或失败
    那些全部属于未来的 ModelRuntimeService（**P0/P1 尚不存在**）。

    它只做四件事：
        1. 取候选 Deployment（走 DeploymentRegistry）
        2. 读 RuntimeState 做过滤（unhealthy/disabled 淘汰，unknown 保留）
        3. 按 strategy 排序
        4. 返回候选顺序

★preferred 语义★
    allow_fallback=False 且指定 preferred_deployment_id → 最多返回这一项；
        它失败与否不在本层判定，本层**不会**偷偷补上别的实例。
    allow_fallback=True  → 最多返回 `limit`（默认 2）个候选。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .runtime_registry import EXCLUDED_STATUSES

__all__ = ["DeploymentResolver", "Resolution"]

STRATEGIES = ("priority", "preferred", "round_robin", "least_requests", "least_queue")
_IMPLEMENTED = ("priority", "preferred")


@dataclass
class Resolution:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    fallback_allowed: bool = True

    @property
    def first(self) -> Optional[Dict[str, Any]]:
        return self.candidates[0] if self.candidates else None


class DeploymentResolver:
    def __init__(self, registry, runtime, models):
        """registry: DeploymentRegistry；runtime: DeploymentRuntimeRegistry；models: ModelCatalog。"""
        self._registry = registry
        self._runtime = runtime
        self._models = models

    def resolve_candidates(self, model_id: str, *,
                           strategy: str = "priority",
                           preferred_deployment_id: Optional[str] = None,
                           allow_fallback: bool = True,
                           requirements: Optional[Dict[str, Any]] = None,
                           limit: int = 2) -> Resolution:
        req = requirements or {}
        limit = max(1, int(limit or 1))

        model = self._models.get(model_id) if self._models is not None else None
        if model is None:
            return Resolution([], f"model not found: {model_id}", allow_fallback)
        if not model.get("enabled", True):
            return Resolution([], f"model disabled: {model_id}", allow_fallback)

        rows = self._registry.list_by_model(model_id, enabled_only=True)
        rows = [r for r in rows if self._match_model(model, req)]
        rows = [r for r in rows if self._healthy_enough(r)]
        rows = [r for r in rows if self._match_runtime(r, req)]

        if not rows:
            return Resolution([], f"no available deployment for model: {model_id}", allow_fallback)

        rows = self._sort(rows, strategy)

        if preferred_deployment_id:
            preferred = [r for r in rows if str(r.get("deployment_id")) == str(preferred_deployment_id)]
            if preferred:
                others = [r for r in rows if r is not preferred[0]]
                if allow_fallback:
                    picked = preferred + others[:limit - 1]
                else:
                    picked = preferred[:1]
                return Resolution(picked, "preferred first", allow_fallback)
            if not allow_fallback:
                # ★严格模式：指定实例不可用时必须如实返回空，禁止换成别的实例★
                return Resolution([], f"preferred deployment unavailable: {preferred_deployment_id}", False)
            return Resolution(rows[:limit], f"preferred not usable, fallback others: {preferred_deployment_id}", True)

        return Resolution(rows[:limit], strategy, allow_fallback)

    # ── 过滤 ────────────────────────────────────────────────────────────────
    def _healthy_enough(self, row: Dict[str, Any]) -> bool:
        """unknown 允许参与首次调用；unhealthy / disabled 淘汰。"""
        st = self._runtime.status(row.get("deployment_id")) if self._runtime is not None else "unknown"
        return st not in EXCLUDED_STATUSES

    @staticmethod
    def _match_model(model: Dict[str, Any], req: Dict[str, Any]) -> bool:
        task = req.get("task")
        if task and task not in (model.get("task_types") or []):
            return False
        return True

    @staticmethod
    def _match_runtime(row: Dict[str, Any], req: Dict[str, Any]) -> bool:
        node_ids = req.get("node_ids")
        if node_ids is not None and str(row.get("node_id")) not in {str(n) for n in node_ids}:
            return False
        if req.get("gpu_required") and row.get("gpu_id") is None:
            return False
        return True

    # ── 排序 ────────────────────────────────────────────────────────────────
    def _sort(self, rows: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        if strategy in ("least_requests", "least_queue"):
            key_field = "active_requests" if strategy == "least_requests" else "queue_length"
            def key(r):
                st = self._runtime.get(r.get("deployment_id")) if self._runtime is not None else None
                value = getattr(st, key_field, 0) or 0 if st else 0
                return (int(value), int(r.get("priority", 100) or 100), str(r.get("deployment_id")))
            return sorted(rows, key=key)
        if strategy == "round_robin":
            # P0/P1 未实现轮询状态，退化为 priority 排序（reason 里会说明）
            return sorted(rows, key=_priority_key)
        return sorted(rows, key=_priority_key)


def _priority_key(r: Dict[str, Any]):
    return (int(r.get("priority", 100) or 100), str(r.get("deployment_id")))
