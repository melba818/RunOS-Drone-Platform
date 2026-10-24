"""ModelRuntimeService —— 真正执行推理并做 failover 的地方（P2B）。

★职责边界（V1.1 拍定）★
    DeploymentResolver  **只排顺序，不推理**（它连 Adapter 都不知道）。
    ModelRuntimeService **才**负责：
        1. 从 ModelCatalog 取 model
        2. 让 Resolver 给出候选 Deployment（最多 2 个）
        3. resolve_options()（request > deployment override > model default）
        4. 按 protocol 取 Adapter
        5. 调 Adapter
        6. 写 DeploymentRuntimeState（healthy / unhealthy / latency / active_requests）
        7. 失败时决定是否试下一个候选（**最多 2 个，禁止无限重试**）

★开关★
    本服务**不读** FEATURE_MODEL_RUNTIME_V1 —— 由调用方决定是否进来。
    （这样单元测试可以直接构造它，不必摆弄全局开关。）
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .options import resolve_options
from .runtime_registry import (
    DeploymentRuntimeRegistry, STATUS_HEALTHY, STATUS_UNHEALTHY,
)
from .target import DetectionTarget
from .auth import redact

__all__ = ["ModelRuntimeService", "RuntimeServiceError", "MAX_ATTEMPTS"]

#: ★最多尝试 2 个实例★ —— 与 Resolver 的 limit=2 一致，禁止无限重试
MAX_ATTEMPTS = 2


class RuntimeServiceError(RuntimeError):
    """所有候选实例都失败（或根本没有可用实例）。消息已脱敏。"""

    def __init__(self, message: str, *, attempts: Optional[List[Dict[str, Any]]] = None,
                 reason: str = ""):
        super().__init__(message)
        self.attempts = attempts or []
        self.reason = reason


class ModelRuntimeService:
    def __init__(self, models, registry, runtime: DeploymentRuntimeRegistry, resolver, *,
                 adapters: Optional[Dict[str, Any]] = None,
                 secrets=None):
        """
        models   —— ModelCatalog
        registry —— DeploymentRegistry
        runtime  —— DeploymentRuntimeRegistry
        resolver —— DeploymentResolver
        adapters —— {protocol: adapter 实例或工厂}；缺省时按 protocol 惰性构造
        """
        self._models = models
        self._registry = registry
        self._runtime = runtime
        self._resolver = resolver
        self._adapters = dict(adapters or {})
        self._secrets = secrets

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def infer(self, model_id: str, image, target: DetectionTarget,
              request_options: Optional[Dict[str, Any]] = None, *,
              preferred_deployment_id: Optional[str] = None,
              allow_fallback: bool = True) -> Dict[str, Any]:
        """返回 dict：{result: AdapterResult, deployment_id, latency_ms,
                       fallback_used, attempts, coord_space}"""
        if target is None:
            raise RuntimeServiceError("DetectionTarget 不能为空")

        model = self._models.get(model_id) if self._models is not None else None
        if model is None:
            raise RuntimeServiceError(f"未知 model_id: {model_id}", reason=f"model not found: {model_id}")
        if not model.get("enabled", True):
            raise RuntimeServiceError(f"模型已停用: {model_id}", reason=f"model disabled: {model_id}")

        resolution = self._resolver.resolve_candidates(
            model_id,
            strategy="priority",
            preferred_deployment_id=preferred_deployment_id,
            allow_fallback=allow_fallback,
            limit=MAX_ATTEMPTS,
        )
        if not resolution.candidates:
            raise RuntimeServiceError(
                f"没有可用部署实例（{resolution.reason}）", reason=resolution.reason)

        attempts: List[Dict[str, Any]] = []
        for idx, dep in enumerate(resolution.candidates[:MAX_ATTEMPTS]):
            did = str(dep.get("deployment_id") or "")
            self._runtime.incr_requests(did, 1)
            t0 = time.time()
            try:
                adapter = self._adapter_for(model, dep)
                options = resolve_options(model, dep, request_options)
                result = adapter.infer(
                    image=image, target=target, deployment=dep,
                    options=options, model=model,
                )
                latency = (time.time() - t0) * 1000.0
                self._runtime.set_status(did, STATUS_HEALTHY, latency_ms=round(latency, 1))
                self._runtime.touch_used(did)
                result.latency_ms = round(latency, 1)
                return {
                    "result": result,
                    "deployment_id": did,
                    "deployment": dep,
                    "latency_ms": round(latency, 1),
                    "coord_space": result.coord_space,
                    "fallback_used": idx > 0,
                    "attempts": attempts + [{"deployment_id": did, "ok": True}],
                    "reason": resolution.reason,
                }
            except Exception as exc:  # noqa: BLE001 - 任何异常都算这个实例失败
                latency = (time.time() - t0) * 1000.0
                safe = redact(f"{type(exc).__name__}: {exc}")
                self._runtime.set_status(did, STATUS_UNHEALTHY,
                                         latency_ms=round(latency, 1), error=safe)
                attempts.append({"deployment_id": did, "ok": False, "error": safe})
                if not allow_fallback:
                    # ★严格模式：指定实例失败就如实失败，绝不换别的实例★
                    break
            finally:
                self._runtime.incr_requests(did, -1)

        last = attempts[-1]["error"] if attempts else resolution.reason
        raise RuntimeServiceError(
            f"全部候选实例均失败：{last}", attempts=attempts, reason=resolution.reason)

    # ── Adapter 选择 ────────────────────────────────────────────────────────
    def _adapter_for(self, model: Dict[str, Any], dep: Dict[str, Any]):
        protocol = str(dep.get("protocol") or model.get("protocol") or "").strip()
        adapter = self._adapters.get(protocol)
        if adapter is None:
            adapter = self._default_adapter(protocol)
            if adapter is None:
                raise RuntimeServiceError(f"没有适配 protocol={protocol} 的 Adapter")
            self._adapters[protocol] = adapter
        return adapter

    def _default_adapter(self, protocol: str):
        if protocol == "vlx_detection":
            from .adapters.vlx import VLXAdapter
            return VLXAdapter(secrets=self._secrets)
        # P2C 再接 openai_compatible（Qwen/VLM）
        return None
