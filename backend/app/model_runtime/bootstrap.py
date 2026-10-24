"""默认 Runtime 组装（惰性单例）。

把 ModelCatalog / DeploymentStore / DeploymentRegistry / DeploymentRuntimeRegistry /
DeploymentResolver / LegacySecretResolver / ModelRuntimeService 串起来。

★P2B 硬约束★
    * Legacy Bridge 只读合成，**不创建 deployments.json**。
    * 所有路径来自 settings.MODELS_DIR，不硬编码。
    * 本模块被 import 时**不做任何 IO**（真正读取发生在第一次 get_runtime()）。
"""

from __future__ import annotations

import threading
from typing import Any, Optional

__all__ = ["build_runtime", "get_runtime", "reset_runtime"]

_lock = threading.RLock()
_SERVICE = None


def _paths():
    from config import settings
    base = settings.PLATFORM_MODELS_DIR
    return base / "lmm_config.json", base / "vlm_providers.json", settings.DEPLOYMENTS_FILE


def build_runtime() -> Any:
    """每次调用都新建一套（测试用；生产请用 get_runtime()）。"""
    from .model_catalog import ModelCatalog
    from .deployment_store import DeploymentStore
    from .registry import DeploymentRegistry
    from .runtime_registry import DeploymentRuntimeRegistry
    from .resolver import DeploymentResolver
    from .secrets import LegacySecretResolver
    from .legacy_bridge import LegacyBridge
    from .runtime import ModelRuntimeService

    lmm_path, vlm_path, dep_path = _paths()
    bridge = LegacyBridge(lmm_config_path=lmm_path, vlm_providers_path=vlm_path)

    models = ModelCatalog(providers=[bridge.legacy_models])
    store = DeploymentStore(dep_path)
    # ★legacy 部署以 provider 形式注入 Registry ★
    #   —— deployments.json 不存在时，Resolver 依然能看到 legacy-vlx-default。
    #   正式 Deployment 默认 priority=100，legacy priority=1000 → 正式永远优先。
    registry = DeploymentRegistry(store, models, legacy_providers=[bridge.legacy_deployments])
    runtime = DeploymentRuntimeRegistry()
    resolver = DeploymentResolver(registry, runtime, models)
    secrets = LegacySecretResolver(lmm_config_path=lmm_path, vlm_providers_path=vlm_path)

    return ModelRuntimeService(models, registry, runtime, resolver, secrets=secrets)


def get_runtime():
    """进程内单例。配置变了可 reset_runtime() 再取。"""
    global _SERVICE
    with _lock:
        if _SERVICE is None:
            _SERVICE = build_runtime()
        return _SERVICE


def reset_runtime() -> None:
    global _SERVICE
    with _lock:
        _SERVICE = None
