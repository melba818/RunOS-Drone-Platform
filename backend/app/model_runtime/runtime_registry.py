"""DeploymentRuntimeState / DeploymentRuntimeRegistry —— **仅内存**的运行时状态。

★P0/P1 硬约束★
    * 状态只存在进程内 dict，禁止写 deployments.json / SQLite / 任何其他 JSON。
    * 健康检查、推理结果都只改这里，不落盘。
    * 后端重启后回到 unknown 属于**正常行为**，不是 bug。
    * 未来若需要保留"最后已知状态"，再引入 deployment_health_cache.json 或 SQLite
      （本轮不实现）。

Deployment（静态配置）与这里是两个概念，不要合并。
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

__all__ = ["DeploymentRuntimeState", "DeploymentRuntimeRegistry", "STATUS_VALUES"]

STATUS_UNKNOWN = "unknown"
STATUS_HEALTHY = "healthy"
STATUS_BUSY = "busy"
STATUS_UNHEALTHY = "unhealthy"
STATUS_DISABLED = "disabled"

STATUS_VALUES = (STATUS_UNKNOWN, STATUS_HEALTHY, STATUS_BUSY, STATUS_UNHEALTHY, STATUS_DISABLED)

# 参与候选时会被过滤掉的状态（unknown 允许参与首次调用）
EXCLUDED_STATUSES = frozenset({STATUS_UNHEALTHY, STATUS_DISABLED})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


@dataclass
class DeploymentRuntimeState:
    deployment_id: str
    status: str = STATUS_UNKNOWN

    runtime_host: Optional[str] = None
    runtime_port: Optional[int] = None      # managed_worker 的 ready.json 随机端口放这里

    latency_ms: Optional[float] = None
    active_requests: int = 0
    queue_length: int = 0

    last_health_check: Optional[str] = None
    health_error: Optional[str] = None
    health_detail: Dict[str, Any] = field(default_factory=dict)

    started_at: Optional[str] = None
    last_used_at: Optional[str] = None
    pid: Optional[int] = None
    remote_root: Optional[str] = None       # managed_worker 的远端工作目录

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeploymentRuntimeRegistry:
    """纯内存的运行时状态登记处。零磁盘 IO。"""

    def __init__(self):
        self._states: Dict[str, DeploymentRuntimeState] = {}
        self._lock = threading.RLock()

    # ── 读 ──────────────────────────────────────────────────────────────────
    def get(self, deployment_id: str) -> DeploymentRuntimeState:
        """不存在则惰性创建一条 status=unknown（纯内存，不落盘）。"""
        did = str(deployment_id)
        with self._lock:
            st = self._states.get(did)
            if st is None:
                st = DeploymentRuntimeState(deployment_id=did)
                self._states[did] = st
            return st

    def snapshot(self, deployment_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self.get(deployment_id).to_dict())

    def all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: copy.deepcopy(v.to_dict()) for k, v in self._states.items()}

    def status(self, deployment_id: str) -> str:
        return self.get(deployment_id).status

    # ── 写 ──────────────────────────────────────────────────────────────────
    def set_status(self, deployment_id: str, status: str, *,
                   latency_ms: Optional[float] = None,
                   error: Optional[str] = None,
                   detail: Optional[Dict[str, Any]] = None) -> None:
        if status not in STATUS_VALUES:
            raise ValueError(f"unknown status: {status}")
        with self._lock:
            st = self.get(deployment_id)
            st.status = status
            st.last_health_check = _now_iso()
            if latency_ms is not None:
                st.latency_ms = float(latency_ms)
            if error is not None:
                st.health_error = error
            if detail is not None:
                st.health_detail = dict(detail)

    def set_runtime_endpoint(self, deployment_id: str, *,
                             host: Optional[str] = None,
                             port: Optional[int] = None,
                             pid: Optional[int] = None,
                             remote_root: Optional[str] = None,
                             started_at: Optional[str] = None) -> None:
        with self._lock:
            st = self.get(deployment_id)
            if host is not None:
                st.runtime_host = host
            if port is not None:
                st.runtime_port = int(port)
            if pid is not None:
                st.pid = int(pid)
            if remote_root is not None:
                st.remote_root = remote_root
            if started_at is not None:
                st.started_at = started_at

    def touch_used(self, deployment_id: str) -> None:
        with self._lock:
            self.get(deployment_id).last_used_at = _now_iso()

    def incr_requests(self, deployment_id: str, delta: int = 1) -> int:
        with self._lock:
            st = self.get(deployment_id)
            st.active_requests = max(0, st.active_requests + int(delta))
            st.last_used_at = _now_iso()
            return st.active_requests

    def set_queue_length(self, deployment_id: str, n: int) -> None:
        with self._lock:
            self.get(deployment_id).queue_length = max(0, int(n))

    def clear(self, deployment_id: str) -> None:
        with self._lock:
            self._states.pop(str(deployment_id), None)

    def clear_all(self) -> None:
        with self._lock:
            self._states.clear()
