# -*- coding: utf-8 -*-
"""R4A：VLX（低频语义）+ OSTrack（高频位置）+ TargetManager（关联/校正/丢失）。

分层（第 3 节）：

    CaptureWorker ──► TrackerFrameCache ──► TrackerWorker ──► track_update
          │                                       ▲
          └─► LatestFrameBuffer ──► VLX ──────────┘（detection → 校正/重捕获）

`task_manager` 只做接线与生命周期，业务逻辑都在这几个模块里（第 45 节）。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

__all__ = ["TrackerBackend", "NullTracker", "OSTrackBackend", "TrackerResult",
           "TrackerFrameCache", "TargetManager", "TrackerWorker", "TrackSnapshot",
           "TrackConfig", "create_backend", "get_shared_backend",
           "shared_backend_stats", "release_shared_backends"]

from .base import NullTracker, TrackerBackend, bbox_clip  # noqa: F401
from .types import TrackerResult, TrackSnapshot  # noqa: F401
from .frame_cache import TrackerFrameCache  # noqa: F401
from .target_manager import TargetManager, TrackConfig  # noqa: F401
from .worker import TrackerWorker  # noqa: F401

try:  # OSTrack 是可选依赖：没装 torch / 权重缺失时 import 本包仍应成功
    from .ostrack import OSTrackBackend  # noqa: F401
except Exception:  # noqa: BLE001
    OSTrackBackend = None  # type: ignore[assignment]


_SHARED_LOCK = threading.RLock()
_SHARED: Dict[str, Any] = {}


def get_shared_backend(mode: str = "ostrack", **kwargs: Any) -> TrackerBackend:
    """**全局单例**后端。两个任务各造一份 OSTrack 会直接 OOM —— 实测
    MX250 上 2 GB 显存，单份后端就占 1.45 GB，第二份必挂。

    而且加载一次要 11 s，每个任务各等一遍完全不可接受。

    注意：后端是**无任务状态**的（track 状态在 TargetManager，按任务各一份），
    所以共享是安全的；任务结束时只能 `clear()` 自己的 track，
    **绝不能**在这里 close 共享后端。
    """
    key = "%s|%s" % (mode, sorted(kwargs.items()))
    with _SHARED_LOCK:
        be = _SHARED.get(key)
        if be is not None:
            return be
        be = create_backend(mode, **kwargs)
        _SHARED[key] = be
        LOG.info("[tracking] 共享后端已创建 mode=%s -> %s", mode, be.name)
        return be


def shared_backend_stats() -> Dict[str, Any]:
    """给 /api 与日志看的共享后端健康状态。"""
    out: Dict[str, Any] = {}
    with _SHARED_LOCK:
        for key, be in _SHARED.items():
            try:
                h = be.health()
            except Exception as exc:  # noqa: BLE001
                h = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            out[key] = h
    return out


def release_shared_backends() -> None:
    """只在进程退出 / 显式重载时用。正常任务结束**不要调**。"""
    with _SHARED_LOCK:
        for be in _SHARED.values():
            try:
                be.close()
            except Exception:  # noqa: BLE001
                pass
        _SHARED.clear()


def create_backend(mode: str = "ostrack", **kwargs: Any) -> TrackerBackend:
    """按模式造后端；**任何失败都退化成 NullTracker，绝不抛**（第 30 节）。

    tracker 是"锦上添花"的通道：它挂了必须让任务继续跑 VLX，
    所以这里不向上抛异常，而是给一个永远 `ok=False` 的占位，
    让上层所有 `if backend` 分支保持原样。
    """
    mode = (mode or "off").strip().lower()
    if mode in ("off", "none", "false", "0", ""):
        return NullTracker("跟踪器未启用")
    if mode in ("ostrack", "ostrack-256", "on", "1", "true"):
        if OSTrackBackend is None:
            return NullTracker("OSTrack 后端不可用（依赖缺失）")
        try:
            be = OSTrackBackend(**kwargs)
            if be.health().get("ok"):
                return be
            return NullTracker("OSTrack 权重加载失败")
        except Exception as exc:  # noqa: BLE001
            return NullTracker(f"OSTrack 初始化失败（{type(exc).__name__}）")
    if mode in ("remote", "ostrack-remote"):
        #: 跟踪计算迁到 A6000 GPU1 上的独立服务，本机只发 JPEG + 收 bbox。
        #: 服务不可用时退化成 NullTracker（tracker 是补充通道，VLX 照跑）。
        try:
            from .remote_backend import RemoteOSTrackBackend
            be = RemoteOSTrackBackend(
                node_id=str(kwargs.get("node_id") or ""),
                port=int(kwargs.get("port") or 8022),
                token=str(kwargs.get("token") or ""),
                #: ★并发通道★：多目标串行时每目标只有 1/N 的预算，必须让不同
                #: track 走不同的 SSH 隧道槽位才可能并行。
                parallel_channels=int(kwargs.get("parallel_channels") or 1),
            )
            if be.health().get("ok"):
                return be
            return NullTracker("远程跟踪服务不可用：%s" % be.health().get("error"))
        except Exception as exc:  # noqa: BLE001
            return NullTracker(f"远程跟踪初始化失败（{type(exc).__name__}）")
    if mode in ("streamhub", "ostrack-streamhub", "ref", "ostrack-ref"):
        #: ★服务器侧共享帧★：与 `remote` 唯一的区别是 **`ref_mode=True`** ——
        #: 不上传任何图片，只发 `{track_id, frame_id, media_time_ms}`，
        #: 由 A6000 上的 OSTrack 服务自己从 StreamHub 取帧。
        #: 旧 image 模式的代码路径一行没删，改回 `remote` 即可原样回退。
        try:
            from .remote_backend import RemoteOSTrackBackend
            be = RemoteOSTrackBackend(
                node_id=str(kwargs.get("node_id") or ""),
                port=int(kwargs.get("port") or 8022),
                token=str(kwargs.get("token") or ""),
                parallel_channels=int(kwargs.get("parallel_channels") or 1),
                ref_mode=True,
            )
            if be.health().get("ok"):
                return be
            return NullTracker("远程跟踪服务不可用：%s" % be.health().get("error"))
        except Exception as exc:  # noqa: BLE001
            return NullTracker(f"StreamHub 共享帧模式初始化失败（{type(exc).__name__}）")
    return NullTracker(f"未知的跟踪模式：{mode}")
