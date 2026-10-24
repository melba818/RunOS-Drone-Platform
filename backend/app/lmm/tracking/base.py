# -*- coding: utf-8 -*-
"""R4A：Tracker 后端抽象层。

为什么 TargetManager 不能直接 import OSTrack：
    这一层的存在只为一个目的 —— **换 tracker 不用重写业务**。SAM2 / CoTracker
    只要实现这五个方法就能接进来，TargetManager / TargetManager 的关联策略 /
    WS 消息结构一行都不用动（第 5 节）。

接口刻意很小：initialize / update / reset / remove / health。
没有 `batch_update`、没有 `reid`：第一版不做，加了就是过度设计。
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import TrackerResult


class TrackerBackend(ABC):
    """单目标跟踪后端的契约。

    实现约定（都会被单测锁住）：
      * `initialize` / `update` 的 `frame` 是 **BGR ndarray**（与 OpenCV 采集侧一致）；
      * 返回的 `bbox_xyxy` 是 **source 图像像素坐标**；
      * **模型权重只加载一次**，多目标只保留各自的 template/state（第 31/32 节）；
      * 任何异常都要转成 `TrackerResult(ok=False, reason=...)`，
        绝不让 TrackerWorker 线程死掉（第 29/30 节）。
    """

    #: 展示名（前端 / 日志用）
    name = "base"

    # ── 生命周期 ──────────────────────────────────────────────────────────
    @abstractmethod
    def initialize(self, track_id: int, frame: Any, bbox_xyxy: Sequence[float]) -> TrackerResult:
        """用某一帧上的某个框**创建**一个跟踪状态。"""

    @abstractmethod
    def update(self, track_id: int, frame: Any) -> TrackerResult:
        """在新的一帧上继续跟（必须先 initialize）。"""

    @abstractmethod
    def reset(self, track_id: int) -> None:
        """丢弃内部状态但保留 track_id（VLX 校正时用：换新模板前先清）。"""

    @abstractmethod
    def remove(self, track_id: int) -> None:
        """彻底删除（目标消失 / 任务结束）。"""

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """后端自检：可用 / 设备 / 显存 / 加载耗时 / 最后一次错误。"""

    # ── 可选能力：服务器侧共享帧（StreamHub / reference frame）──────────────
    #: ★能力开关★：True 表示这一帧**不用调用方给像素**，服务端按 `frame_id`
    #: 自己去 StreamHub 取。TargetManager 据此把 `update()` 换成 `update_at()`，
    #: 其余（状态机 / 漂移守卫 / 校正 / replay）一行都不改。
    supports_ref = False

    def init_by_ref(self, track_id: int, bbox_xyxy: Sequence[float],
                    frame_id: int, media_time_ms: Optional[float] = None,
                    label: str = "", pts_ms: Optional[float] = None) -> TrackerResult:
        """只传 `track_id + bbox + frame_id` 建模板（**服务端自己取帧**）。

        `pts_ms` 是 STEP1 引入的真实流内时间戳，纯附加信息：不支持的实现直接
        忽略即可，不影响既有语义。
        """
        return TrackerResult(track_id, ok=False,
                             reason="该后端不支持 reference frame 初始化")

    def update_at(self, track_id: int, frame_id: int,
                  media_time_ms: Optional[float] = None,
                  pts_ms: Optional[float] = None) -> TrackerResult:
        """只传 `track_id + frame_id` 跟一帧（**服务端自己取帧**）。"""
        return TrackerResult(track_id, ok=False,
                             reason="该后端不支持 reference frame 更新")

    # ── 可选能力 ──────────────────────────────────────────────────────────
    def track_ids(self) -> List[int]:
        return []

    def close(self) -> None:
        pass


class NullTracker(TrackerBackend):
    """永远失败但**不抛异常**的空实现。

    用途有两个：① tracking off 时给 TargetManager 一个占位，业务分支不必写
    `if backend is None`；② 单测里验证"tracker 挂了以后 VLX 仍然工作"（第 30 节）。
    """

    name = "null"

    def __init__(self, reason: str = "跟踪器未启用") -> None:
        self.reason = reason
        self._ids: Dict[int, float] = {}
        self._lock = threading.RLock()

    def _fail(self, track_id: int, what: str) -> TrackerResult:
        return TrackerResult(track_id=track_id, ok=False, reason=f"{self.reason}（{what}）")

    def initialize(self, track_id, frame, bbox_xyxy):
        with self._lock:
            self._ids[track_id] = time.time()
        return self._fail(track_id, "initialize")

    def update(self, track_id, frame):
        return self._fail(track_id, "update")

    def reset(self, track_id):
        pass

    def remove(self, track_id):
        with self._lock:
            self._ids.pop(track_id, None)

    def health(self):
        return {"ok": False, "name": self.name, "reason": self.reason, "tracks": 0}

    def track_ids(self):
        with self._lock:
            return list(self._ids.keys())


def bbox_clip(bbox: Tuple[float, float, float, float], width: int, height: int,
              margin: int = 0) -> Tuple[float, float, float, float]:
    """把框夹回画面内（OSTrack 的 clip_box 语义：保留 margin 的余量）。

    夹取是**进出 tracker 的边界操作**，必须集中在一个函数里：散落在各处会出现
    "有的地方夹了、有的没夹"，然后前端偶发画到画布外面去。
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = min(max(x1, 0.0), float(max(0, width - margin)))
    y1 = min(max(y1, 0.0), float(max(0, height - margin)))
    x2 = min(max(x2, 0.0), float(max(0, width - margin)))
    y2 = min(max(y2, 0.0), float(max(0, height - margin)))
    return (x1, y1, max(x2, x1), max(y2, y1))
