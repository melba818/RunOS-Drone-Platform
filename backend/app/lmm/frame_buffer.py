# -*- coding: utf-8 -*-
"""实时帧缓冲：只保留**最新一帧**。

为什么不能用 `queue.Queue()`：实时场景里"不丢帧"是错的优化目标。
VLX 单帧要约 1~4 秒，而采集是 30 FPS —— 一个 FIFO 队列会在一次推理期间
堆进上百帧，处理完它们时画面已经落后几十秒，而且越跑越糟。

实时模式的正确原则是 **Latest Frame Wins**：推理慢就丢帧，保证下一次推理
拿到的永远是"当前这一刻"的画面，而不是几秒前的历史。

所以这个槽位：
  * 容量恒为 1（写入直接覆盖，永远不可能增长）；
  * `put_latest()` 返回被覆盖掉的帧数（= dropped_frames 的来源）；
  * `take()` 取当前最新且**不清空**（清空会让"读走后到下一帧写入前"出现空窗）。
"""
from __future__ import annotations

import threading
from typing import Any, Optional

__all__ = ["FramePacket", "LatestFrameBuffer"]


class FramePacket:
    """采集侧产出的最小单元。

    `frame_id`     视频里的绝对帧号（`FileSource.position`），跨抽样间隔可比。
    `media_time_ms` 该帧对应的**媒体时间**（毫秒）。它是把"AI 结果"和
                    "浏览器当前播放位置"对齐的唯一桥梁 —— 没有它，前端就
                    无法回答"这个框对应画面上的哪一秒"。
    `capture_ts`   wall clock 采集时刻（算 frame_age 用）。
    """

    __slots__ = ("frame_id", "media_time_ms", "capture_ts", "frame")

    def __init__(self, frame_id: int, media_time_ms: float, capture_ts: float,
                 frame: Any) -> None:
        self.frame_id = int(frame_id)
        self.media_time_ms = float(media_time_ms)
        self.capture_ts = float(capture_ts)
        self.frame = frame


class LatestFrameBuffer:
    """单槽最新帧缓冲（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot: Optional[FramePacket] = None
        self._dropped = 0
        self._put_count = 0

    # ── 写入 ──────────────────────────────────────────────────────────────
    def put_latest(self, packet: FramePacket) -> int:
        """写入并返回**这次被丢弃的帧数**（0 或 1）。

        槽位空 → 没丢；槽位已占 → 旧的那一帧被顶掉，记 1。
        调用方累加它即可得到 dropped_frames。
        """
        dropped = 0
        with self._lock:
            if self._slot is not None:
                dropped = 1
                self._dropped += 1
            self._slot = packet
            self._put_count += 1
        return dropped

    # ── 读取 ──────────────────────────────────────────────────────────────
    def take(self) -> Optional[FramePacket]:
        """取当前最新帧（不清空槽位）。"""
        with self._lock:
            return self._slot

    def peek_frame_id(self) -> int:
        with self._lock:
            return self._slot.frame_id if self._slot is not None else 0

    # ── 观测 ──────────────────────────────────────────────────────────────
    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def put_count(self) -> int:
        with self._lock:
            return self._put_count

    @property
    def queue_depth(self) -> int:
        """恒为 0 或 1。实时模式下它**绝不允许**增长 —— 这是验收项之一。"""
        with self._lock:
            return 1 if self._slot is not None else 0

    def clear(self) -> None:
        with self._lock:
            self._slot = None
