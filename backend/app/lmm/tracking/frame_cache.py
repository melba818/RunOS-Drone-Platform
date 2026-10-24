# -*- coding: utf-8 -*-
"""R4A：TrackerFrameCache —— 给"迟到的 VLX 结果"留的历史帧。

它和 `LatestFrameBuffer` 完全不是一回事：

    LatestFrameBuffer  容量 1，只给 VLX，语义是"最新帧优先、落后不累积"
    TrackerFrameCache  容量 = N 秒的帧，只给 tracker，语义是"VLX 回来时还能
                       找到它检测的那一张图"（第 6/7 节）

为什么必须存在：VLX 端到端约 2.3 s，也就是 VLX 给出 frame=100 的框时，画面
早就走到 frame=165。拿 frame 100 的框直接去 frame 165 上初始化 tracker 是错的
（目标已经移动），正确做法是回到 frame 100 初始化，再沿缓存 replay 到 165。

为什么不缓存原始 1080p（第 8 节）：
    1920×1080×3 × 30fps × 5s ≈ 933 MB —— 一条任务就把内存吃光。
    所以存的是缩到 `longest_side` 的小图（默认 960），且**只存一份坐标换算比例**，
    bbox 换算必须走 `to_source()` / `to_tracker()`，不许散落计算。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class CachedFrame:
    __slots__ = ("frame_id", "media_time_ms", "pts_ms", "capture_ts", "frame",
                 "source_width", "source_height", "tracker_width", "tracker_height")

    def __init__(self, frame_id: int, media_time_ms: float, capture_ts: float,
                 frame: Any, source_width: int, source_height: int,
                 pts_ms: Optional[float] = None) -> None:
        self.frame_id = int(frame_id)
        self.media_time_ms = float(media_time_ms)
        #: ★真实流内 PTS★（STEP1）。None = 未知（旧 hub / 刚启动 / 该帧没抽到）。
        #: 与 `media_time_ms` 并存：后者是 `frame_id × 1000/fps` 的合成值，
        #: 只用于既有索引与外推语义，**不代表真实播放位置**。
        self.pts_ms = None if pts_ms is None else float(pts_ms)
        self.capture_ts = float(capture_ts)
        self.frame = frame
        self.source_width = int(source_width)
        self.source_height = int(source_height)
        th, tw = (frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)
        self.tracker_width = int(tw)
        self.tracker_height = int(th)

    @property
    def scale_x(self) -> float:
        return (self.tracker_width / self.source_width) if self.source_width else 1.0

    @property
    def scale_y(self) -> float:
        return (self.tracker_height / self.source_height) if self.source_height else 1.0

    def to_tracker(self, bbox_xyxy: Tuple[float, float, float, float]
                   ) -> Tuple[float, float, float, float]:
        """source pixel → tracker 图像 pixel。"""
        x1, y1, x2, y2 = bbox_xyxy
        return (x1 * self.scale_x, y1 * self.scale_y,
                x2 * self.scale_x, y2 * self.scale_y)

    def to_source(self, bbox_xyxy: Tuple[float, float, float, float]
                  ) -> Tuple[float, float, float, float]:
        """tracker 图像 pixel → source pixel（**出 tracker 必须走这里**）。"""
        sx = self.scale_x or 1.0
        sy = self.scale_y or 1.0
        x1, y1, x2, y2 = bbox_xyxy
        return (x1 / sx, y1 / sy, x2 / sx, y2 / sy)

    @property
    def nbytes(self) -> int:
        f = self.frame
        return int(getattr(f, "nbytes", 0) or 0)


class TrackerFrameCache:
    """定长环形缓存（**容量恒定**，不会随时间增长 —— 第 40 节的验收项）。"""

    def __init__(self, seconds: float = 5.0, fps: float = 30.0,
                 longest_side: int = 960, max_frames: int = 0) -> None:
        self.seconds = float(seconds or 5.0)
        #: ★存下来★：`fps` 只被用来算容量，之前压根没存 —— 于是任何想按
        #: "帧号差 × 每帧毫秒"换算时间的人（例如时间轴对齐 debug）都拿不到它，
        #: 只能退化成墙钟差，而墙钟差里混着 RTT / 轮询量化 / 外推误差。
        self.fps = float(fps or 0.0)
        self.longest_side = int(longest_side or 960)
        cap = int(max_frames) if max_frames else max(4, int(round(self.seconds * max(1.0, fps))))
        self.capacity = cap
        self._buf: Deque[CachedFrame] = deque(maxlen=cap)
        self._lock = threading.RLock()
        self._puts = 0
        self._dropped = 0
        self._overflow = 0

    # ── 写入 ──────────────────────────────────────────────────────────────
    def put(self, frame_id: Any, media_time_ms: float, frame: Any,
            capture_ts: Optional[float] = None, source_size: Optional[Tuple[int, int]] = None) -> None:
        """采集线程每帧调一次。**写缓存的代价必须很低**，否则会拖慢采集。"""
        if frame is None:
            return
        try:
            import cv2
        except Exception:  # noqa: BLE001
            return
        try:
            h, w = frame.shape[0], frame.shape[1]
        except Exception:  # noqa: BLE001
            return
        small = frame
        side = max(h, w)
        if self.longest_side and side > self.longest_side:
            scale = self.longest_side / float(side)
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        if source_size:
            sw, sh = int(source_size[0]), int(source_size[1])
        else:
            sw, sh = w, h
        item = CachedFrame(int(frame_id), float(media_time_ms),
                           float(capture_ts if capture_ts is not None else time.time()),
                           small, sw, sh)
        with self._lock:
            if len(self._buf) >= self.capacity:
                self._overflow += 1
            elif self._buf and self._buf[-1].frame_id > item.frame_id:
                #: 帧号回退（重跑/循环播放）→ 直接清空，否则查找会命中"上一轮"的帧
                self._buf.clear()
                self._dropped += len(self._buf)
            self._buf.append(item)
            self._puts += 1

    def put_ref(self, frame_id: Any, media_time_ms: float,
                capture_ts: Optional[float] = None,
                source_size: Optional[Tuple[int, int]] = None,
                pts_ms: Optional[float] = None) -> None:
        """★服务器侧共享帧★：只登记"这一帧存在"，**不存像素**。

        StreamHub 模式下像素从头到尾都在 A6000 上：RunOS 只发 `frame_id`，
        服务端自己取帧。但 TargetManager 的 replay / 追帧仍需要一条帧时间线，
        所以这里把 `(frame_id, media_time_ms)` 记进缓存，`frame` 留 None。

        刻意**复用 `TrackerFrameCache` 而不是另造一个结构**：`get()` / `newest()`
        / `replay_plan()` / `since()` 的全部语义（环形淘汰、帧号回退清缓存、
        自适应采样）都能原样继承，TargetManager 因此不需要分两套代码路径。
        """
        item = CachedFrame(int(frame_id), float(media_time_ms),
                           float(capture_ts if capture_ts is not None else time.time()),
                           None,
                           int(source_size[0]) if source_size else 0,
                           int(source_size[1]) if source_size else 0,
                           pts_ms=pts_ms)
        with self._lock:
            if self._buf and self._buf[-1].frame_id > item.frame_id:
                #: 帧号回退（Hub 重启 / 换源）→ 清空，否则查找会命中上一轮的帧
                self._buf.clear()
            self._buf.append(item)
            self._puts += 1

    # ── 读取 ──────────────────────────────────────────────────────────────
    def get(self, frame_id: int) -> Optional[CachedFrame]:
        """精确查找某一帧；找不到返回 None（VLX 结果太旧 → 上层丢弃）。"""
        with self._lock:
            for item in self._buf:
                if item.frame_id == int(frame_id):
                    return item
        return None

    def newest(self) -> Optional[CachedFrame]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def oldest(self) -> Optional[CachedFrame]:
        with self._lock:
            return self._buf[0] if self._buf else None

    def since(self, frame_id: int, include: bool = False) -> List[CachedFrame]:
        """取 frame_id **之后**的所有帧（replay 用）。"""
        with self._lock:
            out = [it for it in self._buf
                   if (it.frame_id >= int(frame_id) if include else it.frame_id > int(frame_id))]
        return out

    def replay_plan(self, from_frame_id: int, max_frames: int) -> List[CachedFrame]:
        """自适应 replay 采样（第 10/39 节）：均匀取样，**最后一帧必跑**。

        不跑缓存里的每一帧：2.3 s × 24 fps ≈ 55 帧，按 20 ms/帧算就是 1.1 s，
        一次 VLX 结果反而制造新的延迟。所以按 `max_frames` 均匀取样，且保证
        最新帧在列 —— 前端要的是"当前附近的位置"，不是"完整轨迹"。
        """
        items = self.since(from_frame_id)
        if not items:
            return []
        if len(items) <= max_frames:
            return items
        step = len(items) / float(max_frames)
        picked = [items[int(round(i * step))] for i in range(max_frames)]
        if picked[-1] is not items[-1]:
            picked[-1] = items[-1]
        return picked

    # ── 观测 ──────────────────────────────────────────────────────────────
    @property
    def oldest_frame_id(self) -> int:
        with self._lock:
            return self._buf[0].frame_id if self._buf else -1

    @property
    def latest_frame_id(self) -> int:
        with self._lock:
            return self._buf[-1].frame_id if self._buf else -1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._buf)
            total = sum(it.nbytes for it in self._buf)
            oldest = self._buf[0].frame_id if self._buf else -1
            newest = self._buf[-1].frame_id if self._buf else -1
            tw = self._buf[-1].tracker_width if self._buf else 0
            th = self._buf[-1].tracker_height if self._buf else 0
        return {
            "capacity_frames": self.capacity,
            "current_frames": n,
            "approx_memory_mb": round(total / 1048576.0, 2),
            "oldest_frame_id": oldest,
            "newest_frame_id": newest,
            "tracker_frame_size": [tw, th],
            "longest_side": self.longest_side,
            "puts": self._puts,
            "overflow": self._overflow,
        }

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
