# -*- coding: utf-8 -*-
"""R4A：TrackerWorker —— 驱动 tracker 的独立 daemon 线程。

为什么必须独立成线程（第 3 节）：

    rt-capture   25 FPS   读帧，绝不能被阻塞
    rt-infer     0.43 FPS VLX，一次 2.3 s
    rt-tracker   ~7 FPS   OSTrack，一次 ~130 ms   ← 本文件

三条链路的**时间尺度差 60 倍**，任何两个塞进同一个线程都会互相饿死：
tracker 塞进 VLX 线程 → 一次 VLX 期间 tracker 停摆 2.3 s（等于没有跟踪）；
VLX 塞进 tracker 线程 → 每 130 ms 就被 2.3 s 的前向打断，采集立刻丢帧。

为什么是"自调度"而不是每帧回调：
    实测 130 ms/帧。如果让采集线程每来一帧就调 update，25 FPS 的输入会
    瞬间堆积 18 个待处理请求，延迟越滚越大。**tracker 只能按自己的节奏
    "追最新帧"**——落后的帧直接跳过，这是设计而非妥协（第 38 节）。

线程安全：
    * `on_vlx()` 由 VLX 推理线程调用，`step()` 由本线程调用，二者都会碰
      TargetManager → 所有共享状态在 TargetManager 内部已加锁；
    * 本线程**绝不碰 asyncio 事件循环**，只调 `on_update` 回调（由上层
      决定是 `hub.publish` 还是别的）。
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .types import TrackSnapshot

LOG = logging.getLogger(__name__)


class TrackerWorker:
    """按 tracker 自己的节奏推进所有 track，并把快照交给 `on_update`。

    典型用法（task_manager 内）：

        worker = TrackerWorker(targets, on_update=lambda snaps: hub.publish(
            task_id, {"type": "track_update", "tracks": [...]}))
        worker.start()
        ...
        worker.stop(timeout=3.0)
    """

    def __init__(self, targets: Any,
                 on_update: Optional[Callable[[List[TrackSnapshot]], None]] = None,
                 interval_ms: float = 0.0,
                 emit_idle_ms: float = 500.0,
                 stats_interval_s: float = 5.0) -> None:
        self.targets = targets
        self.on_update = on_update
        #: 0 = 追完就立刻再追（MX250 上 ~7 FPS 已经吃满，不需要额外限速）；
        #: 在更强的卡上可以设个下限，避免把 GPU 抢光影响 VLX。
        self.interval_ms = float(interval_ms or 0.0)
        #: 即使没有任何 track 被更新，也每 N ms 发一次心跳（让前端知道"跟踪在跑"）
        self.emit_idle_ms = float(emit_idle_ms or 500.0)
        self.stats_interval_s = float(stats_interval_s or 5.0)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._running = False

        self._ticks = 0
        self._emits = 0
        self._errors = 0
        self._last_emit_at = 0.0
        self._last_tick_ms = 0.0
        self._tick_ms_sum = 0.0
        self._started_at = 0.0
        self._last_error = ""

    # ── 生命周期 ──────────────────────────────────────────────────────────
    @property
    def alive(self) -> bool:
        return self._running

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._started_at = time.time()
            self._thread = threading.Thread(
                target=self._run, name="rt-tracker", daemon=True)
            self._running = True
            self._thread.start()
        LOG.info("[TrackerWorker] 已启动 interval_ms=%.1f", self.interval_ms)
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """停线程并 join。**必须能被 join 住**（第 41 节：stop 后不许还有线程在跑）。

        单帧最坏 ~250 ms，超时给 3 s 足够；join 不住就记为异常，但**不抛**——
        任务停止流程不能被 tracker 拖住。
        """
        self._stop.set()
        with self._lock:
            th = self._thread
            self._running = False
        if th is not None and th.is_alive():
            th.join(timeout=max(0.1, float(timeout)))
            if th.is_alive():
                LOG.warning("[TrackerWorker] 线程未在 %.1fs 内退出", timeout)

    # ── 主循环 ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                #: 第 30 节：tracker 出任何问题都不能让线程死掉 ——
                #: 死了就是"跟踪静默消失"，比报错更难查。
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                LOG.exception("[TrackerWorker] tick 异常，继续下一轮")
            elapsed = (time.time() - t0) * 1000.0
            with self._lock:
                self._ticks += 1
                self._last_tick_ms = round(elapsed, 2)
                self._tick_ms_sum += elapsed
            wait = self.interval_ms / 1000.0
            if wait <= 0:
                #: 不设间隔时也要让出 GIL，否则会把采集线程饿死
                wait = 0.001
            if self._stop.wait(wait):
                break

    def _tick(self) -> None:
        now = time.time()
        snaps = self.targets.step() if self.targets is not None else []
        #: 没有 track 被更新时，按心跳周期补发一份全量（前端要靠它显示
        #: "跟踪在跑 / 0 个目标"，而不是一直停在上一帧的框上）
        idle_due = (now - self._last_emit_at) * 1000.0 >= self.emit_idle_ms
        if not snaps and not idle_due:
            return
        if not snaps and idle_due and self.targets is not None:
            snaps = self.targets.active_snapshots(now)
        if not snaps:
            #: 一条 track 都没有 → 也要发一次空列表，让前端清框（第 44 节）
            if not idle_due:
                return
        payload_now = time.time()
        items = [s.to_ws(payload_now) for s in snaps]
        self._emit(items, payload_now)

    def _emit(self, items: List[Dict[str, Any]], now: float) -> None:
        if self.on_update is None:
            self._last_emit_at = now
            return
        try:
            self.on_update(items)
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            self._last_error = f"on_update:{type(exc).__name__}:{exc}"
            LOG.exception("[TrackerWorker] on_update 回调异常")
        self._last_emit_at = now
        self._emits += 1

    # ── 观测 ──────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            ticks = self._ticks
            avg = round(self._tick_ms_sum / ticks, 2) if ticks else 0.0
            up = time.time() - self._started_at if self._started_at else 0.0
            return {
                "running": self._running,
                "ticks": ticks,
                "emits": self._emits,
                "errors": self._errors,
                "last_error": self._last_error,
                "last_tick_ms": self._last_tick_ms,
                "avg_tick_ms": avg,
                "uptime_s": round(up, 1),
                #: 实际跟踪帧率 = 有产出的 tick / 运行时长
                "track_fps": round(self._emits / up, 2) if up > 0.5 else 0.0,
            }
