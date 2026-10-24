"""实时通道的中转站（线程安全队列 → WebSocket）。

为什么用队列而不是直接 await：推理跑在**守护线程**里（与项目其它长任务一致），
而 WebSocket 协程跑在 asyncio 事件循环里。让工作线程直接往循环里塞协程要处理
`run_coroutine_threadsafe` 和循环生命周期，容易在任务结束时残留回调。

这里改成：工作线程只往普通 `queue.Queue` 里放；WS 协程每 100ms 取一次再发出去。
线程侧永不触碰事件循环，任务结束也不会留下悬挂回调。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Set

LOG = logging.getLogger(__name__)

_LOCK = threading.RLock()
_SUBS: Dict[str, Set["queue.Queue"]] = {}


def subscribe(task_id: str, maxsize: int = 8) -> "queue.Queue":
    q: "queue.Queue" = queue.Queue(maxsize=maxsize)
    with _LOCK:
        _SUBS.setdefault(task_id, set()).add(q)
    return q


def unsubscribe(task_id: str, q: "queue.Queue") -> None:
    with _LOCK:
        s = _SUBS.get(task_id)
        if s and q in s:
            s.discard(q)
        if s is not None and not s:
            _SUBS.pop(task_id, None)


def publish(task_id: str, payload: Dict[str, Any]) -> None:
    """广播一帧。队列满时直接丢弃最旧的 —— 实时画面丢一帧无所谓，堆积才是灾难。"""
    with _LOCK:
        subs = list(_SUBS.get(task_id, ()))
    for q in subs:
        try:
            if q.full():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put_nowait(payload)
        except Exception:
            pass


def subscriber_count(task_id: str) -> int:
    with _LOCK:
        return len(_SUBS.get(task_id, ()))
