# -*- coding: utf-8 -*-
"""R0 性能探针：逐帧计时与分位统计。

为什么单独成文件：性能埋点会散在 capture / encode / transport / inference / parse /
render / publish 七处，而「4.4 秒到底花在哪」必须靠**同一套字段定义**才能回答。
把数据结构和统计放在一处，避免各模块各写一份 dict、字段名对不上。

两个最容易搞错的指标，这里显式区分：

- ``frame_age_ms``  = 结果发布时刻 − 这一帧的采集时刻。
  **单帧视角**：一次推理再慢，它也只是"这一帧出生了 4.4 秒才出结果"。

- ``drift_ms``      = 真实经过的时间 − 已处理到的视频时间。
  **进度视角**：这才是"跑一分钟以后落后几十秒"的那个量，R1 真正要压的是它。

同步链路里 frame_age 未必增长（每帧各自等 4.4s），但 drift 一定线性增长；
只看 frame_age 会得出"没问题"的错误结论。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["FrameTiming", "PerfStats", "now_ms"]


def now_ms() -> float:
    """单调毫秒。跨线程/跨进程比较用 `time.time()`，这里只用来求差值。"""
    return time.time() * 1000.0


@dataclass
class FrameTiming:
    """一帧从采集到发布的完整耗时拆分。

    字段刻意与需求一致（`capture/encode/transport/inference/parse/render/publish`），
    缺哪一项就记 0.0（**已知为 0**）或 None（**未知**），绝不填估算值。
    """

    frame_id: int = 0
    capture_timestamp: float = 0.0        # time.time()，与前端/日志对齐用
    capture_ms: float = 0.0               # 读帧（含 grab 跳帧）
    encode_ms: float = 0.0                # BGR → JPEG
    #: 端到端 − 模型侧耗时。注意这是**上界**：服务端只自报模型计算时间
    #: （proposal + vlx），不报整单耗时，所以服务端自己的解码/resize/HTTP 开销
    #: 会被算进这里。要拆得更细必须让服务端加埋点（不在本轮范围）。
    transport_ms: Optional[float] = None
    #: 服务端自报的**模型侧**耗时 = proposal_latency_ms + vlx_latency_ms
    inference_ms: Optional[float] = None
    parse_ms: float = 0.0                 # JSON 解析
    render_ms: float = 0.0                # 画框 / 编码推送图
    publish_ms: float = 0.0               # 推 WS / 写盘
    total_ms: float = 0.0
    frame_age_ms: float = 0.0             # ★最重要★ 发布时刻 − 采集时刻

    def finish(self, publish_ts: Optional[float] = None) -> "FrameTiming":
        """在发布时刻调用：补 total / frame_age。"""
        ts = time.time() if publish_ts is None else publish_ts
        if not self.total_ms:
            known = [v for v in (self.capture_ms, self.encode_ms, self.transport_ms,
                                 self.inference_ms, self.parse_ms, self.render_ms,
                                 self.publish_ms) if v]
            self.total_ms = round(sum(known), 1)
        if self.capture_timestamp:
            self.frame_age_ms = round((ts - self.capture_timestamp) * 1000.0, 1)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "capture_timestamp": round(self.capture_timestamp, 3) if self.capture_timestamp else None,
            "capture_ms": round(self.capture_ms, 1),
            "encode_ms": round(self.encode_ms, 1),
            "transport_ms": (round(self.transport_ms, 1) if self.transport_ms is not None else None),
            "inference_ms": (round(self.inference_ms, 1) if self.inference_ms is not None else None),
            "parse_ms": round(self.parse_ms, 3),
            "render_ms": round(self.render_ms, 1),
            "publish_ms": round(self.publish_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "frame_age_ms": round(self.frame_age_ms, 1),
        }


class PerfStats:
    """环形采样的分位统计（P50 / P95）。

    为什么要 P95 而不是只看平均：实时链路里用户感受到的是**最坏的那几次**
    （画面一卡就是一次长尾），平均值会把长尾抹平。
    """

    def __init__(self, capacity: int = 600) -> None:
        # 尊重入参：曾经这里写死 `max(16, ...)`，调用方传 10 却拿到 16，
        # 静默改写别人的意图。下限只保留到 2（1 个样本算不出分位差）。
        self.capacity = max(2, int(capacity))
        self._vals: List[float] = []
        self._dirty = True
        self._cache: Dict[str, Optional[float]] = {}
        self._last: Optional[float] = None

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        self._last = v
        self._vals.append(v)
        if len(self._vals) > self.capacity:
            del self._vals[: len(self._vals) - self.capacity]
        self._dirty = True

    @property
    def n(self) -> int:
        return len(self._vals)

    @property
    def last(self) -> Optional[float]:
        """最近一次采到的值。UI 上的「当前结果年龄」要的是它，不是平均。"""
        return self._last

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        vals = sorted(self._vals)
        if not vals:
            self._cache = {"n": 0, "avg": None, "min": None, "max": None,
                           "p50": None, "p95": None}
            self._dirty = False
            return

        def pct(p: float) -> float:
            if len(vals) == 1:
                return vals[0]
            idx = int(round((len(vals) - 1) * p))
            return vals[max(0, min(idx, len(vals) - 1))]

        self._cache = {
            "n": len(vals),
            "avg": round(sum(vals) / len(vals), 1),
            "min": round(vals[0], 1),
            "max": round(vals[-1], 1),
            "p50": round(pct(0.50), 1),
            "p95": round(pct(0.95), 1),
        }
        self._dirty = False

    def summary(self) -> Dict[str, Optional[float]]:
        self._rebuild()
        return dict(self._cache)

    def reset(self) -> None:
        self._vals.clear()
        self._dirty = True
        self._cache = {}
