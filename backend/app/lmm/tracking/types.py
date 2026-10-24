# -*- coding: utf-8 -*-
"""R4A：跟踪层的数据类型。

刻意与 `lmm/target_spec.py` 的 DetectionTarget **互不 import**：那边是"要检测什么"
（语义），这边是"这个框现在在哪"（时空）。混在一起会让两边都改不动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── 目标状态 ─────────────────────────────────────────────────────────────
# 第一版刻意只有五个状态：状态机一旦复杂到需要"事件溯源"才能解释某一帧为什么
# 是这个状态，排错成本就超过收益了。
STATE_NEW = "new"
STATE_TRACKING = "tracking"
STATE_UNCERTAIN = "uncertain"
STATE_LOST = "lost"
STATE_REMOVED = "removed"

TRACK_STATES = (STATE_NEW, STATE_TRACKING, STATE_UNCERTAIN, STATE_LOST, STATE_REMOVED)

#: 框的来源。VLX 是"语义权威"，tracker 只是"位置预测"，前端必须能分开显示。
SOURCE_VLX = "vlx"
SOURCE_TRACKER = "tracker"


@dataclass
class TrackerResult:
    """一次 tracker 调用的统一返回。

    `bbox_xyxy` **永远是 source image pixel 坐标**（第 33 节）：
    前端与事件统计都不该知道 OSTrack 内部的 256 crop / resize 比例。
    """

    track_id: int
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    score: float = 0.0
    ok: bool = False
    latency_ms: float = 0.0
    #: 失败原因 / 备注（ok=False 时必填，便于日志与前端提示）
    reason: str = ""
    #: 内部诊断（replay 帧数、stride、模板尺寸…），不上主链路
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "score": round(float(self.score), 4),
            "ok": bool(self.ok),
            "latency_ms": round(float(self.latency_ms), 2),
            "reason": self.reason,
            "extra": dict(self.extra or {}),
        }


@dataclass
class TrackSnapshot:
    """给前端 / WS 的一条 track 记录（第 22 节的消息结构）。"""

    track_id: int
    label: str
    bbox_xyxy: Tuple[float, float, float, float]
    state: str = STATE_TRACKING
    source: str = SOURCE_TRACKER
    track_score: float = 0.0
    #: VLX 最近一次确认这一条 track 的帧号与时刻（第 21 节的 TTL 依据）
    last_vlx_frame_id: Optional[int] = None
    last_vlx_confirmed_at: Optional[float] = None
    #: 事件语义：reacquired / lost / updated … 前端可据此闪一下 "VLX ✓"
    event: str = ""
    #: 上一帧的 VLX 置信度：当前 VLX 不给置信度 → **恒为 None**，不许填 0.9
    vlx_confidence: Optional[float] = None
    frame_id: int = 0
    media_time_ms: float = 0.0
    #: ★真实流内 PTS★（STEP1）。None = 未知（旧 hub / 该帧没抽到 pts）。
    #: 前端 `age_ms = video_pts_ms - tracker_pts_ms` 用它，替代"帧号差"这个
    #: 只能间接反映时间差、且被 fps 假设污染的指标。
    pts_ms: Optional[float] = None

    def to_ws(self, now: Optional[float] = None) -> Dict[str, Any]:
        """`now` 用 time.time()（wall clock），与 `last_vlx_confirmed_at` 同基准。

        年龄必须由**发布时刻**算：在对象里缓存会在消息晚到几百毫秒时给出偏小的
        数字，而前端正是靠它判断"VLX 多久没确认了"（第 21 节的 TTL 可视化）。
        """
        age = None
        if self.last_vlx_confirmed_at and now:
            age = int(max(0.0, now - self.last_vlx_confirmed_at) * 1000.0)
        return {
            "track_id": self.track_id,
            "label": self.label,
            "bbox": [round(float(v), 1) for v in self.bbox_xyxy],
            "source": self.source,
            "track_score": round(float(self.track_score), 3),
            "state": self.state,
            "event": self.event,
            # 第 18 节：VLX 不给置信度 → 恒 None，绝不用 0.9 冒充
            "vlx_confidence": self.vlx_confidence,
            "last_vlx_frame_id": self.last_vlx_frame_id,
            "last_vlx_age_ms": age,
            #: ★追帧闭环★：这两个字段 dataclass 上一直有，但之前没发出来，
            #: 前端因此**算不出 Tracker Age**（只能算 VLX Age）。
            #: `frame_id` = 这个框对应的那一帧；`media_time_ms` = 该帧的流时间。
            #: 前端拿消息里的 `now_media_time_ms` 减它 = Tracker Age。
            "frame_id": int(self.frame_id),
            "media_time_ms": round(float(self.media_time_ms), 1),
            #: STEP1：真实 PTS。未知时发 null，前端诊断行自动降级为只显示帧号差。
            "pts_ms": (None if self.pts_ms is None else round(float(self.pts_ms), 1)),
        }


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_dist(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def box_to_xywh(b: Tuple[float, float, float, float]) -> List[float]:
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def xywh_to_xyxy(b) -> Tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in b)
    return (x, y, x + w, y + h)
