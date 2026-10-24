"""检测事件（DetectionEvent）与事件存储。

事件是「VLX 一次性判定成立」的产物，不是每帧检测结果的拷贝 —— 每帧都写会瞬间撑爆
内存和前端列表。

★1-Hit 即时告警★：VLX 在一个**新的有效帧**上检出目标就立即产出一个事件
（不再等第二/三次命中）；同一目标的重复检出由 cooldown（label + 空间位置）
拦住，不刷屏。每个事件**只对应一张证据图**（触发它的那一帧）。

快照策略：只截目标框区域（带少量边距）并编码为 JPEG base64 直接随事件下发，
不落盘、不占用 temp 目录（`workspace/temp` 会被启动清理器清空，长任务产物不能放那儿）。
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)


def _now_str(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time()).strftime("%H:%M:%S")


@dataclass
class DetectionEvent:
    """一次告警事件。"""

    id: str
    task_id: str
    frame_index: int
    ts: float
    time_str: str
    track_id: Optional[int]
    label: str
    label_zh: str
    display_name: str
    confidence: float
    bbox: List[float]                      # 像素 xyxy
    matched: List[str] = field(default_factory=list)       # 已判定成立的条件（中文）
    unresolved: List[str] = field(default_factory=list)    # 需要二阶段 VLM 确认的条件
    hit_frames: int = 0
    snapshot_b64: Optional[str] = None     # 目标区域 JPEG（base64）
    frame_b64: Optional[str] = None        # 原始帧（缩放后），告警留证用

    # ── 两级置信度：**绝不合并成一个数字** ──
    detector_confidence: float = 0.0       # YOLOE 的分数
    vlm_confidence: Optional[float] = None  # VLM 的分数；未复核时为 None

    # ── 二阶段复核信息 ──
    verified: Optional[bool] = None        # VLM 结论是否可信（None=关闭复核，False=未复核/降级）
    vlm_matched: Optional[bool] = None
    vlm_reason: str = ""
    vlm_attributes: Dict[str, bool] = field(default_factory=dict)
    vlm_config_id: str = ""
    vlm_model_name: str = ""
    vlm_latency_ms: Optional[float] = None
    vlm_attempts: int = 0
    vlm_error: str = ""
    #: 复核状态。空串 = 不展示（无二阶段复核）；非空时优先于 `to_dict` 里的派生值。
    #: **语义检测器（VLX-Seek）必须显式写 "confirmed"** —— 否则 `verified=None`
    #: 会被派生逻辑当成 "pending"，前端就给一条已经定案的告警挂上「待AI复核」。
    vlm_state: str = ""

    # ── 溯源 ──
    original_prompt: str = ""
    spec: Dict[str, Any] = field(default_factory=dict)
    detector_label: str = ""
    #: 该条告警对应的**单帧推理耗时**（毫秒）。存放在事件里而不是只在日志里，
    #: 是为了让"这条告警花了多久才算出来"可以事后追溯 —— VLX-Seek 是秒级模型，
    #: 用户看到画面卡顿时能据此判断是链路慢还是没检出。
    infer_latency_ms: Optional[float] = None
    #: 无轨迹候选的空间归属键（`sp:<label>:<格x>-<格y>`）。跟踪器没给 track_id 时，
    #: 把候选当"伪轨迹"复核；告警里保留它，用户才知道这条告警是按哪块画面出的。
    track_key: str = ""

    # ── ★技术路线切换：三帧确认告警的证据★ ──────────────────────────────
    #: 参与确认的**连续三帧帧号**（旧链路里没有这个概念，恒为空列表）。
    #: ★1-Hit 之后恒为 1 个元素★：检出帧 = 告警帧 = 证据帧。
    frames: List[int] = field(default_factory=list)
    #: ★证据图**原图** URL★（`/api/lmm/alerts/<task>/<alert>/001.png`）。
    #: 取图失败时对应位置为空串 —— **绝不因此丢掉告警本身**。
    #: 这一列**永远指向原图**：卡片和弹窗以前共用它，于是弹窗看到的其实是"卡片
    #: 那张小图"；现在缩略图单独走 `evidence.thumbnail_url`，`images` 留给原图。
    #: ★1-Hit + 单证据图★：**最多 1 个元素**，不再默认构造 3 个槽位。
    images: List[str] = field(default_factory=list)
    #: ★证据图元信息★（与 `images` 一一对应，**最多 1 个**）：
    #: `{index, status, frame_id, pts_ms, source_url, original_url,
    #:   thumbnail_url, bbox, width, height, bytes, format, version}`。
    #:
    #: · `original_url` —— **原图**，弹窗只认它（不缩放、不重编码、不截图）；
    #: · `thumbnail_url` —— 320 px 缩略图，**只给右侧卡片**用；
    #: · `bbox` —— 该帧 VLX 给的 xyxy，**只作留档**（框已烧进像素，前端不画）；
    #: · `width/height/bytes` —— 弹窗底部显示「1920 × 1080 · 583 KB」，
    #:   让人一眼确认"我看的确实是原图"。
    image_meta: List[Dict[str, Any]] = field(default_factory=list)
    #: ★证据就绪状态★（异步证据链）：`pending` / `partial` / `ready` / `error`。
    #: 告警**先出现**，证据图由后台 worker 补上 —— 前端按它决定显示
    #: 「处理中」还是缩略图。空串 = 不启用证据。
    evidence_status: str = ""
    #: ★★1-Hit 之后的**主证据结构：单对象，不是数组**★★
    #: `{status, source_url, original_url, thumbnail_url, width, height, bytes,
    #:   format, frame_id, pts_ms, bbox, version}`。
    #:
    #: 一条告警**只有一张**证据图 —— 检出帧 = 告警帧 = 证据帧，
    #: 所以不需要 `evidence_index` 0/1/2、候选池、三图补齐状态。
    #: 待处理时三个 url 与尺寸全是 `None`，`status=pending`；
    #: worker 完成后就地回填同一个对象（同一个 `event_id`）。
    evidence: Dict[str, Any] = field(default_factory=dict)
    #: 内部留档：每帧证据的轻量状态 `{frame_id, pts_ms, status}`
    #: （`pending` / `processing` / `ready` / `error`）。
    #: ★1-Hit 之后恒 ≤1 个元素★，只是给日志/验收看的，前端不消费。
    evidence_frames: List[Dict[str, Any]] = field(default_factory=list)
    #: 参与确认的三帧置信度（与 `frames` 对齐）。
    frame_confidences: List[float] = field(default_factory=list)
    #: 画面尺寸：前端画红色闪烁光晕时要把像素框映射到画布，缺了它框会错位。
    image_width: int = 0
    image_height: int = 0
    pts_ms: Optional[float] = None

    def to_dict(self, with_snapshot: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "task_id": self.task_id,
            "frame_index": self.frame_index,
            "ts": round(self.ts, 3),
            "time_str": self.time_str,
            "track_id": self.track_id,
            "track_key": self.track_key,
            "label": self.label,
            "label_zh": self.label_zh,
            "display_name": self.display_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "matched": list(self.matched),
            "unresolved": list(self.unresolved),
            "hit_frames": self.hit_frames,
            "detector_confidence": round(float(self.detector_confidence), 4),
            "vlm_confidence": (None if self.vlm_confidence is None
                               else round(float(self.vlm_confidence), 4)),
            "verified": self.verified,
            "vlm_matched": self.vlm_matched,
            "vlm_reason": self.vlm_reason,
            "vlm_attributes": dict(self.vlm_attributes or {}),
            "vlm_config_id": self.vlm_config_id,
            "vlm_model_name": self.vlm_model_name,
            "vlm_latency_ms": self.vlm_latency_ms,
            "vlm_attempts": self.vlm_attempts,
            "vlm_error": self.vlm_error,
            "original_prompt": self.original_prompt,
            "detector_label": self.detector_label,
            # ── ★1-Hit 告警的证据（单张）★ ──
            "frames": [int(f) for f in (self.frames or [])],
            # ── 兼容旧接口：**最多 1 个元素**，不再默认构造 3 个槽位 ──
            "images": list(self.images or [])[:1],
            "image_meta": [dict(m) for m in (self.image_meta or [])][:1],
            "evidence_status": self.evidence_status,
            # ── 主结构：单对象 ──
            "evidence": dict(self.evidence or {}),
            "evidence_frames": [dict(e) for e in (self.evidence_frames or [])][:1],
            "frame_confidences": [round(float(v), 4)
                                  for v in (self.frame_confidences or [])],
            "image_width": int(self.image_width or 0),
            "image_height": int(self.image_height or 0),
            "pts_ms": (None if self.pts_ms is None else round(float(self.pts_ms), 1)),
            "infer_latency_ms": (None if self.infer_latency_ms is None
                                 else round(float(self.infer_latency_ms), 1)),
            "spec": dict(self.spec or {}),
            # 检出状态：confirmed=已检出 / rejected=未匹配 / pending=无结论 / error=失败
            # 语义检测器（VLX-Seek）由事件显式给出 "confirmed"，优先级高于这里按
            # `verified` 的派生（它 verified=None，会被误判成 pending）。
            "vlm_state": (self.vlm_state
                          or ("pending" if self.verified is None
                              else ("confirmed" if self.verified and self.vlm_matched
                                    else ("rejected" if self.verified else "error")))),
        }
        if with_snapshot:
            d["snapshot_b64"] = self.snapshot_b64
            d["frame_b64"] = self.frame_b64
        return d


def crop_snapshot_b64(frame: Any, bbox: List[float], pad_ratio: float = 0.15,
                      max_side: int = 320, quality: int = 80) -> Optional[str]:
    """截目标区域并编码为 JPEG base64。失败返回 None（不能因此丢事件）。"""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
        x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
        x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        # cv2 读入的是 BGR，PIL 需要 RGB
        try:
            img = Image.fromarray(crop[:, :, ::-1])
        except Exception:
            img = Image.fromarray(crop)
        # 限制最长边，避免大图把 WS 消息撑爆
        side = max(img.size)
        if side > max_side:
            scale = max_side / float(side)
            img = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        LOG.debug("[EVENT] 快照编码失败", exc_info=True)
        return None


class EventStore:
    """按任务保存事件（内存 + 可选 JSON 持久化）。

    上限保护：单任务最多保留 `max_events` 条，超出丢最旧的 —— 长时间跑视频/摄像头
    时避免无限增长。
    """

    def __init__(self, max_events: int = 500) -> None:
        self._max = int(max_events)
        self._lock = threading.RLock()
        self._by_task: Dict[str, List[DetectionEvent]] = {}
        self._seq = 0

    def add(self, event: DetectionEvent) -> DetectionEvent:
        with self._lock:
            lst = self._by_task.setdefault(event.task_id, [])
            lst.append(event)
            if len(lst) > self._max:
                del lst[: len(lst) - self._max]
        return event

    def list(self, task_id: str, since_ts: Optional[float] = None,
             limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            lst = self._by_task.get(task_id, [])
            if since_ts:
                lst = [e for e in lst if e.ts > since_ts]
            return [e.to_dict() for e in lst[-int(limit):]]

    def get(self, task_id: str, event_id: str) -> Optional[DetectionEvent]:
        """按 id 取**已入库**的那条告警（异步证据 worker 回填元信息时用）。

        ★为什么必须就地改同一条★：证据是"这条告警的物证"，不是新事件。
        另起一条会让前端出现两条一模一样的告警。
        """
        with self._lock:
            for e in self._by_task.get(task_id, []):
                if e.id == event_id:
                    return e
        return None

    def count(self, task_id: str) -> int:
        with self._lock:
            return len(self._by_task.get(task_id, []))

    def clear(self, task_id: str) -> None:
        with self._lock:
            self._by_task.pop(task_id, None)

    def next_id(self, task_id: str) -> str:
        with self._lock:
            self._seq += 1
            return f"{task_id}-{self._seq:05d}"
