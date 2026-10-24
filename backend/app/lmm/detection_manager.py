"""★VLX 单模型实时识别 + 1-Hit 即时告警★：DetectionManager（**不含任何跟踪器**）。

技术路线切换后的**唯一检测链路**::

    StreamHub ──► VLX ──► Detection ──► DetectionManager ──► AlertEvent

★1-Hit 即时告警（2026-09-26 产品定案）★
    只要 VLX 在一个**新的有效帧**上检出目标 —— **立即**出一条 `AlertEvent`。
    不等待第二次/第三次命中，不等待证据图，不累计 `hit_count`。
    证据图随后由后台 worker 异步补上（一条告警 = 一张证据图）。

★为什么去掉三帧确认★（实测依据见 `.workbuddy/VLX三帧告警确认逻辑诊断报告.md`）：
    VLX 只有 0.563 Hz，一次三连击要 5.3 s 起步；而 StreamHub 实测有
    **49.6% 的轮次帧号是冻结的**（停帧最长 32 s），冻结期间 `gap<=0`
    凑不出任何命中 → 目标第一次进入画面时常常一条告警都不出。

★刻意**不引入任何 tracker 概念**★：没有 track_id、没有状态机、没有 init/模板/
replay/漂移守卫，也没有"候选池 / 命中计数"。本类现在只负责五件事：

  1. 输入 Detection 的合法性（bbox / label / 坐标越界）；
  2. **duplicate / stale 帧保护**（同一个 frame_id 最多处理一次）；
  3. 同帧重复框去重；
  4. **告警去重（cooldown）** —— 按「label + 空间位置」，防止持续可见的目标
     每 1~2 秒刷一条；
  5. 同类不同目标的空间区分（画面里人员 A 与人员 B 要分别告警）。

★为什么 cooldown 不能只看 label★：那样就是 label 级全局冷却，画面里两个人
只会出一条告警。这里按「同 label + 位置重合（IoU 或中心距离）」判定，
位置明显不同就是不同目标，各自出各自的告警。

与旧 `TargetManager` 的关系：旧文件**一行没删**（`backend/lmm/tracking/` 整体
保留），只是不再被任何运行时路径引用。
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

#: 一条 VLX 检出。`bbox` 恒为像素 xyxy。
BBox = Tuple[float, float, float, float]


# ── 数据结构 ─────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """VLX 一次推理给出的一个目标。**字段与 VLX 输出一一对应**，不做二次加工。"""

    frame_id: int
    pts_ms: Optional[float]
    label: str
    bbox: BBox
    confidence: float
    image_width: int = 0
    image_height: int = 0
    media_time_ms: float = 0.0
    label_zh: str = ""
    display_name: str = ""

    @property
    def key_label(self) -> str:
        """匹配用的类别键：小写去空白。中英文混排时避免大小写造成漏匹配。"""
        return (self.label or "").strip().lower()


@dataclass
class AlertEvent:
    """★1-Hit★的产物：一次告警。

    与旧 `DetectionEvent` 的关系：本对象是**纯数据**，由 `task_manager` 转成一个
    `DetectionEvent` 存进 `EventStore` 并推给前端。分层的理由是让「告警判定」与
    「事件存储/截图/落盘」解耦 —— 本文件因此可以脱离后端单独单测。

    ★证据帧就是检出帧★：`frames` / `boxes` / `confidences` 恒为**单元素**，
    且必然等于触发这条告警的那一次 VLX 检出。上层据此取图、画框，
    **禁止回头取最新帧、禁止换帧**。
    """

    label: str
    label_zh: str
    display_name: str
    bbox: BBox
    confidence: float
    frame_id: int
    media_time_ms: float
    pts_ms: Optional[float]
    #: ★告警证据★：恒为 `[frame_id]`（1-Hit：检出帧 = 告警帧 = 证据帧）
    frames: List[int]
    confidences: List[float]
    #: ★与 `frames` 逐帧对齐的 bbox★（VLX 原始像素 xyxy）。
    #: 上层用它**在服务端**把框画进证据图；前端禁止重绘。
    boxes: List[BBox] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    ts: float = 0.0


@dataclass
class _RecentAlert:
    """近期出过告警的目标（冷却判定用）。**不是 track**：不预测、不外推。"""

    label: str
    bbox: BBox
    ts: float
    frame_id: int


# ── 配置 ─────────────────────────────────────────────────────────────────────


@dataclass
class DetectConfig:
    """1-Hit 告警的阈值。**全部可从 `lmm_config.json` 覆盖**。

    ★已废弃的键（`lmm_alert_min_hits` / `max_frame_gap` / `max_time_ms` /
    `expire_ms`）不再参与判定★ —— 1-Hit 之后不存在"连续命中"和"候选存活"这两个
    概念。它们的配置项仍在 `config.DEFAULTS` 白名单里（不会被 `load()` 丢弃），
    只是读了也**不再生效**，留着是为了不让旧配置文件报错。
    """

    #: 同一目标相邻两次告警的 IoU 门槛（判定"是不是同一个目标"）
    iou_threshold: float = 0.5
    #: ★IoU 之外的第二道门★：中心位移 ≤ 目标对角线的这个倍数也算同一个目标。
    #: 0 = 只认 IoU（严格按原始规则）。
    #:
    #: 为什么必须有它：VLX 实测 ~0.5 Hz，而无人机画面本身在摇摄 —— 同一个人在
    #: 相邻两次采样之间能位移好几个身位，IoU 直接掉到 0（实测 IoU 中位数 0.014，
    #: **93.6% 的匹配靠这道兜底门**）。只认 IoU 的结果是该合并的合并不了。
    center_ratio: float = 1.5
    #: ★冷却★：同一「label + 位置」两次告警之间的最小间隔（毫秒）。
    #: 1-Hit 之后这是**唯一**防止刷屏的闸门。
    cooldown_ms: float = 12000.0
    #: 「已确认目标」指标保持多久（毫秒）
    confirm_hold_ms: float = 20000.0
    #: 告警证据：是否抓一张证据图
    evidence_enabled: bool = True
    #: 前端红色闪烁时长（毫秒），只作为默认值下发给前端
    pulse_ms: float = 2600.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "DetectConfig":
        d = dict(d or {})

        def _f(key: str, default: float) -> float:
            try:
                v = d.get(key)
                return default if v is None else float(v)
            except (TypeError, ValueError):
                return default

        return cls(
            iou_threshold=_f("lmm_alert_iou", cls.iou_threshold),
            center_ratio=_f("lmm_alert_center_ratio", cls.center_ratio),
            cooldown_ms=_f("lmm_alert_cooldown_ms", cls.cooldown_ms),
            confirm_hold_ms=_f("lmm_alert_confirm_hold_ms", cls.confirm_hold_ms),
            evidence_enabled=bool(d.get("lmm_alert_evidence", 1)),
            pulse_ms=_f("lmm_alert_pulse_ms", cls.pulse_ms),
        )


# ── 几何 ─────────────────────────────────────────────────────────────────────


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """像素 xyxy 的 IoU。任一框退化（宽或高 ≤ 0）时返回 0。"""
    try:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (IndexError, TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def _center_dist(a: Sequence[float], b: Sequence[float]) -> float:
    """两个框的中心位移（像素）。"""
    try:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (IndexError, TypeError, ValueError):
        return -1.0
    return math.hypot((ax1 + ax2) / 2.0 - (bx1 + bx2) / 2.0,
                      (ay1 + ay2) / 2.0 - (by1 + by2) / 2.0)


def _center_close(a: Sequence[float], b: Sequence[float], ratio: float) -> bool:
    """中心位移是否 ≤ `ratio` × 目标自身对角线（用**较小**的那个框做尺度）。

    用较小框而不是较大框：目标在画面里由远及近时会变大，拿大框当尺度会把
    "很远处的另一个目标"也算进来。
    """
    try:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (IndexError, TypeError, ValueError):
        return False
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    d = math.hypot(acx - bcx, acy - bcy)
    da = math.hypot(ax2 - ax1, ay2 - ay1)
    db = math.hypot(bx2 - bx1, by2 - by1)
    scale = max(1.0, min(da, db))
    return d <= ratio * scale


def _bbox_of(det: Dict[str, Any]) -> Optional[BBox]:
    """从 VLX 的检出 dict 里取 xyxy。兼容三种形态：

    * `{"x1":..,"y1":..,"x2":..,"y2":..}`（VLX-Seek 的 `decorate()` 产出）
    * `{"bbox": [x1,y1,x2,y2]}`
    * 直接的 `[x1,y1,x2,y2]`
    """
    try:
        if all(k in det for k in ("x1", "y1", "x2", "y2")):
            return (float(det["x1"]), float(det["y1"]),
                    float(det["x2"]), float(det["y2"]))
        raw = det.get("bbox")
        if isinstance(raw, (list, tuple)) and len(raw) >= 4:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError, KeyError):
        return None
    return None


def _same_target(a: BBox, b: BBox, cfg: DetectConfig) -> Tuple[bool, float, float]:
    """两个框是不是"同一个目标"（按 IoU 或中心距离）。

    返回 `(是/否, iou, 中心位移px)`。两道门是 **或** 的关系 —— 实测 IoU 对
    移动目标几乎恒为 0，靠中心距离兜底。
    """
    ov = iou_xyxy(a, b)
    if ov >= cfg.iou_threshold:
        return True, ov, _center_dist(a, b)
    if cfg.center_ratio > 0 and _center_close(a, b, cfg.center_ratio):
        return True, ov, _center_dist(a, b)
    return False, ov, _center_dist(a, b)


# ── 管理器 ───────────────────────────────────────────────────────────────────


class DetectionManager:
    """★VLX 检出 → 1-Hit 即时告警★。

    线程模型：VLX 线程调 `on_detections()`，本类内部自带锁，回调 `on_alert`
    **在调用方线程里同步执行** —— 上层若要落盘/取图，请自己在回调里丢给别的
    线程，别在这里做网络 IO（会拖慢 VLX 循环）。

    ★同一个 frame_id 最多处理一次★：这是「重复帧绝不重复告警」的第二层保护
    （第一层在 `task_manager.vlx_loop`，帧号没前进就不调 VLX）。
    """

    def __init__(self, task_id: str = "", cfg: Optional[DetectConfig] = None,
                 on_alert: Optional[Callable[[AlertEvent], None]] = None) -> None:
        self.task_id = str(task_id or "")
        self.cfg = cfg or DetectConfig()
        self.on_alert = on_alert
        self._lock = threading.RLock()
        #: ★duplicate / stale 闸门★：已处理过的最大帧号
        self._last_frame_id = 0
        #: 近期告警（冷却判定用），按 ts 升序
        self._recent: List[_RecentAlert] = []
        #: 近期告警的墙钟（供 `Confirmed Target` 指标）
        self._alert_ts: List[float] = []
        self.stats: Dict[str, Any] = {
            "frames": 0,            # 处理过的 VLX 帧数
            "detections": 0,        # 收到的 VLX 检出总数
            "valid": 0,             # 通过合法性校验的检出数
            "alerts": 0,            # 产出的告警数
            "skipped_frame": 0,     # 整帧被丢弃（duplicate / stale）
            "skipped_invalid": 0,   # 单条检出非法（bbox / label）
            "frame_dupes": 0,       # 同帧内被去重掉的重复框
            "suppressed": 0,        # 被冷却压住的次数
            "vlx_cycles": 0,        # 收到的 VLX 推理轮次
            "vlx_cycle_ms_sum": 0.0,
        }
        self._last_cycle_at: float = 0.0

    # ── 主入口 ───────────────────────────────────────────────────────────
    def on_detections(self, frame_id: int, media_time_ms: float,
                      detections: "List[Dict[str, Any]]",
                      pts_ms: Optional[float] = None,
                      image_width: int = 0, image_height: int = 0,
                      now: Optional[float] = None) -> List[AlertEvent]:
        """喂一帧 VLX 结果。返回**本帧新产生**的告警（1-Hit：检出即返回）。"""
        now = float(now if now is not None else time.time())
        frame_id = int(frame_id or 0)
        media_time_ms = float(media_time_ms or 0.0)
        out: List[AlertEvent] = []

        #: ★诊断★：VLX 每一轮都留痕 —— 帧号 / pts / 与上一轮的墙钟间隔 / 几个框
        dt_ms = 0.0
        if self._last_cycle_at > 0:
            dt_ms = (time.time() - self._last_cycle_at) * 1000.0
        self._last_cycle_at = time.time()

        with self._lock:
            self.stats["vlx_cycles"] = int(self.stats.get("vlx_cycles", 0)) + 1
            if dt_ms > 0:
                self.stats["vlx_cycle_ms_sum"] = (
                    float(self.stats.get("vlx_cycle_ms_sum", 0.0)) + dt_ms)

            #: ★第二层重复帧保护★：同一个 frame_id 最多处理一次。
            #: 第一层在 `vlx_loop`（帧号没前进就不调 VLX），这里是兜底 ——
            #: 两层都在，才敢说"同一个 frame_id 最多触发一次告警"。
            if frame_id <= self._last_frame_id:
                self.stats["skipped_frame"] = (
                    int(self.stats.get("skipped_frame", 0)) + 1)
                LOG.info("[DETECTION SKIP] task=%s reason="
                         "duplicate_or_stale_frame frame=%d last=%d",
                         self.task_id, frame_id, self._last_frame_id)
                return []
            self._last_frame_id = frame_id

            self.stats["frames"] += 1
            parsed = self._parse(detections, frame_id, media_time_ms, pts_ms,
                                 image_width, image_height)
            parsed = self._dedupe_same_frame(parsed)
            self.stats["detections"] += len(parsed)
            self.stats["valid"] += len(parsed)
            for det in parsed:
                self._prune_recent(now)
                ev = self._emit(det, now)
                if ev is not None:
                    out.append(ev)

        for ev in out:
            if self.on_alert is not None:
                try:
                    self.on_alert(ev)
                except Exception:  # noqa: BLE001
                    LOG.exception("[Detect] 任务 %s 告警回调失败", self.task_id)
        return out

    # ── 输入合法性 ────────────────────────────────────────────────────────
    def _parse(self, detections: "List[Any]", frame_id: int,
               media_time_ms: float, pts_ms: Optional[float],
               image_width: int, image_height: int) -> "List[Detection]":
        """把 VLX 的原始 dict 变成 `Detection`；不合法的**逐条**丢弃并留痕。

        ★为什么必须校验★：VLX 偶尔会给出退化框（`x2<=x1`）或越界坐标，
        拿去画证据图会得到一张没有框的"证据" —— 那是假证据，比没有更糟。
        """
        out: List[Detection] = []
        for d in (detections or []):
            if not isinstance(d, dict):
                continue
            label = str(d.get("label") or "").strip()
            if not label:
                self.stats["skipped_invalid"] = (
                    int(self.stats.get("skipped_invalid", 0)) + 1)
                LOG.info("[DETECTION SKIP] task=%s reason=invalid_label frame=%d",
                         self.task_id, frame_id)
                continue
            box = _bbox_of(d)
            if box is None or not self._bbox_ok(box, image_width, image_height):
                self.stats["skipped_invalid"] = (
                    int(self.stats.get("skipped_invalid", 0)) + 1)
                LOG.info("[DETECTION SKIP] task=%s reason=invalid_bbox frame=%d "
                         "bbox=%s", self.task_id, frame_id, box)
                continue
            out.append(Detection(
                frame_id=frame_id,
                pts_ms=pts_ms,
                label=label,
                bbox=box,
                confidence=float(d.get("detector_confidence")
                                 if d.get("detector_confidence") is not None
                                 else (d.get("confidence") or 0.0)),
                image_width=int(image_width or d.get("image_width") or 0),
                image_height=int(image_height or d.get("image_height") or 0),
                media_time_ms=media_time_ms,
                label_zh=str(d.get("label_zh") or "").strip(),
                display_name=str(d.get("display_name") or "").strip(),
            ))
        return out

    @staticmethod
    def _bbox_ok(b: BBox, image_width: int, image_height: int) -> bool:
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            return False
        for v in (x1, y1, x2, y2):
            if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
                return False
        if x2 <= x1 or y2 <= y1:
            return False
        if image_width > 0 and (x1 < -1 or x2 > image_width + 1):
            return False
        if image_height > 0 and (y1 < -1 or y2 > image_height + 1):
            return False
        return True

    def _dedupe_same_frame(self, dets: "List[Detection]") -> "List[Detection]":
        """同一帧里 label 相同且位置重合的框只留第一个。

        保留第一个而不是"置信度最高的那个"：VLX 的置信度常常恒为 1.0，按它排序
        等于随机挑一个；按出现顺序取第一个至少是稳定的。
        """
        kept: List[Detection] = []
        for d in dets:
            dup = False
            for k in kept:
                if k.key_label != d.key_label:
                    continue
                same, _ov, _cd = _same_target(k.bbox, d.bbox, self.cfg)
                if same:
                    dup = True
                    break
            if not dup:
                kept.append(d)
        dropped = len(dets) - len(kept)
        if dropped:
            self.stats["frame_dupes"] = int(self.stats.get("frame_dupes", 0)) + dropped
        return kept

    # ── 冷却 ──────────────────────────────────────────────────────────────
    def _prune_recent(self, now: float) -> None:
        ttl = float(self.cfg.cooldown_ms) / 1000.0
        if ttl <= 0:
            self._recent = []
            return
        self._recent = [r for r in self._recent if (now - r.ts) < ttl]

    def _emit(self, det: Detection, now: float) -> Optional[AlertEvent]:
        """★1-Hit★：过了冷却就**立刻**出告警，不等第二次命中。"""
        for r in self._recent:
            if r.label != det.key_label:
                continue
            same, ov, cd = _same_target(r.bbox, det.bbox, self.cfg)
            if not same:
                continue
            self.stats["suppressed"] = int(self.stats.get("suppressed", 0)) + 1
            LOG.info("[ALERT SUPPRESSED] task=%s reason=cooldown label=%s "
                     "frame=%d iou=%.3f cd=%.0fpx previous_frame=%d "
                     "previous_alert_ms_ago=%.0f cooldown_ms=%.0f "
                     "bbox=[%.0f,%.0f,%.0f,%.0f]",
                     self.task_id, det.key_label[:14], int(det.frame_id),
                     ov, cd, r.frame_id, (now - r.ts) * 1000.0,
                     self.cfg.cooldown_ms,
                     det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3])
            return None

        ev = AlertEvent(
            label=det.label, label_zh=det.label_zh,
            display_name=det.display_name,
            bbox=det.bbox,
            confidence=float(det.confidence),
            frame_id=int(det.frame_id),
            media_time_ms=det.media_time_ms,
            pts_ms=det.pts_ms,
            frames=[int(det.frame_id)],
            confidences=[float(det.confidence)],
            boxes=[det.bbox],
            image_width=det.image_width,
            image_height=det.image_height,
            ts=now,
        )
        self._recent.append(_RecentAlert(label=det.key_label, bbox=det.bbox,
                                         ts=now, frame_id=int(det.frame_id)))
        self._alert_ts.append(now)
        self.stats["alerts"] += 1
        LOG.info("[ALERT] task=%s label=%s frame=%d pts=%s bbox="
                 "[%.0f,%.0f,%.0f,%.0f] conf=%.3f（1-Hit 即时告警）",
                 self.task_id, det.key_label[:14], int(det.frame_id),
                 ("-" if det.pts_ms is None else "%.0f" % float(det.pts_ms)),
                 det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                 ev.confidence)
        return ev

    # ── 观测 ──────────────────────────────────────────────────────────────
    def confirmed_targets(self, now: Optional[float] = None) -> int:
        """当前处于「已确认」状态的目标数（供 `Confirmed Target` 指标）。

        1-Hit 之后它就是"最近 `confirm_hold_ms` 内出过告警的目标数"。
        """
        now = float(now if now is not None else time.time())
        hold = float(self.cfg.confirm_hold_ms) / 1000.0
        with self._lock:
            self._alert_ts = [t for t in self._alert_ts if (now - t) < hold]
            return len(self._alert_ts)

    def snapshot_stats(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self.stats)
            st["confirmed"] = self.confirmed_targets()
            st["live_recent"] = len(self._recent)
            st["last_frame_id"] = int(self._last_frame_id)
            st["iou"] = float(self.cfg.iou_threshold)
            st["center_ratio"] = float(self.cfg.center_ratio)
            st["cooldown_ms"] = float(self.cfg.cooldown_ms)
            st["pulse_ms"] = float(self.cfg.pulse_ms)
            _n = int(self.stats.get("vlx_cycles", 0) or 0)
            st["avg_vlx_cycle_ms"] = (
                round(float(self.stats.get("vlx_cycle_ms_sum", 0.0)) / _n, 1)
                if _n > 1 else 0.0)
            return st

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._alert_ts.clear()
            self._last_frame_id = 0


# ── 离线自检（不依赖后端，可直接 `python detection_manager.py`）──────────────


def _selftest() -> int:
    """★1-Hit★ 的最小验证：首检即告警 / 重复帧不重复 / 冷却不刷屏 / 不同目标分别告警。"""
    got: List[AlertEvent] = []

    #: 1. 第一次有效检出 → 立即告警
    dm = DetectionManager("selftest", DetectConfig(), on_alert=got.append)
    out = dm.on_detections(100, 0.0, [{"label": "person", "bbox": (100, 100, 200, 200),
                                       "confidence": 0.9}])
    assert len(out) == 1, out
    assert len(got) == 1, got
    assert out[0].frames == [100], out[0].frames
    #: ★证据帧 = 检出帧 = 告警帧★
    assert out[0].boxes == [(100, 100, 200, 200)], out[0].boxes
    assert out[0].frame_id == 100

    #: 2. 同一帧再来一次（duplicate）→ 整帧丢弃，不产生第二条告警
    out2 = dm.on_detections(100, 0.0, [{"label": "person", "bbox": (100, 100, 200, 200)}])
    assert out2 == [], out2
    assert len(got) == 1, got

    #: 3. 更旧的帧（stale）→ 同样丢弃
    assert dm.on_detections(99, 0.0, [{"label": "person", "bbox": (100, 100, 200, 200)}]) == []
    assert len(got) == 1, got

    #: 4. 新帧 + 同一目标（冷却内）→ 抑制，不刷屏
    out3 = dm.on_detections(110, 600.0, [{"label": "person", "bbox": (105, 105, 205, 205)}])
    assert out3 == [], out3
    assert len(got) == 1, got
    assert dm.stats["suppressed"] == 1

    #: 5. 新帧 + 同 label 但**位置明显不同** → 属于另一个目标，要分别告警
    out4 = dm.on_detections(111, 660.0, [{"label": "person", "bbox": (900, 500, 960, 560)}])
    assert len(out4) == 1, out4
    assert len(got) == 2, got

    #: 6. 同帧里两个几乎重合的框 → 去重，只出一条
    got.clear()
    dm2 = DetectionManager("selftest2", DetectConfig(), on_alert=got.append)
    same = (300, 300, 360, 360)
    o = dm2.on_detections(200, 0.0, [{"label": "person", "bbox": same},
                                     {"label": "person", "bbox": (302, 302, 362, 362)},
                                     {"label": "person", "bbox": (298, 298, 358, 358)}])
    assert len(o) == 1, o
    assert dm2.stats["frame_dupes"] == 2, dm2.stats

    #: 7. 退化框 / 空 label / 越界坐标 → 逐条丢弃，不出告警
    got.clear()
    dm3 = DetectionManager("selftest3", DetectConfig(), on_alert=got.append)
    o3 = dm3.on_detections(300, 0.0, [
        {"label": "person", "bbox": (50, 50, 40, 90)},      # x2 < x1
        {"label": "", "bbox": (10, 10, 20, 20)},            # 空 label
        {"label": "person", "bbox": (10, 10, 99999, 90)},   # 越界
    ], image_width=1920, image_height=1080)
    assert o3 == [], o3
    assert dm3.stats["skipped_invalid"] == 3, dm3.stats

    print("OK: 1-Hit 首检即告警；重复帧/过期帧不重复告警；冷却内不刷屏；"
          "同类不同位置分别告警；同帧重复框去重；非法检出被丢弃")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
