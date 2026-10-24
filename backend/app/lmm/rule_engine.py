"""规则引擎：把「检测 + 跟踪 + VLM 复核」的结论变成告警事件。

判定链（第二阶段改造后）：

    YOLOE 命中目标标签（高召回候选）
      → 多目标逻辑（TargetSpec.logic：AND 要求每个目标都出现，OR 任一即可）
      → 规则条件（场景 / 缺失 / 数量；属性类留给 VLM）
      → 连续命中 hit_frames 帧
      → 投入 VLM 队列（异步，不阻塞）
      → VLM 结论回来后：matched=true 才产出事件
      → 同一 track 的告警冷却

**不再由 YOLOE 单独定案。** 复杂语义（穿没穿救生衣、是不是在水里）以 VLM 结论为准。

VLM 不可用时的降级（用户已确认策略）：仍然产出事件，但 `verified=False` 且前端标
「待AI复核」，保证离线 / 无额度时模块不会变成哑巴。

**不能判定的条件不假装判定。** 例如没把 life_jacket 下发给 YOLOE，规则侧就不会声称
"缺失成立"，而是放进 `unresolved` 交给 VLM 在 Prompt 里核对。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detectors.base_detector import DetectedBox, FrameResult
from .events import DetectionEvent, EventStore, crop_snapshot_b64
from .target_spec import (
    CONDITION_ABSENCE,
    CONDITION_ATTRIBUTE,
    CONDITION_COUNT,
    CONDITION_SCENE,
    TargetSpec,
)
from .vlm_verifier import VerificationResult

LOG = logging.getLogger(__name__)

#: VLM 不可用时的策略
ON_ERROR_ALLOW = "allow"   # 降级告警并标记「未复核」（默认）
ON_ERROR_BLOCK = "block"   # 严格：拿不到 VLM 结论就不告警


def _encode_frame_small(frame: Any, max_side: int = 640, quality: int = 70) -> Optional[str]:
    """告警留证用的整帧缩略图。"""
    if frame is None:
        return None
    try:
        import base64
        import io

        from PIL import Image
        img = Image.fromarray(frame[:, :, ::-1])
        side = max(img.size)
        if side > max_side:
            s = max_side / float(side)
            img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _cell_of(b: DetectedBox, frame: Any, grid: int) -> str:
    """把检测框的归一化中心落到 grid×grid 的哪一格，返回 `<格x>-<格y>`。

    这是"无轨迹候选"的空间归属键。用画面网格而不是精确坐标，是为了让目标轻微移动
    时仍落在同一格 —— 否则伪轨迹每帧换键，连续命中数永远攒不起来。
    """
    if frame is None or grid <= 1:
        return ""
    try:
        h, w = frame.shape[:2]
    except Exception:  # noqa: BLE001
        return ""
    if not h or not w:
        return ""
    cx = (float(b.x1) + float(b.x2)) * 0.5
    cy = (float(b.y1) + float(b.y2)) * 0.5
    gx = min(grid - 1, max(0, int(cx / float(w) * grid)))
    gy = min(grid - 1, max(0, int(cy / float(h) * grid)))
    return f"{gx}-{gy}"


@dataclass
class _TrackState:
    """单个 track 的累计状态。"""

    hits: int = 0
    last_alert_ts: float = 0.0
    last_frame: int = -1
    #: 最近一帧的单帧推理耗时（毫秒），用于写进事件的 infer_latency_ms
    last_infer_ms: Optional[float] = None
    best: Optional[DetectedBox] = None
    frame: Any = None
    unresolved: List[str] = field(default_factory=list)
    matched_conditions: List[str] = field(default_factory=list)
    #: 已经基于缓存里哪一条结论出过事件（用 cache entry 的 ts 做指纹，避免同一结论重复告警）
    emitted_entry_ts: float = 0.0
    #: 最近一次 VLM 明确判定为「不匹配」
    rejected: bool = False


@dataclass
class RuleEngineConfig:
    hit_frames: int = 3
    cooldown_seconds: float = 30.0
    min_targets: int = 1
    vlm_on_error: str = ON_ERROR_ALLOW
    #: 无轨迹候选兜底：跟踪器没给 track_id 时，按画面网格当"伪轨迹"继续送 VLM 复核。
    #: 关闭则退回旧行为（只计数、不复核、不告警）。
    trackless_verify: bool = True
    #: 空间格边长（画面切成 grid×grid）。
    trackless_grid: int = 6

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RuleEngineConfig":
        return RuleEngineConfig(
            hit_frames=max(1, int(d.get("hit_frames", 3) or 3)),
            cooldown_seconds=float(d.get("cooldown_seconds", 30) or 30),
            min_targets=max(1, int(d.get("min_targets", 1) or 1)),
            vlm_on_error=str(d.get("vlm_on_error", ON_ERROR_ALLOW) or ON_ERROR_ALLOW),
            trackless_verify=bool(d.get("trackless_verify", True)),
            trackless_grid=max(1, int(d.get("trackless_grid", 6) or 6)),
        )


class RuleEngine:
    """有状态：跨帧累计命中 + 冷却 + VLM 结论门控。同一个任务一个实例。

    线程模型：`evaluate()` 在检测线程调用，`on_verification()` 在 VLM worker 线程调用，
    因此内部状态一律用可重入锁保护。
    """

    def __init__(self, spec: TargetSpec, cfg: RuleEngineConfig, store: EventStore,
                 task_id: str, scheduler: Any = None,
                 on_event: Optional[Callable[[DetectionEvent], None]] = None,
                 semantic_final: bool = False) -> None:
        self.spec = spec
        self.cfg = cfg
        self.store = store
        self.task_id = task_id
        self.scheduler = scheduler
        self.on_event = on_event
        #: **检测器本身就是语义模型**（VLX-Seek）。此时整句 Prompt 已经由检测器判定
        #: 完毕，"在水里""没穿救生衣"这类条件不再由规则侧二次核验，因此：
        #:   ① 不做标签白名单过滤（VLX 不产出 YOLOE 类别）；
        #:   ② 多目标逻辑由语义模型保证，不再按标签命中判定；
        #:   ③ 条件不满足不再 hard_fail —— 否则"水面上的船只"会因画面里没有
        #:      water 这个框而被永久否决（VLX 只回目标框，不回场景框）。
        self.semantic_final = bool(semantic_final)
        self._lock = threading.RLock()
        self._tracks: Dict[str, _TrackState] = {}
        self._target_labels: List[str] = list(spec.detection_labels())
        self._observable: set = set(spec.observable_labels())
        self._label_zh: Dict[str, str] = spec.label_to_zh
        self.emitted = 0
        #: 因拿不到 track_id 而被跳过的候选数（仅当兜底关闭 / 没有帧可用于定格子时增长）
        self.skipped_no_track = 0
        #: 走"空间格伪轨迹"送进 VLM 复核的候选数
        self.trackless_verified = 0

    # ── 条件判定（规则侧能判的） ─────────────────────────────────────────────

    def _check_conditions(self, boxes: List[DetectedBox]) -> Tuple[List[str], List[str], bool]:
        """返回 (matched, unresolved, hard_fail)。"""
        matched: List[str] = []
        unresolved: List[str] = []
        present = {b.label for b in boxes}
        hard_fail = False

        for cond in self.spec.conditions or []:
            if cond.type == CONDITION_COUNT:
                n = len([b for b in boxes if b.label in self._target_labels])
                if n >= self.cfg.min_targets:
                    matched.append(f"数量 {n} ≥ {self.cfg.min_targets}")
                else:
                    hard_fail = True
                continue

            if cond.type == CONDITION_SCENE:
                label = cond.text or f"场景：{cond.value}"
                if cond.value in self._observable:
                    if cond.value in present:
                        matched.append(label)
                    else:
                        hard_fail = True
                else:
                    unresolved.append(f"{label}（YOLOE 未下发该类，交由 VLM 复核）")
                continue

            if cond.type == CONDITION_ABSENCE:
                label = cond.text or f"未检出：{cond.target}"
                if cond.target in self._observable:
                    if cond.target not in present:
                        matched.append(label)
                    else:
                        hard_fail = True
                else:
                    # 没有把它当检测类下发 → 规则侧无从判断，交给 VLM 在图像里找
                    unresolved.append(f"{label}（YOLOE 未下发该类，交由 VLM 复核）")
                continue

            if cond.type == CONDITION_ATTRIBUTE:
                unresolved.append(f"{cond.text or ('属性：' + cond.value)}（交由 VLM 复核）")
                continue

            unresolved.append(cond.text or f"{cond.type}：{cond.value or cond.target}")

        return matched, unresolved, hard_fail

    def _targets_satisfied(self, present: set) -> bool:
        """多目标逻辑：AND 要求每个 Target 至少命中一个标签；OR 任一即可。"""
        targets = self.spec.targets or []
        if not targets:
            return True

        def hit(t) -> bool:
            return any(lb in present for lb in (t.labels or [t.detection_label]))

        if self.spec.logic == "OR":
            return any(hit(t) for t in targets)
        return all(hit(t) for t in targets)

    # ── 主入口（检测线程） ───────────────────────────────────────────────────

    def evaluate(self, frame_result: FrameResult, frame: Any = None,
                 now: Optional[float] = None) -> List[DetectionEvent]:
        now = now if now is not None else time.time()
        with self._lock:
            boxes = list(frame_result.boxes or [])
            if not boxes:
                # 全丢帧：命中计数清零（人走了不该继续告警），但缓存留着（目标可能被遮挡）
                for st in self._tracks.values():
                    st.hits = 0
                return []

            present = {b.label for b in boxes}
            if self.semantic_final:
                # VLX-Seek 语义定案：有框即代表整句 Prompt 成立，规则侧不再否决
                targets_ok = True
                unresolved = []
            else:
                targets_ok = self._targets_satisfied(present)
                matched, unresolved, _ = self._check_conditions(boxes)

            groups: Dict[str, List[DetectedBox]] = {}
            for b in boxes:
                if not self.semantic_final and b.label not in self._target_labels:
                    continue
                # 归属键同时决定「状态累计」与「语义缓存」的粒度，必须用同一个函数算
                key = self._vlm_key(b, frame, self.cfg)
                groups.setdefault(key, []).append(b)

            events: List[DetectionEvent] = []
            for key, group in groups.items():
                best = max(group, key=lambda b: b.confidence)
                st = self._tracks.setdefault(key, _TrackState())
                if not targets_ok:
                    st.hits = 0
                    continue

                _m, unresolved_i, hard_fail = self._check_conditions(boxes)
                if hard_fail and not self.semantic_final:
                    st.hits = 0
                    continue
                if self.semantic_final:
                    # 条件已由语义模型判定，不再作为"待确认"挂到事件上
                    unresolved_i = []

                st.hits += 1
                st.best = best
                st.frame = frame
                st.last_frame = frame_result.frame_index
                st.last_infer_ms = getattr(frame_result, "elapsed_ms", None)
                st.unresolved = unresolved_i
                st.matched_conditions = _m

                if st.hits < self.cfg.hit_frames:
                    continue

                if self.scheduler is not None and self.scheduler.enabled:
                    trackless = best.track_id is None
                    if trackless and key.startswith("lb:"):
                        # 兜底关闭（或这一帧拿不到画面、定不出格子）→ 无法归属到语义缓存，
                        # 既不能复核也不能按 track 告警。照旧出"未复核"告警就等于绕开复核门控，
                        # 实测会让一个 `#None` 框在 VLM 一次都没调用的情况下产出告警。只计数。
                        self.skipped_no_track += 1
                        if self.skipped_no_track == 1:
                            LOG.warning(
                                "[RULE] 候选框缺少 track_id 且无法定位空间格（label=%s）→ 跳过；"
                                "复核开启时告警须可归属，不产生未复核告警", best.label)
                        continue
                    if trackless:
                        # 极小/远距离目标：ultralytics 不会为它们建立轨迹，用空间格当伪轨迹复核
                        self.trackless_verified += 1
                    bbox = [best.x1, best.y1, best.x2, best.y2]
                    self.scheduler.remember(key, frame, bbox, frame_result.frame_index)
                    self.scheduler.offer(key, frame, bbox, best.confidence, best.label,
                                         frame_result.frame_index, now)
                    ev = self._emit_from_cache(st, key, best, now)
                    if ev:
                        events.append(ev)
                else:
                    ev = self._emit(st, best, None, now)
                    if ev:
                        events.append(ev)

            return events

    # ── 异步结论回调（VLM worker 线程） ──────────────────────────────────────

    def on_verification(self, job: Any, result: VerificationResult,
                        now: Optional[float] = None) -> Optional[DetectionEvent]:
        """VLM 结论到达。此时 scheduler 已经把结果写进缓存，直接从缓存走同一条出事件路径。"""
        now = now if now is not None else time.time()
        with self._lock:
            # job.track_id 实际承载的是 evaluate() 里的归属键（可能是 id:220 或 sp:person:3-5）
            st = self._tracks.get(str(getattr(job, "track_id", "") or ""))
            if st is None or st.best is None:
                return None
            if st.hits < self.cfg.hit_frames:
                return None
            key = str(getattr(job, "track_id", "") or "")
            ev = self._emit_from_cache(st, key, st.best, now)
            if ev is not None and self.on_event is not None:
                try:
                    self.on_event(ev)
                except Exception:
                    LOG.exception("[RULE] on_event 回调异常")
            return ev

    # ── 出事件 ───────────────────────────────────────────────────────────────

    def _emit_from_cache(self, st: _TrackState, key: str, box: DetectedBox,
                         now: float) -> Optional[DetectionEvent]:
        """依据当前缓存的复核结论决定是否出事件。`key` 必须与 `evaluate` 里的归属键一致。"""
        if not key or self.scheduler is None:
            return None
        entry = self.scheduler.cache.get(key)
        if entry is None:
            return None
        if st.emitted_entry_ts == entry.ts:
            return None                       # 同一条结论已经告过警
        res: VerificationResult = entry.result

        if res.verified and not res.matched:
            st.rejected = True
            return None                       # AI 明确说不是 → 不出事件
        if not res.verified and self.cfg.vlm_on_error == ON_ERROR_BLOCK:
            return None                       # 严格模式：拿不到结论就不告警
        ev = self._emit(st, box, res, now,
                        track_key=("" if box.track_id is not None else key))
        if ev is not None:
            st.emitted_entry_ts = entry.ts
        return ev

    def _emit(self, st: _TrackState, box: DetectedBox,
              res: Optional[VerificationResult], now: float,
              track_key: str = "") -> Optional[DetectionEvent]:
        if st.last_alert_ts and (now - st.last_alert_ts) < self.cfg.cooldown_seconds:
            return None
        st.last_alert_ts = now
        self.emitted += 1

        verified = None if res is None else bool(res.verified)
        attrs = dict(res.attributes) if (res is not None and res.verified) else {}
        unresolved = list(st.unresolved or [])
        if res is not None and res.verified:
            # VLM 已经判定过，规则侧的"待确认"就不再是待确认
            unresolved = []

        ev = DetectionEvent(
            id=self.store.next_id(self.task_id),
            task_id=self.task_id,
            frame_index=st.last_frame if st.last_frame >= 0 else 0,
            ts=now,
            time_str=time.strftime("%H:%M:%S", time.localtime(now)),
            track_id=box.track_id,
            track_key=track_key,
            label=box.label,
            # 展示优先用检测器自己给出的措辞（VLX-Seek 在 <ground> 里写的中文），
            # 而不是 TargetSpec 里的东西 —— 后者现在只是用户原文，不像告警标题。
            label_zh=box.label_zh or self._label_zh.get(box.label, box.label),
            display_name=(box.label_zh or self.spec.display_name
                          or self._label_zh.get(box.label, box.label)),
            confidence=box.confidence,
            bbox=[box.x1, box.y1, box.x2, box.y2],
            matched=list(st.matched_conditions or []),
            unresolved=unresolved,
            hit_frames=st.hits,
            snapshot_b64=crop_snapshot_b64(st.frame, [box.x1, box.y1, box.x2, box.y2])
            if st.frame is not None else None,
            frame_b64=_encode_frame_small(st.frame) if st.frame is not None else None,
            detector_confidence=box.confidence,
            vlm_confidence=(res.confidence if (res is not None and res.verified) else None),
            verified=verified,
            vlm_matched=(res.matched if (res is not None and res.verified) else None),
            vlm_reason=((res.reason if res is not None else "")
                        or ("检测器直接检出" if self.semantic_final else "")),
            # 语义检测器不经过二阶段：`verified=None` 会让 to_dict 派生出 "pending"，
            # 于是前端给一条已经检出的告警挂上「待复核」。这里显式标 confirmed。
            vlm_state=("confirmed" if self.semantic_final else ""),
            vlm_attributes=attrs,
            vlm_config_id=(res.config_id if res is not None else ""),
            vlm_model_name=(res.model_name if res is not None else ""),
            vlm_latency_ms=(round(res.latency_ms, 1) if res is not None else None),
            vlm_attempts=(res.attempts if res is not None else 0),
            vlm_error=("" if (res is None or res.verified) else (res.error or "未复核")),
            original_prompt=self.spec.original_prompt,
            spec=self.spec.to_dict(),
            detector_label=box.label,
            infer_latency_ms=st.last_infer_ms,
        )
        self.store.add(ev)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:
                LOG.exception("[RULE] on_event 回调异常")
        LOG.info("[RULE] 告警 %s | %s #%s | det=%.2f vlm=%s | %s",
                 ev.id, ev.display_name, ev.track_id, ev.detector_confidence,
                 ("-" if ev.vlm_confidence is None else f"{ev.vlm_confidence:.2f}"),
                 ev.vlm_reason or ev.vlm_error or "")
        return ev

    # ── 给前端的画框状态 ─────────────────────────────────────────────────────

    @staticmethod
    def _vlm_key(b: DetectedBox, frame: Any, cfg: "RuleEngineConfig") -> str:
        """候选的归属键 —— **状态累计与语义缓存共用同一个键**，二者必须一致。

        - 有真实轨迹          → `id:<track_id>`
        - 无轨迹 + 兜底开启   → `sp:<label>:<格x>-<格y>`（空间伪轨迹）
        - 无轨迹 + 兜底关闭   → `lb:<label>`（旧行为，仅用于累计命中、不送复核）
        """
        if b.track_id is not None:
            return f"id:{b.track_id}"
        if cfg.trackless_verify:
            cell = _cell_of(b, frame, cfg.trackless_grid)
            if cell:
                return f"sp:{b.label}:{cell}"
        return f"lb:{b.label}"

    def decorate(self, boxes: List[DetectedBox],
                 frame: Any = None) -> List[Dict[str, Any]]:
        """给每个框补三态与两级置信度，供前端按 RunOS 视觉规范渲染。

        `frame` 用来给无轨迹候选定空间格；缺省时退化为「无轨迹=no_track」的旧表现。
        """
        out: List[Dict[str, Any]] = []
        vlm_on = self.scheduler is not None and self.scheduler.enabled
        for b in boxes:
            d = b.to_dict()
            d["detector_confidence"] = d.pop("confidence", b.confidence)
            # 同上：VLX-Seek 自己给的 label 是最终结论，优先于 TargetSpec 的原文
            d["display_name"] = d.get("label_zh") or self.spec.display_name or b.label
            # 与 evaluate() 同源：框上的三态必须查它真正被复核时用的那个键
            key = self._vlm_key(b, frame, self.cfg)
            d["track_key"] = "" if b.track_id is not None else key
            if self.semantic_final:
                # 检测器本身就是语义模型：选中即检出，没有"待复核"这一态
                d["vlm_state"] = "confirmed"
                d["vlm_confidence"] = None
                d["vlm_reason"] = "检测器直接检出"
            elif not vlm_on:
                # 关闭复核时，框由规则引擎直接定案；前端不应显示"待复核"
                d["vlm_state"] = "off"
                d["vlm_confidence"] = None
                d["vlm_reason"] = ""
            elif b.track_id is None and key.startswith("lb:"):
                # 复核开着，但既没有 track_id 也定不出空间格：无法归属到语义缓存，
                # 既不是"待复核"也不是"已确认"。必须与 off 区分，否则用户会以为
                # 复核被关掉了 —— 实测这正是"VLM 调用数为 0"的一个真实来源。
                d["vlm_state"] = "no_track"
                d["vlm_confidence"] = None
                d["vlm_reason"] = ""
            else:
                entry = self.scheduler.cache.get(key)
                if entry is None:
                    d["vlm_state"] = "pending"
                    d["vlm_confidence"] = None
                    d["vlm_reason"] = ""
                else:
                    r = entry.result
                    if r.verified:
                        d["vlm_state"] = "confirmed" if r.matched else "rejected"
                        d["vlm_confidence"] = round(float(r.confidence), 4)
                        d["vlm_reason"] = r.reason
                    else:
                        d["vlm_state"] = "error"
                        d["vlm_confidence"] = None
                        d["vlm_reason"] = r.error or "未复核"
            out.append(d)
        return out

    # ── 可观测性 ─────────────────────────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        if self.semantic_final:
            by_detector = ("detector", "整句 Prompt 已由检测器直接判定（不二次核验）")
        for cond in self.spec.conditions or []:
            if self.semantic_final:
                state, note = by_detector
            elif cond.type == CONDITION_ATTRIBUTE:
                state, note = "vlm", "由视觉大模型判定"
            elif cond.type == CONDITION_SCENE:
                ok = cond.value in self._observable
                state, note = ("live" if ok else "vlm"), (
                    "画面中出现该类别即成立" if ok else "YOLOE 未下发该类，由 VLM 判定")
            elif cond.type == CONDITION_ABSENCE:
                ok = cond.target in self._observable
                state, note = ("live" if ok else "vlm"), (
                    "画面中未出现该类别即成立" if ok else "YOLOE 未下发该类，由 VLM 判定")
            else:
                state, note = "live", "按数量阈值判定"
            checks.append({"type": cond.type, "text": cond.text or cond.value or cond.target,
                           "state": state, "note": note})
        return {
            "labels": list(self._target_labels),
            "hit_frames": self.cfg.hit_frames,
            "cooldown_seconds": self.cfg.cooldown_seconds,
            "min_targets": self.cfg.min_targets,
            "vlm_on_error": self.cfg.vlm_on_error,
            "logic": self.spec.logic,
            "conditions": checks,
            "emitted": self.emitted,
            "skipped_no_track": self.skipped_no_track,
            "trackless_verified": self.trackless_verified,
            "trackless_verify": bool(self.cfg.trackless_verify),
            "trackless_grid": self.cfg.trackless_grid,
            "vlm_gated": bool(self.scheduler is not None and self.scheduler.enabled),
            "semantic_final": bool(self.semantic_final),
        }
