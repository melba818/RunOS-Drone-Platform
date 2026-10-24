"""VLM 调度与语义缓存。

**这是本次改造的性能核心。** 绝对不允许"每帧 → Qwen-VL"：

    主线程（视频解码 + YOLOE + Tracker）   ──投递候选──▶  有界队列 ──▶ VLM worker ──▶ 语义缓存
            ▲                                                                              │
            └──────────────── 每帧只读缓存（不阻塞） ◀───────────────────────────────────┘

设计要点：

- **按 Track 调用，不按帧调用。** 每个 track 有一条语义缓存记录，缓存有效就直接复用结论。
- **主线程永不等待网络。** `offer()` 只做入队（几微秒），推理在 worker 线程里跑。
- **队列有界 + 同 track 去重 + 过期丢弃。** 视频比 VLM 快是常态，堆队列只会让画面
  显示几秒前的目标；宁可丢，也不要滞后。
- **统计必须是真的。** 请求数/命中数/延迟全部来自实际调用，不估算、不填充。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .target_spec import TargetSpec
from .vlm_verifier import NullVerifier, VerificationResult

LOG = logging.getLogger(__name__)

STRATEGY_SMART = "smart"      # 智能触发（默认）：新目标 / 缓存过期 / 形态突变 / 重现 / 置信度下降
STRATEGY_INTERVAL = "interval"  # 固定间隔


def prompt_key_of(spec: TargetSpec) -> str:
    """目标描述指纹。用户改了 Prompt 或标签，指纹就变 → 全部缓存失效。"""
    parts = [spec.original_prompt or ""]
    parts += sorted(spec.detection_labels())
    for c in spec.conditions or []:
        parts.append(f"{c.type}:{c.value or c.target}")
    return "|".join(parts)


def _area(b: List[float]) -> float:
    try:
        return max(1.0, abs(float(b[2]) - float(b[0])) * abs(float(b[3]) - float(b[1])))
    except Exception:
        return 1.0


@dataclass
class CacheEntry:
    result: VerificationResult
    ts: float
    frame_index: int
    bbox: List[float]
    det_conf: float
    prompt_key: str
    count: int = 1


@dataclass
class VLMJob:
    track_id: int
    frame: Any
    bbox: List[float]
    det_conf: float
    det_label: str
    frame_index: int
    ts: float
    prompt_key: str
    history: List[Any] = field(default_factory=list)       # 连续多帧的原始裁剪（BGR）
    visual_prompt: Optional[str] = None                    # Visual Prompt 参考图 data URL
    reason: str = ""


# ── 语义缓存 ─────────────────────────────────────────────────────────────────


class SemanticCache:
    """track_id → VerificationResult 的短期缓存 + 重触发判定。"""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.ttl = float(cfg.get("semantic_cache_ttl", 2.0) or 2.0)
        self.interval = float(cfg.get("vlm_interval_seconds", 5.0) or 5.0)
        self.bbox_area_delta = float(cfg.get("vlm_bbox_area_delta", 0.35) or 0.35)
        self.conf_drop_ratio = float(cfg.get("vlm_conf_drop_ratio", 0.6) or 0.6)
        self.gap_frames = int(cfg.get("vlm_gap_frames", 15) or 15)
        self._d: Dict[int, CacheEntry] = {}
        self._lock = threading.RLock()
        self.reasons: Dict[str, int] = {}
        self.hits = 0
        self.misses = 0

    def _bump(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def decide(self, track_id: int, bbox: List[float], det_conf: float, frame_index: int,
               now: float, prompt_key: str, strategy: str = STRATEGY_SMART) -> Tuple[bool, str]:
        """要不要对这个 track 发起一次 VLM 调用？返回 (是否需要, 原因)。"""
        with self._lock:
            e = self._d.get(track_id)
            if e is None:
                self.misses += 1
                self._bump("new-track")
                return True, "new-track"
            if e.prompt_key != prompt_key:
                self.misses += 1
                self._bump("prompt-changed")
                return True, "prompt-changed"
            # 目标消失一段时间后又出现：之前的结论未必还成立（可能换了人/换了姿态）
            if frame_index - e.frame_index > self.gap_frames:
                self.misses += 1
                self._bump("reappeared")
                return True, "reappeared"
            a_now, a_old = _area(bbox), _area(e.bbox)
            if abs(a_now - a_old) / max(a_old, 1.0) > self.bbox_area_delta:
                self.misses += 1
                self._bump("bbox-area-changed")
                return True, "bbox-area-changed"
            if det_conf < e.det_conf * self.conf_drop_ratio:
                self.misses += 1
                self._bump("confidence-dropped")
                return True, "confidence-dropped"
            if strategy == STRATEGY_INTERVAL and (now - e.ts) >= self.interval:
                self.misses += 1
                self._bump("interval")
                return True, "interval"
            if (now - e.ts) >= self.ttl:
                self.misses += 1
                self._bump("ttl-expired")
                return True, "ttl-expired"
            self.hits += 1
            return False, "cache-hit"

    def put(self, track_id: int, result: VerificationResult, bbox: List[float],
            det_conf: float, frame_index: int, prompt_key: str) -> None:
        with self._lock:
            prev = self._d.get(track_id)
            self._d[track_id] = CacheEntry(
                result=result, ts=time.time(), frame_index=frame_index, bbox=list(bbox),
                det_conf=float(det_conf), prompt_key=prompt_key,
                count=(prev.count + 1) if prev else 1)

    def get(self, track_id: int) -> Optional[CacheEntry]:
        with self._lock:
            return self._d.get(track_id)

    def invalidate(self, track_id: int) -> None:
        with self._lock:
            self._d.pop(track_id, None)

    def invalidate_all(self) -> None:
        with self._lock:
            self._d.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._d)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


# ── 队列 ─────────────────────────────────────────────────────────────────────


class VLMJobQueue:
    """有界队列：满时丢**最旧**的（最旧的最没意义），同 track 只保留一个待处理任务。"""

    def __init__(self, maxsize: int = 16) -> None:
        self.maxsize = max(1, int(maxsize))
        self._q: deque = deque()
        self._pending: set = set()
        self._cv = threading.Condition()
        self.dropped_full = 0
        self.deduped = 0

    def put(self, job: VLMJob) -> bool:
        with self._cv:
            if job.track_id in self._pending:
                self.deduped += 1
                return False
            while len(self._q) >= self.maxsize:
                old = self._q.popleft()
                self._pending.discard(old.track_id)
                self.dropped_full += 1
            self._q.append(job)
            self._pending.add(job.track_id)
            self._cv.notify()
            return True

    def get(self, timeout: float = 0.2) -> Optional[VLMJob]:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            return self._q.popleft()

    def done(self, track_id: int) -> None:
        with self._cv:
            self._pending.discard(track_id)

    def size(self) -> int:
        with self._cv:
            return len(self._q)

    def has_pending(self, track_id: int) -> bool:
        with self._cv:
            return track_id in self._pending

    def clear(self) -> None:
        with self._cv:
            self._q.clear()
            self._pending.clear()


# ── 调度器 ───────────────────────────────────────────────────────────────────


class VLMScheduler:
    """把候选目标变成 VLM 调用，并维护每个 track 的语义结论。"""

    def __init__(self, task_id: str, cfg: Dict[str, Any], spec: TargetSpec,
                 verifier: Any, prompt_key: str,
                 on_result: Optional[Callable[[VLMJob, VerificationResult], None]] = None,
                 unavailable_reason: str = "") -> None:
        self.task_id = task_id
        self.cfg = dict(cfg or {})
        self.spec = spec
        self.verifier = verifier
        self.prompt_key = prompt_key
        self.on_result = on_result
        self.unavailable_reason = unavailable_reason

        self.enabled = not isinstance(verifier, NullVerifier)
        self.strategy = str(self.cfg.get("vlm_strategy", STRATEGY_SMART) or STRATEGY_SMART)
        self.concurrency = max(1, int(self.cfg.get("vlm_concurrency", 1) or 1))
        self.queue = VLMJobQueue(int(self.cfg.get("vlm_queue_size", 16) or 16))
        self.cache = SemanticCache(self.cfg)

        self.temporal_frames = max(0, int(self.cfg.get("vlm_temporal_frames", 0) or 0))
        self.temporal_span = max(1, int(self.cfg.get("vlm_temporal_span", 8) or 8))
        self.visual_prompt = self.cfg.get("visual_prompt_data_url") or None
        self.job_ttl = float(self.cfg.get("vlm_job_ttl_seconds", 3.0) or 3.0)

        self._history: Dict[int, deque] = {}
        self._hist_lock = threading.RLock()
        self._max_tracks = 32

        self._stop = threading.Event()
        self._workers: List[threading.Thread] = []
        self._stat_lock = threading.RLock()
        self.requests = 0
        self.verified = 0
        self.confirmed = 0
        self.rejected = 0
        self.unverified = 0
        self.dropped_stale = 0
        self.failed = 0
        self._lat_sum = 0.0
        self._lat_n = 0
        self.last_latency_ms = 0.0
        self.in_flight = 0
        self.avg_images = 0.0

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled or self._workers:
            return
        for i in range(self.concurrency):
            th = threading.Thread(target=self._loop, name=f"vlm-{self.task_id}-{i}", daemon=True)
            th.start()
            self._workers.append(th)
        LOG.info("[VLM] 调度器启动：%d 并发，策略=%s，缓存 TTL=%.1fs，上限边长=%s",
                 self.concurrency, self.strategy, self.cache.ttl, self.cfg.get("vlm_max_side"))

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self.queue._cv:
            self.queue._cv.notify_all()
        for th in self._workers:
            th.join(timeout=timeout)
        self._workers.clear()
        self.queue.clear()
        self._history.clear()

    # ── 主线程侧 ─────────────────────────────────────────────────────────────

    def remember(self, track_id: int, frame: Any, bbox: List[float],
                 frame_index: int) -> None:
        """为连续多帧行为理解留一小段原始裁剪（关闭时是空操作，不浪费 CPU）。"""
        if self.temporal_frames <= 1 or frame is None:
            return
        import vision_registry as registry
        crop = registry.crop_with_context(
            frame, bbox, self.cache_scale(), int(self.cfg.get("vlm_max_side", 768)),
            int(self.cfg.get("vlm_min_window", 256)))
        if crop is None:
            return
        with self._hist_lock:
            if len(self._history) >= self._max_tracks and track_id not in self._history:
                self._history.pop(next(iter(self._history)), None)
            dq = self._history.setdefault(track_id, deque(maxlen=self.temporal_frames * 3))
            # 按帧号跳采，保证时间上有间隔（相邻帧几乎一样，给 VLM 也没信息量）
            if not dq or (frame_index - dq[-1][0]) >= max(1, self.temporal_span):
                dq.append((frame_index, time.time(), crop))

    def cache_scale(self) -> float:
        return float(self.cfg.get("vlm_context_scale", 1.5) or 1.5)

    def _history_images(self, track_id: int) -> List[Any]:
        if self.temporal_frames <= 1:
            return []
        with self._hist_lock:
            dq = self._history.get(track_id)
            if not dq:
                return []
            items = list(dq)[-self.temporal_frames:]
        return [c for _, _, c in items]

    def offer(self, track_id: int, frame: Any, bbox: List[float], det_conf: float,
              det_label: str, frame_index: int, now: Optional[float] = None,
              force: bool = False) -> Tuple[bool, str]:
        """请求复核。返回 (是否入队, 原因)。**永远不阻塞主线程。**"""
        if not self.enabled:
            return False, "disabled"
        now = now if now is not None else time.time()
        need, reason = self.cache.decide(track_id, bbox, det_conf, frame_index, now,
                                         self.prompt_key, self.strategy)
        if not (need or force):
            return False, reason
        if self.queue.has_pending(track_id):
            return False, "already-queued"
        job = VLMJob(
            track_id=track_id, frame=frame, bbox=list(bbox), det_conf=float(det_conf),
            det_label=det_label, frame_index=frame_index, ts=now, prompt_key=self.prompt_key,
            history=self._history_images(track_id), visual_prompt=self.visual_prompt,
            reason="manual" if force else reason)
        if self.queue.put(job):
            return True, job.reason
        return False, "queue-rejected"

    # ── worker ───────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.queue.get(timeout=0.2)
            if job is None:
                continue
            try:
                self._process(job)
            except Exception:
                LOG.exception("[VLM] worker 异常（已忽略，不影响检测主流程）")
            finally:
                self.queue.done(job.track_id)

    def _process(self, job: VLMJob) -> None:
        now = time.time()
        # 过期任务直接丢：视频已经跑到很前面了，这个结论对当前画面没有意义
        if (now - job.ts) > self.job_ttl:
            self.dropped_stale += 1
            LOG.info("[VLM] track=%s 任务过期（%.1fs）丢弃", job.track_id, now - job.ts)
            return

        with self._stat_lock:
            self.in_flight += 1
            self.requests += 1
        try:
            ctx: Dict[str, Any] = {
                "track_id": job.track_id,
                "frame_index": job.frame_index,
                "detector_confidence": job.det_conf,
                "detector_label": job.det_label,
                "visual_prompt_data_url": job.visual_prompt,
            }
            history = job.history
            if history:
                import vision_registry as registry
                ctx["history"] = [
                    registry.frame_to_data_url(c, int(self.cfg.get("vlm_max_side", 768)),
                                               min_side=int(self.cfg.get("vlm_min_side", 384)))
                    for c in history]
                ctx["temporal_frames"] = len(ctx["history"])

            result = self.verifier.verify(job.frame, job.bbox, self.spec, ctx)
        except Exception as exc:  # Verifier 内部已兜底，这里再兜一层
            LOG.exception("[VLM] track=%s 复核异常", job.track_id)
            result = VerificationResult(verified=False, error=f"复核异常：{exc}")

        self.cache.put(job.track_id, result, job.bbox, job.det_conf, job.frame_index,
                       job.prompt_key)
        with self._stat_lock:
            self.in_flight = max(0, self.in_flight - 1)
            if result.verified:
                self.verified += 1
                if result.matched:
                    self.confirmed += 1
                else:
                    self.rejected += 1
            else:
                self.unverified += 1
                self.failed += 1
            self._lat_sum += float(result.latency_ms or 0.0)
            self._lat_n += 1
            self.last_latency_ms = float(result.latency_ms or 0.0)
            n = result.image_count or 1
            self.avg_images = ((self.avg_images * (self._lat_n - 1)) + n) / self._lat_n

        if self.on_result is not None:
            try:
                self.on_result(job, result)
            except Exception:
                LOG.exception("[VLM] on_result 回调异常（已忽略）")

    # ── 查询 ─────────────────────────────────────────────────────────────────

    def cached_result(self, track_id: int) -> Optional[VerificationResult]:
        e = self.cache.get(track_id)
        return e.result if e else None

    def verification_count(self, track_id: int) -> int:
        e = self.cache.get(track_id)
        return e.count if e else 0

    def invalidate_all(self) -> None:
        self.cache.invalidate_all()

    def invalidate(self, track_id: int) -> None:
        self.cache.invalidate(track_id)

    def stats(self) -> Dict[str, Any]:
        with self._stat_lock:
            avg = (self._lat_sum / self._lat_n) if self._lat_n else 0.0
            d = {
                "enabled": self.enabled,
                "config_id": getattr(self.verifier, "config_id", ""),
                "model_name": getattr(self.verifier, "model_name", ""),
                "unavailable_reason": self.unavailable_reason,
                "strategy": self.strategy,
                "cache_ttl": self.cache.ttl,
                "requests": self.requests,
                "verified": self.verified,
                "confirmed": self.confirmed,
                "rejected": self.rejected,
                "unverified": self.unverified,
                "failed": self.failed,
                "avg_latency_ms": round(avg, 1),
                "last_latency_ms": round(self.last_latency_ms, 1),
                "avg_images": round(self.avg_images, 2),
                "queue_size": self.queue.size(),
                "queue_max": self.queue.maxsize,
                "in_flight": self.in_flight,
                "dropped_stale": self.dropped_stale,
                "dropped_full": self.queue.dropped_full,
                "deduped": self.queue.deduped,
                "cache_size": self.cache.size(),
                "cache_hits": self.cache.hits,
                "cache_misses": self.cache.misses,
                "cache_hit_rate": round(self.cache.hit_rate(), 4),
                "trigger_reasons": dict(self.cache.reasons),
                "temporal_frames": self.temporal_frames,
                "has_visual_prompt": bool(self.visual_prompt),
            }
        return d
