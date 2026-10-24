"""★证据异步链★：把「三帧确认 → 出告警」和「取图 / 画框 / 编码 / 落盘」彻底解耦。

为什么要拆开（这是本轮的主线）：
    旧链路是**同步**的：三次确认一命中，VLX 检测线程就自己去取 JPEG、解码、画框、
    再编码、写磁盘，干完才轮到下一轮推理。证据 IO 一慢，**整条检测链跟着慢**。

改完之后：

    StreamHub → VLX → DetectionManager → 三次确认
        → **立刻创建 AlertEvent（evidence_status=pending，三张都是 pending）**
        → 把 EvidenceJob 丢进队列 → VLX 立刻继续下一轮推理
                                     ↓（后台 worker）
        source.jpg（StreamHub q95 原始字节，原样落盘）
        + 001.png/png（画了 VLX bbox 的展示图）
        + 001_thumb.jpg（卡片缩略图）
        → evidence_update 推给前端，逐张 ready

★三个绝不能让步的约束★
  1. **证据帧号在 job 创建时就钉死**（`frame_id`/`pts_ms`/`bbox`）。worker 执行时
     绝不允许 `get_latest_frame()` —— 异步化最怕的就是"回头取最新的"，那样证据会错帧；
     ring 淘汰了就报 `expired`，宁可缺一张也不能拿别的帧顶替。
  2. **source.jpg 必须是 StreamHub 的原始 JPEG 字节**，原样 `open(...,'wb').write()`，
     不解码不重编码 —— 一次有损都不多加。
  3. **队列有上限**（默认 32）。job 自带 JPEG bytes（~350 KB/张，一条告警 ~1 MB），
     不能无限堆积；满了就丢弃并计数（**丢的是证据图，不是告警**）。

★为什么用 `queue.Queue` 而不是 `asyncio.Queue`★：检测主循环是**线程模型**
（`threading.Event` 控制停止、任务在子线程里跑），不是协程。为了搬字节再引入一条
事件循环只会多一层风险；`queue.Queue` + 固定 worker 线程已经满足"不阻塞 VLX"的目标。

★worker 里任何异常都不能影响检测★：捕获 → 记日志（event_id/frame_id/stage）→
该张 `status=error` → 继续下一张。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOG = logging.getLogger(__name__)

#: ── 状态（与前端一致）─────────────────────────────────────────────────────
PENDING = "pending"
PROCESSING = "processing"
READY = "ready"
ERROR = "error"

#: 队列上限：一条告警 3 张 × ~350 KB ≈ 1 MB，32 个 job ≈ 30 MB 级，可控。
DEFAULT_MAXSIZE = 32
DEFAULT_WORKERS = 2

#: 清晰度比值的告警线：**处理链把图弄糊**的特征是 ratio 明显掉下来。
#: 注意 Laplacian 方差对量化噪声不敏感（q95→q95 实测 ratio≈1.0），
#: 所以这个阈值只在"真的被缩放/被重采样/取错帧"时才会触发。
SHARPNESS_RATIO_WARN = 0.85


@dataclass
class EvidenceJob:
    """一张证据图的作业。

    ★所有字段在**创建时**就固定★：worker 执行时只许照着做，不许重新取帧、
    不许重新推理、不许改 bbox。
    """

    task_id: str
    alert_id: str
    index: int                       # 0/1/2
    frame_id: int                    # ★钉死的帧号★
    pts_ms: Optional[float] = None
    label: str = ""
    label_zh: str = ""
    display_name: str = ""
    bbox: Optional[List[float]] = None
    #: ★StreamHub q95 原始 JPEG 字节★（三次确认时顺手从预取缓存拿的，不额外 IO）
    jpeg: Optional[bytes] = None
    #: 预取没命中时的兜底：worker 里按**同一个 frame_id** 再取一次（不是取最新帧）
    fetcher: Optional[Callable[[int], Optional[bytes]]] = None
    #: 输出目录（<alerts>/<task>/<alert>）
    out_dir: Optional[Path] = None
    #: 状态回调（可选）：worker 开始处理时通知前端变成 processing
    on_status: Optional[Callable[[str, int, str], None]] = None
    #: 完成回调：meta（成功）或 err（失败）
    on_done: Optional[Callable[[str, int, Optional[Dict[str, Any]], Optional[str]], None]] = None
    enqueued_at: float = field(default_factory=time.time)


class EvidenceWorker:
    """后台证据 worker：取字节 → 落 source → 画框 → 编码 → 缩略图 → 回传元信息。"""

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE,
                 workers: int = DEFAULT_WORKERS) -> None:
        self._q: "queue.Queue[EvidenceJob]" = queue.Queue(maxsize=int(maxsize))
        self._n = max(1, int(workers))
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        #: 诊断计数（**只进 debug 日志/metrics，绝不上前端**）
        self.submitted = 0
        self.done = 0
        self.failed = 0
        self.dropped = 0
        self.expired = 0
        self._latencies: List[float] = []
        self._encode_ms: List[float] = []

    # ── 生命周期 ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for i in range(self._n):
            t = threading.Thread(target=self._run, name=f"evidence-w{i}",
                                 daemon=True)
            t.start()
            self._threads.append(t)
        LOG.info("[LMM] 证据 worker 启动：%d 个线程 / 队列上限 %d", self._n, self._q.maxsize)

    def stop(self, wait: float = 2.0) -> None:
        self._stop.set()
        for t in self._threads:
            try:
                t.join(timeout=wait)
            except Exception:  # noqa: BLE001
                pass
        self._threads = []

    # ── 提交（绝不阻塞检测主循环）──────────────────────────────────────────
    def submit(self, job: EvidenceJob) -> bool:
        """入队。**非阻塞**：队列满就丢弃并返回 False（丢图不丢告警）。"""
        try:
            self._q.put_nowait(job)
            with self._lock:
                self.submitted += 1
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            LOG.warning("[LMM] 证据队列已满（%d），丢弃 %s #%d frame=%s",
                        self._q.maxsize, job.alert_id, job.index, job.frame_id)
            if job.on_done:
                try:
                    job.on_done(job.alert_id, job.index, None, "queue_full")
                except Exception:  # noqa: BLE001
                    LOG.debug("[LMM] 证据丢弃回调失败", exc_info=True)
            return False

    def depth(self) -> int:
        return self._q.qsize()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            lat = list(self._latencies[-50:])
            enc = list(self._encode_ms[-50:])
            return {"queued": self._q.qsize(), "submitted": self.submitted,
                    "done": self.done, "failed": self.failed,
                    "dropped": self.dropped, "expired": self.expired,
                    "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
                    "avg_encode_ms": round(sum(enc) / len(enc), 1) if enc else 0.0,
                    "workers": self._n}

    # ── worker 主循环 ─────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                self._notify(job, PROCESSING)
                meta, err = self._process(job)
                lat_ms = (time.time() - t0) * 1000.0
                with self._lock:
                    if err:
                        self.failed += 1
                    else:
                        self.done += 1
                    self._latencies.append(lat_ms)
                if meta is not None:
                    #: ★入队→落盘的端到端耗时★（验收要统计，不进前端）
                    meta["latency_ms"] = round(lat_ms, 1)
                if job.on_done:
                    job.on_done(job.alert_id, job.index, meta, err)
            except Exception as exc:  # noqa: BLE001
                #: ★worker 里的任何异常都不许影响检测★
                LOG.exception("[LMM] 证据作业异常 alert=%s #%d frame=%s",
                              job.alert_id, job.index, job.frame_id)
                with self._lock:
                    self.failed += 1
                if job.on_done:
                    try:
                        job.on_done(job.alert_id, job.index, None, f"{type(exc).__name__}: {exc}")
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                self._q.task_done()

    def _notify(self, job: EvidenceJob, status: str) -> None:
        if not job.on_status:
            return
        try:
            job.on_status(job.alert_id, job.index, status)
        except Exception:  # noqa: BLE001
            LOG.debug("[LMM] 证据状态回调失败", exc_info=True)

    # ── 单张证据的处理 ────────────────────────────────────────────────────
    def _process(self, job: EvidenceJob):
        """返回 `(meta, err)`：成功时 err=None。"""
        from .render import (EVIDENCE_IMAGE_EXT, EVIDENCE_IMAGE_FORMAT,
                             EVIDENCE_JPEG_QUALITY, THUMB_MAX_SIDE, draw_boxes,
                             encode_image_bytes, jpeg_wh, make_thumbnail_jpeg,
                             sharpness)
        import cv2  # noqa: F401  下面解码要用

        data = job.jpeg
        stage = "source"
        try:
            if not data:
                #: ★按同一个 frame_id 兜底取一次★（不是取最新帧）
                if job.fetcher is None:
                    with self._lock:
                        self.expired += 1
                    return None, "frame_expired"
                data = job.fetcher(int(job.frame_id))
                if not data:
                    with self._lock:
                        self.expired += 1
                    LOG.warning("[LMM] 证据帧已淘汰 alert=%s #%d frame=%s",
                                job.alert_id, job.index, job.frame_id)
                    return None, "frame_expired"

            d = Path(job.out_dir) if job.out_dir else None
            if d is None:
                return None, "no_out_dir"
            d.mkdir(parents=True, exist_ok=True)

            base = "%03d" % (job.index + 1)
            #: ── ① source.jpg：StreamHub q95 原始字节，原样落盘，一次都不重编码 ──
            src_name = f"{base}_source.jpg"
            (d / src_name).write_bytes(data)
            src_w, src_h = jpeg_wh(data)

            import numpy as np
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None, "decode_failed"
            #: ★两边都排除框所在区域★：否则"画框"本身的高频边缘会把糊图的
            #: 清晰度抬上去（实测能把 ratio 顶到 1.9，看着像处理链动了手脚）
            box = [float(v) for v in (job.bbox or [])] if job.bbox else None
            src_sharp = sharpness(frame, exclude_box=box)
            h, w = frame.shape[:2]

            #: ── ② 展示图：解码 → 画 VLX bbox → 编码（PNG 无损 / q98）──
            stage = "draw"
            shot = frame.copy()
            if job.bbox and len(job.bbox) >= 4:
                draw_boxes(shot, [{
                    "x1": float(job.bbox[0]), "y1": float(job.bbox[1]),
                    "x2": float(job.bbox[2]), "y2": float(job.bbox[3]),
                    "label": job.label, "label_zh": job.label_zh or job.label,
                    "display_name": job.display_name or job.label_zh or job.label,
                    "vlm_state": "confirmed",
                }], fill_alpha=0.10)
            stage = "encode"
            t_enc = time.perf_counter()
            out = encode_image_bytes(shot, EVIDENCE_IMAGE_FORMAT, EVIDENCE_JPEG_QUALITY)
            enc_ms = (time.perf_counter() - t_enc) * 1000.0
            with self._lock:
                self._encode_ms.append(enc_ms)
            if not out:
                return None, "encode_failed"
            ext = EVIDENCE_IMAGE_EXT
            name = f"{base}.{ext}"
            (d / name).write_bytes(out)

            #: ── ③ 校验：尺寸必须与 source 一致（被缩小 = 严重错误，直接记下来）──
            stage = "verify"
            chk = cv2.imdecode(np.frombuffer(out, dtype=np.uint8), cv2.IMREAD_COLOR)
            if chk is None:
                return None, "verify_failed"
            ch, cw = chk.shape[:2]
            if (cw, ch) != (src_w, src_h):
                LOG.error("[LMM] ★证据图尺寸被改变★ alert=%s #%d source=%dx%d out=%dx%d",
                          job.alert_id, job.index, src_w, src_h, cw, ch)
                return None, f"resized:{cw}x{ch}"
            ev_sharp = sharpness(chk, exclude_box=box)
            ratio = (ev_sharp / src_sharp) if src_sharp > 1e-6 else 1.0

            #: ── ④ 缩略图：只服务卡片 ──
            stage = "thumb"
            thumb_url = ""
            thumb = make_thumbnail_jpeg(out, max_side=THUMB_MAX_SIDE)
            if thumb:
                tname = f"{base}_thumb.jpg"
                (d / tname).write_bytes(thumb)
                thumb_url = f"/api/lmm/alerts/{job.task_id}/{job.alert_id}/{tname}"

            version = str(int(os.path.getmtime(str(d / name))))
            meta = {
                "index": int(job.index),
                "status": READY,
                "frame_id": int(job.frame_id),
                "pts_ms": job.pts_ms,
                "source_url": f"/api/lmm/alerts/{job.task_id}/{job.alert_id}/{src_name}",
                "original_url": f"/api/lmm/alerts/{job.task_id}/{job.alert_id}/{name}",
                "thumbnail_url": thumb_url,
                "bbox": [float(v) for v in (job.bbox or [])],
                "width": int(cw), "height": int(ch),
                "bytes": int(len(out)),
                "source_bytes": int(len(data)),
                "thumb_bytes": int(len(thumb)) if thumb else 0,
                "format": ext,
                "source_sharpness": round(src_sharp, 2),
                "sharpness": round(ev_sharp, 2),
                "quality_ratio": round(ratio, 3),
                "encode_ms": round(enc_ms, 1),
                "version": version,
            }
            LOG.info("[LMM] 证据诊断 alert=%s #%d frame=%s source=%dx%d/%dB "
                     "evidence=%dx%d/%dB(%s) thumb=%dB "
                     "source_sharpness=%.1f evidence_sharpness=%.1f ratio=%.3f encode=%.0fms",
                     job.alert_id, job.index, job.frame_id, src_w, src_h, len(data),
                     cw, ch, len(out), ext, len(thumb) if thumb else 0,
                     src_sharp, ev_sharp, ratio, enc_ms)
            if ratio < SHARPNESS_RATIO_WARN:
                LOG.warning("[LMM] ★证据清晰度下降★ alert=%s #%d ratio=%.3f "
                            "（source=%.1f → evidence=%.1f，怀疑处理链有损）",
                            job.alert_id, job.index, ratio, src_sharp, ev_sharp)
            return meta, None
        except Exception as exc:  # noqa: BLE001
            LOG.exception("[LMM] 证据处理失败 alert=%s #%d frame=%s stage=%s",
                          job.alert_id, job.index, job.frame_id, stage)
            return None, f"{stage}:{type(exc).__name__}: {exc}"


#: 进程内单例（两个 worker 线程够用：JPEG/PNG 编解码 + 磁盘 IO 本身吃 CPU，
#: 开太多反而会跟检测抢资源）
_WORKER: Optional[EvidenceWorker] = None
_WLOCK = threading.Lock()


def get_worker() -> EvidenceWorker:
    global _WORKER
    with _WLOCK:
        if _WORKER is None:
            _WORKER = EvidenceWorker()
            _WORKER.start()
        return _WORKER


def evidence_status_of(statuses: "List[str]") -> str:
    """三张的状态 → 告警级 `evidence_status`。"""
    if not statuses:
        return PENDING
    if all(s == READY for s in statuses):
        return READY
    if all(s == ERROR for s in statuses):
        return ERROR
    if any(s == READY for s in statuses):
        return "partial"
    if any(s in (PROCESSING, ERROR) for s in statuses):
        return PROCESSING
    return PENDING
