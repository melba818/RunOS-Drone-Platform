# -*- coding: utf-8 -*-
"""R1：Realtime Latest-Frame Fast Path —— 把「采集」和「AI 推理」拆成两条执行流。

为什么必须拆：VLX 单帧要约 1~4 秒，而采集是 25~30 FPS。同步循环里
`read → detect → publish` 串行执行，等一帧推理完，视频时间已经走掉几秒；
处理进度因此**越跑越落后**（R0 实测 45 秒落后 34 秒）。

拆成两条流之后：

    CaptureWorker   按媒体时间匀速读帧 → LatestFrameBuffer（容量 1）
                                     ↘ PreviewSampler → JPEG 预览（UI 用）
    InferenceWorker 只从 buffer 取最新帧 → VLX → detection metadata（overlay 用）

推理再慢也只影响"框多久更新一次"，**不影响画面**；而且因为 buffer 只有一帧，
下一次推理拿到的永远是"当前"画面，落后不会累积。

★本模块刻意不 import task_manager / RuleEngine★
    检测动作以 `infer_fn` 回调注入，这里只负责流水线与计时 —— 否则会形成
    循环导入，而且流水线逻辑也没法单独测试。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .frame_buffer import FramePacket, LatestFrameBuffer
from .timing import PerfStats

LOG = logging.getLogger(__name__)

__all__ = ["RealtimeOptions", "RealtimePipeline", "encode_preview"]


def encode_preview(frame: Any, max_side: int = 960, quality: int = 75,
                   encode_fn: Optional[Callable[..., Optional[str]]] = None) -> Optional[str]:
    """预览帧 → JPEG base64。

    `encode_fn` 可注入（`render.encode_jpeg_b64`），缺省时在本文件内做一次最小实现，
    保证这个模块单独 import 也能跑（单测不依赖渲染栈）。
    **预览参数与送给 VLX 的原图无关**：VLX 拿的始终是原始分辨率帧。
    """
    if frame is None:
        return None
    if encode_fn is not None:
        try:
            return encode_fn(frame, max_side=max_side, quality=quality)
        except Exception:  # noqa: BLE001
            LOG.debug("[RT] 注入的编码函数失败，回退内置实现", exc_info=True)
    try:
        import io

        import numpy as np  # noqa: F401
        from PIL import Image

        img = Image.fromarray(frame[:, :, ::-1])
        side = max(img.size)
        if side > max_side:
            scale = max_side / float(side)
            img = img.resize((max(1, int(img.size[0] * scale)),
                              max(1, int(img.size[1] * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        import base64

        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        LOG.debug("[RT] 预览帧编码失败", exc_info=True)
        return None


@dataclass
class RealtimeOptions:
    """实时模式的采集与模型调用节流参数。"""

    #: 预览帧率。VLX 只有 0.2~1 FPS，但画面必须持续动，所以预览独立定频。
    preview_fps: float = 8.0
    preview_quality: int = 75
    preview_max_side: int = 960
    #: 推理侧发现"没有新帧"时的等待间隔（秒）。太大会增加延迟，太小会空转。
    inference_idle_sleep: float = 0.02
    #: 在线视觉模型的最小调用间隔；等待时仍由单槽缓冲持续保留最新帧。
    inference_interval_s: float = 0.0
    #: 采集是否按媒体时间匀速推进。**本地文件必须 True**，否则后端会一路读到
    #: 文件尾，而浏览器还停在第 10 秒 —— detection 与画面对不上。
    realtime_pacing: bool = True
    #: 允许的最大"追帧"落后（秒）：pacing 落后超过它就跳过 sleep 尽快追上，
    #: 避免轻微抖动被无限放大成持续落后。
    pacing_catchup_s: float = 0.25
    #: ★直播流★允许的**最大连续读空次数**。超过才判定断流。
    #:
    #: 为什么必须有：RTSP 抖动/重缓冲时 `read()` 会偶发返回空，而采集循环原先
    #: 见 `None` 就 `break` —— 于是任务跑几秒就"正常结束"，UI 表现为画面冻住。
    #: 本地文件读到 None 才是真 EOF，必须立刻结束，所以只对流生效。
    #: 取值要 < StreamSource 内部的 `max_empty`（默认 60），否则会先被它抛
    #: SourceError，任务从"正常结束"变成"失败"。
    max_empty_frames: int = 30


class RealtimePipeline:
    """Capture / Inference 双线程流水线。"""

    def __init__(
        self,
        *,
        src: Any,
        fps: float,
        stop: threading.Event,
        infer_fn: Callable[[Any, int], Tuple[List[Dict[str, Any]], Dict[str, Any]]],
        on_preview: Optional[Callable[[int, float, Optional[str]], None]] = None,
        on_detection: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_finished: Optional[Callable[[str], None]] = None,
        options: Optional[RealtimeOptions] = None,
        capture_fn: Optional[Callable[[], Tuple[Optional[Any], int]]] = None,
        #: ★R4A★ 每一帧的原始回调（给 TrackerFrameCache 喂历史帧）。
        #: **只此一处改动**：本文件刻意不 import task_manager / tracking，
        #: 帧缓存必须在采集侧写（每帧一次、不能等推理），但不能把 tracker
        #: 语义（label 权威 / TargetManager）带进来，所以只留一个空回调。
        #: 回调必须**极轻量**（只是 resize + 入队），慢了会直接拖慢采集。
        on_frame: Optional[Callable[[FramePacket], None]] = None,
    ) -> None:
        self.src = src
        self.fps = float(fps or 0.0) or 30.0
        self.stop = stop
        self.infer_fn = infer_fn
        self.on_preview = on_preview
        self.on_detection = on_detection
        self.on_finished = on_finished
        self.on_frame = on_frame
        self.opts = options or RealtimeOptions()
        #: 可注入的读帧函数（默认 `src.read(1)`），单测用它喂假帧
        self.capture_fn = capture_fn

        self.buffer = LatestFrameBuffer()
        #: ★预览专用最新帧槽★（与推理用的 `self.buffer` 分开，各用一把锁）。
        #: 采集侧每次只做一次 `put_latest`：临界区就是"赋值 + 计数"，无 IO、
        #: 无等待，**不会阻塞采集线程**（详见 `_preview_loop` 的说明）。
        self._preview_buf = LatestFrameBuffer()
        self._lock = threading.RLock()
        self._dropped = 0
        #: 连续读空计数（只对直播流有意义）。拿到一帧就清零。
        self._empty = 0
        #: 源是不是直播流：`kind == "stream"`（StreamSource）。本地文件是 "file"。
        self._is_stream = getattr(src, "kind", "") == "stream"
        self._capture_frames = 0
        self._processed_frames = 0
        self._latest_frame_id = 0
        self._processed_frame_id = 0
        self._preview_count = 0
        self._t0 = 0.0
        self._started = False
        self._threads: List[threading.Thread] = []
        #: 采集结束（读到文件尾 / 断流 / 出错）。推理线程靠它决定**何时收工**：
        #: 否则文件读完后 inference 会永远停在"等新帧"上，任务结束不了。
        self._capture_done = threading.Event()

        self.perf: Dict[str, PerfStats] = {
            "capture": PerfStats(), "encode": PerfStats(), "transport": PerfStats(),
            "inference": PerfStats(), "parse": PerfStats(), "render": PerfStats(),
            "publish": PerfStats(), "total": PerfStats(), "frame_age": PerfStats(),
        }

    # ── 生命周期 ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._t0 = time.time()
        #: ★rt-preview★：JPEG 编码 + WS 组包独立成线程，采集线程不再被它拖慢。
        #: 三条线程都 daemon，退出条件一致（`stop` 或 `capture_done` 且无新帧）。
        self._threads = [
            threading.Thread(target=self._capture_loop, name="rt-capture", daemon=True),
            threading.Thread(target=self._inference_loop, name="rt-infer", daemon=True),
            threading.Thread(target=self._preview_loop, name="rt-preview", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def join(self, timeout: Optional[float] = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # ── Capture ───────────────────────────────────────────────────────────
    def _capture_loop(self) -> None:
        """持续读帧，按媒体时间匀速推进。

        ★这里**只做采集**★：读帧 → 写最新帧槽 → 喂 tracker 缓存 → pacing。
        JPEG 编码与 WS 推送已移到 `rt-preview` 线程（`_preview_loop`），
        采集线程不再承担任何 CPU 密集活 —— 否则采集跟不上源帧率，
        RTSP 缓冲堆帧 → 连续读空 → 任务被判断流自动退出。
        """
        idx = 0
        t_start = time.monotonic()
        #: 预览定频已移到 rt-preview 线程，这两个变量不再在采集侧使用
        #: （旧实现内联节流时才需要，保留注释以说明去向）。
        # last_preview = 0.0
        # preview_interval = 1.0 / max(0.1, float(self.opts.preview_fps))
        try:
            while not self.stop.is_set():
                t_read = time.monotonic()
                if self.capture_fn is not None:
                    frame, frame_id = self.capture_fn()
                else:
                    frame = self.src.read(1)
                    try:
                        frame_id = int(self.src.position)
                    except Exception:  # noqa: BLE001
                        frame_id = idx + 1
                if frame is None:
                    # 直播流的偶发读空（抖动 / 重缓冲）不能当成"流结束"：
                    # 原来这里直接 break，任务跑几秒就退，UI 看着像卡死。
                    # 本地文件读到 None 才是真 EOF，必须立刻结束。
                    self._empty += 1
                    if not self._is_stream:
                        break
                    if self._empty >= max(1, int(self.opts.max_empty_frames)):
                        LOG.warning("[RT] 直播流连续读空 %d 次，判定断流并结束采集",
                                    self._empty)
                        break
                    if self._empty == 1 or self._empty % 10 == 0:
                        LOG.info("[RT] 直播流读空 %d/%d 次，继续等待",
                                 self._empty, self.opts.max_empty_frames)
                    self.stop.wait(0.05)
                    continue
                self._empty = 0
                idx += 1
                capture_ms = (time.monotonic() - t_read) * 1000.0

                media_time_ms = (float(frame_id) / self.fps) * 1000.0
                pkt = FramePacket(frame_id=frame_id, media_time_ms=media_time_ms,
                                  capture_ts=time.time(), frame=frame)
                dropped = self.buffer.put_latest(pkt)
                with self._lock:
                    self._dropped += dropped
                    self._capture_frames += 1
                    self._latest_frame_id = frame_id
                self.perf["capture"].add(capture_ms)

                # ── ★R4A★ 历史帧缓存（tracker 的 replay 靠它）──
                #: 必须在采集侧写：VLX 结果回来时（2.3 s 后）要能找到它检测的那
                #: 一帧，等推理再写就晚了。异常一律吞掉 —— 缓存写失败最多是
                #: "这一轮跟踪用不上历史帧"，绝不能把采集线程带崩。
                if self.on_frame is not None:
                    try:
                        self.on_frame(pkt)
                    except Exception:  # noqa: BLE001
                        LOG.debug("[RT] on_frame 回调失败", exc_info=True)

                # ── 预览通道：采集侧只"投递最新帧"，编码交给 rt-preview ──
                #: ★旧实现（内联在采集线程里，已停用，保留备查）★
                #:   now = time.monotonic()
                #:   if self.on_preview is not None and (now - last_preview) >= preview_interval:
                #:       last_preview = now
                #:       t_enc = time.monotonic()
                #:       b64 = encode_preview(frame, self.opts.preview_max_side,
                #:                            self.opts.preview_quality)
                #:       with self._lock:
                #:           self._preview_count += 1
                #:       self.perf["render"].add((time.monotonic() - t_enc) * 1000.0)
                #:       try:
                #:           self.on_preview(frame_id, media_time_ms, b64)
                #:       except Exception:
                #:           LOG.debug("[RT] 预览回调失败", exc_info=True)
                #:
                #: 为什么要搬走：JPEG 编码（resize + 编码 + base64）一帧 10~20 ms，
                #: 内联在 30 FPS 的采集循环里 = 每帧白付 30%~60% 的时间预算；
                #: 采集一慢，RTSP 的 TCP 接收缓冲就堆帧，`read()` 开始连续返回空，
                #: 最后被 `max_empty_frames` 判成断流 → 任务自动退出。
                self._preview_buf.put_latest(pkt)

                # ── realtime pacing：不要让后端跑到浏览器前面去 ──
                if self.opts.realtime_pacing and self.fps > 0:
                    target = t_start + (float(frame_id) / self.fps)
                    remaining = target - time.monotonic()
                    if remaining > 0:
                        # 用 Event.wait 而不是 sleep：stop 一到立刻醒来
                        self.stop.wait(remaining)
                    elif remaining < -self.opts.pacing_catchup_s:
                        # 落后太多（解码跟不上）→ 重新对齐时间基准，
                        # 否则每一帧都"欠债"，pacing 会退化成空转。
                        t_start = time.monotonic() - (float(frame_id) / self.fps)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("[RT] capture 线程异常")
            if self.on_finished:
                try:
                    self.on_finished("capture-error:%s" % exc)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._capture_done.set()
            if self.on_finished:
                try:
                    self.on_finished("capture-end")
                except Exception:  # noqa: BLE001
                    pass

    # ── Preview ───────────────────────────────────────────────────────────
    def _preview_loop(self) -> None:
        """★预览专用线程★：JPEG 编码 + WS 组包（原先内联在采集线程里）。

        搬出来的原因：编码一帧 10~20 ms，内联在 30 FPS 采集循环里等于每帧
        白付 30%~60% 的预算，采集被拖慢到 20 FPS 以下 → RTSP 的 TCP 接收缓冲
        堆帧 → `read()` 连续返回空 → 被 `max_empty_frames` 判成断流 → 任务自退。
        独立成线程后，编码再慢也只影响"预览多久更新一次"，**绝不影响采集**。

        节流仍按 `preview_fps` 定频：采集每帧写、这里按频率读 → 天然
        Latest Frame Wins（槽位容量恒为 1），预览慢就丢帧，绝不堆积。

        ★三条硬约束★
          1. `daemon=True`（在 `start()` 里设置），主进程退出不留悬挂线程；
          2. 退出条件双保险：`stop` 事件 / `capture_done` 且无新帧 —— 与
             `rt-infer` 完全一致，任务结束时能收干净；
          3. **任何异常都吞在自己线程里**，绝不让编码/回调失败波及 capture。
        """
        last_id = -1
        interval = 1.0 / max(0.1, float(self.opts.preview_fps))
        next_at = time.monotonic()
        try:
            while not self.stop.is_set():
                # ── 定频：到点才编码，别把 CPU 吃满（编码是纯 CPU 活）──
                now = time.monotonic()
                if now < next_at:
                    self.stop.wait(next_at - now)
                    if self.stop.is_set():
                        return
                next_at = max(time.monotonic(), next_at) + interval

                pkt = self._preview_buf.take()
                if pkt is None or pkt.frame_id == last_id:
                    # 没有新帧 → 采集已结束就可以收工；否则短暂等待再试，
                    # 绝不重复编码同一帧（那是纯浪费 CPU）。
                    if self._capture_done.is_set():
                        return
                    self.stop.wait(min(interval, 0.02))
                    continue
                last_id = pkt.frame_id

                t_enc = time.monotonic()
                b64 = encode_preview(pkt.frame, self.opts.preview_max_side,
                                     self.opts.preview_quality)
                with self._lock:
                    self._preview_count += 1
                self.perf["render"].add((time.monotonic() - t_enc) * 1000.0)
                if self.on_preview is not None:
                    try:
                        #: ★接口签名与旧实现逐字一致★
                        #: `on_preview(frame_id, media_time_ms, b64)`，
                        #: task_manager 侧不需要任何改动。
                        self.on_preview(pkt.frame_id, pkt.media_time_ms, b64)
                    except Exception:  # noqa: BLE001
                        LOG.debug("[RT] 预览回调失败", exc_info=True)
        except Exception:  # noqa: BLE001
            #: 预览是"锦上添花"的通道：它挂了最多是画面不动，
            #: 绝不能把整个流水线带崩，也绝不能污染 capture / infer。
            LOG.exception("[RT] preview 线程异常")
        finally:
            LOG.debug("[RT] preview 线程结束（frame_id=%s）", last_id)

    # ── Inference ─────────────────────────────────────────────────────────
    def _inference_loop(self) -> None:
        """只处理最新帧；慢模型丢弃过期帧，并按需节流远程请求。"""
        last_id = -1
        next_inference_at = 0.0
        while not self.stop.is_set():
            pkt = self.buffer.take()
            if pkt is None or pkt.frame_id == last_id:
                # 采集已经结束且没有新帧可拿 → 收工（文件读完 / 断流）
                if self._capture_done.is_set():
                    return
                # 没有新帧 → 短暂等待；不要忙等，也不要在这里重复推理同一帧
                self.stop.wait(self.opts.inference_idle_sleep)
                continue
            remaining = next_inference_at - time.monotonic()
            if remaining > 0:
                self.stop.wait(remaining)
                if self.stop.is_set():
                    return
                # 等待期间采集线程持续覆盖单槽，只推理 interval 到期时的最新画面。
                pkt = self.buffer.take() or pkt
                if pkt.frame_id == last_id:
                    continue
            last_id = pkt.frame_id
            interval_s = max(0.0, float(self.opts.inference_interval_s or 0.0))
            if interval_s:
                next_inference_at = time.monotonic() + interval_s
            started = time.time()
            try:
                detections, timing = self.infer_fn(pkt.frame, pkt.frame_id)
            except Exception as exc:  # noqa: BLE001
                # ★推理失败绝不能把 capture 线程一起带走★
                LOG.exception("[RT] 推理失败（frame_id=%s）", pkt.frame_id)
                with self._lock:
                    pass
                self.stop.wait(0.2)
                continue
            finished = time.time()

            tm = dict(timing or {})
            for k in ("capture", "encode", "transport", "inference", "parse", "render", "publish"):
                if tm.get(k + "_ms") is not None:
                    self.perf[k].add(tm.get(k + "_ms"))

            finished_ms = (finished - started) * 1000.0
            total_ms = tm.get("total_ms")
            if total_ms is None:
                total_ms = finished_ms
            age_ms = (finished - pkt.capture_ts) * 1000.0
            self.perf["total"].add(total_ms)
            self.perf["frame_age"].add(age_ms)

            with self._lock:
                self._processed_frames += 1
                self._processed_frame_id = pkt.frame_id

            if self.on_detection is not None:
                payload = {
                    "frame_id": pkt.frame_id,
                    "media_time_ms": round(pkt.media_time_ms, 1),
                    "inference_started_at": round(started, 3),
                    "inference_finished_at": round(finished, 3),
                    "detections": detections or [],
                    "timing": {
                        **tm,
                        "total_ms": round(float(total_ms or 0.0), 1),
                        "frame_age_ms": round(age_ms, 1),
                        "capture_timestamp": round(pkt.capture_ts, 3),
                    },
                }
                try:
                    self.on_detection(payload)
                except Exception:  # noqa: BLE001
                    LOG.debug("[RT] detection 回调失败", exc_info=True)

    # ── 观测 ──────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            dropped = self._dropped
            capture_frames = self._capture_frames
            processed = self._processed_frames
            latest = self._latest_frame_id
            processed_id = self._processed_frame_id
            preview_count = self._preview_count
        elapsed = (time.time() - self._t0) if self._t0 else 0.0
        age = self.perf["frame_age"].summary()
        inf = self.perf["inference"].summary()
        media_pos_ms = (float(processed_id) / self.fps) * 1000.0 if self.fps else 0.0
        return {
            "mode": "realtime_latest",
            "capture_fps": round(capture_frames / elapsed, 2) if elapsed > 0 else None,
            "preview_fps": round(preview_count / elapsed, 2) if elapsed > 0 else None,
            "ai_fps": round(processed / elapsed, 3) if elapsed > 0 else None,
            "last_inference_ms": self.perf["inference"].last,
            "average_inference_ms": inf["avg"],
            "frame_age_ms": self.perf["frame_age"].last,
            "average_frame_age_ms": age["avg"],
            "frame_age_p50_ms": age["p50"],
            "frame_age_p95_ms": age["p95"],
            "frame_age_max_ms": age["max"],
            "media_position_ms": round(media_pos_ms, 1),
            "wall_elapsed_ms": round(elapsed * 1000.0, 1),
            "drift_ms": round(elapsed * 1000.0 - media_pos_ms, 1),
            "dropped_frames": int(dropped),
            "processed_frames": int(processed),
            "latest_frame_id": int(latest),
            "processed_frame_id": int(processed_id),
            "queue_depth": self.buffer.queue_depth,
            "segments": {k: self.perf[k].summary() for k in
                         ("capture", "encode", "transport", "inference", "parse",
                          "render", "publish", "total", "frame_age")},
        }
