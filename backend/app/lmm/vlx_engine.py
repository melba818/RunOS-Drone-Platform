# -*- coding: utf-8 -*-
"""VLX-Seek 推理适配层。

改造后「多模态目标检测」不再经过 YOLOE + Qwen-VL 复核，而是把整帧交给云端
VLX-Seek（WeDetect 自己生成 proposal，再由 reasoning_detection 决定选哪些 region）。

本层只做三件事：
1. 把 BGR 帧编码成 JPEG 发给 ``VLXSeekClient``；
2. 把返回的真实 region bbox 转成项目通用的 ``DetectedBox``（**像素 xyxy**）；
3. 补上前端需要的字段（`detector_confidence` / `vlm_state` / `display_name` …），
   字段形状与 ``RuleEngine.decorate()`` 保持一致，**前端画布零改动**。

重要约定：**绝不伪造 bbox**。VLX-Seek 没选中任何 region 就返回空列表，前端不画框。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2

from lmm.detectors.base_detector import DetectedBox, FrameResult
from lmm.vlx_client import VLXError, VLXSeekClient

LOG = logging.getLogger(__name__)


class VLXEngineError(RuntimeError):
    """VLX-Seek 链路不可用。"""


class VLXEngine:
    """与 ``BaseDetector`` 同形的 VLX-Seek 适配器（不依赖 ultralytics）。"""

    name = "vlx-seek"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.client = VLXSeekClient(config)
        cfg = self.client.cfg
        self.prompt = ""
        # **默认值一律取自 lmm_config.json**（vlx_seek_lang / vlx_seek_task /
        # vlx_seek_max_new_tokens）。不要在这里硬编码 reasoning_detection/1024 ——
        # 单图接口（/api/lmm/vlx/infer-image）用的是配置里的 detection/256，
        # 两条路径默认值不一致时，同一句 Prompt 会给出不同结果，极难排查。
        self.lang = cfg.get("lang") or "zh"
        self.task = cfg.get("task") or "detection"
        self.debug = False
        self.max_new_tokens = int(cfg.get("max_new_tokens") or 256)
        #: 回填给检测框的**归组标签**（写进 DetectedBox.label）。由调用方用用户原文
        #: 覆盖，缺省才退回模型原文。它只影响规则引擎的空间伪轨迹归组，不用于展示。
        self.label = ""
        #: 展示标签的后备值。留空表示"优先用 VLX 自己在 <ground> 里返回的中文档案"。
        self.label_zh = ""
        self.jpeg_quality = 92
        self._lock = threading.RLock()
        self._stats: Dict[str, Any] = {
            "requests": 0, "errors": 0, "last_latency_ms": None,
            "avg_latency_ms": None, "last_proposal_count": 0,
            "last_selected_count": 0, "total_selected": 0,
            "last_proposal_latency_ms": None, "last_vlx_latency_ms": None,
            "last_error": "", "endpoint": self._endpoint_label(),
        }
        self._lat_sum = 0.0
        self._lat_n = 0

    # ── 配置 ──────────────────────────────────────────────────────────────
    def _endpoint_label(self) -> str:
        c = self.client.cfg
        if (c.get("transport") or "").lower() == "direct":
            return c.get("base_url") or ""
        return f"ssh:{c.get('node_id') or '3060'} → 127.0.0.1:{c.get('port')}"

    def set_prompt(self, prompt: str, lang: Optional[str] = None,
                   task: Optional[str] = None, debug: bool = False,
                   max_new_tokens: Optional[int] = None,
                   label: Optional[str] = None,
                   label_zh: Optional[str] = None) -> None:
        """入参为 None 时**保留原有默认值**，不退回硬编码常量。

        `prompt` 必须是**用户原文**：这里不存在关键词抽取 / 中英转换 / 外部 LLM 调用，
        客户端只会按已保存配置追加附加提示词，不做关键词抽取或翻译。
        """
        with self._lock:
            self.prompt = prompt
            if lang:
                self.lang = lang
            if task:
                self.task = task
            self.debug = debug
            if max_new_tokens:
                self.max_new_tokens = int(max_new_tokens)
            if label is not None:
                self.label = str(label)
            if label_zh is not None:
                self.label_zh = str(label_zh)
            # 每个任务只 set_prompt 一次 → 正好在这里打印一次实际下发的三要素。
            # 验收时用它核对服务端收到的 query 有没有被拆成英文关键词。
            LOG.info("[VLX] task: %s", self.task)
            LOG.info("[VLX] lang: %s", self.lang)
            LOG.info("[VLX] raw query: %s", self.prompt)

    # ── 状态 ──────────────────────────────────────────────────────────────
    def health(self, timeout_note: str = "") -> Dict[str, Any]:
        t0 = time.time()
        try:
            info = self.client.health()
            model_info = {}
            try:
                model_info = self.client.model_info()
            except VLXError:
                model_info = {}
            return {"ok": True, "health": info, "model_info": model_info,
                    "probe_ms": round((time.time() - t0) * 1000, 1),
                    "endpoint": self._endpoint_label(), "note": timeout_note}
        except VLXError as exc:
            return {"ok": False, "error": str(exc), "detail": exc.detail,
                    "probe_ms": round((time.time() - t0) * 1000, 1),
                    "endpoint": self._endpoint_label()}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            out = dict(self._stats)
        out["avg_latency_ms"] = round(self._lat_sum / self._lat_n, 1) if self._lat_n else None
        return out

    def describe(self) -> Dict[str, Any]:
        """给前端「引擎状态」卡片用。"""
        return {
            "engine": "vlx-seek",
            "model": "VLX-Seek-1.5-10B",
            "task": self.task,
            "lang": self.lang,
            "prompt": self.prompt,
            "endpoint": self._endpoint_label(),
            "debug": self.debug,
            **self.stats(),
        }

    # ── 推理 ──────────────────────────────────────────────────────────────
    @staticmethod
    def encode(frame_bgr, quality: int = 92) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise VLXEngineError("画面编码失败")
        return buf.tobytes()

    def detect(self, frame_bgr, frame_index: int = 0,
               request_id: Optional[str] = None) -> FrameResult:
        #: ★R2A★ 适配器整单墙钟的起点：从**进入 detect** 算起，含 JPEG 编码。
        t_adapter = time.time()
        h, w = frame_bgr.shape[:2]
        #: ★R2A★ request_id —— 一次 VLX 请求的全局追踪 ID，形如
        #: `vlx-<task_id>-<frame_id>-<uuid8>`。调用方（task_manager）带 task_id /
        #: frame_id 进来；没有就自己造一个，保证任何路径发出的请求都可追踪。
        #: 刻意**不用 token 关联**（token 是凭据，进日志等于泄漏）。
        rid = request_id or f"vlx-adhoc-{frame_index}-{uuid.uuid4().hex[:8]}"
        # ★R0★ JPEG 编码**必须单独计时**：它发生在"发出请求"之前，过去整个被算进
        # 「推理耗时」里，于是 4.4 秒到底多少是模型、多少是编码永远说不清。
        t_enc = time.time()
        payload = self.encode(frame_bgr, self.jpeg_quality)
        encode_ms = (time.time() - t_enc) * 1000.0
        with self._lock:
            prompt = self.prompt
            lang = self.lang
            task = self.task
            debug = self.debug
            mnt = self.max_new_tokens
            label_override = self.label
            label_zh_override = self.label_zh
        if not prompt:
            raise VLXEngineError("未设置目标描述")

        t0 = time.time()
        cli_timing: Dict[str, Any] = {}
        try:
            res = self.client.infer_image(payload, prompt, task=task, lang=lang,
                                          debug=debug, max_new_tokens=mnt,
                                          timing=cli_timing, request_id=rid)
        except VLXError as exc:
            with self._lock:
                self._stats["errors"] += 1
                self._stats["last_error"] = str(exc)
            raise VLXEngineError(str(exc)) from exc
        elapsed = (time.time() - t0) * 1000.0

        # ── R2A：三个口径全部实测 ──────────────────────────────────────────────
        # R0 时服务端只有 `latency_ms`（模型侧自报），于是只能做减法：
        #   transport = 端到端 − 模型侧
        # 这个差值里混着**服务端的收包/拆包/解码/序列化**，被当成"网络"看，结论
        # 必然跑偏（"外围 1.35 秒"就是这么来的，且说不清是哪一段）。
        #
        # R2A 服务端新增 `server_total_ms`（整单墙钟）后，三个口径互不重叠：
        #   model_ms          = 服务端自报 latency_ms（proposal + vlx 计算）
        #   server_total_ms   = 服务端整单墙钟（handler 进入 → response 准备完成）
        #   wire_overhead_ms  = client_total − server_total
        #                     ★真正的"线路"开销★（SSH 隧道 + HTTP + 在途字节）
        #   non_model_total_ms= client_total − model_ms
        #                     除模型计算之外的一切（原 `transport_ms` 改名 —— 它
        #                     包含服务端前后处理，叫"传输"会误导人去查网络）
        #   client_total_ms   = 客户端从发请求到解析完 JSON
        #   adapter_total_ms  = 含 JPEG 编码的适配器整单（前端"一次检测"的真实体感）
        try:
            server_total = (float(res.get("server_total_ms"))
                            if res.get("server_total_ms") is not None else None)
        except (TypeError, ValueError):
            server_total = None
        try:
            model_ms = float(res.get("latency_ms")) if res.get("latency_ms") is not None else None
        except (TypeError, ValueError):
            model_ms = None
        # 旧版服务端没有 server_total_ms 时 wire_overhead **留 None**，不拿旧口径
        # 的差值冒充 —— 两种口径混在同一列里，P95 就是假数据。
        wire_overhead_ms = (max(0.0, elapsed - server_total)
                            if server_total is not None else None)
        non_model_total_ms = (max(0.0, elapsed - model_ms) if model_ms is not None else None)
        #: 旧名保留：`timing.py / task_manager` 的 R0/R1 消费者仍在读 `transport_ms`。
        transport_ms = non_model_total_ms

        boxes: List[DetectedBox] = []
        for det in res.get("detections") or []:
            bb = det.get("bbox_xyxy") or []
            if len(bb) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bb)
            if x2 <= x1 or y2 <= y1:
                continue
            # 归组标签（跨帧必须稳定）与展示标签（VLX 自己的措辞）分开：
            # 前者写进 bbox.label，规则引擎用它算空间伪轨迹键，一帧一变会导致
            # 连续命中永远攒不起来；后者直接显示给用户，就是"VLX-Seek 最终返回的信息"。
            stable_label = label_override or prompt
            model_label = str(det.get("label") or "").strip()
            show_label = model_label or label_zh_override or prompt
            boxes.append(DetectedBox(
                x1=max(0.0, min(x1, w)), y1=max(0.0, min(y1, h)),
                x2=max(0.0, min(x2, w)), y2=max(0.0, min(y2, h)),
                # VLX-Seek 不输出置信度 → 按项目约定填 1.0，绝不伪造分数
                confidence=1.0, confidence_source="synthetic",
                label=stable_label,
                label_zh=show_label,
                track_id=None,
            ))

        with self._lock:
            self._stats["requests"] += 1
            self._stats["last_latency_ms"] = round(elapsed, 1)
            self._stats["last_proposal_count"] = int(res.get("proposal_count") or 0)
            self._stats["last_selected_count"] = len(boxes)
            self._stats["total_selected"] += len(boxes)
            self._stats["last_proposal_latency_ms"] = res.get("proposal_latency_ms")
            self._stats["last_vlx_latency_ms"] = res.get("vlx_latency_ms")
            #: ★R2A★ 引擎状态里也留一份追踪 ID 与三个口径，前端 debug 行可直接读，
            #: 不用等下一帧的 timing 才能看到"刚才那次请求到底慢在哪"。
            self._stats["last_request_id"] = res.get("request_id") or rid
            self._stats["last_server_total_ms"] = res.get("server_total_ms")
            self._stats["last_wire_overhead_ms"] = (round(wire_overhead_ms, 1)
                                                    if wire_overhead_ms is not None else None)
            self._stats["last_queue_wait_ms"] = (res.get("timing") or {}).get("queue_wait_ms")
            self._stats["last_error"] = ""
            self._lat_sum += elapsed
            self._lat_n += 1

        result = FrameResult(frame_index=frame_index, width=w, height=h,
                            boxes=boxes, elapsed_ms=elapsed)
        #: ★R0★ 细分耗时（encode / transport / inference / parse …），上层直接读它做统计。
        #: 与 `result.vlx` 分开是因为这份是**稳定的性能契约字段**，不该和服务端的
        #: debug 字段混在一起（服务端加字段不会动到它）。
        def _ms(key: str) -> Optional[float]:
            v = cli_timing.get(key)
            return round(float(v), 3) if v is not None else None

        #: ★R2A★ 客户端细分。命名按"实际发生的动作"排，能直接读成一条时间线：
        #:   编码 → 组包 → 取隧道 → 开 channel → 建连接 → 写请求 → 等首字节
        #:   → 读响应 → 解码 → 解析
        result.timing = {                               # type: ignore[attr-defined]
            "request_id": res.get("request_id") or rid,
            "encode_ms": round(encode_ms, 1),
            "client_prepare_ms": _ms("multipart_ms"),
            "tunnel_acquire_ms": _ms("tunnel_acquire_ms"),
            "ssh_channel_open_ms": _ms("ssh_channel_open_ms"),
            "http_connection_create_ms": _ms("http_connection_create_ms"),
            "request_write_ms": _ms("request_write_ms"),
            "wait_first_byte_ms": _ms("wait_first_byte_ms"),
            "response_read_ms": _ms("response_read_ms"),
            "bytes_decode_ms": _ms("bytes_decode_ms"),
            "json_parse_ms": _ms("json_parse_ms"),
            "client_total_ms": round(elapsed, 1),
            "adapter_total_ms": round((time.time() - t_adapter) * 1000.0, 1),
            # 服务端实测（None = 服务端没报，绝不拿客户端的减法冒充）
            "server_total_ms": round(server_total, 1) if server_total is not None else None,
            "model_ms": round(model_ms, 1) if model_ms is not None else None,
            # 服务端细分的两个关键项也透出到 timing：排队（GPU 串行化程度）与
            # synchronize 后的真实模型墙钟。它们决定"慢是慢在算还是慢在等"。
            "queue_wait_ms": (res.get("timing") or {}).get("queue_wait_ms"),
            "model_wall_ms": (res.get("timing") or {}).get("model_wall_ms"),
            "wire_overhead_ms": (round(wire_overhead_ms, 1)
                                 if wire_overhead_ms is not None else None),
            "non_model_total_ms": (round(non_model_total_ms, 1)
                                   if non_model_total_ms is not None else None),
            # 复用标记（不是毫秒，但是"这段耗时为什么忽大忽小"的解释变量）
            "tunnel_reused": cli_timing.get("tunnel_reused"),
            "channel_reused": cli_timing.get("channel_reused"),
            "http_keepalive": cli_timing.get("http_keepalive"),
            # ── 兼容 R0/R1 的旧键：字段名与语义都不变，消费者无需改动 ──
            "transport_ms": round(transport_ms, 1) if transport_ms is not None else None,
            "inference_ms": round(model_ms, 1) if model_ms is not None else None,
            "total_ms": round(elapsed, 1),
            "parse_ms": round(float(cli_timing.get("parse_ms") or 0.0), 3),
            "decode_ms": round(float(cli_timing.get("decode_ms") or 0.0), 3),
            "io_ms": round(float(cli_timing.get("io_ms") or 0.0), 3),
            "multipart_ms": round(float(cli_timing.get("multipart_ms") or 0.0), 3),
        }
        # 额外元数据挂在对象上（不污染 FrameResult 定义，其他调用方不受影响）
        result.vlx = {                                  # type: ignore[attr-defined]
            "request_id": res.get("request_id") or rid,
            "proposal_count": res.get("proposal_count"),
            "selected_region_count": res.get("selected_region_count"),
            "proposal_latency_ms": res.get("proposal_latency_ms"),
            "vlx_latency_ms": res.get("vlx_latency_ms"),
            "latency_ms": res.get("latency_ms"),
            "raw_output": res.get("raw_output"),
            "vram_allocated_mb": res.get("vram_allocated_mb"),
            "debug_images": res.get("debug_images"),
            "image_width": res.get("image_width"),
            "image_height": res.get("image_height"),
            "server_total_ms": res.get("server_total_ms"),
            # ★R2A★ 服务端细分（request_receive / body_read / multipart_parse /
            # jpeg_decode / queue_wait / proposal / vlx_model / postprocess / …）
            "server_timing": res.get("timing"),
            # ★R2A★ 尺寸：判断"是不是图太大"的唯一硬依据（q92 / 1920x1080 下
            # 一帧 JPEG 多大、请求体多大、响应体多大）
            "sizes": {
                "source_width": int(w),
                "source_height": int(h),
                "jpeg_quality": int(self.jpeg_quality),
                "jpeg_bytes": len(payload),
                "request_bytes": cli_timing.get("request_bytes") or len(payload),
                "response_bytes": cli_timing.get("response_bytes"),
                "server_sizes": res.get("sizes"),
            },
            # R0：与 result.timing 同源，这里一起给前端 debug 行用
            "encode_ms": round(encode_ms, 1),
            "transport_ms": round(transport_ms, 1) if transport_ms is not None else None,
        }
        LOG.debug("[VLX] rid=%s client=%.0fms server=%.0fms model=%.0fms wire=%.0fms "
                  "queue=%s wait1st=%.0fms jpeg=%dB",
                  rid, elapsed, server_total or -1, model_ms or -1,
                  wire_overhead_ms if wire_overhead_ms is not None else -1,
                  (res.get("timing") or {}).get("queue_wait_ms"),
                  float(cli_timing.get("wait_first_byte_ms") or 0.0), len(payload))
        return result

    def detect_at(self, frame_id: int, media_time_ms: Optional[float] = None,
                  request_id: Optional[str] = None,
                  pts_ms: Optional[float] = None) -> FrameResult:
        """★服务器侧共享帧★：只发 `frame_id`，**不上传图片**。

        VLX 服务端自己向 StreamHub 要这一帧。与 `detect()` 的三点差别：

          1. 没有 `frame_bgr`，因此**没有 JPEG 编码**（`encode_ms = 0`）；
          2. 宽高来自服务端回带（本机没有像素，量不出来）；
          3. 结果里带 `frame_id` / `media_time_ms` —— **这就是与 OSTrack 对齐
             的那一帧**，两者都来自同一个 StreamHub。

        其余（标签归组、置信度约定、`timing` 口径）与 `detect()` 完全一致，
        因此上层 `on_detection` 的 WS 消息结构一行都不用改。
        """
        t_adapter = time.time()
        with self._lock:
            prompt = self.prompt
            lang = self.lang
            task = self.task
            debug = self.debug
            mnt = self.max_new_tokens
            label_override = self.label
            label_zh_override = self.label_zh
        if not prompt:
            raise VLXEngineError("未设置目标描述")

        rid = request_id or f"vlx-at-{frame_id}-{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        cli_timing: Dict[str, Any] = {}
        try:
            res = self.client.infer_at(int(frame_id), prompt,
                                       media_time_ms=media_time_ms,
                                       pts_ms=pts_ms,
                                       task=task, lang=lang,
                                       max_new_tokens=mnt,
                                       timing=cli_timing, request_id=rid)
        except VLXError as exc:
            with self._lock:
                self._stats["errors"] += 1
                self._stats["last_error"] = str(exc)
            raise VLXEngineError(str(exc)) from exc
        elapsed = (time.time() - t0) * 1000.0

        #: 本机没有像素 → 尺寸只能信服务端。它没报就留 0（上层有兜底配置值），
        #: 绝不拿 1920×1080 之类的猜测值冒充 —— 猜错会让前端整片框错位。
        w = int(res.get("image_width") or 0)
        h = int(res.get("image_height") or 0)

        boxes: List[DetectedBox] = []
        for det in res.get("detections") or []:
            bb = det.get("bbox_xyxy") or []
            if len(bb) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bb)
            if x2 <= x1 or y2 <= y1:
                continue
            stable_label = label_override or prompt
            model_label = str(det.get("label") or "").strip()
            show_label = model_label or label_zh_override or prompt
            if w > 0 and h > 0:
                x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
                y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            boxes.append(DetectedBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                # VLX-Seek 不输出置信度 → 按项目约定填 1.0，绝不伪造分数
                confidence=1.0, confidence_source="synthetic",
                label=stable_label, label_zh=show_label, track_id=None))

        with self._lock:
            self._stats["requests"] += 1
            self._stats["last_latency_ms"] = round(elapsed, 1)
            self._stats["last_proposal_count"] = int(res.get("proposal_count") or 0)
            self._stats["last_selected_count"] = len(boxes)
            self._stats["total_selected"] += len(boxes)
            self._stats["last_request_id"] = res.get("request_id") or rid
            self._stats["last_error"] = ""
            self._lat_sum += elapsed
            self._lat_n += 1

        result = FrameResult(frame_index=int(frame_id), width=w, height=h,
                             boxes=boxes, elapsed_ms=elapsed)
        #: ★与 OSTrack 对齐的锚点★：服务端实际用的那一帧（可能比请求的晚一点）
        result.frame_id = int(res.get("frame_id") or frame_id)      # type: ignore[attr-defined]
        result.media_time_ms = float(res.get("media_time_ms") or 0.0)  # type: ignore[attr-defined]
        #: ★真实 PTS★：服务端若没回带，就用请求时带上去的那个；都没有则 None。
        _pts = res.get("pts_ms")
        result.pts_ms = (                                    # type: ignore[attr-defined]
            None if (_pts is None and pts_ms is None)
            else float(_pts if _pts is not None else pts_ms))
        result.timing = {                                    # type: ignore[attr-defined]
            "request_id": res.get("request_id") or rid,
            "encode_ms": 0.0,
            "client_total_ms": round(elapsed, 1),
            "adapter_total_ms": round((time.time() - t_adapter) * 1000.0, 1),
            "server_total_ms": res.get("server_total_ms"),
            "model_ms": res.get("latency_ms"),
            "hub_fetch_ms": (res.get("timing") or {}).get("hub_fetch_ms"),
            "request_bytes": cli_timing.get("request_bytes"),
            "response_bytes": cli_timing.get("response_bytes"),
            "total_ms": round(elapsed, 1),
            "inference_ms": res.get("latency_ms"),
        }
        result.vlx = {                                       # type: ignore[attr-defined]
            "request_id": res.get("request_id") or rid,
            "frame_id": result.frame_id,                     # type: ignore[attr-defined]
            "media_time_ms": result.media_time_ms,           # type: ignore[attr-defined]
            "proposal_count": res.get("proposal_count"),
            "selected_region_count": res.get("selected_region_count"),
            "latency_ms": res.get("latency_ms"),
            "raw_output": res.get("raw_output"),
            "image_width": w, "image_height": h,
            "server_total_ms": res.get("server_total_ms"),
            "server_timing": res.get("timing"),
        }
        LOG.debug("[VLX] rid=%s infer_at frame=%s client=%.0fms dets=%d req=%sB",
                  rid, result.frame_id, elapsed, len(boxes),       # type: ignore[attr-defined]
                  cli_timing.get("request_bytes"))
        return result

    # ── 前端字段（与 RuleEngine.decorate 同形）───────────────────────────
    def decorate(self, boxes: List[DetectedBox], frame: Any = None,
                 display_name: str = "") -> List[Dict[str, Any]]:
        name = display_name or self.prompt
        out: List[Dict[str, Any]] = []
        for b in boxes:
            d = b.to_dict()
            d["detector_confidence"] = d.pop("confidence", 1.0)
            d["display_name"] = name or d.get("label_zh") or b.label
            d["track_key"] = ""
            # VLX-Seek 本身就是语义判定，没有"待复核"这一态：选中即检出。
            d["vlm_state"] = "confirmed"
            d["vlm_confidence"] = None
            d["vlm_reason"] = f"检测器直接检出（{self.task}）"
            out.append(d)
        return out

    def release(self) -> None:
        """客户端不持有本地权重，无需释放显存。"""
        return None
