"""LMM 推理任务管理。

生命周期：

    created → ready → running → stopped / failed
                        ↑
                    stopping（用户点停止）

与项目其它长任务保持一致的实现方式：**守护线程 + 模块级 dict**，不用 asyncio、不用进程池。
停止靠 `threading.Event`，而不是杀线程（杀线程会让显存无法释放）。

显存：任务结束（正常/异常/停止）都会走 `_cleanup()` —— 清 tracker 状态 + `empty_cache()`。
2GB 显存的机器不释放就开第二个任务会直接 OOM（项目既有教训）。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import config as lmm_config
from . import hub, registry
from . import video_index as vindex
from .detectors import get_detector, unload_all
from .detectors.base_detector import DetectorError
from .events import DetectionEvent, EventStore
#: `encode_jpeg_b64` 必须一起 import 进来：曾经只导入了 draw_boxes/output_dir，
#: 而 `_encode_frame` 里写的是 `render.encode_jpeg_b64(...)` → 运行时 NameError，
#: 且只炸 legacy 路径（实时模式用自己的编码函数），很容易被漏掉。
from .evidence import (ERROR, PENDING, PROCESSING, READY, EvidenceJob,
                       evidence_status_of, get_worker)
from .render import (EVIDENCE_JPEG_QUALITY, draw_boxes, encode_jpeg_b64,
                     encode_jpeg_bytes, output_dir)
from .rule_engine import RuleEngine, RuleEngineConfig
from .realtime import RealtimeOptions, RealtimePipeline
#: ★技术路线切换（2026-09-25）★：跟踪层**已整条下线**。
#:
#: - OSTrack / RemoteOSTrackBackend / TargetManager / TrackerWorker /
#:   AsyncTrackerRuntime 的**文件一行没删**（`backend/lmm/tracking/` 整体保留），
#:   将来要重新接入只需把 `_setup_tracking` 的调用加回来；
#: - 但本模块**不再 import 它们**，也不再有任何运行时调用 —— import 它们会连带
#:   拉起 `_vendor/timm` 等一大坨依赖，纯粹是不必要的启动成本；
#: - 唯一检测链路 = StreamHub → VLX → DetectionManager → AlertEvent。
#:
#: ★保留 `StreamHubClient`★：它是**取帧时间线**的客户端，不是 OSTrack 的东西
#: （VLX 的 `/infer_at` 与告警证据取图都要用它），StreamHub 链路保持不变。
from .tracking.streamhub_client import StreamHubClient
#: ★新的检测链路★：VLX 直连 + 三帧确认告警（无任何跟踪器）
from .detection_manager import DetectConfig, DetectionManager
from .sources import SourceError, open_source
from .target_spec import TargetSpec
from .timing import FrameTiming, PerfStats
from .qwen_engine import VisionModelEngine, online_vision_models, require_online_vision_model
from .vlx_client import local_availability as vlx_local_availability

LOG = logging.getLogger(__name__)

STATUS_CREATED = "created"
STATUS_PARSING = "parsing"
STATUS_READY = "ready"
STATUS_RUNNING = "running"
STATUS_STOPPING = "stopping"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

#: R1 执行模式。`legacy_sync` 是 R1 之前的全部行为，一行都没删；
#: `realtime_latest` 才是「采集 / 推理解耦 + 最新帧优先」。开关在
#: lmm_config 的 `lmm_realtime_fast_path`（默认 0），出问题一键回退。
MODE_LEGACY_SYNC = "legacy_sync"
MODE_REALTIME_LATEST = "realtime_latest"

STATUS_ZH = {
    STATUS_CREATED: "已创建",
    STATUS_PARSING: "解析中",
    STATUS_READY: "就绪",
    STATUS_RUNNING: "运行中",
    STATUS_STOPPING: "停止中",
    STATUS_STOPPED: "已停止",
    STATUS_FAILED: "失败",
}

_LOCK = threading.RLock()
_TASKS: Dict[str, Dict[str, Any]] = {}
_EVIDENCE_REPAIR_LOCK = threading.Lock()
_EVIDENCE_REPAIRING: set = set()
_THREADS: Dict[str, threading.Thread] = {}
_STOP: Dict[str, threading.Event] = {}
_STORE = EventStore()
#: 每个任务保留的最新一帧（单槽覆盖，供轮询兜底使用）
_LAST: Dict[str, Dict[str, Any]] = {}

#: 每个任务的运行日志（环形缓冲，随任务一起消亡，不落盘）。
#: 这是给**用户**看的运行轨迹（什么时候开始、抽样怎么定的、跑到哪、写了什么文件），
#: 不是 debug 转储 —— logging 那一份仍然是 LOG.info 打给开发者的。
#: 前端按 `since` 增量拉取只做 append，所以进度日志必须**稀疏**（每 10% 一条），
#: 否则逐帧任务的 400 行缓冲会被进度条冲掉，真正有用的「已写入结果视频」反而留不住。
_LOGS: Dict[str, List[Dict[str, Any]]] = {}
_LOG_MAX = 400

#: 实时推送节流：推理再快也不超过这个频率往外发
#: （改上面那段常量时**别把它一起删掉** —— 漏了就是每个任务都炸 NameError）
_PUSH_INTERVAL = 0.08


# ── 工具 ─────────────────────────────────────────────────────────────────────


def _encode_frame(frame: Any, max_side: int = 960, quality: int = 72) -> Optional[str]:
    """帧 → JPEG base64。缩到 max_side 以内，避免 WS 消息过大。

    实现统一放在 `render.encode_jpeg_b64`：R1 的预览通道要用同一份，
    两处各写一份会导致缩略尺寸改一半、画面大小对不上。
    """
    return encode_jpeg_b64(frame, max_side=max_side, quality=quality)


def _filter_sporadic(samples: List[Dict[str, Any]], min_hits: int, size: tuple) -> int:
    """就地精简每个抽样的框：只留「连续出现 ≥ min_hits 个抽样帧」的目标，返回删掉的个数。

    为什么不能一遍写完：写第 k 帧时还不知道这一框后面会不会再出现，所以先跑完所有抽样帧、
    统计每条链的长度，再**重新逐帧读一遍**写盘 —— 代价是多一次解码+编码，换来的是真过滤。

    关联用**框中心距离**（阈值 = 画面对角线的 12%），而不是规则引擎的空间格：
    慢速小目标一天内可能跨过格子边界，中心距离不会因为网格划分把同一目标断成两条链；
    同时要求**必须在相邻抽样帧**出现，中间断一帧就收链（这才叫"连续"）。
    """
    if min_hits <= 1 or not samples:
        return 0
    width = int(size[0] or 0)
    height = int(size[1] or 0)
    limit = ((width * width + height * height) ** 0.5) * 0.12 if width and height else 0.0

    tracks: List[Dict[str, Any]] = []
    for pos, sample in enumerate(samples):
        for box_i, b in enumerate(sample.get("boxes") or []):
            try:
                cx = (float(b["x1"]) + float(b["x2"])) / 2.0
                cy = (float(b["y1"]) + float(b["y2"])) / 2.0
            except (KeyError, TypeError, ValueError):
                continue
            label = str(b.get("label") or "")
            best = None
            best_d = None
            for t in tracks:
                # 只接相邻抽样帧：断一次就收链，不允许"隔帧续上"
                if t["label"] != label or t["last_pos"] != pos - 1:
                    continue
                d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
                if d <= limit and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is None:
                tracks.append({"label": label, "cx": cx, "cy": cy,
                               "last_pos": pos, "pts": [(pos, box_i)]})
            else:
                best["cx"], best["cy"], best["last_pos"] = cx, cy, pos
                best["pts"].append((pos, box_i))

    keep = set()
    for t in tracks:
        if len(t["pts"]) >= int(min_hits):
            keep.update(t["pts"])

    dropped = 0
    for pos, sample in enumerate(samples):
        boxes = sample.get("boxes") or []
        kept = [b for i, b in enumerate(boxes) if (pos, i) in keep]
        dropped += len(boxes) - len(kept)
        sample["boxes"] = kept
    return dropped


def _cleanup(task_id: str) -> None:
    """任务收尾：复位 tracker + 释放显存。

    在线 Qwen-VL 任务**不加载本地检测权重**，所以没有 tracker 要复位；
    这里只在任务用了本地检测器时才去碰 ultralytics 单例。
    """
    try:
        det = None
        with _LOCK:
            model_path = _TASKS.get(task_id, {}).get("model_path")
            used_vlx = bool(_TASKS.get(task_id, {}).get("vlx"))
        if model_path and not used_vlx:
            from .detectors.yoloe_detector import _SINGLETON
            det = _SINGLETON.get(model_path)
        if det is not None:
            det.reset_tracker()
    except Exception:
        LOG.debug("[LMM] tracker 复位失败", exc_info=True)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _set(task_id: str, **fields: Any) -> None:
    with _LOCK:
        t = _TASKS.get(task_id)
        if t is not None:
            t.update(fields)


def log_line(task_id: str, text: str, level: str = "info") -> None:
    """记一条**面向用户**的运行日志（前端「识别日志」面板直接显示）。

    刻意与 `LOG.info` 分开：那一份是给开发者的，字段多、噪音大，直接丢给用户看只会变成
    一屏天书。这里每条都必须是用户能读懂的一句话，并且**数量要少**（进度每 5% 一条）。
    """
    try:
        entry = {"ts": round(time.time(), 3), "level": level, "text": text}
        with _LOCK:
            buf = _LOGS.setdefault(task_id, [])
            buf.append(entry)
            if len(buf) > _LOG_MAX:
                del buf[: len(buf) - _LOG_MAX]
        # 落盘那份只服务"重启后还能看到日志"：索引里没有这一行就自动跳过
        # （多模态页的实时任务不落盘，不会被这里带上）。
        vindex.add_log(task_id, entry)
    except Exception:  # noqa: BLE001 —— 记日志绝不能把任务搞挂
        LOG.debug("[LMM] 任务 %s 写日志失败", task_id, exc_info=True)


def log_lines(task_id: str, since_ts: Optional[float] = None,
              limit: int = _LOG_MAX) -> List[Dict[str, Any]]:
    """取任务日志。传 `since_ts`（上次拿到的最大 ts）即为增量。

    内存里没有（后端重启过）就回落到落盘索引 —— 否则用户在「识别日志」里
    看到的是一片空白，而磁盘上明明躺着那次任务的完整记录。
    """
    with _LOCK:
        buf = list(_LOGS.get(task_id) or [])
    if not buf:
        buf = vindex.logs(task_id)
    if since_ts is not None:
        try:
            buf = [x for x in buf if float(x.get("ts") or 0) > float(since_ts)]
        except (TypeError, ValueError):
            pass
    return buf[-int(limit):]


#: 疑似凭据的键名片段。
_SECRET_HINTS = ("token", "api_key", "apikey", "secret", "password", "passwd", "credential")


def _is_secret_key(key: str) -> bool:
    """键名是否疑似凭据。

    **必须按"词边界"匹配，不能简单子串包含。** 子串匹配会把
    `vlx_seek_max_new_tokens` 也判成凭据（它含 "token"）→ 脱敏成 `"***"`，
    而它随后要走 `int(...)`，于是任务一启动就
    `invalid literal for int() with base 10: '***'`。这个坑真实发生过。
    """
    k = str(key).lower()
    for h in _SECRET_HINTS:
        if (k == h or k.endswith("_" + h) or k.startswith(h + "_")
                or ("_" + h + "_") in k):
            return True
    return False


def _redact(value: Any) -> Any:
    """递归把疑似凭据的字段换成 `"***"`（保留键，仍能看出"配过了"）。

    `effective` / `config_snapshot` 存的是整份 `lmm_config`，而它现在含
    `vlx_seek_token`（云端 VLX-Seek 服务的鉴权凭据）—— `GET /api/lmm/tasks/{id}`
    会原样返回这两个字段，**不脱敏就等于把凭据发给浏览器**。
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(k):
                out[k] = "***" if v else v
            else:
                out[k] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _publish_event(ev: DetectionEvent) -> None:
    """事件立刻推给前端，不等下一次帧推送。"""
    try:
        hub.publish(ev.task_id, {"type": "event", "task_id": ev.task_id,
                                 "event": ev.to_dict()})
    except Exception:
        LOG.debug("[LMM] 事件推送失败", exc_info=True)


_gpu_cache: Dict[str, Any] = {"ts": 0.0, "value": None}


def _gpu_stats(min_interval: float = 1.0) -> Dict[str, Any]:
    """真实显存占用（1 秒最多刷一次，避免每帧都问 CUDA）。"""
    now = time.time()
    if _gpu_cache["value"] is not None and (now - _gpu_cache["ts"]) < min_interval:
        return _gpu_cache["value"]
    val: Dict[str, Any] = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            val = {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
                "free_mb": round(free_b / 1048576, 1),
                "total_mb": round(total_b / 1048576, 1),
            }
    except Exception:
        val = {"available": False}
    _gpu_cache["ts"] = now
    _gpu_cache["value"] = val
    return val


def _metrics(idx: int, elapsed: float, read_ms_sum: float, infer_ms_sum: float,
             detections: int, detector: str = "vision-model") -> Dict[str, Any]:
    """性能指标。每一项都来自实测计时，**不做任何估算或填充**。

    `video_fps` 与 `detector_fps` 分开：前者是解码/读取能力上限，后者是检测吞吐，
    两者之差就是"画面被推理拖慢了多少"。

    在线视觉模型的单帧延迟可能是秒级，所以 `detector_fps` 会明显低于本地解码速度；
    视频侧按 `qwen_interval_seconds` 抽样，并用最新帧槽避免积压。
    """
    video_fps = (idx / (read_ms_sum / 1000.0)) if read_ms_sum > 0 else None
    det_fps = (idx / (infer_ms_sum / 1000.0)) if infer_ms_sum > 0 else None
    latency = round(infer_ms_sum / idx, 1) if idx else None
    return {
        "processed": idx,
        "detector": detector,
        "pipeline_fps": round(idx / elapsed, 2) if elapsed > 0 else None,
        "video_fps": round(video_fps, 2) if video_fps else None,
        "detector_fps": round(det_fps, 2) if det_fps else None,
        "infer_latency_ms": latency,
        #: 兼容旧前端字段名，值与 infer_latency_ms 相同（新代码请用 infer_latency_ms）
        "yoloe_latency_ms": latency,
        "detections": detections,
        "gpu": _gpu_stats(),
        # 兼容字段：旧客户端若还在读 metrics.vlm，会看到它已被移除而不是 None
        "vlm": {"enabled": False, "removed": True,
                "unavailable_reason": "当前任务由所选检测引擎直接完成语义判定"},
    }


def _perf_block(perf: Dict[str, PerfStats], *, dropped_frames: int,
                latest_frame_id: int, processed_frame_id: int, queue_depth: int,
                elapsed_s: float, src_fps: float, capture_frames: int,
                processed_frames: int, mode: str = "legacy_sync") -> Dict[str, Any]:
    """R0/R1 的实时性能字段（`metrics.perf`）。

    ★两个"落后"必须分开看，混在一起会得出完全错误的结论★

        ``frame_age_ms`` —— 单帧从采集到出结果多久（≈ 一次推理延迟）。
        ``drift_ms``     —— 真实经过的时间 − 已处理到的**视频时间**。

    同步（legacy）链路里 frame_age 可以一直稳定在 4.4s，看着"没问题"；
    但 drift 会随跑的时间线性增长（跑 60s 只处理了 10s 的视频）。
    R1 的目标正是把 drift 也压到 ≈ 单次推理延迟，而**不是**继续只盯 frame_age。
    """
    inf = perf["inference"].summary()
    age = perf["frame_age"].summary()
    video_pos_s = (float(processed_frame_id) / float(src_fps)) if src_fps else 0.0
    return {
        "mode": mode,
        "capture_fps": round(capture_frames / elapsed_s, 2) if elapsed_s > 0 else None,
        "ai_fps": round(processed_frames / elapsed_s, 3) if elapsed_s > 0 else None,
        "last_inference_ms": perf["inference"].last,
        "average_inference_ms": inf["avg"],
        "frame_age_ms": perf["frame_age"].last,
        "average_frame_age_ms": age["avg"],
        "frame_age_p50_ms": age["p50"],
        "frame_age_p95_ms": age["p95"],
        "frame_age_max_ms": age["max"],
        "drift_ms": round((elapsed_s - video_pos_s) * 1000.0, 1),
        "dropped_frames": int(dropped_frames),
        "processed_frames": int(processed_frames),
        "latest_frame_id": int(latest_frame_id),
        "processed_frame_id": int(processed_frame_id),
        "queue_depth": int(queue_depth),
        "segments": {k: perf[k].summary() for k in
                     ("capture", "encode", "transport", "inference", "parse",
                      "render", "publish", "total", "frame_age")},
    }


# ── 创建 ─────────────────────────────────────────────────────────────────────


def create_task(prompt: str, source_type: str = "file", source: str = "",
                provider: str = "auto", options: Optional[Dict[str, Any]] = None,
                detection_labels: Optional[List[str]] = None,
                ai_config_id: Optional[str] = None) -> Dict[str, Any]:
    """创建任务。**不再有任何 Prompt 解析步骤。**

    `provider` / `detection_labels` / `ai_config_id` 只为兼容旧调用方保留（例如仍在
    用旧参数发请求的脚本），**全部忽略**：过去它们会把中文拆成 `person / swimmer /
    water` 这类英文关键词，反而丢掉了「水面上 / 未穿 / 救生衣」之间的语义关系。

    新任务默认使用已保存的在线 Qwen-VL 配置；本地 YOLOE 仍可显式选择。
    """
    task_id = uuid.uuid4().hex[:12]
    cfg = lmm_config.load()
    opts = dict(options or {})

    clean_prompt = (prompt or "").strip()
    spec = TargetSpec.from_raw(clean_prompt)

    task: Dict[str, Any] = {
        "id": task_id,
        "prompt": clean_prompt,
        "source_type": source_type,
        "source": source,
        "status": STATUS_CREATED,
        "message": "已创建任务",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "progress": 0.0,
        "processed": 0,
        "total_frames": 0,
        "fps": None,
        "proc_fps": None,
        "elapsed_ms": None,
        "detections": 0,
        "events_count": 0,
        "error": None,
        "spec": None,
        "labels": [],
        "options": opts,
        "config_snapshot": cfg,
        # 二阶段复核状态（未启动前也能让前端显示"用什么模型复核"）
        "vlm": None,
        "warnings": [],
        "metrics": None,
        # 结果落盘：只有「本地文件 + save_video」才会赋值，前端据此显示下载入口。
        # 任务在跑的时候刻意留 None —— 半个 mp4 能下载也没意义。
        "save_video": False,
        "result_video": None,
        "result_json": None,
    }
    # 合并任务级覆盖参数
    eff = dict(cfg)
    for k in ("imgsz", "confidence", "iou", "tracker", "frame_stride",
              "hit_frames", "cooldown_seconds", "min_targets", "half", "device",
              "trackless_verify", "trackless_grid",
              # 结果视频：本地文件任务把检出框画好写成 mp4，用户可下载（推理页的
              # 「视频 + 大模型」走这条；LMM 页实时看画面，不开这个开关）
              "save_video",
              # 结果视频里「同一目标至少连续出现几个抽样帧才画」：1 = 不过滤。
              # 云端 VLX 不给置信度（后端统一填 1.0），想压掉零星单帧的框只能靠这个。
              "video_min_hits",
              # VLX-Seek 的键也在这里合并：前端「高级配置」直接调抽样间隔
              "engine", "vision_provider_id", "qwen_provider_id", "qwen_interval_seconds",
              "vlx_seek_enabled", "vlx_seek_task", "vlx_seek_lang",
              "vlx_seek_frame_interval", "vlx_seek_max_new_tokens",
              # R1：实时模式开关与预览参数（任务级可覆盖，便于 A/B 对比）
              "lmm_realtime_fast_path", "lmm_realtime_preview_fps",
              "lmm_realtime_preview_quality", "lmm_realtime_preview_max_side",
              "lmm_realtime_pacing",
              # R4A：跟踪通道（任务级可覆盖，便于 A/B 对比）
              "lmm_tracking_enabled", "lmm_tracking_mode",
              "lmm_track_score_threshold", "lmm_track_uncertain_frames",
              "lmm_track_lost_frames", "lmm_track_max_tracks",
              "lmm_track_max_without_vlx_ms", "lmm_track_cache_seconds",
              "lmm_track_frame_longest_side", "lmm_track_max_replay_frames",
              "lmm_track_match_iou", "lmm_track_match_center_ratio",
              # ★STEP5★ init 超时（任务级可覆盖，便于 A/B 对比）
              "lmm_track_pending_init_timeout_ms",
              # 服务器侧共享帧（StreamHub）
              "lmm_streamhub_node_id", "lmm_streamhub_port",
              "lmm_streamhub_token", "lmm_streamhub_poll_hz",
              "lmm_streamhub_cache_seconds", "lmm_streamhub_vlx_interval",
              "lmm_streamhub_fallback_mode"):
        if k in opts and opts[k] is not None:
            eff[k] = opts[k]

    engine_name = str(eff.get("engine") or "vlm").strip().lower()
    # Old saved configs and the upload-inference route still send qwen-vl.
    if engine_name == "qwen-vl":
        engine_name = "vlm"
    if engine_name not in {"vlm", "yoloe", "vlx"}:
        raise ValueError("不支持该检测引擎；请选择已配置的在线视觉大模型、VLX-Seek 或本地 YOLOE")
    eff["engine"] = engine_name
    #: VLX 只由显式选择 `engine=vlx` 开启；读取配置、打开页面和选中下拉项
    #: 都不发健康探测或模型请求。
    eff["vlx_seek_enabled"] = engine_name == "vlx"
    use_remote_vision = engine_name in {"vlm", "vlx"}
    if engine_name == "vlx":
        availability = vlx_local_availability()
        if not availability.get("available"):
            raise ValueError(str(availability.get("reason") or "VLX-Seek 当前不可用"))
    if engine_name == "vlm":
        provider_id = str(
            eff.get("vision_provider_id") or eff.get("qwen_provider_id")
            or cfg.get("vision_provider_id") or cfg.get("qwen_provider_id") or ""
        ).strip()
        if not provider_id:
            providers = online_vision_models()
            provider_id = str(providers[0]["config_id"]) if providers else ""
        if not provider_id:
            raise ValueError("未选择在线视觉大模型，请先在大模型配置中保存可用模型")
        require_online_vision_model(provider_id)
        eff["vision_provider_id"] = provider_id
        # 实时监控走最新帧模式；上传文件且需要导出结果视频时保留逐帧写盘模式。
        eff["lmm_realtime_fast_path"] = int(not (source_type == "file" and eff.get("save_video")))

    with _LOCK:
        _TASKS[task_id] = task

    model_path = None
    if not use_remote_vision:
        model_path = registry.resolve_model_path(eff.get("model_name"))
        if model_path is None:
            _set(task_id, status=STATUS_FAILED,
                 message=f"YOLOE 模型未安装（缺少 {eff.get('model_name')}）",
                 error="model-missing", spec=spec.to_dict(), finished_at=time.time())
            return get_task(task_id) or task
        spec.warnings.append("当前使用本地 YOLOE；复杂自然语言条件的语义判断能力有限")

    # ── 二阶段复核**已删除** ────────────────────────────────────────────────
    # 语义判定由已选引擎完成，不再 build_verifier、不再建复核调度器、
    # 不再读 Visual Prompt 参考图。
    # `vlm` 字段保留并显式标 removed：旧前端/旧脚本取到它时能看到"已停用"而不是 None。
    vlm_info = {
        "enabled": False,
        "removed": True,
            "reason": "当前检测任务不使用独立的二阶段复核",
    }
    vlx_info = ({"enabled": True, "manual_selection": True,
                 "reason": "VLX-Seek 仅在本任务手动选择后调用"}
                if engine_name == "vlx" else {
                    "enabled": False,
                    "reason": "本任务未选择 VLX-Seek，不会调用 VLX",
                })
    _set(task_id, vlm=vlm_info, vlx=vlx_info)

    _set(task_id,
         status=STATUS_READY,
         message="已就绪，可以开始检测",
         spec=spec.to_dict(),
         labels=spec.detection_labels(),
         observable=spec.observable_labels(),
         display_name=spec.display_name,
         # 在线 Qwen-VL 任务没有本地权重 → 空串，不是字符串 "None"
         model_path=str(model_path or ""),
         detector=engine_name,
         # 建任务时就把「会不会产出结果视频」告诉前端：真开始跑才置 True 的话，
         # 前端在 running 状态下无法决定要不要显示下载入口。
         save_video=bool(eff.get("save_video", False)),
         #: 建任务时就告诉前端这次跑哪种模式：前端要靠它决定"用 <video> 播
         #: 还是等 WS 帧"，running 之后再给就晚了（画面通道已经选错）。
         mode=_resolve_mode(eff),
         effective=eff)

    # 「视频推理记录」落盘：只有会产出结果视频的任务才进索引（见 video_index 的说明）。
    # 建任务时就写，是为了"跑着的过程中后端被重启"也不至于整条记录凭空消失。
    if eff.get("save_video"):
        vindex.upsert(task_id,
                      prompt=clean_prompt,
                      created_at=task.get("created_at") or time.time(),
                      source_name=Path(str(source or "")).name,
                      status=STATUS_CREATED,
                      logs=[],
                      video_min_hits=int(eff.get("video_min_hits", 1) or 1))
    return get_task(task_id) or task


# ── 运行 ─────────────────────────────────────────────────────────────────────


def start_task(task_id: str) -> Dict[str, Any]:
    task = get_task(task_id)
    if task is None:
        raise KeyError(task_id)
    if str((task.get("effective") or {}).get("engine") or "").lower() == "vlx":
        availability = vlx_local_availability()
        if not availability.get("available"):
            raise RuntimeError(str(availability.get("reason") or "VLX-Seek 当前不可用"))
    if task["status"] == STATUS_RUNNING:
        return task
    if task["status"] not in (STATUS_READY, STATUS_STOPPED, STATUS_FAILED):
        raise RuntimeError(f"任务状态 {task['status']} 不支持启动")

    with _LOCK:
        if task_id in _THREADS and _THREADS[task_id].is_alive():
            return task
        _STOP[task_id] = threading.Event()
        _STORE.clear(task_id)
        # 重跑同一个任务时旧日志必须清掉：否则第二遍的日志接在第一遍后面，
        # 「已写入结果视频」会出现两次，用户分不清哪条是这次的。落盘那份一起清。
        _LOGS.pop(task_id, None)
        vindex.update(task_id, status=STATUS_RUNNING, logs=[])
        th = threading.Thread(target=_run_task, args=(task_id,),
                              name=f"lmm-{task_id}", daemon=True)
        _THREADS[task_id] = th
    _set(task_id, status=STATUS_RUNNING, message="启动中…", started_at=time.time(),
         finished_at=None, error=None, progress=0.0, processed=0, detections=0,
         events_count=0)
    prompt = str(task.get("prompt") or "")
    log_line(task_id, f"开始识别：{prompt}" if prompt else "开始识别")
    log_line(task_id, f"输入源：{Path(str(task.get('source') or '')).name or task.get('source')}")
    th.start()
    return get_task(task_id) or task


def stop_task(task_id: str) -> Dict[str, Any]:
    """请求停止。**已经结束的任务不再改状态。**

    之前这里无条件把状态写成 `stopping`：任务其实早就跑完并清理了（`_STOP` 已 pop），
    于是状态被永久钉在"停止中"，前端只能一直转圈 —— 对已结束的任务再点一次停止就会触发。
    """
    task = get_task(task_id)
    if task is None:
        return {}
    if task.get("status") not in (STATUS_RUNNING, STATUS_STOPPING, STATUS_CREATED,
                                  STATUS_PARSING, STATUS_READY):
        return task
    ev = _STOP.get(task_id)
    if ev is not None:
        ev.set()
        log_line(task_id, "收到停止请求，正在收尾（结果视频只含已处理部分）", level="warn")
        _set(task_id, status=STATUS_STOPPING, message="正在停止…")
    return get_task(task_id) or task


def _resolve_mode(eff: Dict[str, Any]) -> str:
    """选执行模式。只认 `lmm_realtime_fast_path`，默认回到 legacy_sync。"""
    try:
        if bool(eff.get("lmm_realtime_fast_path", 0)):
            return MODE_REALTIME_LATEST
    except Exception:  # noqa: BLE001
        pass
    return MODE_LEGACY_SYNC


def _remote_request_id(task_id: str, frame_id: Any) -> str:
    """Generate an opaque request ID for a remote visual-model call."""
    return f"lmm-{task_id}-{frame_id}-{uuid.uuid4().hex[:8]}"


def _detect_config(eff: Dict[str, Any]) -> DetectConfig:
    """★三帧确认★的参数构造入口（两条执行链路共用：StreamHub 与本地实时）。

    ★所有键都必须登记进 `config.DEFAULTS`★：那份 `DEFAULTS` 同时是配置白名单
    （`_ALLOWED = set(DEFAULTS.keys())`），漏登记的键会被 `load()` **静默丢弃**
    —— 表现为"明明配了却没效果"，排查起来极费时间。
    """
    return DetectConfig.from_dict(eff)


# ── ★服务器侧共享帧（StreamHub）★ ─────────────────────────────────────────
#: 技术路线切换之后，StreamHub 只剩两个职责（**都不涉及跟踪器**）：
#:   1. 给 VLX 提供 `/infer_at` 的帧号（服务端自己取帧，上行几百字节）；
#:   2. 给告警证据提供 `/frame.jpg?id=` 的按需取图（一条告警三张）。
#: MediaMTX / RTSP / WebRTC / StreamHub **一律不动**，只替换算法处理链。


def _streamhub_requested(eff: Dict[str, Any]) -> bool:
    """任务是否要求走 StreamHub 共享帧链路。

    ★不再看 `lmm_tracking_enabled`★：以前 StreamHub 是"给 OSTrack 供帧"的通道，
    所以挂在跟踪开关下面。现在它是**唯一检测链路的取帧端**，与跟踪器无关 ——
    继续挂在那个开关下会导致"关掉跟踪就连 VLX 也不跑"，是错的。
    """
    mode = str(eff.get("lmm_tracking_mode") or "").strip().lower()
    return mode in ("streamhub", "ostrack-streamhub", "ref", "ostrack-ref",
                    "vlx", "vlx-streamhub")


def _make_streamhub_client(eff: Dict[str, Any]) -> StreamHubClient:
    return StreamHubClient(
        #: 没单独配 StreamHub 节点时复用 OSTrack 的节点（同一个 A6000）
        node_id=(str(eff.get("lmm_streamhub_node_id") or "")
                 or str(eff.get("lmm_ostrack_node_id") or "")),
        port=int(eff.get("lmm_streamhub_port") or 8030),
        token=str(eff.get("lmm_streamhub_token") or ""),
        timeout=float(eff.get("vlx_seek_timeout") or 10.0),
    )


# ── ★告警：证据图落盘★ ─────────────────────────────────────────────────────


def _alert_dir(task_id: str, alert_id: str) -> Path:
    from config import settings
    return Path(settings.ALERTS_DIR) / str(task_id) / str(alert_id)


#: ★证据取图重试★ 为什么必须重试：`/health` 的 `last_frame_id` 会**先于** JPEG
#: 写进 ring 更新，于是「拿到帧号就立刻取最新那一帧」存在竞态 —— 实测表现为
#: **每次缺的都是告警三帧里最新那帧**（真实 RTSP 300 s：缺 1~2 张/轮）。隔
#: 0.3 s 再取一般就有；ring 实测保留 15~40 帧（15 fps 下 ≈ 1~2.7 s），
#: 三次重试总共 ~0.6 s，仍在窗口内。
_EVIDENCE_RETRY_ATTEMPTS = 3
_EVIDENCE_RETRY_GAP_S = 0.3
#: ★预取用更宽的窗口★：StreamHub 实测是**停帧十几秒再一次性灌几百帧**，
#: 跳跃那一刻服务端还在追写积压帧，JPEG 落 ring 的滞后远不止 0.3 s。
#: 给预取 5 次 × 0.7 s ≈ 2.8 s 的窗口（取到就立刻返回，不白等）。
_PF_FETCH_ATTEMPTS = 5
_PF_FETCH_GAP_S = 0.7
#: ★停帧补投★：同一个帧号最多额外补投几次、两次之间至少隔多久。
#: StreamHub 停帧实测 ~15 s，隔 1.5 s 补投、最多 3 次 → 只占停帧期的前 4.5 s，
#: 取到就停（画面静止时必中），取不到也不再纠缠（免得白占隧道带宽）。
_PF_RETRY_MAX = 3
_PF_RETRY_GAP_S = 1.5
#: ★停帧时的最小轮询问隔★：`shub.latest()` 是**一次真实的 `/health` 隧道往返**
#: （无任何缓存）。1-Hit 之前每轮都要跑 2 s 的 VLX 推理，天然给轮询踩了刹车；
#: 加了 `[VLX SKIP]` 之后停帧期不再有推理，若还按 `interval`（实测 0.05 s）等，
#: 轮询频率会从 ~0.5 Hz 飙到 ~10 Hz —— 隧道被自己的心跳占满，帧反而更取不到。
#: 停帧时 0.5 s 问一次足够（实测停帧长达十几秒，漏不掉帧号跳变）。
_SKIP_WAIT_S = 0.5


def _fetch_frame(shub: StreamHubClient, frame_id: int,
                 stop: Optional[threading.Event] = None,
                 attempts: Optional[int] = None,
                 gap_s: Optional[float] = None) -> Optional[bytes]:
    """按帧号取一张证据图；失败隔 `gap_s` 重试，取不到返回 None。

    `attempts` / `gap_s` 不传就用默认档；**预取**传更宽的那一档
    （`_PF_FETCH_ATTEMPTS` / `_PF_FETCH_GAP_S`）。
    """
    n = int(attempts or _EVIDENCE_RETRY_ATTEMPTS)
    g = float(_EVIDENCE_RETRY_GAP_S if gap_s is None else gap_s)
    for k in range(n):
        try:
            data = shub.frame_jpeg(int(frame_id))
        except Exception:  # noqa: BLE001
            data = None
        if data:
            return data
        if k + 1 < n:
            if stop is not None:
                stop.wait(g)
            else:
                time.sleep(g)
    return None


def _stamp_dicts(boxes: Any, label: str, label_zh: str,
                 display_name: str) -> "List[Dict[str, Any]]":
    """★证据图要画什么★：把 VLX 的 xyxy + 类别名转成 `draw_boxes` 的入参。

    ★1-Hit 之后这里恒为单元素★：检出帧 = 告警帧 = 证据帧，画的就是触发这条
    告警的那一次 VLX 自己给的框 —— 不外推、不平滑、不换帧。
    """
    out: List[Dict[str, Any]] = []
    name = (display_name or label_zh or label or "").strip()
    for b in (boxes or []):
        try:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
        except (TypeError, ValueError, IndexError):
            out.append({})
            continue
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": str(label or ""),
                    "label_zh": str(label_zh or label or ""),
                    "display_name": name,
                    "vlm_state": "confirmed"})
    return out


def _pending_evidence(seed: "Optional[Dict[str, Any]]") -> "Dict[str, Any]":
    """★1-Hit：一条告警 = 一个证据 job = 一个 `evidence` 单对象（pending 态）★

    三个 url 与尺寸全是 `None`（**不是空串**）—— 空串会让前端误以为"图已就绪、
    只是路径为空"，从而去加载一个 404；`None` 才能明确表达"还没生成"。
    worker 完成后由 `_sync_evidence_fields` 就地回填**同一个对象**。
    """
    s = dict(seed or {})
    return {"status": PENDING,
            "source_url": None, "original_url": None, "thumbnail_url": None,
            "width": None, "height": None, "bytes": None, "format": None,
            "frame_id": s.get("frame_id"), "pts_ms": s.get("pts_ms"),
            "bbox": list(s.get("bbox") or []), "version": None}


def _publish_evidence_update(ev: DetectionEvent) -> None:
    """★证据逐张 ready★：更新**同一个 event_id**，绝不再造一条新告警。

    为什么单独一个消息类型：告警是"发生了什么"，证据是"这件事的物证补上了"，
    语义完全不同。复用 `event` 会让前端把它当新告警再插一条（同一件事变两条）。
    """
    try:
        hub.publish(ev.task_id, {"type": "evidence_update",
                                 "task_id": ev.task_id,
                                 "event": ev.to_dict()})
    except Exception:  # noqa: BLE001
        LOG.debug("[LMM] 证据更新推送失败", exc_info=True)


def _sync_evidence_fields(rec: DetectionEvent, metas: "List[Dict[str, Any]]") -> None:
    """把 worker 回填的元信息同步到事件字段上（`evidence` 单对象 / 状态）。

    ★1-Hit + 单证据图★：`metas` 恒 ≤1 个元素，所以 `evidence` 是**单对象**：
    `{status, source_url, original_url, thumbnail_url, width, height, …}`。

    `images` 仍**只放 `original_url`**，没 ready 时是空串 —— 前端据此显示
    「处理中」，绝不会拿缩略图顶上（`original_url ?? thumbnail_url` 那种静默
    回退是明确禁止的）。
    """
    ordered = sorted([m for m in metas if m],
                     key=lambda m: int(m.get("index") or 0))[:1]
    rec.image_meta = ordered
    rec.images = [str(m.get("original_url") or "") for m in ordered]
    rec.evidence_frames = [{"frame_id": m.get("frame_id"),
                            "pts_ms": m.get("pts_ms"),
                            "status": str(m.get("status") or PENDING)}
                           for m in ordered]
    rec.evidence_status = evidence_status_of(
        [str(m.get("status") or PENDING) for m in ordered])
    m0 = dict(ordered[0] or {}) if ordered else {}
    rec.evidence = {
        "status": str(m0.get("status") or rec.evidence_status or PENDING),
        "source_url": m0.get("source_url") or None,
        "original_url": m0.get("original_url") or None,
        "thumbnail_url": m0.get("thumbnail_url") or None,
        "width": m0.get("width"), "height": m0.get("height"),
        "bytes": m0.get("bytes"), "format": m0.get("format"),
        "frame_id": m0.get("frame_id"), "pts_ms": m0.get("pts_ms"),
        "bbox": list(m0.get("bbox") or []),
        "version": m0.get("version"),
        "source_original_fallback": bool(m0.get("source_original_fallback")),
    }


def _evidence_callbacks(task_id: str):
    """worker 回调：就地更新**已入库**的那条告警（同一个 id），然后推 WS。"""
    from .evidence import PENDING  # noqa: F401  局部 import 避免循环依赖

    def _on_status(alert_id: str, index: int, status: str) -> None:
        try:
            rec = _STORE.get(task_id, alert_id)
            if rec is None:
                return
            im = list(rec.image_meta or [])
            while len(im) <= index:
                im.append({"index": len(im)})
            cur = dict(im[index] or {})
            cur.update({"index": index, "status": status})
            im[index] = cur
            _sync_evidence_fields(rec, im)
            _publish_evidence_update(rec)
        except Exception:  # noqa: BLE001
            LOG.debug("[LMM] 证据状态回写失败", exc_info=True)

    def _on_done(alert_id: str, index: int,
                 meta: "Optional[Dict[str, Any]]", err: Optional[str]) -> None:
        try:
            rec = _STORE.get(task_id, alert_id)
            if rec is None:
                return
            im = list(rec.image_meta or [])
            while len(im) <= index:
                im.append({"index": len(im)})
            cur = dict(im[index] or {})
            if meta:
                cur.update(meta)
                cur["status"] = READY
            else:
                cur.update({"index": index, "status": ERROR, "error": str(err or "")})
                #: 标注图后续处理失败时，原始 JPEG 往往已经落盘。保留它作为完整分辨率
                #: 的兜底图，让告警仍可打开；缩略图接口会按需生成卡片小图。
                source_path = _alert_dir(task_id, alert_id) / ("%03d_source.jpg" % (index + 1))
                if source_path.is_file():
                    try:
                        from .render import jpeg_wh
                        source_bytes = source_path.read_bytes()
                        sw, sh = jpeg_wh(source_bytes)
                        source_url = f"/api/lmm/alerts/{task_id}/{alert_id}/{source_path.name}"
                        cur.update({
                            "source_url": source_url,
                            "original_url": source_url,
                            "thumbnail_url": source_url,
                            "width": int(sw or 0), "height": int(sh or 0),
                            "bytes": len(source_bytes), "format": "jpg",
                            "version": str(source_path.stat().st_mtime_ns),
                            "source_original_fallback": True,
                        })
                    except Exception:  # noqa: BLE001
                        LOG.debug("[LMM] 读取原始证据帧失败：%s", source_path, exc_info=True)
            im[index] = cur
            _sync_evidence_fields(rec, im)
            _publish_evidence_update(rec)
            log_line(task_id, "证据 %s #%d %s" % (alert_id, index + 1,
                                                  "已就绪" if meta else "失败：" + str(err)))
        except Exception:  # noqa: BLE001
            LOG.exception("[LMM] 证据完成回写失败")

    return _on_status, _on_done


def _make_alert_sink(task_id: str, shub: Optional[StreamHubClient],
                     cfg: DetectConfig, width_ref: "List[int]",
                     height_ref: "List[int]",
                     prefetched: Optional[Dict[int, bytes]] = None):
    """构造「AlertEvent → DetectionEvent → 事件存储 + WS 推送」的出口。

    `width_ref` / `height_ref` 是**单元素 list**，用来在 VLX 回带真实尺寸后
    就地更新（闭包里改不了外层 int，只能改容器内容）。
    """

    def _sink(ev: Any) -> None:
        #: ★验收指标★：三次确认命中 → 告警落库 + 推送 + 证据入队的总耗时。
        #: 目标是 < 50 ms —— 这条路径上**不允许有任何磁盘/网络 IO 等待**。
        _t0 = time.perf_counter()
        try:
            aid = _STORE.next_id(task_id)
            #: ★1-Hit：一条告警 = 一张证据图★，证据帧 = 检出帧 = 告警帧。
            frames = list(ev.frames)[:1]
            stamps: List[Dict[str, Any]] = []
            if cfg.evidence_enabled and shub is not None:
                stamps = _stamp_dicts(getattr(ev, "boxes", None) or [],
                                      str(ev.label or ""),
                                      str(ev.label_zh or ""),
                                      str(ev.display_name or ""))
            #: ★先落一条"证据待处理"的告警并立刻推送★ —— 检测主循环不在这里等 IO
            seeds: List[Dict[str, Any]] = []
            for i, fid in enumerate(frames):
                bb: List[float] = []
                s = stamps[i] if i < len(stamps) else None
                if s:
                    bb = [float(s.get("x1", 0)), float(s.get("y1", 0)),
                          float(s.get("x2", 0)), float(s.get("y2", 0))]
                seeds.append({"index": i, "status": PENDING,
                              "frame_id": int(fid),
                              "pts_ms": (float(ev.pts_ms) if (ev.pts_ms is not None
                                                              and i == len(frames) - 1) else None),
                              "bbox": bb})
            name = (ev.display_name or ev.label_zh or ev.label or "目标").strip()
            if not name.startswith("疑似"):
                name = "疑似%s" % name
            record = DetectionEvent(
                id=aid,
                task_id=task_id,
                frame_index=int(ev.frame_id),
                ts=float(ev.ts or time.time()),
                time_str=_now_hms(ev.ts),
                #: ★没有 tracker ⇒ 没有 track_id★（前端也不再显示 `#id`）
                track_id=None,
                label=str(ev.label or ""),
                label_zh=str(ev.label_zh or ev.label or ""),
                display_name=name,
                confidence=float(ev.confidence or 0.0),
                bbox=[float(v) for v in ev.bbox],
                matched=["VLX 检出"],
                unresolved=[],
                hit_frames=1,
                detector_confidence=float(ev.confidence or 0.0),
                #: 语义检测器一次定案，与旧 VLX 链路保持一致
                vlm_state="confirmed",
                frames=[int(f) for f in ev.frames],
                images=[""] * len(seeds),
                image_meta=list(seeds),
                evidence_frames=[{"frame_id": s["frame_id"], "pts_ms": s["pts_ms"],
                                  "status": PENDING} for s in seeds],
                evidence=(_pending_evidence(seeds[0]) if seeds else {}),
                evidence_status=(PENDING if seeds else ""),
                frame_confidences=[float(c) for c in ev.confidences],
                image_width=int(ev.image_width or width_ref[0] or 0),
                image_height=int(ev.image_height or height_ref[0] or 0),
                pts_ms=ev.pts_ms,
            )
            _STORE.add(record)
            _publish_event(record)
            log_line(task_id,
                     "告警：%s · 帧 %d · 证据 %d 张待处理"
                     % (name, int(ev.frame_id), len(seeds)))

            #: ★证据作业入队（非阻塞）★：VLX 立刻回去跑下一轮推理
            queued = 0
            if cfg.evidence_enabled and shub is not None and seeds:
                queued = _enqueue_evidence(task_id=task_id, alert_id=aid, seeds=seeds,
                                           label=str(ev.label or ""),
                                           label_zh=str(ev.label_zh or ""),
                                           display_name=name,
                                           prefetched=prefetched,
                                           fetcher=(lambda f: _fetch_frame(shub, int(f))))
            LOG.info("[LMM] 告警落库耗时 alert=%s %.1fms（证据入队 %d/%d）",
                     aid, (time.perf_counter() - _t0) * 1000.0, queued, len(seeds))
        except Exception:  # noqa: BLE001
            LOG.exception("[LMM] 任务 %s 告警落库失败", task_id)

    return _sink


def _enqueue_evidence(*, task_id: str, alert_id: str,
                      seeds: "List[Dict[str, Any]]", label: str,
                      label_zh: str, display_name: str,
                      prefetched: "Optional[Dict[int, bytes]]" = None,
                      fetcher: "Optional[Callable[[int], Optional[bytes]]]" = None,
                      out_dir: "Optional[Path]" = None) -> int:
    """★把**一张**证据丢进后台队列★ —— 这里**不等待任何磁盘/网络 IO**。

    1-Hit 之后一条告警只对应一个 job（`seeds` 恒为单元素）。job 带着**钉死的
    frame_id / pts_ms / bbox**，外加（预取命中时）StreamHub 的 q95 JPEG 字节；
    worker 之后只许照着做，**不许回头取最新帧、不许换帧**。
    """
    on_status, on_done = _evidence_callbacks(task_id)
    d = out_dir or _alert_dir(task_id, alert_id)
    n = 0
    for s in seeds:
        fid = int(s.get("frame_id") or 0)
        jpeg = prefetched.get(fid) if prefetched is not None else None
        job = EvidenceJob(task_id=task_id, alert_id=alert_id,
                          index=int(s.get("index") or 0), frame_id=fid,
                          pts_ms=s.get("pts_ms"), label=label,
                          label_zh=label_zh, display_name=display_name,
                          bbox=list(s.get("bbox") or []), jpeg=jpeg,
                          fetcher=fetcher, out_dir=d,
                          on_status=on_status, on_done=on_done)
        if get_worker().submit(job):
            n += 1
    return n


def _now_hms(ts: Optional[float] = None) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts if ts else time.time()).strftime("%H:%M:%S")


def _harvest_inference_jpegs(frames: "List[int]", cache: "Dict[int, bytes]",
                             lock: threading.Lock) -> "Dict[int, bytes]":
    """Copy the exact JPEG captured when the matching inference request began."""
    with lock:
        return {int(fid): cache[int(fid)] for fid in (frames or [])[:1]
                if int(fid) in cache}



def _make_alert_sink_local(task_id: str, cfg: DetectConfig,
                           width_ref: "List[int]", height_ref: "List[int]",
                           inference_jpegs: "Dict[int, bytes]",
                           inference_jpegs_lock: threading.Lock):
    """本地采集链路版的告警出口（证据图来自该次本机推理帧，不经 StreamHub）。

    与 StreamHub 版一样：**先出告警，再异步补证据**。
    """

    def _sink(ev: Any) -> None:
        try:
            aid = _STORE.next_id(task_id)
            #: ★1-Hit：一条告警 = 一张证据图★，证据帧 = 检出帧 = 告警帧。
            frames = list(ev.frames)[:1]
            stamps: List[Dict[str, Any]] = []
            harvested: "Dict[int, bytes]" = {}
            if cfg.evidence_enabled:
                stamps = _stamp_dicts(getattr(ev, "boxes", None) or [],
                                      str(ev.label or ""),
                                      str(ev.label_zh or ""),
                                      str(ev.display_name or ""))
                #: 推理请求开始时已经保存了对应帧，模型数秒后返回也不会误取最新帧。
                harvested = _harvest_inference_jpegs(
                    frames, inference_jpegs, inference_jpegs_lock)
            seeds: List[Dict[str, Any]] = []
            for i, fid in enumerate(frames):
                bb: List[float] = []
                s = stamps[i] if i < len(stamps) else None
                if s:
                    bb = [float(s.get("x1", 0)), float(s.get("y1", 0)),
                          float(s.get("x2", 0)), float(s.get("y2", 0))]
                seeds.append({"index": i, "status": PENDING,
                              "frame_id": int(fid),
                              "pts_ms": (float(ev.pts_ms) if (ev.pts_ms is not None
                                                              and i == len(frames) - 1) else None),
                              "bbox": bb})
            name = (ev.display_name or ev.label_zh or ev.label or "目标").strip()
            if not name.startswith("疑似"):
                name = "疑似%s" % name
            record = DetectionEvent(
                id=aid, task_id=task_id, frame_index=int(ev.frame_id),
                ts=float(ev.ts or time.time()), time_str=_now_hms(ev.ts),
                track_id=None,
                label=str(ev.label or ""),
                label_zh=str(ev.label_zh or ev.label or ""),
                display_name=name,
                confidence=float(ev.confidence or 0.0),
                bbox=[float(v) for v in ev.bbox],
                matched=["VLX 检出"],
                unresolved=[], hit_frames=1,
                detector_confidence=float(ev.confidence or 0.0),
                vlm_state="confirmed",
                frames=[int(f) for f in ev.frames],
                images=[""] * len(seeds),
                image_meta=list(seeds),
                evidence_frames=[{"frame_id": s["frame_id"], "pts_ms": s["pts_ms"],
                                  "status": PENDING} for s in seeds],
                evidence=(_pending_evidence(seeds[0]) if seeds else {}),
                evidence_status=(PENDING if seeds else ""),
                frame_confidences=[float(c) for c in ev.confidences],
                image_width=int(ev.image_width or width_ref[0] or 0),
                image_height=int(ev.image_height or height_ref[0] or 0),
                pts_ms=ev.pts_ms,
            )
            _STORE.add(record)
            _publish_event(record)
            log_line(task_id,
                     "告警：%s · 帧 %d · 证据 %d 张待处理"
                     % (name, int(ev.frame_id), len(seeds)))
            if cfg.evidence_enabled and seeds:
                _enqueue_evidence(task_id=task_id, alert_id=aid, seeds=seeds,
                                  label=str(ev.label or ""),
                                  label_zh=str(ev.label_zh or ""),
                                  display_name=name,
                                  prefetched=harvested, fetcher=None)
        except Exception:  # noqa: BLE001
            LOG.exception("[LMM] 任务 %s 告警落库失败", task_id)

    return _sink


def recover_failed_evidence_from_local_video(task_id: str, alert_id: str) -> Dict[str, Any]:
    """Rebuild a failed alert frame from its original local video, without model calls."""
    task = _TASKS.get(task_id)
    if task is None:
        raise KeyError(task_id)
    if str(task.get("source_type") or "").lower() != "file":
        raise ValueError("只能从本地视频恢复告警原帧")
    source = Path(str(task.get("source") or ""))
    if not source.is_file():
        raise ValueError("原始本地视频当前不可访问，无法恢复此帧")
    rec = _STORE.get(task_id, alert_id)
    if rec is None:
        raise KeyError(alert_id)

    key = (task_id, alert_id)
    with _EVIDENCE_REPAIR_LOCK:
        if key in _EVIDENCE_REPAIRING:
            return rec.to_dict()
        _EVIDENCE_REPAIRING.add(key)
    try:
        metas = list(rec.image_meta or [])
        meta = dict(metas[0] if metas else {})
        if meta.get("status") == READY:
            return rec.to_dict()
        index = int(meta.get("index") or 0)
        raw_frame_id = meta.get("frame_id")
        if raw_frame_id is None:
            raw_frame_id = (rec.frames[0] if rec.frames else rec.frame_index)
        frame_id = int(raw_frame_id)
        if frame_id < 0:
            raise ValueError("告警未记录有效原帧位置，无法恢复")

        import cv2
        cap = cv2.VideoCapture(str(source))
        try:
            if not cap.isOpened():
                raise ValueError("无法打开原始本地视频")
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
                raise ValueError("原始视频不支持定位到告警帧")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError("原始视频中找不到告警帧")
        finally:
            cap.release()

        jpeg = encode_jpeg_bytes(frame, quality=EVIDENCE_JPEG_QUALITY)
        if not jpeg:
            raise ValueError("无法编码恢复的原始帧")
        seeds = [{"index": index, "status": PENDING, "frame_id": frame_id,
                  "pts_ms": meta.get("pts_ms"), "bbox": list(meta.get("bbox") or rec.bbox or [])}]
        meta.update({"index": index, "status": PENDING, "error": "",
                     "source_url": None, "original_url": None,
                     "thumbnail_url": None, "source_original_fallback": False})
        rec.image_meta = [meta]
        _sync_evidence_fields(rec, rec.image_meta)
        _publish_evidence_update(rec)

        job = EvidenceJob(
            task_id=task_id, alert_id=alert_id, index=index,
            frame_id=frame_id, pts_ms=meta.get("pts_ms"),
            label=str(rec.label or ""), label_zh=str(rec.label_zh or ""),
            display_name=str(rec.display_name or rec.label_zh or rec.label or ""),
            bbox=list(seeds[0]["bbox"]), jpeg=jpeg, fetcher=None,
            out_dir=_alert_dir(task_id, alert_id),
            on_status=_evidence_callbacks(task_id)[0],
            on_done=_evidence_callbacks(task_id)[1],
        )
        worker = get_worker()
        worker._notify(job, PROCESSING)
        image_meta, error = worker._process(job)
        if image_meta is not None:
            with worker._lock:
                worker.done += 1
        else:
            with worker._lock:
                worker.failed += 1
        if job.on_done:
            job.on_done(alert_id, index, image_meta, error)
        return rec.to_dict()
    finally:
        with _EVIDENCE_REPAIR_LOCK:
            _EVIDENCE_REPAIRING.discard(key)


def recover_orphaned_alert_from_local_video(*, task_id: str, alert_id: str,
                                            source_path: str,
                                            event: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild an alert whose in-memory task/event was lost during an API restart.

    The browser supplies the selected task's local source and the event already
    displayed on screen. This path never invokes a detector or remote model.
    """
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(task_id or "")):
        raise ValueError("任务标识无效")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", str(alert_id or "")):
        raise ValueError("告警标识无效")

    raw_source = str(source_path or "").strip()
    if not raw_source or "://" in raw_source or raw_source.startswith(("\\\\", "//")):
        raise ValueError("仅支持原始本地视频文件")
    source = Path(raw_source).expanduser()
    if not source.is_absolute() or source.suffix.lower() not in {
            ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}:
        raise ValueError("仅支持绝对路径的本地视频文件")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError("原始本地视频当前不可访问") from exc
    if not source.is_file():
        raise ValueError("原始本地视频当前不可访问")

    evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
    image_meta = event.get("image_meta") if isinstance(event.get("image_meta"), list) else []
    first_meta = image_meta[0] if image_meta and isinstance(image_meta[0], dict) else {}
    raw_frame_id = (evidence.get("frame_id") if evidence else None)
    if raw_frame_id is None:
        raw_frame_id = first_meta.get("frame_id")
    if raw_frame_id is None:
        frames = event.get("frames") if isinstance(event.get("frames"), list) else []
        raw_frame_id = frames[0] if frames else event.get("frame_index")
    try:
        frame_id = int(raw_frame_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("告警没有有效的原始帧位置") from exc
    if frame_id < 0:
        raise ValueError("告警没有有效的原始帧位置")

    raw_bbox = evidence.get("bbox") if evidence else None
    if raw_bbox is None:
        raw_bbox = first_meta.get("bbox")
    if raw_bbox is None:
        raw_bbox = event.get("bbox")
    bbox = [float(v) for v in raw_bbox] if isinstance(raw_bbox, (list, tuple)) else []

    import cv2
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError("无法打开原始本地视频")
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
            raise ValueError("原始视频不支持定位到告警帧")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("原始视频中找不到告警帧")
    finally:
        cap.release()

    jpeg = encode_jpeg_bytes(frame, quality=EVIDENCE_JPEG_QUALITY)
    if not jpeg:
        raise ValueError("无法编码恢复的原始帧")

    seed = dict(first_meta)
    seed.update({"index": 0, "status": PENDING, "frame_id": frame_id,
                 "bbox": bbox, "pts_ms": evidence.get("pts_ms", seed.get("pts_ms"))})
    job = EvidenceJob(
        task_id=task_id, alert_id=alert_id, index=0, frame_id=frame_id,
        pts_ms=seed.get("pts_ms"), label=str(event.get("label") or ""),
        label_zh=str(event.get("label_zh") or ""),
        display_name=str(event.get("display_name") or event.get("label_zh") or ""),
        bbox=bbox, jpeg=jpeg, fetcher=None, out_dir=_alert_dir(task_id, alert_id),
    )
    worker = get_worker()
    metadata, error = worker._process(job)
    if metadata is None:
        raise ValueError(f"本地原帧恢复失败：{error or 'unknown'}")
    with worker._lock:
        worker.done += 1

    updated = dict(event)
    updated.update({"id": alert_id, "task_id": task_id,
                    "image_meta": [metadata], "images": [metadata.get("original_url")],
                    "evidence_status": READY,
                    "evidence": {
                        "status": READY,
                        "source_url": metadata.get("source_url"),
                        "original_url": metadata.get("original_url"),
                        "thumbnail_url": metadata.get("thumbnail_url"),
                        "width": metadata.get("width"),
                        "height": metadata.get("height"),
                        "bytes": metadata.get("bytes"),
                        "format": metadata.get("format"),
                        "frame_id": frame_id, "bbox": bbox,
                    }})
    return updated

def _run_streamhub_task(task_id: str, *, task: Dict[str, Any], det: Any,
                        eff: Dict[str, Any], spec: Any, stop: threading.Event,
                        shub: StreamHubClient, src_width: int, src_height: int,
                        use_vlx: bool) -> None:
    """★VLX 单模型实时识别★ 执行体：本机**不采集、不上传任何像素**。

    与 `_run_realtime_task` 的本质差别：

      1. 没有本地 Source / RealtimePipeline —— 帧的读者是 A6000 上的 StreamHub；
      2. 没有 `preview_frame`（本机没有像素可编码；画面仍由前端 WebRTC 直连呈现）；
      3. VLX 走 `/infer_at`（只发 `frame_id`，服务端自己取帧）；
      4. ★没有任何跟踪器★：VLX 结果直接进 `DetectionManager`，**1-Hit 即时**
         出 `AlertEvent`（不再累计 1/2/3 次命中）。不发 `track_update`、
         不调 `init_by_ref` / `update_at`。

    ★为什么不开 RuleEngine★：事件判定（`hit_frames` / 空间格归组）需要像素，
    本机在这一模式下一帧都没有；而且"是不是同一个目标"已经由
    `DetectionManager` 按 IoU / 中心距离做了 —— 它比按画面网格归组更准
    （真正按目标，不按位置）。
    """
    t0 = time.time()
    width = int(src_width or 0) or int(eff.get("lmm_streamhub_width") or 0)
    height = int(src_height or 0) or int(eff.get("lmm_streamhub_height") or 0)
    #: 单元素容器：VLX 回带真实尺寸后要在闭包里就地更新（int 是值语义，改不了外层）
    w_ref = [width]
    h_ref = [height]
    state = {"detections": 0, "last_detection_frame_id": 0}
    #: ★单元素容器★：VLX 循环每轮已经轮询过一次 StreamHub，把"画面当前帧号"
    #: 就地存下来给 `on_vlx_result` 复用。**每轮只轮询一次**是有意为之 ——
    #: 一次 health 要走一趟 A6000 隧道（实测均值 600ms、峰值 3.8s），轮询两次
    #: 等于把 VLX 采样周期平白拉长约 40%，直接压缩"连续三帧"的时间窗。
    vid_ref: List[Optional[int]] = [None]
    #: ★第一层重复帧保护★：最近一次**真正发起过 VLX 推理**的帧号。
    #: StreamHub 实测会停帧（49.6% 的轮次 `last_frame_id` 不变），帧号没前进就
    #: 别调 VLX —— 对同一张静止画面反复推理既烧 GPU，又会让 DetectionManager
    #: 拿到 `gap<=0` 的重复结果（那是"重复帧重复告警"的源头）。
    last_req_frame: List[int] = [0]
    lock = threading.RLock()

    dcfg = _detect_config(eff)
    #: ★证据预取缓存★：帧号 → JPEG 字节。请求推理的同时就后台取，免得 VLX 结果
    #: 回来时那一帧已被 StreamHub 的环形缓冲挤出（1-Hit：等结果再取就晚了）。
    _prefetched: Dict[int, bytes] = {}
    _pf_lock = threading.Lock()
    _pf_q: "queue.Queue[int]" = queue.Queue(maxsize=64)
    #: ★停帧补投表★：帧号 → [已补投次数, 上次补投时刻]
    _pf_retry: Dict[int, List[float]] = {}

    def _queue_prefetch(fid: int) -> None:
        """把帧号丢进预取队列（幂等：已经在缓存里就不投）。"""
        if not dcfg.evidence_enabled:
            return
        fid = int(fid)
        with _pf_lock:
            if fid in _prefetched:
                return
        try:
            _pf_q.put_nowait(fid)
        except Exception:  # noqa: BLE001
            pass

    def _retry_prefetch(fid: int) -> None:
        """★停帧期间补投预取★ —— 帧号没前进时画面是静止的，StreamHub 的 ring
        不会被新帧冲刷，这时候取帧**必中**。

        为什么必须补这一刀：StreamHub 实测会停帧（本轮最长 ~15 s）然后一次性
        灌进来几百帧（实测 +563 / +767）。**跳跃发生的那一刻 ring 正被整批
        覆盖**，那一帧的取图请求大概率 404；而"只在帧号前进时排一次"的老策略
        恰好把取图安排在最糟的时刻。停帧补投把重试挪到画面静止期，命中率极高。
        """
        if not dcfg.evidence_enabled:
            return
        fid = int(fid)
        with _pf_lock:
            if fid in _prefetched:
                return
        st = _pf_retry.get(fid)
        now = time.time()
        if st is None:
            st = [0.0, 0.0]
            _pf_retry[fid] = st
            while len(_pf_retry) > 64:
                _pf_retry.pop(min(_pf_retry.keys()), None)
        if st[0] >= _PF_RETRY_MAX or (now - st[1]) < _PF_RETRY_GAP_S:
            return
        st[0] += 1
        st[1] = now
        try:
            _pf_q.put_nowait(fid)
            LOG.info("[LMM] 任务 %s 证据预取补投 frame=%d 第 %d 次（停帧期）",
                     task_id, fid, int(st[0]))
        except Exception:  # noqa: BLE001
            pass

    def _evidence_worker() -> None:
        while not stop.is_set():
            try:
                fid = _pf_q.get(timeout=0.5)
            except Exception:  # noqa: BLE001
                continue
            try:
                data = _fetch_frame(shub, int(fid), stop,
                                    attempts=_PF_FETCH_ATTEMPTS,
                                    gap_s=_PF_FETCH_GAP_S)
                if not data:
                    continue
                with _pf_lock:
                    _prefetched[int(fid)] = data
                    #: 只留最近 64 帧（够覆盖一个 VLX 采样周期 + 长间隔取样）
                    while len(_prefetched) > 64:
                        _prefetched.pop(min(_prefetched.keys()), None)
            except Exception:  # noqa: BLE001
                pass

    _pf_thread = None
    if dcfg.evidence_enabled:
        _pf_thread = threading.Thread(target=_evidence_worker,
                                      name="streamhub-evidence", daemon=True)
        _pf_thread.start()

    dm = DetectionManager(task_id, dcfg,
                          on_alert=_make_alert_sink(task_id, shub, dcfg,
                                                    w_ref, h_ref, _prefetched))
    _set(task_id, tracking_status="off",
         tracking_detail="VLX 单模型 + 1-Hit 即时告警（无跟踪器）")
    log_line(task_id, f"告警：1-Hit 即时告警 · 同目标冷却 "
                      f"{dcfg.cooldown_ms / 1000:g}s · IoU ≥ {dcfg.iou_threshold:g}"
                      + ("（存一张证据图）" if dcfg.evidence_enabled else ""))

    #: VLX FPS / 检测延迟的滚动统计
    vlx_ts: List[float] = []
    last_latency: List[float] = [0.0]
    video_fps: List[float] = [float(shub.fps or 0.0)]

    def on_vlx_result(frame_id: int, media_time_ms: float,
                      detections: "List[Dict[str, Any]]",
                      timing: Dict[str, Any], started: float,
                      finished: float, pts_ms: Optional[float] = None,
                      img_w: int = 0, img_h: int = 0) -> None:
        """发布一条 `detection` 并喂给 `DetectionManager`（★1-Hit 即时告警★）。

        ★STEP4★ `img_w/img_h` = VLX 服务端**实际推理的那一帧**的像素尺寸。
        拿到就用它替换 `lmm_streamhub_width/height` 那个**兜底猜测值** —— 本机在
        共享帧模式下一帧像素都没有，尺寸只能靠服务端回带。猜错的后果是纯视觉的、
        且极难定位：后端所有坐标都是对的，但前端 Canvas 按 `sx = cw / width`
        缩放，width 小 1.5 倍 → **告警光晕整体放大 1.5 倍并往右下偏移**。
        实测：兜底 1280×720，真实 1920×1080。
        """
        if img_w > 0 and img_h > 0 and (img_w, img_h) != (w_ref[0], h_ref[0]):
            if w_ref[0] and h_ref[0]:
                LOG.warning("[LMM] 任务 %s 画面尺寸修正 %dx%d -> %dx%d（服务端回带）",
                            task_id, w_ref[0], h_ref[0], img_w, img_h)
                log_line(task_id,
                         f"画面尺寸 {w_ref[0]}×{h_ref[0]} → {img_w}×{img_h}（服务端回带）")
            w_ref[0], h_ref[0] = int(img_w), int(img_h)
        with lock:
            if int(frame_id) < state["last_detection_frame_id"]:
                return
            state["last_detection_frame_id"] = int(frame_id)
            state["detections"] += len(detections)
            vlx_ts.append(time.time())
            #: 只留最近 20 次，够算 10 s 窗口的帧率就行
            del vlx_ts[:-20]

        #: ★唯一检测链路★：VLX → DetectionManager（**不经过任何 tracker**）
        try:
            dm.on_detections(int(frame_id), float(media_time_ms),
                             detections, pts_ms=pts_ms,
                             image_width=w_ref[0], image_height=h_ref[0])
        except Exception:  # noqa: BLE001
            LOG.exception("[LMM] 任务 %s DetectionManager 处理失败", task_id)

        #: ★证据预取★ 不在这里排队 —— 见 `vlx_loop` 里的说明：**等结果回来再取
        #: 就晚了**（推理 ~1.6s + ring 只留 15~40 帧 = 1~2.7s，正好越窗）。

        lat = 0.0
        try:
            lat = float((timing or {}).get("total_ms")
                        or max(0.0, (finished - started) * 1000.0))
        except (TypeError, ValueError):
            lat = max(0.0, (finished - started) * 1000.0)
        last_latency[0] = lat
        try:
            video_fps[0] = float(shub.fps or 0.0)
        except Exception:  # noqa: BLE001
            pass

        #: 画面当前帧号（WebRTC 播放的那条流）— 画面与算法是两条独立流，
        #: 帧号本来不可比，这里只作为"流还活着"的观测值下发。
        #: ★不再二次轮询 StreamHub★：这个字段只是"画面那条流还活着"的观测值，
        #: 复用 `vlx_loop` 本轮刚拿到的那次 health 结果即可（见 `vid_ref` 注释）。
        now_fid = vid_ref[0]

        st = dm.snapshot_stats()
        msg = {
            "type": "detection", "task_id": task_id,
            "frame_id": int(frame_id),
            "media_time_ms": float(media_time_ms),
            "inference_started_at": started,
            "inference_finished_at": finished,
            "detections": detections, "timing": timing,
            "width": w_ref[0], "height": h_ref[0],
            "detections_total": state["detections"],
            "video_frame_id": now_fid,
            "video_pts_ms": (None if pts_ms is None else round(float(pts_ms), 1)),
            #: ★新指标★：Video FPS / VLX FPS / Detection Latency /
            #: Confirmed Target / Alert Count 全在这里，前端不再需要 track_update
            "vlx_fps": _rate(vlx_ts),
            "detection_latency_ms": round(lat, 1),
            "video_fps": round(video_fps[0], 2),
            "confirmed": int(st.get("confirmed") or 0),
            "alert_count": int(st.get("alerts") or 0),
            "pulse_ms": float(dcfg.pulse_ms),
            "metrics": {"perf": {
                "processed_frames": int(frame_id),
                "latest_frame_id": int(now_fid or frame_id),
                "dropped_frames": 0, "ai_fps": _rate(vlx_ts),
                "queue_depth": 0,
            }, "streamhub": shub.stats()},
        }
        with _LOCK:
            _LAST[task_id] = dict(msg)
        #: 注意：模块级的 `hub` 是 **WS 广播器**，StreamHub 客户端在本文里叫
        #: `shub` —— 两者同名会让"广播"静默变成"取帧"，是最难查的一类 bug。
        hub.publish(task_id, msg)
        _set(task_id, processed=int(frame_id),
             detections=state["detections"],
             events_count=_STORE.count(task_id),
             metrics=msg["metrics"],
             message=f"识别中 · 帧 {frame_id} · 告警 {st.get('alerts', 0)} 条")

    def vlx_loop() -> None:
        interval = float(eff.get("lmm_streamhub_vlx_interval", 1.5) or 1.5)
        err_run = 0
        while not stop.is_set():
            try:
                cur = shub.latest()
                if cur is None:
                    stop.wait(0.2)
                    continue
                vid_ref[0] = int(cur.frame_id)
                #: ★第一层重复帧保护★：同一个 frame_id 最多只做一次有效 VLX 检测。
                if int(cur.frame_id) <= last_req_frame[0]:
                    LOG.info("[LMM] 任务 %s [VLX SKIP] reason=no_new_frame "
                             "frame=%d last=%d", task_id,
                             int(cur.frame_id), last_req_frame[0])
                    #: ★画面静止 ⇒ ring 不会被冲刷 ⇒ 此刻补投预取必中★
                    _retry_prefetch(int(cur.frame_id))
                    #: ★停帧就别再高频轮询★（见 `_SKIP_WAIT_S` 的注释）
                    stop.wait(max(_SKIP_WAIT_S, interval))
                    continue
                last_req_frame[0] = int(cur.frame_id)
                #: ★证据预取要**在这里**排队，不能等 VLX 结果回来★
                #: 实测：推理本身 ~1.6s，而 StreamHub 的 ring 只留 15~40 帧
                #: （15 fps 下 1~2.7s）。等结果回来再取图，正好卡在窗口边缘 ——
                #: 这就是"一条告警只存到 1~2 张证据图"的根因（日志里成片的
                #: `取证据帧失败 status=404`）。请求时就排队，真正取图发生在
                #: 这一帧被采样后的 0.1~0.7s 内，稳稳落在窗口里。
                #: 代价：每个采样帧都要取一张（~0.22 Hz、每张 ~150 KB），
                #: 相当于 ~30 KB/s —— 与"逐帧追帧"的 2.2 Mbps 不是一个量级。
                #: ★证据预取★：帧号没变时由 `_retry_prefetch` 在停帧期补投，
                #: 这里只负责"帧号前进"的那一次首投。
                _queue_prefetch(int(cur.frame_id))
                t_start = time.time()
                res = det.detect_at(cur.frame_id, cur.media_time_ms,
                                    pts_ms=getattr(cur, "pts_ms", None))
                t_end = time.time()
                err_run = 0
                dets = det.decorate(res.boxes) if hasattr(det, "decorate") else []
                #: 服务端回带的是它**实际用的那一帧**，用它而不是请求的 frame_id
                fid = int(getattr(res, "frame_id", 0) or cur.frame_id)
                mt = float(getattr(res, "media_time_ms", 0.0) or cur.media_time_ms)
                _pts = getattr(res, "pts_ms", None)
                if _pts is None:
                    _pts = getattr(cur, "pts_ms", None)
                #: ★诊断★：请求帧 vs 服务端实际用的帧。两者**相同**意味着
                #: StreamHub 的环形缓冲这一轮没往前走 —— 下一轮若还相同，
                #: `vlx_loop` 的 `[VLX SKIP] reason=no_new_frame` 会直接跳过，
                #: `DetectionManager` 的第二层保护也会把它丢掉，
                #: 因此"同一帧重复告警"不可能发生。这是必须实测的一项。
                LOG.info("[LMM] 任务 %s VLX req_frame=%d -> got_frame=%d "
                         "latency=%.0fms dets=%d media_ms=%.0f",
                         task_id, int(cur.frame_id), int(fid),
                         (t_end - t_start) * 1000.0, len(dets), mt)
                on_vlx_result(fid, mt, dets,
                              dict(getattr(res, "timing", None) or {}),
                              t_start, t_end, pts_ms=_pts,
                              #: ★STEP4★ 真实画面尺寸由服务端回带，不再停留在兜底猜测值
                              img_w=int(getattr(res, "width", 0) or 0),
                              img_h=int(getattr(res, "height", 0) or 0))
            except Exception as exc:  # noqa: BLE001
                err_run += 1
                LOG.warning("[LMM] 任务 %s 共享帧 VLX 失败(%d)：%s",
                            task_id, err_run, exc)
                if err_run == 1:
                    log_line(task_id, f"共享帧 VLX 调用失败：{exc}", level="warn")
                #: 连续失败 5 次 → 停止（不要空转刷日志）
                if err_run >= 5:
                    log_line(task_id, "共享帧 VLX 连续失败 5 次，停止推理",
                             level="error")
                    break
            stop.wait(max(0.05, interval))

    vlx_thread = None
    if use_vlx:
        vlx_thread = threading.Thread(target=vlx_loop, name="streamhub-vlx",
                                      daemon=True)
        vlx_thread.start()

    log_line(task_id, "服务器侧共享帧：本机不采集 / 不上传像素，"
                      "VLX 唯一检测链路（无跟踪器）")
    while not stop.is_set():
        stop.wait(0.5)

    if vlx_thread is not None and vlx_thread.is_alive():
        vlx_thread.join(timeout=3.0)
    dm.reset()

    elapsed = time.time() - t0
    st = dm.snapshot_stats()
    _set(task_id, status=STATUS_STOPPED,
         message=(f"识别结束：处理帧号至 {state['last_detection_frame_id']}，"
                  f"用时 {elapsed:.1f}s，检出 {state['detections']} 个目标，"
                  f"告警 {st.get('alerts', 0)} 条"),
         finished_at=time.time(), processed=state["last_detection_frame_id"],
         detections=state["detections"], events_count=_STORE.count(task_id),
         metrics={"perf": {"processed_frames": state["last_detection_frame_id"]},
                  "streamhub": shub.stats()},
         progress=100.0, result_video=None, result_json=None, save_video=False)
    log_line(task_id,
             f"识别完成：用时 {elapsed:.1f}s，"
             f"检出 {state['detections']} 个目标，告警 {st.get('alerts', 0)} 条"
             + ("（已手动停止）" if stop.is_set() else ""),
             level="warn" if stop.is_set() else "ok")


def _rate(ts: "List[float]", window: float = 10.0) -> float:
    """最近 `window` 秒内的到达频率（Hz）。样本 <2 时返回 0。"""
    if len(ts) < 2:
        return 0.0
    now = ts[-1]
    kept = [t for t in ts if (now - t) <= window]
    span = max(0.001, now - (kept[0] if kept else now))
    return round((len(kept) - 1) / span, 2) if len(kept) >= 2 else 0.0




def _run_realtime_task(task_id: str, *, task: Dict[str, Any], src: Any, det: Any,
                       eff: Dict[str, Any], spec: Any, stop: threading.Event,
                       src_fps: float, detector_label: str, remote_detector: bool,
                       src_width: int, src_height: int) -> None:
    """R1 `realtime_latest` 执行体：采集与推理解耦，最新帧优先。

    与同步循环的三点本质差别：

      1. **采集独立线程 + realtime pacing** —— 画面完全不等推理；本地文件按媒体
         时间匀速推进，后端不会跑到浏览器前面去（否则 detection 对不上画面）。
      2. **LatestFrameBuffer 容量 1** —— 推理慢就丢帧，落后**不累积**。
      3. **WS 分两类消息** —— `preview_frame` 只更新画面，`detection` 只更新框，
         且 detection **绝不带整张 JPEG**（那是把画面更新又绑回 AI 频率）。

    本模式**不写结果视频**：实时优先低延迟，逐帧编码写盘会拖慢实时链路。

    ★检测帧与预览帧独立★：告警证据使用触发检测的那一帧，不受预览帧率与在线
    模型推理延迟影响，也不需要再去 StreamHub 取。
    """
    interval_key = ("vlx_seek_frame_interval"
                    if str(eff.get("engine") or "").lower() == "vlx"
                    else "qwen_interval_seconds")
    opts = RealtimeOptions(
        preview_fps=float(eff.get("lmm_realtime_preview_fps", 8.0) or 8.0),
        preview_quality=int(eff.get("lmm_realtime_preview_quality", 75) or 75),
        preview_max_side=int(eff.get("lmm_realtime_preview_max_side", 960) or 960),
        inference_interval_s=(max(0.0, float(eff.get(interval_key, 2.0) or 0.0))
                              if remote_detector else 0.0),
        realtime_pacing=bool(eff.get("lmm_realtime_pacing", 1)),
    )
    #: ★告警出口只剩一个★：技术路线切换后，实时链路的告警由 `DetectionManager`
    #: （连续三帧确认）产生并落库 —— 见下面 `dm = DetectionManager(...)`。
    #: 这里**刻意传 `on_event=None`**：RuleEngine 自己也有一套"连续命中 hit_frames
    #: 帧就出事件"的逻辑，两边同时开着会**同一目标告两次警**，而且 RuleEngine 那条
    #: 事件没有 `frames` / `images`（没有三张连续证据图），右侧告警栏会出现一批
    #: 无证据图的空卡片，与"每条告警 3 张证据"的验收直接冲突。
    #: RuleEngine 在这条链路里只剩 `decorate()` 的措辞/条件标注职责。
    engine = RuleEngine(spec, RuleEngineConfig.from_dict(eff), _STORE, task_id,
                        scheduler=None, on_event=None, semantic_final=remote_detector)
    #: 尺寸来自 probe 阶段缓存的元信息 —— 实时流是"先探测再正式 open"的两阶段
    #: 生命周期，那时再读 `src.size` 就已经太晚。
    width = int(src_width or 0)
    height = int(src_height or 0)
    w_ref = [width]
    h_ref = [height]

    #: Keep the exact JPEG for the active request; its callback runs after inference,
    #: even when the remote model takes several seconds.
    inference_jpegs: Dict[int, bytes] = {}
    inference_jpegs_lock = threading.Lock()

    dcfg = _detect_config(eff)
    dm = DetectionManager(task_id, dcfg,
                          on_alert=_make_alert_sink_local(
                              task_id, dcfg, w_ref, h_ref,
                              inference_jpegs, inference_jpegs_lock))
    _set(task_id, tracking_status="off",
         tracking_detail="VLX 单模型 + 1-Hit 即时告警（无跟踪器）")

    t0 = time.time()
    state = {"detections": 0, "last_detection_frame_id": 0}
    lock = threading.RLock()
    vlx_ts: List[float] = []

    def infer_fn(frame: Any, frame_id: int):
        ts = time.time()
        if dcfg.evidence_enabled:
            try:
                raw_jpeg = encode_jpeg_bytes(frame, quality=EVIDENCE_JPEG_QUALITY)
                with inference_jpegs_lock:
                    inference_jpegs.clear()
                    if raw_jpeg:
                        inference_jpegs[int(frame_id)] = raw_jpeg
            except Exception:  # noqa: BLE001
                LOG.debug("[LMM] 推理帧 %s 的原始证据 JPEG 编码失败", frame_id,
                          exc_info=True)
        if remote_detector:
            #: R2A：把 task_id / frame_id 带进 request_id，跨端可追踪
            res = det.detect(frame, frame_index=frame_id,
                             request_id=_remote_request_id(task_id, frame_id))
        else:
            res = det.track(frame, float(eff.get("confidence", 0.25)),
                            float(eff.get("iou", 0.45)), int(eff.get("imgsz", 640)),
                            str(eff.get("tracker", "botsort")), frame_index=frame_id)
        engine.evaluate(res, frame=frame, now=ts)
        decorated = engine.decorate(res.boxes, frame=frame)
        tm = dict(getattr(res, "timing", None) or {})
        tm.setdefault("total_ms", round(float(getattr(res, "elapsed_ms", 0.0) or 0.0), 1))
        return decorated, tm

    def on_preview(frame_id: int, media_time_ms: float, b64: Optional[str]) -> None:
        if not b64:
            return
        preview = {"type": "preview_frame", "task_id": task_id,
                   "frame_id": frame_id,
                   "media_time_ms": round(media_time_ms, 1),
                   "frame_b64": b64}
        with _LOCK:
            latest = _LAST.setdefault(task_id, {})
            latest.update(preview)
        hub.publish(task_id, {"type": "preview_frame", "task_id": task_id,
                              "frame_id": frame_id,
                              "media_time_ms": round(media_time_ms, 1),
                              "frame_b64": b64})

    # ── ★阶段 B：重型准备全部结束，现在才建立正式实时流★ ──────────────────────
    #: 从这里到 `pipeline.start()` 之间只允许毫秒级的软件初始化。严禁再夹：
    #: 模型加载 / CUDA 初始化 / 网络 health check / tracker 装载 / 大文件读取 / 长 sleep。
    #: 违反任何一条，空窗就会重新长回 10~20 s，MediaMTX 会再一次把会话判死。
    _t_open = None
    if getattr(src, "kind", "") == "stream":
        _t_open = time.monotonic()
        src.open()
        _fw, _fh = (list(src.size) + [0, 0])[:2]
        _ffps = float(src.fps or 0)
        #: 与 probe 值不一致时**只告警，不静默覆盖**：静默覆盖会让后续 detection
        #: 的框继续按旧尺寸归一化 → 画面上的坐标整体错位，而且极难排查。
        if (int(_fw or 0), int(_fh or 0)) != (int(width), int(height)):
            LOG.warning("[LMM] 任务 %s 正式流尺寸与探测值不一致：探测 %d×%d → 实际 %d×%d"
                        "（沿用探测值）", task_id, width, height,
                        int(_fw or 0), int(_fh or 0))
            log_line(task_id, f"正式流尺寸变化：探测 {width}×{height} → "
                              f"实际 {int(_fw or 0)}×{int(_fh or 0)}（沿用探测值）",
                     level="warn")
        #: 帧率只在"两者都非零且差得出来"时告警 —— RTSP 的 SDP 帧率经常是近似值，
        #: 不能拿它当真值去覆盖，也不该为 30.0 vs 30.000001 刷警告。
        if _ffps and src_fps and abs(_ffps - float(src_fps)) > 0.5:
            LOG.warning("[LMM] 任务 %s 正式流帧率与探测值不一致：探测 %.2f → 实际 %.2f"
                        "（沿用探测值）", task_id, float(src_fps), _ffps)

    def on_detection(payload: Dict[str, Any]) -> None:
        # ★旧结果绝不覆盖新结果★：未来多 worker / 网络抖动 / fallback 都可能让
        # 先发的那一帧晚到，顺着写会把新框退回成旧框（画面往回跳）。
        with lock:
            if int(payload.get("frame_id") or 0) < state["last_detection_frame_id"]:
                LOG.info("[LMM] 任务 %s 丢弃过期检测结果 frame_id=%s（已有 %s）",
                         task_id, payload.get("frame_id"), state["last_detection_frame_id"])
                return
            state["last_detection_frame_id"] = int(payload.get("frame_id") or 0)
            state["detections"] += len(payload.get("detections") or [])
            vlx_ts.append(time.time())
            del vlx_ts[:-20]

        #: ★唯一检测链路★：VLX → DetectionManager（**不经过任何 tracker**）
        try:
            dm.on_detections(int(payload.get("frame_id") or 0),
                             float(payload.get("media_time_ms") or 0.0),
                             payload.get("detections") or [],
                             pts_ms=payload.get("pts_ms"),
                             image_width=w_ref[0], image_height=h_ref[0])
        except Exception:  # noqa: BLE001
            LOG.exception("[LMM] 任务 %s DetectionManager 处理失败", task_id)

        snap = pipeline.snapshot()
        tm_timing = dict(payload.get("timing") or {})
        _lat = 0.0
        try:
            _lat = float(tm_timing.get("total_ms") or 0.0)
        except (TypeError, ValueError):
            _lat = 0.0
        st = dm.snapshot_stats()
        msg = {
            "type": "detection",
            "task_id": task_id,
            "frame_id": payload["frame_id"],
            "video_frame_id": snap.get("latest_frame_id"),
            "media_time_ms": payload["media_time_ms"],
            "inference_started_at": payload["inference_started_at"],
            "inference_finished_at": payload["inference_finished_at"],
            "detections": payload["detections"],
            "timing": payload["timing"],
            "width": w_ref[0], "height": h_ref[0],
            "detections_total": state["detections"],
            #: ★新指标★（与 StreamHub 链路同构，前端因此不用分两套）
            "vlx_fps": _rate(vlx_ts),
            "detection_latency_ms": round(_lat, 1),
            "video_fps": round(float(snap.get("capture_fps")
                                     or snap.get("preview_fps") or 0.0), 2),
            "confirmed": int(st.get("confirmed") or 0),
            "alert_count": int(st.get("alerts") or 0),
            "pulse_ms": float(dcfg.pulse_ms),
            "metrics": {"perf": snap},
        }
        with _LOCK:
            # 轮询兜底同时带最近画面与最近检测；预览帧与检测帧独立更新。
            latest = _LAST.setdefault(task_id, {})
            latest.update(msg)
        hub.publish(task_id, msg)
        _set(task_id, processed=snap["processed_frames"],
             detections=state["detections"],
             events_count=_STORE.count(task_id),
             metrics={"perf": snap},
             message=f"实时识别中 · AI {snap['ai_fps'] or 0} fps · "
                     f"告警 {st.get('alerts', 0)} 条")

    pipeline = RealtimePipeline(src=src, fps=src_fps, stop=stop, infer_fn=infer_fn,
                                on_preview=on_preview, on_detection=on_detection,
                                options=opts)
    log_line(task_id, f"实时模式：预览 {opts.preview_fps:g} fps"
                      + ("（按媒体时间匀速采集）" if opts.realtime_pacing else "（不限速）"))
    pipeline.start()

    #: T6：rt-capture 第一次 `src.read()` 成功。
    #: 轮询 `snapshot()` 的 `latest_frame_id`（采集侧每帧更新），精度 ~20 ms ——
    #: 足够验证"正式 open → 首帧"没有再掉回 10~20 s 的老坑。每任务只记一行，低频。
    if _t_open is not None:
        _open_to_start_ms = (time.monotonic() - _t_open) * 1000.0
        _first_frame_ms = None
        for _ in range(250):
            if int((pipeline.snapshot() or {}).get("latest_frame_id") or 0) > 0:
                _first_frame_ms = (time.monotonic() - _t_open) * 1000.0
                break
            if stop.is_set():
                break
            stop.wait(0.02)
        _ff_txt = (f"{_first_frame_ms:.0f} ms" if _first_frame_ms is not None
                   else "5 s 内未读到首帧")
        log_line(task_id, f"开流时序：正式 open→pipeline.start {_open_to_start_ms:.0f} ms，"
                          f"正式 open→首帧 {_ff_txt}")
        LOG.info("[LMM] 任务 %s 开流时序：open→start %.0f ms，open→首帧 %s",
                 task_id, _open_to_start_ms, _ff_txt)

    # 主线程只等结束：capture 读到尾 / 用户停止
    while pipeline.alive and not stop.is_set():
        if pipeline.buffer.queue_depth == 0 and not pipeline.alive:
            break
        stop.wait(0.5)
    pipeline.join(timeout=5.0)
    dm.reset()

    elapsed = time.time() - t0
    snap = pipeline.snapshot()
    _alerts = dm.stats.get("alerts", 0)
    _set(task_id, status=STATUS_STOPPED,
         message=(f"实时识别结束：处理 {snap['processed_frames']} 帧，"
                  f"跳帧 {snap['dropped_frames']}，用时 {elapsed:.1f}s，"
                  f"检出 {state['detections']} 个目标，告警 {_alerts} 条"),
         finished_at=time.time(), processed=snap["processed_frames"],
         detections=state["detections"], events_count=_STORE.count(task_id),
         progress=100.0, metrics={"perf": snap},
         # 实时模式刻意不产出结果视频（低延迟优先）
         result_video=None, result_json=None, save_video=False)
    log_line(task_id,
             f"识别完成：处理 {snap['processed_frames']} 帧，跳帧 {snap['dropped_frames']}，"
             f"用时 {elapsed:.1f}s，检出 {state['detections']} 个目标，告警 {_alerts} 条"
             + ("（已手动停止）" if stop.is_set() else ""),
             level="warn" if stop.is_set() else "ok")


def _run_task(task_id: str) -> None:
    # 刻意**不用 get_task()**：那份是给 HTTP 的脱敏副本（`vlx_seek_token` 会变成
    # `"***"`）。跑推理要的是真实配置，脱敏副本只会让任务在启动瞬间炸掉。
    with _LOCK:
        raw = _TASKS.get(task_id)
    if raw is None:
        return
    task = dict(raw)
    eff = dict(task.get("effective") or {})
    spec = TargetSpec.from_dict(task.get("spec") or {})
    stop = _STOP.get(task_id) or threading.Event()
    src = None
    # ── ★两阶段 Source 生命周期★ ────────────────────────────────────────────
    #: RTSP 是「服务端主动推」模型：`src.open()` 一发出 PLAY，MediaMTX 立刻开始往
    #: 我们这里灌帧。开流到 capture 首次 `read()` 之间一旦空窗过长（实测 ≥ 15 s），
    #: `writeQueueSize:512` 撑满 → `reader is too slow` 开始丢帧 → `writeTimeout:10s`
    #: → 会话被销毁 → 读端永远读空 → `max_empty_frames` 判成断流 → 任务"正常结束"。
    #:
    #: 所以：**先用一个短命的 probe_src 只取元信息就立刻关掉**，重型准备（VLX 体检 /
    #: OSTrack 装载 / 模型初始化）全部做完后，才建立正式流并紧邻 `pipeline.start()`。
    #:
    #: 为什么**重新创建** src 而不是 reopen 同一个对象：`StreamSource.open()` 不会
    #: 先关闭已有的 `_cap`（连调两次会覆盖引用、旧会话要等 GC 才释放），`close()` 也
    #: 不重置 `_empty`，基类契约里没有任何 reopen 语义的保证 —— 不赌生命周期行为，
    #: 用两个互不干扰的对象最稳。
    #:
    #: 只有 `kind == "stream"` 走两阶段：本地文件与摄像头是"拉"模型，不打开就没人推，
    #: 不存在空窗；摄像头额外还有二次 open 的硬件独占/释放风险，不为统一而引入。
    probe_src = None
    #: probe 阶段缓存下来的媒体元信息 —— 正式流打开之前 `_setup_tracking` 就要用
    src_width = 0
    src_height = 0
    src_fps_meta = 0.0
    src_total_frames = 0
    # 在这两行就声明：异常可能发生在写盘分支之前，finally 里要能安全地 release。
    writer = None
    video_path = None
    try:
        #: mode 提前算出来（纯函数，只读 `eff`）：probe 分支必须知道接下来会不会走
        #: 实时路径。**只有实时路径**才分两阶段 —— legacy_sync 是同步逐帧读，
        #: 它本来就要立刻开流，跳过 open() 会让它一帧都读不到。
        mode = _resolve_mode(eff)
        # 多模态任务全部由当前机器本地采集，绕过旧 StreamHub/算力节点分支。
        src = open_source(task.get("source_type", "file"), task.get("source", ""))
        if getattr(src, "kind", "") == "stream" and mode == MODE_REALTIME_LATEST:
            _t1 = time.monotonic()
            probe_src = open_source(task.get("source_type", "file"),
                                    task.get("source", ""))
            try:
                probe_src.open()
                _pw, _ph = (list(probe_src.size) + [0, 0])[:2]
                src_width = int(_pw or 0)
                src_height = int(_ph or 0)
                src_fps_meta = float(probe_src.fps or 0)
                src_total_frames = int(probe_src.total_frames or 0)
            finally:
                #: 探测完**立刻**关：留下任何残留 reader 都会让 MediaMTX 那边
                #: 多一条"只推不读"的会话，正是我们要避开的情形。
                probe_src.close()
            _probe_ms = (time.monotonic() - _t1) * 1000.0
        else:
            src.open()
            _w, _h = (list(src.size) + [0, 0])[:2]
            src_width = int(_w or 0)
            src_height = int(_h or 0)
            src_fps_meta = float(src.fps or 0)
            src_total_frames = int(src.total_frames or 0)
            _probe_ms = 0.0
        log_line(task_id,
                 f"已探测视频：{src_width}×{src_height} @ {float(src_fps_meta or 0):.2f} fps"
                 + (f"，共 {int(src_total_frames)} 帧"
                    if src_total_frames else "（实时流，无总帧数）")
                 + (f"，探测会话 {_probe_ms:.0f} ms" if _probe_ms else ""))
        # 源路径可能被「补全」过（例如用户从资源管理器复制文件名，Windows 默认隐藏 .mp4）
        # → 把实际使用的路径回写，原始输入留在 source_original，便于溯源与排查。
        src_path = getattr(src, "path", None)
        if src_path and str(src_path) != str(task.get("source") or ""):
            LOG.info("[LMM] 任务 %s 源路径已解析：%r → %r",
                     task_id, task.get("source"), str(src_path))
            _set(task_id, source=str(src_path),
                 source_original=str(task.get("source") or ""))

        engine_name = str(eff.get("engine") or "vlm").strip().lower()
        use_remote_detector = engine_name in {"vlm", "qwen-vl", "vlx"}
        interval_key = "vlx_seek_frame_interval" if engine_name == "vlx" else "qwen_interval_seconds"
        try:
            raw_interval = float(eff.get(interval_key, 2.0))
        except (TypeError, ValueError):
            raw_interval = 2.0
        #: 用 probe 阶段缓存的帧率（file/device 分支里就是从已打开的 src 读的，等价）
        src_fps = float(src_fps_meta or 0) or 30.0
        interval = raw_interval if raw_interval > 0 else (1.0 / src_fps if src_fps else 1.0)

        if engine_name in {"vlm", "qwen-vl"}:
            provider_id = str(eff.get("vision_provider_id") or eff.get("qwen_provider_id") or "")
            det = VisionModelEngine(provider_id)
            det.set_prompt(task.get("prompt") or spec.original_prompt)
            interval = max(0.25, raw_interval)
            stride = max(1, int(round(src_fps * interval)))
            detector_label = "vision-model"
            log_line(task_id, f"在线视觉模型 {det.provider.name} 已配置：每 {interval:g}s 发送一帧")
        elif engine_name == "vlx":
            availability = vlx_local_availability()
            if not availability.get("available"):
                raise RuntimeError(str(availability.get("reason") or "VLX-Seek 当前不可用"))
            from .vlx_engine import VLXEngine
            det = VLXEngine()
            det.set_prompt(task.get("prompt") or spec.original_prompt,
                           lang=str(eff.get("vlx_seek_lang") or "zh"),
                           task=str(eff.get("vlx_seek_task") or "detection"),
                           max_new_tokens=int(eff.get("vlx_seek_max_new_tokens") or 256))
            interval = max(0.25, raw_interval)
            stride = max(1, int(round(src_fps * interval)))
            detector_label = "vlx-seek"
            log_line(task_id, f"VLX-Seek 已手动选择：每 {interval:g}s 发送一帧")
        else:
            det = get_detector(task["model_path"], device=eff.get("device", "auto"),
                               half=bool(eff.get("half", False)))
            # 下发的是 observable_labels（目标 + 场景）：场景类别不下发，"在水里"这类
            # 条件就永远无法判定，只能一律标"待确认"。
            det.set_prompts(spec.observable_labels())
            det.reset_tracker()
            det.warmup(int(eff.get("imgsz", 640)))
            stride = max(1, int(eff.get("frame_stride", 1)))
            detector_label = "yoloe"

        # ── R1：模式分支 ──────────────────────────────────────────────────────
        # 必须在**结果视频初始化之前**分流：实时模式不写 mp4（低延迟优先），
        # 若先建了 VideoWriter 再走实时，等于白白占着一个文件句柄。
        # `mode` 已在建源阶段算过（probe 分支要用它决定是否两阶段），这里直接复用。
        _set(task_id, mode=mode)
        if mode == MODE_REALTIME_LATEST:
            _run_realtime_task(task_id, task=task, src=src, det=det, eff=eff,
                               spec=spec, stop=stop, src_fps=src_fps,
                               detector_label=detector_label, remote_detector=use_remote_detector,
                               src_width=src_width, src_height=src_height)
            return

        # ── 结果视频（可选）────────────────────────────────────────────────────
        # 只有本地文件才写：RTSP / 摄像头没有终点，写出来是无上限增长的文件。
        # 为了下载到的视频与原片**同帧率、同时长**，写盘模式改成逐帧读取，只有抽样帧
        # 真推理、其余帧沿用上一次的框 —— 观感上框每 N 秒刷新一次，而不是变成幻灯片。
        writer = None
        video_path = None
        json_path = None
        saved_frames: List[Dict[str, Any]] = []
        last_boxes: List[Dict[str, Any]] = []
        #: 连续出现过滤删掉的框数（JSON 与结束语都要用，先给默认值）
        dropped = 0
        save_video = bool(eff.get("save_video", False)) and bool(src.total_frames)
        min_hits = max(1, int(eff.get("video_min_hits", 1) or 1))
        # 「连续出现次数」过滤必须两遍：写第 k 帧时还不知道这一框后面会不会再出现，
        # 所以第一遍只推理、把框攒下来，第二遍重读视频时才决定画不画。
        two_pass = save_video and min_hits > 1
        if save_video:
            video_path = output_dir() / f"{task_id}_result.mp4"
            json_path = output_dir() / f"{task_id}_detections.json"
        if save_video and not two_pass:
            try:
                import cv2

                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (int(src.size[0]), int(src.size[1])))
                if not writer.isOpened():
                    raise RuntimeError("VideoWriter 打不开")
                LOG.info("[LMM] 任务 %s 结果视频：%s（单遍）", task_id, video_path)
            except Exception:  # noqa: BLE001
                # 写盘失败不该让检测白跑：降级为"不写视频"，但要让任务里看得见原因。
                LOG.warning("[LMM] 任务 %s 无法创建结果视频，改为不落盘", task_id, exc_info=True)
                writer = None
                video_path = json_path = None
                save_video = False

        if save_video:
            log_line(task_id,
                     f"结果视频：原帧率、原时长输出（共 {int(src.total_frames)} 帧）"
                     + (f"；最少连续出现 {min_hits} 帧过滤已开启 → 需第二遍写盘"
                        if two_pass else ""))
        elif eff.get("save_video") and not src.total_frames:
            log_line(task_id, "输入不是本地文件（拿不到总帧数），本次不生成结果视频",
                     level="warn")
        elif eff.get("save_video"):
            log_line(task_id, "结果视频创建失败，已降级为只做识别（原因见后端日志）",
                     level="warn")

        # ── 二次复核**已删除** ─────────────────────────────────────────────────
        # VLX-Seek 自己就是那个"大模型"：region 由它按 Prompt 选，选中即语义结论。
        # 因此不再有 VLM 队列 / 语义缓存 / 三态框，`scheduler=None` 让规则引擎直接
        # 按 hit_frames + 冷却定案（事件里的 verified 为 None，前端按"不涉及复核"渲染）。
        scheduler = None
        engine = RuleEngine(spec, RuleEngineConfig.from_dict(eff), _STORE, task_id,
                            scheduler=None, on_event=_publish_event,
                            semantic_final=use_remote_detector)

        imgsz = int(eff.get("imgsz", 640))
        conf = float(eff.get("confidence", 0.25))
        iou = float(eff.get("iou", 0.45))
        tracker = str(eff.get("tracker", "botsort"))

        # 进度按"抽样后要处理的帧数"算，否则跳 30 帧时进度条永远停在 3%。
        # 写结果视频时进度必须按**全部帧**算：那一遍是真的逐帧读、逐帧写。
        if src.total_frames:
            total = int(src.total_frames) if save_video else int(src.total_frames / stride)
        else:
            total = 0
        _set(task_id, total_frames=total, message="检测中",
             source_size=list(src.size), source_fps=src.fps,
             source_total_frames=src.total_frames or 0,
             sample_stride=stride, sample_interval=interval, detector=detector_label,
             save_video=save_video)
        # `update`（不是 upsert）：多模态页的实时任务没有索引行，不能被这里凭空建一条
        vindex.update(task_id, status=STATUS_RUNNING, total_frames=total,
                      sample_stride=stride, detector=detector_label, source_fps=src_fps)

        # ── R0 性能探针 ────────────────────────────────────────────────────
        # 每段一个 PerfStats（环形采样 + P50/P95）。刻意放在任务内、随任务消亡，
        # 不做模块级全局：两个任务同时跑时统计不会互相污染。
        perf: Dict[str, PerfStats] = {
            "capture": PerfStats(), "encode": PerfStats(), "transport": PerfStats(),
            "inference": PerfStats(), "parse": PerfStats(), "render": PerfStats(),
            "publish": PerfStats(), "total": PerfStats(), "frame_age": PerfStats(),
        }
        #: 实时链路的四件套。R0 阶段流水线还是同步的，所以 dropped / queue_depth
        #: 恒为 0 —— 保留字段是为了让 baseline 与 R1 的报表**字段完全一致**，
        #: 否则 A/B 对比时两边对不上列。
        dropped_frames = 0
        latest_frame_id = 0
        processed_frame_id = 0
        queue_depth = 0

        idx = 0
        t0 = time.time()
        last_push = 0.0
        detections = 0
        recent_ms: List[float] = []
        read_ms_sum = 0.0
        infer_ms_sum = 0.0
        #: 进度日志的下一档（每 10% 一条）。刻意稀疏：逐帧任务日志会迅速被进度刷屏，
        #: 400 行环形缓冲一满，「已写入结果视频」这种真正有用的尾行就会被冲掉。
        next_log_pct = 10.0
        first_infer_logged = False

        while not stop.is_set():
            t_read = time.time()
            frame = src.read(1 if save_video else stride)
            if frame is None:
                break
            capture_ms = (time.time() - t_read) * 1000.0
            read_ms_sum += capture_ms
            idx += 1
            # ── R0：这一帧的身份 ──────────────────────────────────────────
            # frame_id 用**视频里的绝对帧号**（不是"处理过的第几帧"），否则抽样
            # 间隔一改，同一个 frame_id 指的就不是一个画面，A/B 无法对比。
            capture_ts = t_read
            try:
                latest_frame_id = int(src.position) or latest_frame_id
            except Exception:  # noqa: BLE001
                pass
            frame_id = latest_frame_id or idx
            if save_video and (idx - 1) % stride:
                # 未抽样帧：本轮不推理，沿用上一次的框。写成 copy 是因为 draw_boxes
                # 就地改像素，而这一帧马上就要交给 VideoWriter。
                # 两遍模式下 writer 还是 None（第一遍只收集），这里什么都不写。
                if writer is not None:
                    writer.write(draw_boxes(frame.copy(), last_boxes) if last_boxes else frame)
                continue
            ts = time.time()
            if use_remote_detector:
                # 在线 Qwen-VL 直接输出整帧候选框，不走本地跟踪器。
                res = det.detect(frame, frame_index=idx,
                                 request_id=_remote_request_id(task_id, frame_id))
            else:
                res = det.track(frame, conf, iou, imgsz, tracker, frame_index=idx)
            infer_ms_sum += res.elapsed_ms
            #: 远程视觉模型会附带调用耗时；本地 YOLOE 使用空字典。
            _tm: Dict[str, Any] = getattr(res, "timing", None) or {}
            processed_frame_id = frame_id
            detections += len(res.boxes)
            recent_ms.append(res.elapsed_ms)
            if len(recent_ms) > 30:
                recent_ms.pop(0)
            # 首个真推理帧就报一行：逐帧任务的耗时几乎是首帧耗时 × 总帧数，
            # 用户看到 1.2 s 就知道该不该让它跑完，而不是等半小时才发现。
            if not first_infer_logged:
                first_infer_logged = True
                # 全程预估：逐帧任务的耗时 ≈ 首帧耗时 × 总帧数，用户看到这个数才知道
                # 该不该让它跑完（30 帧的测试片说"约 0 分钟"是废话，所以分秒分开写）。
                _eta_s = res.elapsed_ms * total / 1000.0 if total > 1 else 0.0
                _eta_txt = ""
                if _eta_s >= 60:
                    _eta_txt = f"，按此推算全程约 {_eta_s / 60:.0f} 分 {int(_eta_s % 60)} 秒"
                elif _eta_s > 0:
                    _eta_txt = f"，按此推算全程约 {_eta_s:.0f} 秒"
                log_line(task_id,
                         f"首帧推理完成：{res.elapsed_ms:.0f} ms，{len(res.boxes)} 个候选框"
                         + _eta_txt)
            # 主循环只管投递候选（几微秒）；VLM 在 worker 线程里跑，绝不阻塞画面
            engine.evaluate(res, frame=frame, now=ts)
            # 装饰只算一次：实时推送与结果视频共用同一份（含空间格归属）
            decorated = engine.decorate(res.boxes, frame=frame)
            if save_video:
                last_boxes = decorated
                if writer is not None:
                    writer.write(draw_boxes(frame.copy(), decorated))
                saved_frames.append({
                    "frame_index": idx,
                    "time_s": round((idx - 1) / src_fps, 3),
                    "boxes": [{
                        "x1": b.get("x1"), "y1": b.get("y1"),
                        "x2": b.get("x2"), "y2": b.get("y2"),
                        "label": b.get("label"), "display_name": b.get("display_name"),
                    } for b in decorated],
                })

            now = time.time()
            if now - last_push >= _PUSH_INTERVAL:
                last_push = now
                avg = sum(recent_ms) / len(recent_ms) if recent_ms else 0.0
                #: 开发者调试信息（界面上只放在小字 debug 行）：任务类型 / 原文 query /
                #: proposal 数 / 推理耗时。**验证"服务端收到的 query 有没有被改写"就看它**。
                vlx_raw = getattr(res, "vlx", None) or {}
                vlx_debug = {
                    "task": getattr(det, "task", ""),
                    "lang": getattr(det, "lang", ""),
                    "query": getattr(det, "prompt", ""),
                    "proposal_count": vlx_raw.get("proposal_count"),
                    "inference_ms": round(res.elapsed_ms, 1),
                    "server_total_ms": vlx_raw.get("server_total_ms"),
                    # R0：把"编码"和"传输"从推理耗时里摘出来
                    "encode_ms": _tm.get("encode_ms"),
                    "transport_ms": _tm.get("transport_ms"),
                    # ★R2A★ 追踪 ID + 线路开销：前端 debug 行能直接看到"这一帧的
                    # 请求在服务端排了多久队、路上花了多少"，不用翻服务端日志
                    "request_id": _tm.get("request_id"),
                    "wire_overhead_ms": _tm.get("wire_overhead_ms"),
                    "queue_wait_ms": ((vlx_raw.get("server_timing") or {}).get("queue_wait_ms")),
                }
                t_render = time.time()
                frame_b64 = _encode_frame(frame)
                render_ms = (time.time() - t_render) * 1000.0
                payload = {
                    "type": "frame",
                    "task_id": task_id,
                    "frame_index": idx,
                    # R0：视频里的**绝对帧号**（抽样间隔怎么改都指同一个画面）
                    "frame_id": frame_id,
                    "vlx_debug": vlx_debug,
                    "frame_b64": frame_b64,
                    # 三态框 + 两级置信度（detector / VLM 分开）
                    "boxes": decorated,
                    "width": res.width,
                    "height": res.height,
                    "fps": round(1000.0 / avg, 2) if avg > 0 else None,
                    "proc_fps": round(idx / (now - t0), 2) if now > t0 else None,
                    "elapsed_ms": round(avg, 1),
                    "processed": idx,
                    "total_frames": total,
                    "progress": round(idx / total * 100, 1) if total else None,
                    "detections": detections,
                    "events_count": _STORE.count(task_id),
                    "status": STATUS_RUNNING,
                    # 规则侧的可观测状态：哪些条件真的在生效、有多少候选因缺少
                    # track_id 被当成"空间格伪轨迹"处理（VLX-Seek 没有跟踪器，这一项
                    # 会持续增长，属正常；它不是"复核被跳过"，而是"用格子累计命中"）
                    "rule": engine.describe(),
                    "metrics": _metrics(idx, now - t0, read_ms_sum, infer_ms_sum,
                                        detections, detector_label),
                }
                t_pub = time.time()
                with _LOCK:
                    _LAST[task_id] = payload
                hub.publish(task_id, payload)
                publish_ms = (time.time() - t_pub) * 1000.0

                # ── R0：这一帧的完整计时 ──────────────────────────────────
                # 必须在 publish **之后**收尾：frame_age 的定义就是「发布时刻 −
                # 采集时刻」，publish 之前算会少掉推送本身的耗时。
                # 队列里放的是 payload 引用，WS 协程稍后才序列化，所以这里回填
                # 的 publish_ms / frame_age 来得及被发出去。
                ft = FrameTiming(
                    frame_id=frame_id, capture_timestamp=capture_ts,
                    capture_ms=capture_ms,
                    encode_ms=float(_tm.get("encode_ms") or 0.0),
                    transport_ms=_tm.get("transport_ms"),
                    inference_ms=_tm.get("inference_ms"),
                    parse_ms=float(_tm.get("parse_ms") or 0.0),
                    render_ms=render_ms, publish_ms=publish_ms,
                )
                ft.finish(publish_ts=time.time())
                payload["timing"] = ft.to_dict()
                for _k, _v in (("capture", ft.capture_ms), ("encode", ft.encode_ms),
                               ("transport", ft.transport_ms), ("inference", ft.inference_ms),
                               ("parse", ft.parse_ms), ("render", ft.render_ms),
                               ("publish", ft.publish_ms), ("total", ft.total_ms),
                               ("frame_age", ft.frame_age_ms)):
                    perf[_k].add(_v)
                payload["metrics"]["perf"] = _perf_block(
                    perf, dropped_frames=dropped_frames,
                    latest_frame_id=latest_frame_id,
                    processed_frame_id=processed_frame_id, queue_depth=queue_depth,
                    elapsed_s=(now - t0), src_fps=src_fps,
                    # legacy 同步链路里"采集"是被动的：每次循环 read(stride) 推进
                    # stride 帧，所以真实采集帧数 = 已处理抽样帧 × stride。
                    capture_frames=idx * stride, processed_frames=idx,
                    mode="legacy_sync")

                _set(task_id, progress=payload["progress"] or 0.0, processed=idx,
                     detections=detections, events_count=payload["events_count"],
                     fps=payload["fps"], proc_fps=payload["proc_fps"],
                     elapsed_ms=payload["elapsed_ms"],
                     metrics=payload["metrics"],
                     message=f"检测中 {idx}" + (f"/{total}" if total else "") + " 帧")

                pct = float(payload["progress"] or 0.0)
                if pct >= next_log_pct:
                    while next_log_pct <= pct:
                        next_log_pct += 10.0
                    remain_s = ((total - idx) * (now - t0) / idx) if (total and idx) else 0.0
                    remain_txt = ""
                    if remain_s >= 60:
                        remain_txt = f" · 预计还需 {int(remain_s // 60)} 分 {int(remain_s % 60)} 秒"
                    elif remain_s > 0:
                        remain_txt = f" · 预计还需 {remain_s:.0f} 秒"
                    log_line(task_id,
                             f"已处理 {idx}/{total} 帧（{pct:.0f}%）· 检出 {detections} 个目标"
                             f" · 单帧 {avg:.0f} ms" + remain_txt)

        elapsed = time.time() - t0
        # 收尾也要带 perf：任务跑完停在最后一帧上，前端常常是这时候才去看指标
        _final_metrics = _metrics(idx, elapsed, read_ms_sum, infer_ms_sum,
                                  detections, detector_label)
        _final_metrics["perf"] = _perf_block(
            perf, dropped_frames=dropped_frames, latest_frame_id=latest_frame_id,
            processed_frame_id=processed_frame_id, queue_depth=queue_depth,
            elapsed_s=elapsed, src_fps=src_fps, capture_frames=idx * stride,
            processed_frames=idx, mode="legacy_sync")

        # ── 两遍模式：先过滤零星框，再重读一遍视频写盘 ──────────────────────────
        if two_pass:
            # 过滤只作用于**视频**：检出数据（JSON）保持原样，方便事后核对"模型到底给了什么"。
            # 所以这里过的是副本，`saved_frames` 仍是未过滤的原始结果。
            video_frames = [{**s, "boxes": [dict(b) for b in (s.get("boxes") or [])]}
                            for s in saved_frames]
            dropped = _filter_sporadic(video_frames, min_hits, src.size)
            _set(task_id, message="写结果视频…",
                 sporadic_dropped=dropped, video_min_hits=min_hits)
            LOG.info("[LMM] 任务 %s 连续出现过滤：删掉 %d 个零星框（min_hits=%d）",
                     task_id, dropped, min_hits)
            src2 = None
            try:
                import cv2

                src2 = open_source(task.get("source_type", "file"), task.get("source", ""))
                src2.open()
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (int(src.size[0]), int(src.size[1])))
                if not writer.isOpened():
                    raise RuntimeError("VideoWriter 打不开")
                si = 0
                cur: List[Dict[str, Any]] = []
                out_idx = 0
                while True:
                    f = src2.read(1)
                    if f is None:
                        break
                    out_idx += 1
                    # 该帧落在这个抽样结果之后 → 换用它的框（两遍的映射必须与第一遍一致）
                    if si < len(video_frames) and video_frames[si]["frame_index"] <= out_idx:
                        cur = video_frames[si]["boxes"]
                        si += 1
                    writer.write(draw_boxes(f.copy(), cur) if cur else f)
            except Exception:  # noqa: BLE001
                LOG.warning("[LMM] 任务 %s 第二遍写盘失败", task_id, exc_info=True)
                video_path = None
            finally:
                if src2 is not None:
                    try:
                        src2.close()
                    except Exception:
                        pass
            log_line(task_id,
                     f"第二遍写盘完成：按「最少连续出现 {min_hits} 帧」过滤掉 {dropped} 个零星框"
                     if dropped else
                     f"第二遍写盘完成：没有零星框达到过滤条件（阈值 {min_hits} 帧）")

        # 收尾落盘：视频句柄必须 release，否则 MP4 的索引写不进去、播放器打不开
        written = 0
        if writer is not None:
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                LOG.debug("[LMM] 结果视频释放失败", exc_info=True)
            writer = None
            try:
                # 小测试片只有几百 KB，一律按 MB 写会显示成 "0.0 MB"，看着像写空了
                _size = float(video_path.stat().st_size)
                _size_txt = (f"{_size / 1048576.0:.1f} MB" if _size >= 1048576
                             else f"{_size / 1024.0:.0f} KB")
                log_line(task_id,
                         f"结果视频已写入：{video_path.name}（{int(src.total_frames or idx)} 帧"
                         f" / {src_fps:.2f} fps / {_size_txt}）")
            except OSError:
                log_line(task_id, "结果视频已释放，但读取文件大小失败（文件可能不可用）",
                         level="warn")
            try:
                json_path.write_text(json.dumps({
                    "task_id": task_id,
                    "prompt": task.get("prompt"),
                    "source": task.get("source"),
                    "detector": detector_label,
                    "fps": src_fps,
                    "total_frames": src.total_frames or 0,
                    "sample_interval_s": interval,
                    "sample_stride": stride,
                    "processed_frames": idx,
                    "detections": detections,
                    # 结果视频的过滤口径：1 = 不过滤；>1 时视频只画连续出现够次数的框，
                    # 但下面的 sampled 仍是**未过滤**的原始检出（数据不被视频口径污染）。
                    "video_min_hits": min_hits,
                    "sporadic_dropped": dropped,
                    # 只记**真推理过**的帧：未抽样帧的框是从上一帧续用的，写进来会
                    # 让人误以为模型每一帧都算过。
                    "sampled": saved_frames,
                    "events": _STORE.list(task_id, limit=500),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                written = len(saved_frames)
                log_line(task_id,
                         f"检出数据已写入：{json_path.name}（{written} 个抽样帧的原始检出）")
            except Exception:  # noqa: BLE001
                LOG.warning("[LMM] 任务 %s 检出数据写入失败", task_id, exc_info=True)
                json_path = None
                log_line(task_id, "检出数据写入失败（结果视频不受影响）", level="warn")
        sample_note = "逐帧识别" if stride == 1 else f"每 {interval:g}s 抽样 1 帧"
        _set(task_id, status=STATUS_STOPPED,
             # 复核已删除 → 结束语里不再提 VLM 次数，改报"每帧平均推理耗时"
             message=(f"已处理 {idx} 帧（{sample_note}），用时 {elapsed:.1f}s，"
                      f"检出 {detections} 个目标"),
             finished_at=time.time(), processed=idx, detections=detections,
             events_count=_STORE.count(task_id), progress=100.0 if total else 0.0,
             rule_state=engine.describe(),
             metrics=_final_metrics,
             # 只有真的写出来才把路径交出去：失败时前端只看到"没有下载入口"，
             # 而不是一个点开就 404 的按钮。
             result_video=(str(video_path) if video_path and video_path.is_file() else None),
             result_json=(str(json_path) if json_path and json_path.is_file() else None),
             sampled_frames=written)
        # 落盘索引的收尾：结果文件的**路径**也要存 —— 后端重启后内存里的任务没了，
        # 但「下载结果视频」还得能用（router 会回落到这里取路径）。
        vindex.update(task_id,
                      status=STATUS_STOPPED,
                      processed=idx, total_frames=total, detections=detections,
                      finished_at=time.time(),
                      result_video=(str(video_path) if video_path and video_path.is_file() else None),
                      result_json=(str(json_path) if json_path and json_path.is_file() else None),
                      sporadic_dropped=dropped, video_min_hits=min_hits,
                      stopped_by_user=bool(stop.is_set()))
        log_line(task_id,
                 "识别完成：已处理 "
                 + (f"{idx}/{total} 帧" if total else f"{idx} 帧")
                 + f"，用时 {elapsed:.1f}s，检出 {detections} 个目标"
                 + ("（已手动停止，结果视频只含已处理部分）" if stop.is_set() else ""),
                 level="warn" if stop.is_set() else "ok")
        LOG.info("[LMM] 任务 %s 结束：%d 帧 / %d 检出 / %d 事件 / 检测器 %s",
                 task_id, idx, detections, _STORE.count(task_id), detector_label)
    except (SourceError, DetectorError) as exc:
        LOG.exception("[LMM] 任务 %s 失败", task_id)
        log_line(task_id, f"任务失败：{exc}", level="error")
        vindex.update(task_id, status=STATUS_FAILED, message=str(exc),
                      finished_at=time.time())
        _set(task_id, status=STATUS_FAILED, message=str(exc), error=str(exc),
             finished_at=time.time())
    except Exception as exc:
        LOG.exception("[LMM] 任务 %s 异常", task_id)
        log_line(task_id, f"任务异常终止：{exc}", level="error")
        vindex.update(task_id, status=STATUS_FAILED, message=f"任务异常：{exc}",
                      finished_at=time.time())
        _set(task_id, status=STATUS_FAILED, message=f"任务异常：{exc}", error=str(exc),
             finished_at=time.time())
    finally:
        if writer is not None:
            # 异常路径也必须 release，否则文件句柄泄漏、mp4 尾部索引缺失
            try:
                writer.release()
            except Exception:
                pass
        #: probe_src 正常情况下在它自己的 try/finally 里已经关了；这里再兜一次底，
        #: 保证任何异常路径都不会留下残留的 RTSP reader / MediaMTX 会话 / socket。
        if probe_src is not None:
            try:
                probe_src.close()
            except Exception:
                pass
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
        _cleanup(task_id)
        with _LOCK:
            _THREADS.pop(task_id, None)
            _STOP.pop(task_id, None)


# ── 查询 ─────────────────────────────────────────────────────────────────────


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        t = _TASKS.get(task_id)
        if t is None:
            return None
        out = dict(t)
    out["status_zh"] = STATUS_ZH.get(out.get("status"), out.get("status"))
    out["events_count"] = _STORE.count(task_id)
    # 脱敏：这两个字段是整份 lmm_config，含 vlx_seek_token
    for key in ("effective", "config_snapshot"):
        if isinstance(out.get(key), dict):
            out[key] = _redact(out[key])
    return out


def list_tasks() -> List[Dict[str, Any]]:
    with _LOCK:
        ids = list(_TASKS.keys())
    out = [get_task(i) for i in ids]
    return [t for t in out if t]


def delete_task(task_id: str) -> bool:
    stop_task(task_id)
    with _LOCK:
        _TASKS.pop(task_id, None)
        _LAST.pop(task_id, None)
        _LOGS.pop(task_id, None)
    _STORE.clear(task_id)
    vindex.remove(task_id)
    return True


def snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    """轮询兜底用：拿最近一帧（含检测框与统计）。"""
    with _LOCK:
        p = _LAST.get(task_id)
        return dict(p) if p else None


def events(task_id: str, since_ts: Optional[float] = None,
           limit: int = 100) -> List[Dict[str, Any]]:
    return _STORE.list(task_id, since_ts=since_ts, limit=limit)


def events_count(task_id: str) -> int:
    return _STORE.count(task_id)


def engine_state(task_id: str) -> Optional[Dict[str, Any]]:
    """给前端展示"哪些条件真的在生效"。运行期才有实例，否则回退到 spec 静态描述。"""
    task = get_task(task_id)
    if task is None:
        return None
    return {"labels": task.get("labels") or [],
            "conditions": (task.get("spec") or {}).get("conditions", [])}
