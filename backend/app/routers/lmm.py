"""LMM（多模态目标检测）HTTP 接口。

路由风格**刻意沿用项目现有约定**：`APIRouter()` 不带 prefix，装饰器写全路径，
前缀统一 `/api/lmm/*`。

为什么不叫 `/api/vlm/*`：该前缀已被「大模型标注」（`routers/vlm.py`，OpenAI 兼容视觉
大模型做自动标注）占用，两者职责不同（前者是逐帧实时检测，后者是离线打标签）。

第一阶段已实现：
    GET  /api/lmm/model-status      YOLOE 权重 / 文本编码器就绪状态
    GET  /api/lmm/config            可调参数读取
    POST /api/lmm/config            可调参数写入
    POST /api/lmm/tasks                 创建任务（用户原文直通，无 Prompt 解析）
    GET  /api/lmm/tasks                 任务列表
    GET  /api/lmm/tasks/{id}            任务详情
    DELETE /api/lmm/tasks/{id}          删除任务
    POST /api/lmm/tasks/{id}/start      开始检测
    POST /api/lmm/tasks/{id}/stop       停止检测
    GET  /api/lmm/tasks/{id}/events     告警事件（?since=<ts> 增量拉取）
    GET  /api/lmm/tasks/{id}/logs       运行日志（面向用户，?since=<ts> 增量拉取）
    GET  /api/lmm/tasks/{id}/snapshot   最近一帧（轮询兜底）
    WS   /api/lmm/ws/tasks/{id}         实时帧推送（前端断线自动降级为 snapshot 轮询）

结果下载（推理页「视频 + 大模型」用）：
    POST /api/lmm/video-tasks           上传视频 → 建任务 → 启动（一次请求）
    GET  /api/lmm/video-tasks           「视频推理记录」（含重启后仍在的历史）
    GET  /api/lmm/tasks/{id}/result-video  结果视频（检出框已画好）
    GET  /api/lmm/tasks/{id}/result-json   检出数据（只记真推理过的抽样帧）

Prompt 链路（2026-09-23 起）
--------------------------
`POST /api/lmm/parse-prompt` 与 `GET /api/lmm/prompt-providers` **已删除**，
`lmm/prompt_analyzer.py` 一并删除：把「寻找水面上未穿救生衣的人员」拆成
`person / swimmer / water / life_jacket` 会丢掉这些词之间的语义关系；在线视觉检测
引擎直接消费整句中文。用户输入的字符串从接口到服务端**逐字不变**。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from config import settings
from core.nexushive import token_from_request, verify_session
from lmm import config as lmm_config
from lmm import hub, registry
from lmm.detectors.base_detector import DetectorError
from lmm.qwen_engine import (
    VisionModelEngine,
    configured_vision_models,
    online_qwen_models,
    online_vision_models,
)
from lmm import task_manager as tm
from lmm import video_index as vi

LOG = logging.getLogger(__name__)

router = APIRouter()


# ── 请求体 ───────────────────────────────────────────────────────────────────


class ConfigPatch(BaseModel):
    """可调参数。只提交要改的键，未提交的保持原值。"""

    model_config = {"extra": "allow"}


class VlxConnectionConfigPatch(BaseModel):
    """保存 VLX 接入参数；本接口只读写本地配置，不探测或请求 VLX。"""

    transport: Literal["ssh", "direct"] | None = None
    node_id: str | None = Field(default=None, max_length=128)
    port: int | None = Field(default=None, ge=1, le=65535)
    base_url: str | None = Field(default=None, max_length=2048)
    token: SecretStr | None = None
    timeout: float | None = Field(default=None, gt=0, le=1800)
    task: Literal["detection", "reasoning_detection", "grounding_single", "counting"] | None = None
    lang: Literal["zh", "en"] | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=8192)
    frame_interval: float | None = Field(default=None, ge=0.1, le=3600)
    extra_instruction: str | None = Field(default=None, max_length=2048)


class CreateTaskRequest(BaseModel):
    """创建检测任务。

    `prompt` 是**发给所选检测引擎的原文**，不做切词、不做翻译、不调用解析模型。
    这里刻意不再提供 `detection_labels` / `provider` / `ai_config_id` —— 它们只在
    「先解析成关键词」的旧链路里有用，留着就是给语义降级留后门。
    """

    prompt: str = Field(default="", description="自然语言目标描述（原样下发）")
    source_type: str = Field(default="file", description="file | rtsp | camera")
    source: str = Field(default="", description="视频路径 / RTSP 地址 / 摄像头索引")
    #: 任务级参数覆盖（imgsz / confidence / iou / tracker / frame_stride / hit_frames
    #: / cooldown_seconds / min_targets / half / device）
    options: Optional[Dict[str, Any]] = None


class RecoverEvidenceRequest(BaseModel):
    """Context retained by the browser to recover a visible alert after restart."""

    source: str = ""
    event: Optional[Dict[str, Any]] = None


class VisionInferenceFrameRequest(BaseModel):
    """Single browser-captured frame for a configured visual model."""

    provider_id: str
    prompt: str
    frame_b64: str
    confidence: float = 0.25


# ── 模型与配置 ─�────────────────────────────────────────────────────────────


# ── 「视觉大模型复核」相关接口（vlm-models / vlm-probe / reverify / vlm-stats）已删除 ──
# 语义判定由 VLX-Seek 一次完成，不再有独立的二阶段复核链路与语义缓存。


@router.get("/api/lmm/model-status")
def model_status() -> Dict[str, Any]:
    """YOLOE 权重与文本编码器是否就绪。未就绪时前端提示"模型未安装"。"""
    cfg = lmm_config.load()
    status = registry.model_status(cfg)
    status["device_setting"] = cfg.get("device")
    status["half"] = cfg.get("half")
    return status


@router.get("/api/lmm/config")
def get_config() -> Dict[str, Any]:
    return {"config": lmm_config.load(), "defaults": lmm_config.DEFAULTS,
            "config_file": str(lmm_config.CONFIG_FILE)}


def _vlx_connection_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return sanitized saved VLX metadata only; never probe or expose tokens."""
    transport = str(cfg.get("vlx_seek_transport") or "ssh").strip().lower()
    node_id = str(cfg.get("vlx_seek_node_id") or "")
    base_url = str(cfg.get("vlx_seek_base_url") or "")
    token_configured = bool(cfg.get("vlx_seek_token"))
    has_target = bool(node_id and cfg.get("vlx_seek_port")) if transport == "ssh" else bool(base_url) if transport == "direct" else False
    return {"configured": bool(has_target and token_configured),
            "enabled": bool(cfg.get("vlx_seek_enabled", False)),
            "transport": transport, "node_id": node_id,
            "port": int(cfg.get("vlx_seek_port") or 8021),
            "base_url": base_url, "token_configured": token_configured,
            "timeout": float(cfg.get("vlx_seek_timeout") or 300),
            "task": cfg.get("vlx_seek_task") or "detection",
            "lang": cfg.get("vlx_seek_lang") or "zh",
            "max_new_tokens": int(cfg.get("vlx_seek_max_new_tokens") or 256),
            "frame_interval": float(cfg.get("vlx_seek_frame_interval") or 1.0),
            "extra_instruction": cfg.get("vlx_seek_extra_instruction") or ""}


@router.get("/api/lmm/vlx/config")
def get_vlx_connection_config() -> Dict[str, Any]:
    return {"config": _vlx_connection_config(lmm_config.load())}

@router.post("/api/lmm/vlx/config")
def save_vlx_connection_config(patch: VlxConnectionConfigPatch) -> Dict[str, Any]:
    """Save VLX connection metadata only; never probes, calls, or starts VLX."""
    submitted = patch.model_dump(exclude_unset=True, exclude={"token"})
    mapping = {
        "transport": "vlx_seek_transport",
        "node_id": "vlx_seek_node_id",
        "port": "vlx_seek_port",
        "base_url": "vlx_seek_base_url",
        "timeout": "vlx_seek_timeout",
        "task": "vlx_seek_task",
        "lang": "vlx_seek_lang",
        "max_new_tokens": "vlx_seek_max_new_tokens",
        "frame_interval": "vlx_seek_frame_interval",
        "extra_instruction": "vlx_seek_extra_instruction",
    }
    config_patch = {mapping[key]: value for key, value in submitted.items() if key in mapping}
    secret = patch.token if "token" in patch.model_fields_set else None
    if secret is not None and secret.get_secret_value():
        config_patch["vlx_seek_token"] = secret.get_secret_value()
    try:
        cfg = lmm_config.save(config_patch)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("[LMM] 写入 VLX 接入参数失败")
        raise HTTPException(status_code=500, detail="写入 VLX 接入参数失败") from exc
    return {"config": _vlx_connection_config(cfg)}


@router.post("/api/lmm/config")
def update_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    """更新可调参数（模型 / imgsz / confidence / IoU / tracker / frame_stride /
    连续命中帧 / 告警冷却 / 数量阈值 …）。未知键会被忽略。"""
    if not isinstance(patch, dict):
        raise HTTPException(status_code=400, detail="请求体必须是对象")
    try:
        cfg = lmm_config.save(patch)
    except Exception as exc:
        LOG.exception("[LMM] 写入配置失败")
        raise HTTPException(status_code=500, detail=f"写入配置失败：{exc}") from exc
    # 节点 / 端口 / 传输方式可能变了 —— 旧隧道必须丢掉，否则会连到上一个节点去。
    try:
        from lmm.vlx_client import VLXSeekClient, drop_persistent
        VLXSeekClient.drop_tunnels()
        #: ★R2B★ 持久 HTTP 连接握着的是**旧隧道上的 channel**，只丢隧道不丢它的话，
        #: 复用路径会继续往已经作废的节点发请求（而且本地健康检查看不出来）。
        drop_persistent()
    except Exception:  # noqa: BLE001
        pass
    return {"config": cfg, "config_file": str(lmm_config.CONFIG_FILE)}


# ── 任务 ─────────────────────────────────────────────────────────────────────


def _require(task_id: str) -> Dict[str, Any]:
    task = tm.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.post("/api/lmm/tasks")
def create_task(req: CreateTaskRequest) -> Dict[str, Any]:
    """Create a local-capture task using online Qwen-VL or local YOLOE.

    输入源非法等情况下任务仍会创建，但 status=failed 且带可读原因，
    前端直接展示即可（不吞错误）。
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt 为空")
    if not (req.source or "").strip():
        raise HTTPException(status_code=400, detail="未指定输入源")
    if req.source_type=='file':
        from media_upload import owned_file
        owned_file(req.source,settings.WORKSPACE_DIR)
    try:
        task = tm.create_task(prompt, req.source_type, req.source, options=req.options)
    except (ValueError, DetectorError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    LOG.info("[LMM] 创建任务 %s：%s", task["id"], prompt)
    return task


@router.get("/api/lmm/qwen-models")
def qwen_models() -> Dict[str, Any]:
    """List saved online Qwen-VL providers without exposing endpoints or secrets."""
    providers = online_qwen_models()
    configured_default = str(lmm_config.load().get("qwen_provider_id") or "")
    valid_ids = {item["config_id"] for item in providers}
    default_id = configured_default if configured_default in valid_ids else ""
    return {"providers": providers, "default_config_id": default_id}


@router.get("/api/lmm/vision-models")
def vision_models() -> Dict[str, Any]:
    """Use the same saved model list for all visual-model selection pages; never probe."""
    return configured_vision_model_list()


@router.get("/api/lmm/configured-vision-models")
def configured_vision_model_list() -> Dict[str, Any]:
    """List all ready saved visual models, including providers hosted locally."""
    providers = configured_vision_models()
    cfg = lmm_config.load()
    vlx = _vlx_connection_config(cfg)
    if vlx.get("configured") and not any(p.get("config_id") == "vlx-seek" for p in providers):
        providers.append({"config_id": "vlx-seek", "display_name": "VLX-Seek",
                         "model": "VLX-Seek-1.5-10B", "kind_label": "Dedicated VLX API"})
    configured_default = str(cfg.get("vision_provider_id") or cfg.get("qwen_provider_id") or "")
    valid_ids = {item["config_id"] for item in providers}
    default_id = configured_default if configured_default in valid_ids else ""
    if not default_id and providers:
        default_id = providers[0]["config_id"]
    return {"providers": providers, "default_config_id": default_id}


@router.get("/api/lmm/vlx/availability")
def vlx_availability() -> Dict[str, Any]:
    """Local-only routing check; unlike /health, this never probes the VLX service."""
    from lmm.vlx_client import local_availability
    return local_availability()


def _run_configured_vision_inference(
    payload: bytes,
    provider_id: str,
    prompt: str,
    confidence: float,
    *,
    base64_encoded: bool = False,
) -> Dict[str, Any]:
    """Dispatch an explicitly selected saved model using its own protocol."""
    import base64
    import cv2
    import numpy as np

    if not payload:
        raise ValueError("图片内容为空")
    if base64_encoded:
        encoded = payload.decode("ascii", errors="ignore") if isinstance(payload, bytes) else str(payload)
        if encoded.startswith("data:") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("图片编码无效") from exc
    else:
        raw = payload
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法读取图片")

    if provider_id == "vlx-seek":
        if not _vlx_connection_config(lmm_config.load()).get("configured"):
            raise DetectorError("VLX-Seek 接入参数未配置完整")
        from lmm.vlx_engine import VLXEngine
        engine = VLXEngine()
    else:
        engine = VisionModelEngine(provider_id, allow_local=True)
    engine.set_prompt(prompt)
    result = engine.detect(frame)
    # Legacy confidence parameter is ignored for unscored visual-language models.
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in (box.x1, box.y1, box.x2, box.y2)]
        label = str(box.label_zh or box.label or prompt)
        score = float(box.confidence)
        detections.append({
            "class_name": label,
            "confidence": score,
            "confidence_source": "synthetic",
            "bbox": [x1, y1, x2, y2],
        })
        cv2.rectangle(frame, (x1, y1), (x2, y2), (37, 99, 235), 2)
        # Chinese labels are shown in the adjacent list. Do not paint synthetic scores.

    ok, encoded_image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("推理结果图片编码失败")
    return {
        "annotated_b64": base64.b64encode(encoded_image.tobytes()).decode("ascii"),
        "detections": detections,
        "inference_ms": result.elapsed_ms,
    }


def _configured_vision_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DetectorError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    LOG.error("Configured visual-model inference failed: %s", exc,
              exc_info=(type(exc), exc, exc.__traceback__))
    return HTTPException(status_code=502, detail=f"视觉大模型推理失败：{exc}")


@router.post("/api/lmm/vision/infer-image")
async def configured_vision_infer_image(
    image: UploadFile = File(..., description="待识别图片"),
    provider_id: str = Form(..., description="已保存的视觉大模型配置 ID"),
    prompt: str = Form(..., description="自然语言目标描述"),
    confidence: float = Form(default=0.25),
) -> Dict[str, Any]:
    clean_prompt = (prompt or "").strip()
    if not provider_id.strip():
        raise HTTPException(status_code=400, detail="请先选择视觉大模型")
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="请输入目标描述")
    payload = await image.read()
    try:
        return await asyncio.to_thread(
            _run_configured_vision_inference, payload, provider_id.strip(), clean_prompt, confidence)
    except (DetectorError, ValueError) as exc:
        raise _configured_vision_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise _configured_vision_error(exc) from exc


@router.post("/api/lmm/vision/infer-frame")
async def configured_vision_infer_frame(req: VisionInferenceFrameRequest) -> Dict[str, Any]:
    clean_prompt = (req.prompt or "").strip()
    if not req.provider_id.strip():
        raise HTTPException(status_code=400, detail="请先选择视觉大模型")
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="请输入目标描述")
    try:
        return await asyncio.to_thread(
            _run_configured_vision_inference,
            req.frame_b64.encode("ascii", errors="ignore"),
            req.provider_id.strip(), clean_prompt, req.confidence, base64_encoded=True)
    except (DetectorError, ValueError) as exc:
        raise _configured_vision_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise _configured_vision_error(exc) from exc


@router.get("/api/lmm/tasks")
def list_tasks() -> Dict[str, Any]:
    return {"tasks": tm.list_tasks()}


@router.get("/api/lmm/recent-videos")
def recent_videos(limit: int = 8) -> Dict[str, Any]:
    """把推理页用过的视频列出来，省得用户手抄绝对路径。

    只看 `workspace/temp/inference/*_input.mp4`（`*_result.mp4` 是上一次的输出，
    拿它当输入会递归放大），按修改时间倒序。
    """
    try:
        root = Path(settings.TEMP_DIR) / "inference"
        if not root.is_dir():
            return {"videos": []}
        files = [p for p in root.glob("*_input.mp4") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return {"videos": [str(p) for p in files[: int(limit)]]}
    except Exception as exc:
        LOG.debug("[LMM] 列举最近视频失败", exc_info=True)
        return {"videos": [], "error": str(exc)}


@router.get("/api/lmm/tasks/{task_id}")
def get_task(task_id: str) -> Dict[str, Any]:
    return _require(task_id)


@router.delete("/api/lmm/tasks/{task_id}")
def delete_task(task_id: str) -> Dict[str, Any]:
    _require(task_id)
    tm.delete_task(task_id)
    return {"ok": True, "id": task_id}


@router.post("/api/lmm/tasks/{task_id}/start")
def start_task(task_id: str) -> Dict[str, Any]:
    task = _require(task_id)
    if task.get('source_type')=='file':
        from media_upload import owned_file
        owned_file(task.get('source',''),settings.WORKSPACE_DIR)
    if task["status"] == tm.STATUS_FAILED and not task.get("labels"):
        raise HTTPException(status_code=400, detail=task.get("message") or "任务不可用")
    try:
        return tm.start_task(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/lmm/tasks/{task_id}/stop")
def stop_task(task_id: str) -> Dict[str, Any]:
    _require(task_id)
    return tm.stop_task(task_id)


@router.post("/api/lmm/tasks/{task_id}/alerts/{alert_id}/recover-evidence")
def recover_alert_evidence(task_id: str, alert_id: str,
                           req: Optional[RecoverEvidenceRequest] = None) -> Dict[str, Any]:
    """Restore a failed local-video alert from its exact source frame, without AI calls."""
    from media_upload import owned_file
    if req and req.source:owned_file(req.source,settings.WORKSPACE_DIR)
    existing=tm.get_task(task_id)
    if existing and existing.get('source_type')=='file':owned_file(existing.get('source',''),settings.WORKSPACE_DIR)
    try:
        if tm.get_task(task_id) is not None:
            event = tm.recover_failed_evidence_from_local_video(task_id, alert_id)
        elif req and req.event:
            event = tm.recover_orphaned_alert_from_local_video(
                task_id=task_id, alert_id=alert_id, source_path=req.source,
                event=req.event,
            )
        else:
            raise KeyError(task_id)
    except KeyError as exc:
        if not (req and req.event):
            raise HTTPException(status_code=404, detail="告警或任务不存在") from exc
        try:
            event = tm.recover_orphaned_alert_from_local_video(
                task_id=task_id, alert_id=alert_id, source_path=req.source,
                event=req.event,
            )
        except ValueError as recover_exc:
            raise HTTPException(status_code=400, detail=str(recover_exc)) from recover_exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOG.exception("[LMM] 本地告警证据恢复失败 task=%s alert=%s", task_id, alert_id)
        raise HTTPException(status_code=500, detail="恢复原始帧失败，请检查本地视频是否仍可读取") from exc
    return {"event": event}


@router.get("/api/lmm/tasks/{task_id}/events")
def task_events(task_id: str, since: Optional[float] = None, limit: int = 100) -> Dict[str, Any]:
    """告警事件。传 `since`（上次拿到的最大 ts）即可增量拉取。"""
    _require(task_id)
    return {"events": tm.events(task_id, since_ts=since, limit=limit),
            "count": tm.events_count(task_id)}


@router.get("/api/lmm/tasks/{task_id}/logs")
def task_logs(task_id: str, since: Optional[float] = None, limit: int = 400) -> Dict[str, Any]:
    """任务的**面向用户**运行日志（推理页「识别日志」面板）。

    传 `since`（上次拿到的最大 ts）即为增量拉取，前端只做 append —— 这样进度日志
    即使在轮询间隙里成对产生也不会乱序。与事件接口分开：事件是告警，日志是过程。

    **刻意不用 `_require`**：后端重启后任务已不在内存里，但日志在落盘索引里 ——
    这正是"记录要能对上"的关键，此处报 404 就等于把历史日志藏起来。
    """
    lines = tm.log_lines(task_id, since_ts=since, limit=limit)
    if not lines and tm.get_task(task_id) is None and vi.get(task_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"logs": lines, "count": len(lines)}


@router.get("/api/lmm/tasks/{task_id}/snapshot")
def task_snapshot(task_id: str) -> Dict[str, Any]:
    """轮询兜底：最近一帧 + 检测框 + 统计。WS 不可用时前端改用它。"""
    task = _require(task_id)
    snap = tm.snapshot(task_id)
    if snap is None:
        return {"task": task, "frame_b64": None, "boxes": [], "events": []}
    return {"task": task, "events": tm.events(task_id, limit=10), **snap}


# ── ★告警证据图★ ────────────────────────────────────────────────────────────
#: 每条告警存连续三帧 JPEG（`workspace/alerts/<task>/<alert>/001~003.jpg`），
#: 告警记录里的 `images` 就是这里的 URL。刻意**走独立路由而不是塞 base64**：
#: base64 会把 WS 消息撑大好几倍，而证据图只在用户点开告警时才需要。

#: ★三种文件★（异步证据链）：
#:   · `NNN_source.jpg` —— StreamHub q95 **原始字节**，一次都没重编码；
#:   · `NNN.png` / `NNN.jpg` —— 画了 VLX bbox 的**展示图**（弹窗用，默认 PNG 无损）；
#:   · `NNN_thumb.jpg` —— 320 px 缩略图（**只给卡片**，弹窗绝不用）。
#: **必须白名单匹配**，不能放开成任意文件名 —— 这里是直接拼路径读盘的接口。
_ALERT_IMG_RE = re.compile(r"^\d{3}(_source|_thumb)?\.(jpg|png)$")


@router.get("/api/lmm/alerts/{task_id}/{alert_id}/{filename}")
def alert_image(task_id: str, alert_id: str, filename: str,
                thumbnail: bool = False) -> FileResponse:
    """Serve the original image or a cached/generated list thumbnail."""
    if not _ALERT_IMG_RE.match(filename or ""):
        raise HTTPException(status_code=404, detail="证据图不存在")
    #: 任务号与告警号都当作目录名校验：它们由后端自己生成，出现路径分隔符
    #: 只可能是请求被伪造，直接 404。
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise HTTPException(status_code=404, detail="证据图不存在")
    if "/" in alert_id or "\\" in alert_id or ".." in alert_id:
        raise HTTPException(status_code=404, detail="证据图不存在")
    root=Path(settings.ALERTS_DIR).resolve()
    fp = (root / task_id / alert_id / filename).resolve()
    if not fp.is_relative_to(root):raise HTTPException(404,'证据图不存在。')
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="证据图不存在")
    if thumbnail and not filename.endswith("_thumb.jpg"):
        thumb = fp.with_name(f"{fp.stem}_thumb.jpg")
        if not thumb.is_file():
            from lmm.render import make_thumbnail_jpeg

            thumb_bytes = make_thumbnail_jpeg(fp.read_bytes())
            if not thumb_bytes:
                raise HTTPException(status_code=422, detail="无法生成证据缩略图")
            temp = thumb.with_name(f"{thumb.name}.{uuid.uuid4().hex}.tmp")
            try:
                temp.write_bytes(thumb_bytes)
                temp.replace(thumb)
            finally:
                temp.unlink(missing_ok=True)
        return FileResponse(str(thumb), media_type="image/jpeg",
                            headers={"Cache-Control": "private, no-store"})
    media_type = "image/png" if fp.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(str(fp), media_type=media_type,
                        headers={"Cache-Control": "private, no-store"})


# ── VLX-Seek（自然语言目标检测，替代 YOLOE + Qwen-VL 二阶段）────────────────


@router.get("/api/lmm/vlx/health")
def vlx_health() -> Dict[str, Any]:
    """云端 VLX-Seek 常驻服务状态。

    RunOS 不直连公网：请求经算力节点 SSH 隧道打到远端 127.0.0.1。
    服务不可用时**返回 error 而不是抛 500**，前端据此显示"未连接"。
    """
    from lmm.vlx_client import VLXError, VLXSeekClient
    availability = tm.vlx_local_availability()
    if not availability.get("available"):
        return {"connected": False, "status": "blocked",
                "message": str(availability.get("reason") or "VLX-Seek 当前不可用")}

    try:
        client = VLXSeekClient()
        info = client.health()
    except VLXError as exc:
        return {"connected": False, "status": "error", "message": str(exc),
                "detail": (exc.detail or "")[:300]}
    except Exception as exc:  # noqa: BLE001
        return {"connected": False, "status": "error", "message": f"{type(exc).__name__}: {exc}"}

    try:
        meta = client.model_info()
    except Exception:  # noqa: BLE001
        meta = {}
    return {
        "connected": bool(info.get("model_loaded")),
        "status": info.get("status", ""),
        "model": info.get("model", ""),
        "gpu": info.get("gpu", ""),
        "device": info.get("device", ""),
        "load_seconds": info.get("load_seconds"),
        "vram_free_mb": info.get("vram_free_mb"),
        "vram_total_mb": info.get("vram_total_mb"),
        "load_error": info.get("load_error"),
        "runner": meta,
    }


@router.post("/api/lmm/vlx/infer-image")
async def vlx_infer_image(
    image: UploadFile = File(..., description="待识别图片"),
    prompt: str = Form(..., description="中文目标描述，例如「水面上的船」"),
    task: str = Form(default=""),
    lang: str = Form(default=""),
    debug: bool = Form(default=False),
    max_new_tokens: int = Form(default=0),
) -> Dict[str, Any]:
    """单张图片 → VLX-Seek → 真实 bbox（原图像素 xyxy）。

    返回结构与云端服务一致，额外补上前端画框需要的字段。**不伪造 bbox**：
    `matched=false` 时 `boxes` 为空，由前端显示"本帧未发现匹配目标"。
    """
    from lmm.vlx_client import VLXError, VLXSeekClient

    availability = tm.vlx_local_availability()
    if not availability.get("available"):
        raise HTTPException(status_code=409,
                            detail=str(availability.get("reason") or "VLX-Seek 当前不可用"))

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="图片为空")
    if not (prompt or "").strip():
        raise HTTPException(status_code=400, detail="prompt 为空")

    cfg = lmm_config.load()
    try:
        res = VLXSeekClient().infer_image(
            payload,
            prompt.strip(),
            task=(task or cfg.get("vlx_seek_task") or "detection"),
            lang=(lang or cfg.get("vlx_seek_lang") or "zh"),
            debug=debug,
            max_new_tokens=int(max_new_tokens or cfg.get("vlx_seek_max_new_tokens") or 256),
            filename=image.filename or "frame.jpg",
        )
    except VLXError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    width = int(res.get("image_width") or 0)
    height = int(res.get("image_height") or 0)
    boxes = [{
        "x1": round(det["bbox_xyxy"][0], 2),
        "y1": round(det["bbox_xyxy"][1], 2),
        "x2": round(det["bbox_xyxy"][2], 2),
        "y2": round(det["bbox_xyxy"][3], 2),
        "confidence": 1.0,
        "confidence_source": "synthetic",
        "label": det.get("label") or "",
        "region_id": det.get("region_id") or "",
        "vlm_state": "confirmed",
        "display_name": prompt.strip(),
    } for det in res.get("detections", [])]

    return {
        "success": True,
        "prompt": res.get("prompt"),
        "task": res.get("task"),
        "lang": res.get("lang"),
        "width": width,
        "height": height,
        "matched": bool(boxes),
        "boxes": boxes,
        "detections": res.get("detections", []),
        "proposal_count": res.get("proposal_count"),
        #: 调试用：WeDetect 的全部候选框（原图像素 xyxy），索引即 <objN> 的 N
        "proposal_boxes": res.get("proposal_boxes") or [],
        #: 原始 proposal 里面积为 0 的退化框个数（服务端就地补成 1 像素细条以保住 region
        #: 索引对齐）；tiny 是短边不足阈值（默认 4 px）的候选数（含前者）；dropped 是
        #: 其中**被模型选中又被服务端丢弃**的个数。不静默：前端看到 dropped>0 就知道
        #: "少框"是服务端主动过滤，不是漏检。
        "degenerate_proposals": res.get("degenerate_proposals"),
        "tiny_proposals": res.get("tiny_proposals"),
        "dropped_invalid_regions": res.get("dropped_invalid_regions"),
        "min_region_side": res.get("min_region_side"),
        "selected_region_count": res.get("selected_region_count"),
        "latency_ms": res.get("latency_ms"),
        "proposal_latency_ms": res.get("proposal_latency_ms"),
        "vlx_latency_ms": res.get("vlx_latency_ms"),
        "raw_output": res.get("raw_output"),
        "completion_tokens": res.get("completion_tokens"),
        "vram_allocated_mb": res.get("vram_allocated_mb"),
        "debug_images": res.get("debug_images", []),
        "request_id": res.get("request_id"),
    }


# ── 上传视频 → 检测 → 下载结果（推理页「视频 + 大模型」走这条）────────────────

#: 认得出的视频后缀。不在这里的一律拒绝：写进 temp 的文件类型必须可控。
_VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".flv", ".wmv",
                   ".webm", ".mpg", ".mpeg", ".ts")


@router.post('/api/lmm/upload-media')
async def upload_detection_media(file: UploadFile = File(...)):
    from media_upload import save_upload, VIDEO_SUFFIXES, IMAGE_SUFFIXES
    name = Path(file.filename or '').name
    path, size = await save_upload(file, Path(settings.TEMP_DIR) / 'detection-inputs', VIDEO_SUFFIXES | IMAGE_SUFFIXES)
    return {'name': name, 'source': str(path), 'size': size,'preview_url':'/api/lmm/uploaded-media/'+path.name,
            'kind': 'image' if path.suffix in IMAGE_SUFFIXES else 'video', 'started': False}


@router.get('/api/lmm/uploaded-media/{ident}')
async def preview_uploaded_media(ident:str):
    from media_upload import owned_file,VIDEO_SUFFIXES,IMAGE_SUFFIXES
    if not re.fullmatch('[a-f0-9]{32}\\.[a-z0-9]+',ident) or Path(ident).suffix not in VIDEO_SUFFIXES|IMAGE_SUFFIXES:
        raise HTTPException(404,'上传文件不存在。')
    path=owned_file(Path(settings.TEMP_DIR)/'detection-inputs'/ident,Path(settings.TEMP_DIR)/'detection-inputs')
    return FileResponse(path,headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})


@router.post('/api/lmm/detect-uploaded-image')
async def detect_uploaded_image(body: dict):
    from media_upload import IMAGE_SUFFIXES
    root = (Path(settings.TEMP_DIR) / 'detection-inputs').resolve()
    path = Path(str(body.get('source', ''))).resolve()
    if path.parent != root or path.suffix not in IMAGE_SUFFIXES or not path.is_file():
        raise HTTPException(422, '请选择当前账号上传的图片。')
    provider = str(body.get('provider_id', '')).strip()
    prompt = str(body.get('prompt', '')).strip()
    if not provider or not prompt or len(prompt) > 3000:
        raise HTTPException(422, '请选择模型并填写目标描述。')
    try:
        return await asyncio.to_thread(_run_configured_vision_inference, path.read_bytes(), provider, prompt, .25)
    except (DetectorError, ValueError) as exc:
        raise _configured_vision_error(exc) from exc


@router.post("/api/lmm/video-tasks")
async def create_video_task(
    video: UploadFile = File(..., description="待检测视频"),
    prompt: str = Form(..., description="中文目标描述，原样下发给所选在线视觉模型"),
    vision_provider_id: str = Form(default="", description="已保存的在线视觉模型配置 ID"),
    engine: str = Form(default="vlm", description="vlm | vlx"),
    qwen_provider_id: str = Form(default="", description="兼容旧前端的 Qwen-VL 配置 ID"),
    task: str = Form(default="", description="detection / reasoning_detection / …"),
    # 0 = 逐帧；>0 = 每 N 秒取 1 帧；<0 = 使用已保存的 qwen_interval_seconds。
    interval: float = Form(default=-1.0, description="在线检测间隔（秒）；0 = 逐帧，-1 = 用配置默认"),
    max_new_tokens: int = Form(default=0),
    lang: str = Form(default=""),
    # 轨迹 / 告警（规则引擎参数，0 / -1 表示"用配置默认"）
    hit_frames: int = Form(default=0, description="连续命中帧才定案"),
    cooldown_seconds: int = Form(default=-1, description="同一目标告警冷却秒数"),
    min_targets: int = Form(default=0, description="最少目标数"),
    trackless_verify: int = Form(default=-1, description="1/0：无轨迹候选按空间格累计命中"),
    trackless_grid: int = Form(default=0, description="空间格数"),
    tracker: str = Form(default="", description="跟踪器；在线 Qwen-VL 不使用"),
    min_hits: int = Form(default=0, description="同一目标连续出现几个抽样帧才画进结果视频；0/1 = 不过滤"),
) -> Dict[str, Any]:
    """上传视频 → 建任务 → 立即启动，一次请求完成。

    本机不加载检测权重：视频由本机解码，抽样帧发给已保存的 OpenAI 兼容视觉模型，
    检出框写进结果视频，完成后用 `GET /api/lmm/tasks/{id}/result-video` 下载。

    为什么不合进 `POST /api/lmm/tasks`：那个接口收的是**服务器上的路径**（LMM 页的用法），
    而这里的视频来自浏览器上传，必须先落盘。合成一次请求，前端只要一次上传就能看到进度。

    返回里的 `video_path` 是落在 `workspace/temp/inference/` 的输入副本，便于事后溯源。
    """
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="目标描述为空")

    suffix = Path(video.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in _VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"不支持的文件类型：{suffix}")

    save_dir = Path(settings.TEMP_DIR) / "inference"
    from media_upload import save_upload
    video_path,_ = await save_upload(video,save_dir,set(_VIDEO_SUFFIXES))

    cfg = lmm_config.load()
    provider_id = (vision_provider_id or qwen_provider_id
                   or cfg.get("vision_provider_id") or cfg.get("qwen_provider_id") or "")
    selected_engine = str(engine or "vlm").strip().lower()
    if (vision_provider_id or qwen_provider_id) == "vlx-seek":
        selected_engine = "vlx"
    if selected_engine not in {"vlm", "vlx"}:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Unsupported visual model engine")
    if selected_engine == "vlx":
        provider_id = ""
    qwen_interval = (float(cfg.get("qwen_interval_seconds", 2.0) or 2.0)
                     if float(interval) < 0 else max(0.0, float(interval)))
    options: Dict[str, Any] = {
        "engine": selected_engine,
        "vision_provider_id": provider_id,
        "qwen_interval_seconds": qwen_interval,
        "lmm_realtime_fast_path": 0,
        "save_video": True,
    }
    # 轨迹 / 告警：0 或 -1 表示"没传"，沿用配置默认；冷却 0 是合法值（=不冷却）。
    if hit_frames and int(hit_frames) > 0:
        options["hit_frames"] = int(hit_frames)
    if int(cooldown_seconds) >= 0:
        options["cooldown_seconds"] = int(cooldown_seconds)
    if min_targets and int(min_targets) > 0:
        options["min_targets"] = int(min_targets)
    if int(trackless_verify) >= 0:
        options["trackless_verify"] = bool(trackless_verify)
    if trackless_grid and int(trackless_grid) > 0:
        options["trackless_grid"] = int(trackless_grid)
    if tracker:
        options["tracker"] = tracker
    # >1 时结果视频只画「连续出现够次数」的框（代价：多读一遍视频）。
    if min_hits and int(min_hits) > 1:
        options["video_min_hits"] = int(min_hits)

    try:
        created = tm.create_task(clean_prompt, "file", str(video_path), options=options)
    except (ValueError, DetectorError) as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    LOG.info("[LMM] 视频任务 %s：%s（%s，%.1f MB）",
             created["id"], clean_prompt, video_path.name, len(payload) / 1048576.0)

    # 创建就能失败（例如源不合法），这种任务不该再启动 —— 直接把失败原因交回前端。
    if created.get("status") == tm.STATUS_FAILED:
        return {"task": created, "video_path": str(video_path), "started": False}

    started = tm.start_task(created["id"])
    return {"task": started, "video_path": str(video_path), "started": True}


#: 「视频推理记录」的状态文案。与 `tm.STATUS_ZH` 只差 `stopped`/`ready` 两条：
#: 这两个词在**实时检测**语境下是"已停止/就绪"，在**上传视频做识别**语境下应该是
#: "已完成/待开始"。同一个枚举两套文案，别把后者照抄成前者。
_VIDEO_STATUS_ZH: Dict[str, str] = {
    **tm.STATUS_ZH,
    tm.STATUS_STOPPED: "已完成",
    tm.STATUS_READY: "待开始",
}


@router.get("/api/lmm/video-tasks")
def list_video_tasks(limit: int = 20) -> Dict[str, Any]:
    """推理页「视频 + 大模型」跑过的任务（= 开了 `save_video` 的那些）。

    刻意**不直接返回** `/api/lmm/tasks` 的全量列表：那里混着多模态页的实时摄像头任务
    （没有结果视频、没有终点），放进「视频推理记录」全是干扰项 —— 用户看到的
    就是"记录跟自己的操作对不上"。这里只留真正能对应「上传视频 → 识别 → 下载」的任务。
    """
    # 内存态（活着的任务，字段最全）∪ 落盘索引（后端重启后仍在的历史记录）。
    # 同一个 id 以内存为准：索引里的 status 是最后一次回写的快照，可能落后。
    stored = {r.get("id"): r for r in vi.rows()}
    live = {t.get("id"): t for t in tm.list_tasks()
            if t.get("save_video") or t.get("result_video")}

    rows: List[Dict[str, Any]] = []
    for tid in set(stored) | set(live):
        t = live.get(tid) or {}
        s = stored.get(tid) or {}
        eff = t.get("effective") or {}
        rows.append({
            "id": tid,
            "prompt": t.get("prompt") or s.get("prompt"),
            "status": t.get("status") or s.get("status"),
            "status_zh": t.get("status_zh"),
            "message": t.get("message") or s.get("message"),
            "created_at": t.get("created_at") or s.get("created_at"),
            "finished_at": t.get("finished_at") or s.get("finished_at"),
            "processed": t.get("processed") if t else s.get("processed"),
            "total_frames": t.get("total_frames") or s.get("total_frames"),
            "detections": t.get("detections") if t else s.get("detections"),
            "detector": t.get("detector") or s.get("detector"),
            "sample_stride": t.get("sample_stride") or s.get("sample_stride"),
            # 只回布尔：路径是本机绝对路径，对浏览器没意义
            "has_video": bool(t.get("result_video") or s.get("result_video")),
            "has_json": bool(t.get("result_json") or s.get("result_json")),
            "video_min_hits": t.get("video_min_hits") or eff.get("video_min_hits")
                              or s.get("video_min_hits"),
            "sporadic_dropped": t.get("sporadic_dropped") if t else s.get("sporadic_dropped"),
            "source_name": (Path(str(t.get("source") or "")).name
                            or s.get("source_name") or ""),
        })
    # 状态中文统一按**推理页**的语义：lmm 的 `stopped` 在这里就是"已完成"
    # （用户点的是「开始识别」，任务正常收尾显示"已完成"才对；照抄 tm.STATUS_ZH
    #   会写成"已停止"，看着像被中断了）。
    for r in rows:
        r["status_zh"] = _VIDEO_STATUS_ZH.get(r.get("status"), r.get("status"))
    rows.sort(key=lambda r: float(r.get("created_at") or 0), reverse=True)
    return {"tasks": rows[: int(limit)]}


def _stored_result_path(task_id: str, kind: str) -> Optional[Path]:
    """后端重启后任务已不在内存里 → 去落盘索引找结果文件路径。

    **只信索引里记的、且必须是 `temp/inference` 下的普通文件**：索引是本地 JSON，
    但它毕竟是"磁盘上的一个文件"，直接把里面的路径交给 FileResponse 等于开了一个
    任意文件下载口（改一行 JSON 就能下系统文件）。
    """
    row = vi.get(task_id) or {}
    p = row.get(kind)
    if not p:
        return None
    try:
        path = Path(str(p)).resolve()
        root = (Path(settings.TEMP_DIR) / "inference").resolve()
        if path.is_file() and path.is_relative_to(root):
            return path
    except Exception:  # noqa: BLE001 —— 路径不合法就当没有
        LOG.debug("[LMM] 结果路径不可用：%s", p, exc_info=True)
    return None


@router.get("/api/lmm/tasks/{task_id}/result-video")
def task_result_video(task_id: str) -> FileResponse:
    """下载结果视频（检出框已画好，原帧率、原时长）。

    任务不在内存里（后端重启过）就回落到落盘索引：磁盘上明明有文件却报 404 是用户
    最不能理解的一类错（"我刚跑完的，怎么就没了"）。
    """
    task = tm.get_task(task_id)
    if task is not None:
        path = task.get("result_video")
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=404,
                                detail="结果视频尚未生成（任务未完成或未开启结果落盘）")
        from media_upload import owned_file
        path=owned_file(path,settings.WORKSPACE_DIR)
        return FileResponse(path, media_type="video/mp4", filename=f"{task_id}_result.mp4",headers={'Cache-Control':'private, no-store'})

    stored = _stored_result_path(task_id, "result_video")
    if stored is None:
        raise HTTPException(status_code=404,
                            detail="结果视频不存在（任务已不在内存中，且索引里没有可用记录）")
    return FileResponse(str(stored), media_type="video/mp4", filename=f"{task_id}_result.mp4",headers={'Cache-Control':'private, no-store'})


@router.get("/api/lmm/tasks/{task_id}/result-json")
def task_result_json(task_id: str) -> FileResponse:
    """下载检出数据（只在**真推理过**的抽样帧上记框，不伪造逐帧结果）。"""
    task = tm.get_task(task_id)
    if task is not None:
        path = task.get("result_json")
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=404,
                                detail="检出数据尚未生成（任务未完成或未开启结果落盘）")
        from media_upload import owned_file
        path=owned_file(path,settings.WORKSPACE_DIR)
        return FileResponse(path, media_type="application/json",
                            filename=f"{task_id}_detections.json",headers={'Cache-Control':'private, no-store'})

    stored = _stored_result_path(task_id, "result_json")
    if stored is None:
        raise HTTPException(status_code=404,
                            detail="检出数据不存在（任务已不在内存中，且索引里没有可用记录）")
    return FileResponse(str(stored), media_type="application/json",
                        filename=f"{task_id}_detections.json",headers={'Cache-Control':'private, no-store'})


#: 源视频的媒体类型（浏览器能不能直接播，取决于 Content-Type 对不对）
_MEDIA_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/x-m4v", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv", ".flv": "video/x-flv", ".ts": "video/mp2t",
    ".mpg": "video/mpeg", ".mpeg": "video/mpeg", ".m2ts": "video/mp2t",
    ".3gp": "video/3gpp", ".mts": "video/mp2t", ".rmvb": "application/vnd.rn-realmedia-vbr",
}
_CHUNK = 262144


def _iter_file(path: Path, start: int, end: int, chunk: int = _CHUNK):
    """按 [start, end] 区间产出字节（闭区间）。"""
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/api/lmm/tasks/{task_id}/media")
async def task_media(task_id: str, request: Request):
    """源视频流式播放（**支持 HTTP Range / 206 Partial Content**）。

    存在的理由（R1 §A）：实时模式要把「画面」和「AI 结果」拆成两条通道，浏览器
    必须能**自己连续播放原始视频**；否则画面更新频率仍然等于 AI 推理频率 ——
    那正是"4 秒动一次"的根因。

    ★安全★
        路径**只从任务记录里取**，不接受 `?path=` 之类的任意路径参数 ——
        否则这个端点就是个任意文件下载口（改一个 query 就能下系统文件）。
        任务里的 `source` 是后端自己 resolve 过的真实路径；RTSP / 摄像头没有
        本地文件，直接 404（那些输入源走 `preview_frame` 预览通道）。
    """
    task = tm.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if (task.get("source_type") or "file").lower() not in ("file", "video", ""):
        raise HTTPException(status_code=404,
                            detail="该输入源不是本地文件，浏览器无法直接播放（请用预览通道）")
    from media_upload import owned_file
    path=owned_file(task.get('source',''),settings.WORKSPACE_DIR)
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=415, detail=f"不支持的媒体类型：{path.suffix}")

    size = path.stat().st_size
    headers = {"Accept-Ranges": "bytes",'Cache-Control':'private, no-store'}
    range_hdr = (request.headers.get("range") or "").strip()

    start, end = 0, size - 1
    status_code = 200
    if range_hdr.lower().startswith("bytes="):
        spec = range_hdr[6:].strip()
        try:
            s, _, e = spec.partition("-")
            start = int(s) if s else 0
            end = min(int(e), size - 1) if e else size - 1
        except ValueError:
            raise HTTPException(status_code=416, detail="Range 头格式不合法")
        if start < 0 or end < start or start >= size:
            raise HTTPException(status_code=416, detail="Range 超出文件范围")
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)

    return StreamingResponse(_iter_file(path, start, end), status_code=status_code,
                             media_type=media_type, headers=headers)


@router.websocket("/api/lmm/ws/tasks/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str) -> None:
    """实时帧推送。

    推的是"最近一帧"而不是"每一帧"：推理快于网络时丢帧才是正确行为，堆积会让画面
    越来越滞后。前端拿 WS 失败时要自动降级到 `/snapshot` 轮询。
    """
    try:
        claims = verify_session(token_from_request(websocket))
    except ValueError:
        await websocket.close(code=4401, reason="RunHive sign-in required")
        return
    if int(claims.get("sub", 0)) != int(os.environ.get("RUNOS_OWNER_ADMIN_ID", "1")):
        await websocket.close(code=4403, reason="RunOS workspace account mismatch")
        return

    await websocket.accept()
    q = hub.subscribe(task_id)
    try:
        while True:
            try:
                payload = q.get_nowait()
            except Exception:
                await asyncio.sleep(0.08)
                continue
            try:
                await websocket.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        LOG.debug("[LMM] WS 异常", exc_info=True)
    finally:
        hub.unsubscribe(task_id, q)
