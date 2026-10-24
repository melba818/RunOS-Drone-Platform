"""YOLOE 开放词汇检测器封装。

关键实现约束（都来自需求文档 + 本机实测）：

1. **模型只初始化一次。** 模块级单例 `_SINGLETON`，按权重路径复用；换权重才重建。
2. **Prompt 编码只在标签变化时执行。** `_labels_key` 与上次一致就直接返回 —— 实测
   同一组标签重复 `set_classes` 耗时 0.0000s，而首次编码（含加载 MobileCLIP2-B
   TorchScript，254MB）约 1.3s。视频每帧都重设会把 FPS 打到个位数。
3. **标签必须是单 token。** `YOLOE.set_classes` 内部 `assert " " not in classes`，
   含空格直接 AssertionError。多词概念由 `target_spec.sanitize_label()` 转成下划线形式
   （LVIS 风格），文本编码器可正常分词。
4. **不做 masked attention / 不做逐帧 VLM。** 第一阶段只走文本 Prompt。

Windows 中文路径注意：ultralytics 加载 TorchScript 文本编码器走窄字符 I/O，权重目录若含
非 ASCII 字符会失败 —— 与项目既有的 `ascii_safe_name()` 约束同源，这里不额外处理，
只把可读的报错抛给上层。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .base_detector import BaseDetector, DetectorError, DetectedBox, FrameResult

LOG = logging.getLogger(__name__)

_LOCK = threading.RLock()

#: 模块级单例：key = 权重绝对路径
_SINGLETON: Dict[str, "YOLOEDetector"] = {}


def _resolve_device(device: str) -> Any:
    """'auto' → 有 CUDA 就用 0。MX250 2GB 也能跑 YOLOE-s（实测 0.74GB reserved）。"""
    try:
        import torch
    except Exception:
        return "cpu"
    if device and device not in ("auto", ""):
        return device
    return 0 if torch.cuda.is_available() else "cpu"


def get_detector(model_path: str, *, device: str = "auto", half: bool = False) -> "YOLOEDetector":
    """按权重路径取（或创建）单例检测器。"""
    key = str(model_path)
    with _LOCK:
        det = _SINGLETON.get(key)
        if det is None:
            det = YOLOEDetector()
            _SINGLETON[key] = det
        det.load_model(key, device=device, half=half)
        return det


def unload_all() -> None:
    """释放所有已加载的检测器（任务结束 / 切换模型时调用）。"""
    with _LOCK:
        for det in list(_SINGLETON.values()):
            try:
                det.unload()
            except Exception:
                LOG.exception("[YOLOE] unload 失败")
        _SINGLETON.clear()


class YOLOEDetector(BaseDetector):
    """ultralytics YOLOE 封装。

    ultralytics 8.4.146 自带 `YOLOE` 类，具备 `get_text_pe` / `set_classes` / `predict`
    / `track`，不需要我们持有算法副本 —— 与项目对 ultralytics 内置跟踪器的态度一致。
    """

    name = "yoloe"

    def __init__(self) -> None:
        self._model = None
        self._model_path: Optional[str] = None
        self._device: Any = None
        self._half = False
        self._labels: Tuple[str, ...] = ()
        self._text_pe = None
        self._load_seconds: float = 0.0
        self._encode_seconds: float = 0.0

    # ── 加载 ──────────────────────────────────────────────────────────────────

    def load_model(self, model_path: str, device: str = "auto", half: bool = False) -> None:
        if self._model is not None and self._model_path == str(model_path):
            return
        t0 = time.time()
        try:
            from ultralytics import YOLOE
        except Exception as exc:  # pragma: no cover - 环境缺失
            raise DetectorError(f"ultralytics 不可用或版本过低（需要 YOLOE）：{exc}") from exc

        resolved = _resolve_device(device)
        try:
            model = YOLOE(model_path)
        except Exception as exc:
            raise DetectorError(f"加载 YOLOE 权重失败（{model_path}）：{exc}") from exc

        self._model = model
        self._model_path = str(model_path)
        self._device = resolved
        self._half = bool(half)
        self._labels = ()
        self._text_pe = None
        self._load_seconds = time.time() - t0
        LOG.info("[YOLOE] 权重加载完成 %s（%.2fs, device=%s, half=%s）",
                 model_path, self._load_seconds, resolved, self._half)

    # ── Prompt ───────────────────────────────────────────────────────────────

    def set_prompts(self, labels: List[str]) -> None:
        """下发标签。**标签未变则直接返回**（这就是「Prompt 编码不重复执行」的落点）。"""
        if self._model is None:
            raise DetectorError("模型尚未加载")
        clean: List[str] = []
        for lb in labels or []:
            s = str(lb).strip()
            if not s:
                continue
            # set_classes 内部 assert " " not in classes，这里兜底替换而不是抛错
            s = s.replace(" ", "_")
            if s not in clean:
                clean.append(s)
        if not clean:
            raise DetectorError("没有可用的检测标签")

        key = tuple(clean)
        if key == self._labels:
            LOG.debug("[YOLOE] 标签未变化，跳过文本编码：%s", ",".join(key))
            return

        t0 = time.time()
        try:
            pe = self._model.get_text_pe(clean)
            self._model.set_classes(clean, pe)
        except Exception as exc:
            raise DetectorError(f"Prompt 编码失败（labels={clean}）：{exc}") from exc
        self._labels = key
        self._text_pe = pe
        self._encode_seconds = time.time() - t0
        LOG.info("[YOLOE] 文本编码完成 %s（%.2fs）", ",".join(clean), self._encode_seconds)

    # ── 推理 ─────────────────────────────────────────────────────────────────

    def _kwargs(self, confidence: float, iou: float, imgsz: int) -> Dict[str, Any]:
        return {
            "conf": float(confidence),
            "iou": float(iou),
            "imgsz": int(imgsz),
            "device": self._device,
            "half": self._half,
            "verbose": False,
        }

    def predict(self, frame: Any, confidence: float, iou: float, imgsz: int,
                frame_index: int = 0) -> FrameResult:
        if self._model is None:
            raise DetectorError("模型尚未加载")
        t0 = time.time()
        try:
            results = self._model.predict(frame, **self._kwargs(confidence, iou, imgsz))
        except Exception as exc:
            raise DetectorError(f"YOLOE 推理失败：{exc}") from exc
        return self._to_result(results, frame_index, time.time() - t0)

    def track(self, frame: Any, confidence: float, iou: float, imgsz: int,
              tracker: str, frame_index: int = 0) -> FrameResult:
        if self._model is None:
            raise DetectorError("模型尚未加载")
        from .tracker_config import resolve_tracker_yaml
        t0 = time.time()
        try:
            results = self._model.track(
                frame, persist=True, tracker=resolve_tracker_yaml(tracker, confidence),
                **self._kwargs(confidence, iou, imgsz))
        except Exception as exc:
            raise DetectorError(f"YOLOE 跟踪失败：{exc}") from exc
        return self._to_result(results, frame_index, time.time() - t0)

    # ── 结果转换 ─────────────────────────────────────────────────────────────

    def _to_result(self, results: Any, frame_index: int, elapsed: float) -> FrameResult:
        r = results[0] if isinstance(results, (list, tuple)) else results
        h, w = (int(r.orig_shape[0]), int(r.orig_shape[1])) if getattr(r, "orig_shape", None) else (0, 0)
        names = getattr(r, "names", {}) or {}
        boxes: List[DetectedBox] = []
        b = getattr(r, "boxes", None)
        if b is not None and len(b) > 0:
            xyxy = b.xyxy.tolist()
            conf = b.conf.tolist() if b.conf is not None else [0.0] * len(xyxy)
            cls = b.cls.tolist() if b.cls is not None else [0] * len(xyxy)
            ids = b.id.int().tolist() if getattr(b, "id", None) is not None else [None] * len(xyxy)
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                idx = int(cls[i]) if i < len(cls) else 0
                label = names.get(idx, str(idx)) if isinstance(names, dict) else (
                    names[idx] if isinstance(names, (list, tuple)) and idx < len(names) else str(idx))
                boxes.append(DetectedBox(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    confidence=float(conf[i]) if i < len(conf) else 0.0,
                    label=str(label),
                    track_id=int(ids[i]) if i < len(ids) and ids[i] is not None else None,
                ))
        return FrameResult(frame_index=frame_index, width=w, height=h,
                           boxes=boxes, elapsed_ms=elapsed * 1000.0)

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    def warmup(self, imgsz: int = 640) -> None:
        """跑一帧纯色图，把 CUDA kernel / cuDNN 的选择提前完成。"""
        if self._model is None or not self._labels:
            return
        try:
            import numpy as np
            blank = np.zeros((int(imgsz), int(imgsz), 3), dtype=np.uint8)
            self._model.predict(blank, **self._kwargs(0.99, 0.45, imgsz))
        except Exception:
            LOG.debug("[YOLOE] 预热跳过", exc_info=True)

    def reset_tracker(self) -> None:
        """复位跟踪器状态。

        `persist=True` 会让 tracker 在模型对象里长期存活，如果不复位，上一个任务的
        track_id / 轨迹会带进下一个任务（新视频第 1 帧就可能冒出 #37）。
        """
        try:
            pred = getattr(self._model, "predictor", None)
            for tr in getattr(pred, "trackers", None) or []:
                tr.reset()
        except Exception:
            LOG.debug("[YOLOE] tracker 复位跳过", exc_info=True)

    def unload(self) -> None:
        self._model = None
        self._model_path = None
        self._labels = ()
        self._text_pe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_path": self._model_path,
            "device": str(self._device),
            "half": self._half,
            "labels": list(self._labels),
            "load_seconds": round(self._load_seconds, 2),
            "encode_seconds": round(self._encode_seconds, 2),
        }
