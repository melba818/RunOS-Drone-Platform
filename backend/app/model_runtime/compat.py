"""DetectionTarget 归一化入口（P2A）。

★为什么要有这个统一入口★
    禁止多个 router 各自手写一套"新结构 / 旧 text_prompt"兼容逻辑。
    所有标注入口（单图 / 文本批量）都必须调用 `normalize_detection_target()`。

优先级：
    1. 显式 `target` 对象
    2. 独立字段 `label` / `description` / `extra_prompt`
    3. legacy `text_prompt`

★legacy 语义★
    只有 `text_prompt` 时走 `DetectionTarget.from_legacy_text_prompt()`：
    label 为空、description = 整句。调用方**必须**继续沿用旧的写盘语义
    （P2A 不改变 legacy 行为），label 的取值由上层按
    `LEGACY_LABEL_FROM_DESCRIPTION` 决定。

★新结构★
    新 API（传了 target 或 label/description）**必须**有 label；
    只有 description 没有 label 时抛 `MissingLabelError`（"请选择类别"），
    绝不偷偷把 description 当成 label —— 这正是本次要消灭的 data.yaml 污染。

★无提示词★
    四者皆空 → MODE_NONE，返回空 DetectionTarget 且**不抛错**。
    YOLO / RF-DETR / SAM 等本来就不接收 text_prompt，不能因为标准化而回归。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .target import DetectionTarget

__all__ = [
    "normalize_detection_target",
    "MissingLabelError",
    "MODE_TARGET",
    "MODE_FIELDS",
    "MODE_LEGACY",
    "MODE_NONE",
]

MODE_TARGET = "target"      # 传了 target 对象
MODE_FIELDS = "fields"      # 传了 label/description/extra_prompt
MODE_LEGACY = "legacy"      # 只有 text_prompt（旧行为，保持不变）
MODE_NONE = "none"          # 什么都没传（YOLO 等不需要提示词的模型）——不能报错

MSG_SELECT_CLASS = "请选择类别"


class MissingLabelError(ValueError):
    """新结构请求缺少 label。前端应提示用户先选类别。"""


def _has(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    return True


def normalize_detection_target(
    *,
    target: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    description: Optional[str] = None,
    extra_prompt: Optional[str] = None,
    text_prompt: Optional[str] = None,
    confidence: Optional[float] = None,
    default_confidence: float = 0.5,
) -> Tuple[DetectionTarget, str]:
    """返回 (DetectionTarget, mode)。缺 label 的新结构请求抛 MissingLabelError。"""
    conf = default_confidence if confidence is None else float(confidence)

    # 0) 什么都没传 —— YOLO / RF-DETR 等不需要提示词，绝不能报错
    if not (
        (isinstance(target, dict) and target)
        or _has(label) or _has(description) or _has(extra_prompt) or _has(text_prompt)
    ):
        return DetectionTarget(label="", description="", confidence=conf), MODE_NONE

    # 1) 显式 target 对象
    if isinstance(target, dict) and target:
        payload = dict(target)
        if not _has(payload.get("label")):
            raise MissingLabelError(MSG_SELECT_CLASS)
        payload.setdefault("confidence", conf)
        return DetectionTarget.from_dict(payload), MODE_TARGET

    # 2) 独立字段
    if _has(label) or _has(description) or _has(extra_prompt):
        if not _has(label):
            # 新 API 给了描述却没给类别 —— 明确报错，不回落成"描述即类别"
            raise MissingLabelError(MSG_SELECT_CLASS)
        return DetectionTarget(
            label=str(label).strip(),
            description=str(description or "").strip(),
            extra_prompt=str(extra_prompt or "").strip(),
            confidence=conf,
        ), MODE_FIELDS

    # 3) legacy：只有 text_prompt
    if _has(text_prompt):
        tg = DetectionTarget.from_legacy_text_prompt(str(text_prompt))
        tg.confidence = conf
        return tg, MODE_LEGACY

    raise MissingLabelError("缺少目标描述或类别")
