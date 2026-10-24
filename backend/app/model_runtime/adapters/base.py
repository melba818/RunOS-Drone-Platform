"""InferenceAdapter 抽象与 AdapterResult。

★职责边界★
    Adapter 只做"拿一个 Deployment + 一个 DetectionTarget，调一次模型，返回统一 box"。
    它**不负责**：选实例（Resolver）、重试/切换实例（RuntimeService）、
    写 data.yaml（调用方）、决定类别（target.label 已经在 DetectionTarget 里）。

★坐标★
    Adapter 统一输出 `pixel_xyxy`（原始图像像素、绝对坐标、左上原点）。
    **禁止在 Adapter 内转成 YOLO cxcywh 归一化** —— 转格式是写盘层的事，
    在这里转会出现"二次归一化"（VLX 本来就返回像素坐标）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "InferenceAdapter",
    "AdapterResult",
    "AdapterError",
    "COORD_PIXEL_XYXY",
    "to_image_bytes",
]

COORD_PIXEL_XYXY = "pixel_xyxy"
COORD_NORM_0_1000 = "norm_0_1000"
COORD_NORM_0_1 = "norm_0_1"


class AdapterError(RuntimeError):
    """Adapter 内部错误（配置缺失 / 协议错误 / 调用失败）。
    消息**不得包含** token / api_key / 密码。"""


@dataclass
class AdapterResult:
    """一次模型调用的统一结果。"""
    boxes: List[Dict[str, Any]] = field(default_factory=list)
    coord_space: str = COORD_PIXEL_XYXY
    latency_ms: float = 0.0
    deployment_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boxes": self.boxes,
            "coord_space": self.coord_space,
            "latency_ms": self.latency_ms,
            "deployment_id": self.deployment_id,
            "raw": self.raw,
        }


def to_image_bytes(image: Union[bytes, bytearray, str, Path]) -> bytes:
    """接受 bytes / 文件路径，统一成 bytes。"""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if isinstance(image, io.BytesIO):  # pragma: no cover
        return image.getvalue()
    raise AdapterError(f"unsupported image type: {type(image).__name__}")


class InferenceAdapter:
    """所有 Adapter 的基类。子类只需实现 infer()。"""

    #: 与 Model.protocol 对应，RuntimeService 用它挑 Adapter
    protocol: str = ""

    def infer(self, *, image, target, deployment: Dict[str, Any],
              options: Optional[Dict[str, Any]] = None,
              model: Optional[Dict[str, Any]] = None) -> AdapterResult:
        raise NotImplementedError
