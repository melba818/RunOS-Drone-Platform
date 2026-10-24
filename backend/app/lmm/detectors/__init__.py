"""检测器实现。对外只暴露 `get_detector()` / `unload_all()`。"""

from .base_detector import BaseDetector, DetectorError, DetectedBox, FrameResult
from .yoloe_detector import YOLOEDetector, get_detector, unload_all

__all__ = [
    "BaseDetector",
    "DetectorError",
    "DetectedBox",
    "FrameResult",
    "YOLOEDetector",
    "get_detector",
    "unload_all",
]
