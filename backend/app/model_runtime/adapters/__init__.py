"""Adapter 层（P2B）。

★为什么要有这一层★
    `model_integration._run_inference_unlocked` 现在是 `if model_type == ... elif ...`
    的大分发。Adapter 的目的是把它逐步拆开，但**不一次性重写**：
    P2B 只让 VLX 单图走 Adapter，YOLO / SAM / RF-DETR / Qwen 继续走旧分支。

★统一输出契约★
    AdapterResult.boxes 里每个 box：
        {
            "xyxy": [x1, y1, x2, y2],   # 由 AdapterResult.coord_space 声明其含义
            "label": <类别名>,           # ★永远来自 target.label，不是 prompt★
            "label_zh": <模型自己的措辞>,
            "confidence": 1.0,          # VLX 不输出置信度 → synthetic
            "confidence_source": "synthetic",
            "raw": {...}
        }
"""

from .base import InferenceAdapter, AdapterResult, AdapterError, COORD_PIXEL_XYXY, to_image_bytes
from .vlx import VLXAdapter

__all__ = [
    "InferenceAdapter",
    "AdapterResult",
    "AdapterError",
    "COORD_PIXEL_XYXY",
    "to_image_bytes",
    "VLXAdapter",
]
