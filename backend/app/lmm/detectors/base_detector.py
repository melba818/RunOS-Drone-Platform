"""检测器抽象层。

为什么要这一层：需求文档明确要求「模型必须独立封装」，并且为后续接入别的开放词汇
检测器（Grounding DINO / OWL-ViT …）留出位置。业务逻辑（规则引擎、任务管理）只依赖
这里定义的契约，不直接 import ultralytics。

坐标约定：**像素 xyxy**，与项目现有模块一致（不归一化），前端按画布尺寸等比缩放即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectedBox:
    """单个检测结果。"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    #: YOLOE 实际消费的英文单 token（无空格）
    label: str
    #: 前端展示用的中文名；为空时前端回退显示 label
    label_zh: str = ""
    #: 跟踪 ID；未开启跟踪时为 None
    track_id: Optional[int] = None
    #: Synthetic values preserve older consumers; never display or threshold them.
    confidence_source: str = "detector"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(float(self.confidence), 4),
            "confidence_source": self.confidence_source,
            "label": self.label,
            "label_zh": self.label_zh or self.label,
            "track_id": self.track_id,
        }


@dataclass
class FrameResult:
    """一帧的检测输出。"""

    frame_index: int
    width: int
    height: int
    boxes: List[DetectedBox] = field(default_factory=list)
    #: 本帧推理耗时（毫秒），用于前端显示实时 FPS
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "boxes": [b.to_dict() for b in self.boxes],
        }


class DetectorError(RuntimeError):
    """检测器不可用（权重缺失 / 加载失败 / 推理异常）。"""


class BaseDetector(ABC):
    """检测器契约。

    生命周期（由调用方遵守）：

        load_model() → set_prompts(labels) → predict()/track()*N → unload()

    `set_prompts` 必须**幂等且廉价**：标签没变时不得重跑文本编码（需求文档硬性要求
    「只有用户修改目标 Prompt 时才重新执行 set_classes / Prompt 编码」）。
    """

    #: 子类填入，用于日志与前端展示
    name: str = "base"

    @abstractmethod
    def load_model(self, model_path: str, device: str = "auto", half: bool = False) -> None:
        """加载权重。重复调用必须是空操作（模型只能初始化一次并复用）。"""

    @abstractmethod
    def set_prompts(self, labels: List[str]) -> None:
        """下发开放词汇标签。标签未变化时直接返回。"""

    @abstractmethod
    def predict(self, frame: Any, confidence: float, iou: float, imgsz: int) -> FrameResult:
        """单帧检测（不出 track_id）。"""

    @abstractmethod
    def track(self, frame: Any, confidence: float, iou: float, imgsz: int,
              tracker: str, frame_index: int = 0) -> FrameResult:
        """单帧检测 + 跟踪（出 track_id）。`persist=True` 由实现内部保证。"""

    @abstractmethod
    def unload(self) -> None:
        """释放模型与显存。"""

    def warmup(self, imgsz: int = 640) -> None:
        """可选：跑一帧空图预热，避免首帧把 CUDA kernel 编译时间算进 FPS。"""
        return None

    def info(self) -> Dict[str, Any]:
        """给前端「模型状态」卡片用的描述。"""
        return {"name": self.name}
