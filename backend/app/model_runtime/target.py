"""DetectionTarget —— 统一检测目标。

★命名约定★
    本模块的类名是 `DetectionTarget`，**不要**叫 `TargetSpec`。
    `backend/lmm/target_spec.py` 里已有一个同名 `TargetSpec`（多模态视频链路的语义结构），
    保持原名、原逻辑、原 import，本模块与它**互不 import**。

★写盘规则★
    label            —— 唯一允许作为 YOLO / Dataset 类别写入 data.yaml 的字段。
    description      —— 自然语言任务描述。
    extra_prompt     —— 高级语义约束。
    description / extra_prompt 允许未来保存到 TargetProfile / batch job，
                      但**绝对禁止**写入 data.yaml 的 names 或 YOLO class name。
                      （注意：不是"任何 JSON 都不能存"，只是不能进 dataset 类别。）
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

__all__ = ["DetectionTarget"]

# 允许写入 data.yaml 的字段 —— 有且只有一个
LABEL_ONLY_FIELDS = ("label",)


@dataclass
class DetectionTarget:
    """统一检测目标：把"类别"与"给模型看的描述"彻底分离。"""

    label: str
    description: str = ""
    extra_prompt: str = ""
    confidence: float = 0.5
    aliases: List[str] = field(default_factory=list)

    # ── 构造 ────────────────────────────────────────────────────────────────
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DetectionTarget":
        d = d or {}
        aliases = d.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        return DetectionTarget(
            label=str(d.get("label") or "").strip(),
            description=str(d.get("description") or ""),
            extra_prompt=str(d.get("extra_prompt") or ""),
            confidence=float(d.get("confidence", 0.5) or 0.5),
            aliases=[str(a) for a in aliases],
        )

    @staticmethod
    def from_legacy_text_prompt(text: str) -> "DetectionTarget":
        """旧 text_prompt 兼容入口。

        旧 VLX 链路是"描述即类别"。回放历史 job 时由上层决定是否采用
        LEGACY_LABEL_FROM_DESCRIPTION=true（此时 label 被赋值为整句描述）。
        这里保持 `label=""`，由调用方按开关显式填入，避免隐式行为。
        """
        t = str(text or "").strip()
        return DetectionTarget(label="", description=t, extra_prompt="", confidence=0.5)

    # ── 输出 ────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def prompt_text(self) -> str:
        """送给模型的文本。

        description 有值 → description；extra_prompt 有值 → 换行追加。
        两者都空 → 回退 label（**只是当提示词用**，不代表 label 可以反过来被描述覆盖）。
        绝不把 label 自动扩写成不存在的语义。
        """
        parts = [self.description.strip(), self.extra_prompt.strip()]
        text = "\n".join(p for p in parts if p)
        return text if text else self.label.strip()

    def effective_label(self, legacy_label_from_description: bool = False) -> str:
        """返回应当写入 data.yaml 的类别名。

        legacy_label_from_description=True 且 label 为空时，回退成整句 description
        （旧 VLX 行为），用于历史 job 回放。默认新行为必须是短 label。
        """
        if self.label.strip():
            return self.label.strip()
        if legacy_label_from_description:
            return self.description.strip()
        return ""

    def is_valid(self) -> bool:
        return bool(self.label.strip())
