"""TargetSpec —— 用户 Prompt 与后端链路之间的唯一契约。

**2026-09-23 起：`TargetSpec` 不再由关键词解析产出。**
原先那层「中文 → person / swimmer / water」的拆解（本地规则词表 + 外部 LLM）会把
「水面上 / 未穿 / 救生衣 / 人员」之间的语义关系全部拆掉，而 VLX-Seek 本身就能直接消费
整句中文。因此现在唯一入口是 `TargetSpec.from_raw(prompt)`：**用户原文原样透传**。

保留下来的职责只有两条：

    detection_label  跨帧稳定的归属标签；本机 YOLOE 回退时它就是下发给
                     `set_classes()` 的类别，**必须保证无空格**。
    original_prompt  用户原文，全程原样透传，用于事件溯源。

（`conditions` 已不再参与：语义判定由 VLX-Seek 一次完成，规则侧不再二次核验。）
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ── 条件类型 ─────────────────────────────────────────────────────────────────

CONDITION_SCENE = "scene"        # 目标所处场景，如 water / road
CONDITION_ABSENCE = "absence"    # 期望"未检出"的伴随目标，如未穿救生衣
CONDITION_ATTRIBUTE = "attribute"  # 目标属性，如红色（第一阶段只能交给 VLM 确认）
CONDITION_COUNT = "count"        # 数量阈值，如"多艘船只"

CONDITION_TYPES = (
    CONDITION_SCENE,
    CONDITION_ABSENCE,
    CONDITION_ATTRIBUTE,
    CONDITION_COUNT,
)

# ── 词表 ─────────────────────────────────────────────────────────────────────
# TARGET: 需要被 YOLOE 检测的实体。labels 为**单 token**（无空格）候选，按优先级排列。
#         多词概念用下划线连接（LVIS 风格），仍是单 token，文本编码器可正常分词。
# SCENE:  场景词，不交给 YOLOE，作为 RuleEngine 的条件。
# ATTR:   属性词（颜色 / 衣着），第一阶段只能作为条件交给未来的 VLMVerifier。
# ABSENCE: 否定标记，决定把后面的词归到 "absence" 条件而不是新目标。

TARGET_LEXICON: List[Dict[str, Any]] = [
    {"zh": "人员", "en": "person", "labels": ["person"],
     "aliases": ["人员", "人", "行人", "人影", "目标人物", "人类"]},
    {"zh": "游泳者", "en": "swimmer", "labels": ["swimmer", "person"],
     "aliases": ["游泳者", "游泳人员", "游泳的人", "野泳", "游泳"]},
    {"zh": "救生衣", "en": "life jacket", "labels": ["life_jacket"],
     "aliases": ["救生衣", "救生背心", "浮力衣", "救生服"]},
    {"zh": "安全帽", "en": "helmet", "labels": ["helmet"],
     "aliases": ["安全帽", "头盔", "安全头盔", "安全帽的"]},
    {"zh": "船只", "en": "boat", "labels": ["boat"],
     "aliases": ["船只", "船", "小船", "船舶", "快艇", "渔船", "游船", "皮划艇", "木船"]},
    {"zh": "汽车", "en": "car", "labels": ["car"],
     "aliases": ["汽车", "轿车", "小轿车", "小汽车", "车辆", "车"]},
    {"zh": "卡车", "en": "truck", "labels": ["truck"],
     "aliases": ["卡车", "货车", "渣土车", "工程车"]},
    {"zh": "客车", "en": "bus", "labels": ["bus"],
     "aliases": ["客车", "巴士", "公交车", "大巴"]},
    {"zh": "摩托车", "en": "motorcycle", "labels": ["motorcycle"],
     "aliases": ["摩托车", "电动车", "电瓶车", "机车"]},
    {"zh": "自行车", "en": "bicycle", "labels": ["bicycle"],
     "aliases": ["自行车", "单车", "脚踏车"]},
    {"zh": "飞机", "en": "airplane", "labels": ["airplane"],
     "aliases": ["飞机", "飞行器", "客机"]},
    {"zh": "无人机", "en": "drone", "labels": ["drone"],
     "aliases": ["无人机", "航拍器"]},
    {"zh": "火焰", "en": "fire", "labels": ["fire"],
     "aliases": ["火焰", "明火", "着火", "起火", "火"]},
    {"zh": "烟雾", "en": "smoke", "labels": ["smoke"],
     "aliases": ["烟雾", "浓烟", "冒烟", "烟"]},
    {"zh": "挖掘机", "en": "excavator", "labels": ["excavator"],
     "aliases": ["挖掘机", "挖机", "工程机械", "施工机械"]},
    {"zh": "起重机", "en": "crane", "labels": ["crane"],
     "aliases": ["起重机", "塔吊", "吊车", "吊机"]},
    {"zh": "建筑", "en": "building", "labels": ["building"],
     "aliases": ["建筑物", "建筑", "楼房", "房屋", "大棚"]},
    {"zh": "道路", "en": "road", "labels": ["road"],
     "aliases": ["道路", "路面", "马路", "公路"]},
    {"zh": "桥梁", "en": "bridge", "labels": ["bridge"],
     "aliases": ["桥梁", "桥"]},
    {"zh": "树木", "en": "tree", "labels": ["tree"],
     "aliases": ["树木", "树", "树林", "林木"]},
    {"zh": "狗", "en": "dog", "labels": ["dog"], "aliases": ["狗", "犬"]},
    {"zh": "猫", "en": "cat", "labels": ["cat"], "aliases": ["猫"]},
    {"zh": "牛", "en": "cow", "labels": ["cow"], "aliases": ["牛", "水牛", "黄牛"]},
    {"zh": "羊", "en": "sheep", "labels": ["sheep"], "aliases": ["羊", "绵羊", "山羊"]},
    {"zh": "鸟", "en": "bird", "labels": ["bird"], "aliases": ["鸟", "飞鸟", "鸟群"]},
    {"zh": "垃圾桶", "en": "trash can", "labels": ["trash_can"],
     "aliases": ["垃圾桶", "垃圾箱", "废弃物"]},
    {"zh": "漂浮物", "en": "floating debris", "labels": ["debris"],
     "aliases": ["漂浮物", "浮垃圾", "水面垃圾", "水藻", "浮萍"]},
]

SCENE_LEXICON: List[Dict[str, str]] = [
    {"zh": "水域", "en": "water",
     "aliases": ["水面", "水里", "水中", "水域", "河道", "河中", "河里", "河", "湖",
                 "湖泊", "海面", "海里", "池塘", "水池", "水库", "江", "水"]},
    {"zh": "岸边", "en": "shore", "aliases": ["岸边", "岸上", "堤岸", "河岸", "海滨"]},
    {"zh": "道路", "en": "road", "aliases": ["道路上", "路面", "公路上", "马路"]},
    {"zh": "空中", "en": "sky", "aliases": ["空中", "天空", "高空"]},
]

ATTR_LEXICON: List[Dict[str, str]] = [
    {"zh": "红色", "en": "red", "aliases": ["红色", "红的", "红衣", "红"]},
    {"zh": "蓝色", "en": "blue", "aliases": ["蓝色", "蓝的", "蓝"]},
    {"zh": "黑色", "en": "black", "aliases": ["黑色", "黑的", "黑"]},
    {"zh": "白色", "en": "white", "aliases": ["白色", "白的", "白"]},
    {"zh": "黄色", "en": "yellow", "aliases": ["黄色", "黄的", "黄"]},
    {"zh": "绿色", "en": "green", "aliases": ["绿色", "绿的", "绿"]},
    {"zh": "橙色", "en": "orange", "aliases": ["橙色", "橘色"]},
    {"zh": "小型", "en": "small", "aliases": ["小型", "小型船只", "小的"]},
    {"zh": "大型", "en": "large", "aliases": ["大型", "大的"]},
]

#: 否定标记 —— 出现在目标词之前时，该目标转为 absence 条件（"未穿救生衣"）
ABSENCE_MARKERS: List[str] = [
    "未穿", "没穿", "没有穿", "未佩戴", "没佩戴", "不戴", "未戴",
    "没有", "未", "不带", "无",
]

#: 常见的"组合条件 → 中文展示名"映射，命中后给出更贴近业务的告警文案
DISPLAY_NAME_RULES: List[Dict[str, Any]] = [
    {"require_targets": {"person"}, "require_scenes": {"water"},
     "require_absence": {"life_jacket"}, "display_name": "疑似野泳人员"},
    {"require_targets": {"person"}, "require_scenes": {"water"},
     "display_name": "水域人员"},
    {"require_targets": {"boat"}, "require_scenes": {"water"},
     "display_name": "水面船只"},
    {"require_targets": {"car"}, "require_scenes": {"road"},
     "display_name": "道路车辆"},
]

# ── 数据模型 ─────────────────────────────────────────────────────────────────


@dataclass
class Target:
    """一个待检测目标。

    Attributes:
        name_zh: 中文展示名，用于前端 chips 与告警文案。
        name_en: 英文名（人类可读，可含空格——仅用于展示）。
        detection_label: **下发 YOLOE 的单 token**，必须无空格。
        labels: 该目标可用的全部单 token（按优先级），detection_label 取第一个。
        phrases: 自然语言英文描述，供前端"高级"区展示。
        dropped_phrases: 因含空格而无法下发 YOLOE 的短语，保留以便追溯。
    """

    name_zh: str
    name_en: str
    detection_label: str
    labels: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    dropped_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Target":
        return Target(
            name_zh=d.get("name_zh", ""),
            name_en=d.get("name_en", ""),
            detection_label=d.get("detection_label") or (d.get("labels") or [""])[0],
            labels=list(d.get("labels") or []),
            phrases=list(d.get("phrases") or []),
            dropped_phrases=list(d.get("dropped_phrases") or []),
        )


@dataclass
class Condition:
    """RuleEngine 消费的语义条件。

    Attributes:
        type: `scene` / `absence` / `attribute` / `count` 之一。
        value: 条件的值，如 water / red。
        target: 条件作用的目标 label（absence 用），如 life_jacket。
        text: 中文说明，直接给前端展示。
        verifiable: 是否可由规则引擎独立判定；False 表示需要二阶段 VLM 确认。
    """

    type: str
    value: str = ""
    target: str = ""
    text: str = ""
    verifiable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Condition":
        return Condition(
            type=d.get("type", ""),
            value=d.get("value", ""),
            target=d.get("target", ""),
            text=d.get("text", ""),
            verifiable=bool(d.get("verifiable", True)),
        )


@dataclass
class TargetSpec:
    """一次自然语言目标描述的完整、可执行表示。"""

    original_prompt: str
    display_name: str
    targets: List[Target] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)
    logic: str = "AND"
    provider: str = "local-rule"
    warnings: List[str] = field(default_factory=list)

    # ── 便捷视图 ────────────────────────────────────────────────────────────

    def detection_labels(self) -> List[str]:
        """真正下发给 YOLOE 的单 token 列表（去重、保持顺序）。"""
        out: List[str] = []
        for t in self.targets:
            for label in (t.labels or [t.detection_label]):
                if label and label not in out:
                    out.append(label)
        return out

    @property
    def label_to_zh(self) -> Dict[str, str]:
        """单 token → 中文展示名，供检测结果回填中文标签。"""
        m: Dict[str, str] = {}
        for t in self.targets:
            for label in (t.labels or [t.detection_label]):
                m.setdefault(label, t.name_zh)
        return m

    def candidate_prompts(self) -> List[str]:
        """主目标的英文表述（单 token 优先，其后是人可读短语）。

        这些是要交给 YOLOE **高召回**去找的候选，不承担最终语义判定 —— 最终判定
        交给 VLMVerifier。
        """
        out: List[str] = []
        for t in self.targets or []:
            for lb in (t.labels or [t.detection_label]):
                if lb and lb not in out:
                    out.append(lb)
            for ph in t.phrases or []:
                if ph and ph not in out:
                    out.append(ph)
        return out

    def support_prompts(self) -> List[str]:
        """辅助目标的英文表述（如 life jacket / life vest）。

        它们**不下发给 YOLOE 当检测类**（会增加类别、拖慢推理且召回不稳），
        而是写进结构化 Prompt 作为 VLM 要核对的要素。
        """
        out: List[str] = []
        for c in self.conditions or []:
            if c.type == CONDITION_ABSENCE and c.target:
                spaced = c.target.replace("_", " ")
                for v in (c.target, spaced):
                    if v not in out:
                        out.append(v)
        return out

    def summary(self) -> Dict[str, Any]:
        """给前端「目标 / 环境 / 排除 / 判断逻辑」四组展示用的结构化摘要。"""
        targets_zh = [t.name_zh for t in self.targets or [] if t.name_zh]
        scenes_zh = [c.text or c.value for c in self.conditions or []
                     if c.type == CONDITION_SCENE]
        exclude_zh = [c.text or c.target for c in self.conditions or []
                      if c.type == CONDITION_ABSENCE]
        attrs_zh = [c.text or c.value for c in self.conditions or []
                    if c.type == CONDITION_ATTRIBUTE]

        joiner = " OR " if self.logic == "OR" else " AND "
        # 注意：exclude_zh 取的是 condition.text，本身已是「未检出：life jacket」这类
        # 完整表述，不能再套一层「未发现」前缀（否则出现「未发现未检出：…」）。
        logic_parts = list(targets_zh) + list(scenes_zh) + list(exclude_zh)
        logic_parts += attrs_zh
        return {
            "targets": targets_zh,
            "scenes": scenes_zh,
            "exclusions": exclude_zh,
            "attributes": attrs_zh,
            "needs_vlm": [c.text or c.value or c.target for c in self.conditions or []
                          if not c.verifiable],
            "logic": self.logic,
            "logic_text": joiner.join(p for p in logic_parts if p) or "—",
        }

    def observable_labels(self) -> List[str]:
        """真正下发给 YOLOE 的全部类别 = 目标 + 场景。

        场景类别（water / road / sky / shore）不下发就无法判定"在水里"这个条件，
        只能永远显示"待确认"。把它们一并下发，条件才真的可判定。
        注意它们**不是目标**：不参与告警归组，只在 `_observable` 里用于条件判定。
        """
        out: List[str] = self.detection_labels()
        for scene in self.scene_values():
            if scene and scene not in out:
                out.append(scene)
        return out

    def absence_targets(self) -> List[str]:
        return [c.target for c in self.conditions if c.type == CONDITION_ABSENCE and c.target]

    def scene_values(self) -> List[str]:
        return [c.value for c in self.conditions if c.type == CONDITION_SCENE and c.value]

    # ── 校验 / 序列化 ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """返回问题清单（空表示合法）。重点是 YOLOE 的无空格约束。"""
        issues: List[str] = []
        if not self.targets:
            issues.append("未解析出任何目标")
        for t in self.targets:
            if not t.detection_label:
                issues.append(f"目标「{t.name_zh}」缺少 detection_label")
            for label in (t.labels or [t.detection_label]):
                if " " in label:
                    issues.append(
                        f"目标「{t.name_zh}」的标签 {label!r} 含空格，YOLOE 会拒绝；"
                        f"请改用下划线形式（如 life_jacket）")
        for c in self.conditions:
            if c.type not in CONDITION_TYPES:
                issues.append(f"未知条件类型 {c.type!r}")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "display_name": self.display_name,
            "targets": [t.to_dict() for t in self.targets],
            "conditions": [c.to_dict() for c in self.conditions],
            "logic": self.logic,
            "provider": self.provider,
            "warnings": list(self.warnings),
            # 便利字段：前端"高级"区与调试都用它
            "detection_labels": self.detection_labels(),
            # 实际下发给 YOLOE 的类别（目标 + 场景）
            "observable_labels": self.observable_labels(),
            # 需求文档约定的两段式 Prompt 表述
            "candidate_prompts": self.candidate_prompts(),
            "support_prompts": self.support_prompts(),
            # 前端四组展示（目标 / 环境 / 排除 / 判断逻辑）
            "summary": self.summary(),
        }

    @staticmethod
    def from_raw(prompt: str) -> "TargetSpec":
        """**用户原文就是目标** —— 唯一合法的构造入口（替代 PromptAnalyzer）。

        这里刻意不做任何关键词抽取 / 中英转换 / 规则推导 / 外部 LLM 调用：
        VLX-Seek 吃的就是整句自然语言，「水面上」「未穿」「救生衣」「人员」之间的
        语义关系一旦被拆成 `person / swimmer / water` 就永久丢失。

        * `targets` 只放一个 Target，`detection_label` 是原文本身（跨帧稳定，
          规则引擎靠它做空间格归组，label 每帧变一次累计命中就永远攒不起来）；
        * `conditions` 为空 —— 语义判定由检测器一次完成，规则侧不再二次核验；
        * `display_name` 留原文，展示优先级低于「VLX 自己返回的 label」。

        `sanitize_label` 只做一件事：把空格换成下划线。YOLOE 回退路径的
        `set_classes` 有一句 `assert " " not in classes`，中文天然不含空格，
        所以中文原文会原样保留。
        """
        text = (prompt or "").strip()
        label = sanitize_label(text) or "target"
        return TargetSpec(
            original_prompt=text,
            display_name=text,
            targets=[Target(name_zh=text, name_en=text,
                            detection_label=label, labels=[label],
                            phrases=[text], dropped_phrases=[])],
            conditions=[],
            logic="AND",
            provider="raw-query",
            warnings=[],
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TargetSpec":
        return TargetSpec(
            original_prompt=d.get("original_prompt", ""),
            display_name=d.get("display_name", ""),
            targets=[Target.from_dict(t) for t in d.get("targets") or []],
            conditions=[Condition.from_dict(c) for c in d.get("conditions") or []],
            logic=d.get("logic", "AND"),
            provider=d.get("provider", "local-rule"),
            warnings=list(d.get("warnings") or []),
        )


def sanitize_label(text: str) -> str:
    """把任意英文短语转成 YOLOE 可接受的单 token（下划线连接、小写）。"""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in (text or "").strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def looks_like_detection_label(text: str) -> bool:
    """判断一个串是否可以直接作为 YOLOE 类别（无空格且非空）。"""
    return bool(text) and " " not in text


def spec_summary(spec: "TargetSpec") -> Optional[str]:
    """给日志用的一行摘要。"""
    if not spec:
        return None
    labels = ",".join(spec.detection_labels()) or "-"
    return f"{spec.display_name} | labels={labels} | logic={spec.logic} | provider={spec.provider}"
