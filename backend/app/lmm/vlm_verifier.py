"""VLM 二阶段语义复核。

YOLOE 只负责"这里有东西"，本模块负责"这到底是不是用户要找的东西"。

三条硬约束（都来自需求）：

1. **不看整张图、也不看紧贴目标的裁剪。** 以候选框为中心扩 `context_scale`（默认 1.5）
   再裁，且长边限制在 `max_side`（默认 768）。4K 无人机原图直接上传既慢又贵，VLM 也
   未必看得更准。
2. **Prompt 由 TargetSpec 动态生成，并强制 JSON 输出。** 属性键是从条件反推出来的
   （scene=water → `in_water`；absence=life_jacket → `life_jacket`），不是写死的模板。
   输出用 Pydantic 校验。
3. **解析失败不能炸任务。** 模型吐 Markdown / 思考块 / 前后夹带解释都是常态，这里做
   剥壳 + 平衡括号抽取 + 有限次（默认 1 次）针对性修复；仍失败就返回 `verified=False`
   的降级结果，由规则引擎按"未复核"处理。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

import vision_registry as registry
from .target_spec import CONDITION_ABSENCE, CONDITION_ATTRIBUTE, CONDITION_SCENE, TargetSpec

LOG = logging.getLogger(__name__)

#: 常见标签 → (JSON 键, 中文问法)。没命中的标签用兜底问法。
_ATTR_QUESTIONS: Dict[str, Tuple[str, str]] = {
    "person": ("person", "画面中心是否存在人员"),
    "swimmer": ("swimming", "人员是否处于游泳或涉水状态"),
    "boat": ("boat", "是否存在船只"),
    "car": ("car", "是否存在汽车"),
    "truck": ("truck", "是否存在卡车"),
    "bus": ("bus", "是否存在客车"),
    "motorcycle": ("motorcycle", "是否存在摩托车"),
    "bicycle": ("bicycle", "是否存在自行车"),
    "drone": ("drone", "是否存在无人机"),
    "airplane": ("airplane", "是否存在飞机"),
    "fire": ("fire", "是否存在明火"),
    "smoke": ("smoke", "是否存在烟雾"),
    "life_jacket": ("life_jacket", "是否观察到救生衣"),
    "helmet": ("helmet", "是否观察到安全帽"),
    "debris": ("debris", "是否存在漂浮物"),
    "trash_can": ("trash_can", "是否存在垃圾桶"),
    "building": ("building", "是否存在建筑物"),
    "tree": ("tree", "是否存在树木"),
    "crane": ("crane", "是否存在起重机"),
    "excavator": ("excavator", "是否存在挖掘机"),
}

_SCENE_QUESTIONS: Dict[str, Tuple[str, str]] = {
    "water": ("in_water", "目标是否处于水域之中"),
    "road": ("on_road", "目标是否位于道路上"),
    "sky": ("in_sky", "目标是否位于空中"),
    "shore": ("on_shore", "目标是否位于岸边"),
}


# ── 输出模型 ─────────────────────────────────────────────────────────────────


class VerificationResult(BaseModel):
    """VLM 的判定结论 + 我们补的运行信息。

    前四个字段来自模型；后面的是运行元数据，**不参与模型校验**，出错时用它解释原因。
    """

    model_config = ConfigDict(extra="ignore")

    matched: bool = False
    confidence: float = 0.0
    attributes: Dict[str, bool] = Field(default_factory=dict)
    reason: str = ""

    # ── 运行元数据（我们自己填） ──
    verified: bool = False          # False = 未拿到可信结论（未复核/降级）
    error: str = ""
    raw: str = ""
    latency_ms: float = 0.0
    attempts: int = 0
    config_id: str = ""
    model_name: str = ""
    image_count: int = 0

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_conf(cls, v: Any) -> float:
        """模型常把置信度写成 91、0.91、"0.91"、"91%" —— 一律归一到 0~1。"""
        try:
            if isinstance(v, str):
                v = v.strip().rstrip("%")
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if f > 1.0:
            f = f / 100.0 if f <= 100.0 else 1.0
        return max(0.0, min(1.0, f))

    @field_validator("attributes", mode="before")
    @classmethod
    def _coerce_attrs(cls, v: Any) -> Dict[str, bool]:
        """把 "true"/"yes"/"不确定"/1 这类宽松值转成 bool；无法判定的键直接丢掉。"""
        if not isinstance(v, dict):
            return {}
        out: Dict[str, bool] = {}
        for k, raw in v.items():
            key = str(k).strip()
            if not key:
                continue
            if isinstance(raw, bool):
                out[key] = raw
                continue
            s = str(raw).strip().lower()
            if s in ("true", "1", "yes", "y", "有", "是"):
                out[key] = True
            elif s in ("false", "0", "no", "n", "无", "否"):
                out[key] = False
            # 其余（"uncertain" / "不确定"）视为未判定，不塞进来
        return out

    @field_validator("reason", "raw", "error", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        return "" if v is None else str(v)

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["latency_ms"] = round(float(self.latency_ms), 1)
        return d


# ── 属性键推导 + Prompt ──────────────────────────────────────────────────────


def attribute_spec(spec: TargetSpec) -> List[Tuple[str, str]]:
    """从 TargetSpec 反推要问 VLM 哪些是非题，返回 [(json_key, 中文问法)]。

    顺序：目标 → 场景 → 缺失 → 属性。去重，保持可读顺序（JSON 里键的顺序也稳定）。
    """
    keys: List[Tuple[str, str]] = []
    seen: set = set()

    def add(key: str, question: str) -> None:
        if key and key not in seen:
            seen.add(key)
            keys.append((key, question))

    for t in spec.targets or []:
        labels = list(t.labels or [t.detection_label])
        added = False
        for lb in labels:
            if lb in _ATTR_QUESTIONS:
                k, q = _ATTR_QUESTIONS[lb]
                add(k, q)
                added = True
        if not added and t.detection_label:
            add(t.detection_label, f"是否存在 {t.name_zh or t.detection_label}")

    for c in spec.conditions or []:
        if c.type == CONDITION_SCENE:
            k, q = _SCENE_QUESTIONS.get(c.value, (f"in_{c.value}", f"目标是否处于{c.text or c.value}"))
            add(k, q)
        elif c.type == CONDITION_ABSENCE:
            k, q = _ATTR_QUESTIONS.get(c.target, (c.target, f"是否观察到{c.text or c.target}"))
            add(k, q)
        elif c.type == CONDITION_ATTRIBUTE:
            add(f"has_{c.value}", f"目标是否符合「{c.text or c.value}」这一外观特征")

    if not keys:
        add("target_visible", "画面中心是否为用户描述的目标")
    return keys


def build_verification_prompt(spec: TargetSpec, attrs: List[Tuple[str, str]],
                              extra_instruction: str = "", temporal: int = 0) -> str:
    """生成严格的结构化判断 Prompt。

    `temporal > 0` 时说明这是连续多帧行为理解，提示词要改成"按时间顺序观察"。
    """
    lines: List[str] = []
    lines.append("你是无人机巡检视觉分析模型。请重点分析图像中心位置的候选目标。")
    if temporal > 1:
        lines.append(
            f"后面会给你同一目标按时间先后排列的 {temporal} 张裁剪图，"
            "请把它们当作一段连续画面来理解目标的行为与状态，"
            "不要只根据其中一张下结论。")
    else:
        lines.append("图中心的目标是检测器给出的候选目标，你需要独立判断它是否符合用户描述。")
    lines.append(f"用户原始描述：「{spec.original_prompt}」。")
    if spec.display_name:
        lines.append(f"业务上的目标名称是「{spec.display_name}」。")

    lines.append("请逐条判断以下问题，并全部给出 true 或 false：")
    for i, (key, q) in enumerate(attrs, 1):
        lines.append(f"  {i}. {q}（JSON 键：{key}）")

    lines.append("判定要求：")
    lines.append("  · 只依据图像中**可见**的证据判断，不要根据常识补全看不见的信息。")
    lines.append("  · 如果画面模糊、遮挡严重或证据不足，把对应的键设为 false，"
                 "并在 reason 里说明「证据不足」，不要猜测。")
    lines.append("  · matched 表示该目标是否**整体符合**用户描述"
                 "（即上面所有必需条件都成立）。")

    lines.append("输出要求：")
    lines.append("  · 只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释文字。")
    lines.append("  · JSON 结构：")
    lines.append('    {"matched": true/false, "confidence": 0.0~1.0, '
                 '"attributes": {' + ", ".join(f'"{k}": true/false' for k, _ in attrs) + '}, '
                 '"reason": "一句话中文说明，写明你看到了什么"}')
    lines.append("  · confidence 是你对自己结论的置信度，0 到 1 之间的小数。")
    if extra_instruction:
        lines.append(f"补充要求：{extra_instruction}")

    if spec.logic == "OR" and len(spec.targets) > 1:
        lines.append("  · 本次为「或」逻辑：命中任意一个目标即视为符合描述。")
    elif len(spec.targets) > 1:
        lines.append("  · 本次为「且」逻辑：列出的目标需同时出现才算符合描述。")
    return "\n".join(lines)


_REPAIR_PROMPT = (
    "你上一次的输出无法被解析为 JSON。请重新输出，**只输出一个合法的 JSON 对象**，"
    "不要 Markdown 代码块、不要解释、不要思考过程。\n"
    "上一次的输出是：\n{}\n"
)


# ── 容错解析 ─────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_THINK_RE = re.compile(r"<(?:think|thinking|reasoning)[^>]*>.*?</(?:think|thinking|reasoning)>", re.S)


def strip_wrappers(text: str) -> str:
    """去掉思考块与 Markdown 代码围栏 —— 思考型模型（qwen3-vl 等）两个都会带。"""
    t = _THINK_RE.sub("", text or "")
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1)
    return t.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从任意文本里抽出第一个**括号平衡**的 JSON 对象。

    比 `find('{') ... rfind('}')` 稳：模型在 JSON 后面继续写解释时，后者会把解释也切进去。
    这里做一次带字符串/转义感知的扫描。
    """
    s = strip_wrappers(text)
    if not s:
        return None
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                    except Exception:
                        break
                    return obj if isinstance(obj, dict) else None
        start = s.find("{", start + 1)
    # 最后再试一次直接 loads（应对没有大括号包裹数组等边缘情况）
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ── Verifier ─────────────────────────────────────────────────────────────────


class VLMVerifier:
    """一次复核 = 裁剪 → 组 Prompt → 调模型 → 校验（失败则有限次修复）。"""

    def __init__(self, config_id: str, *, context_scale: float = 1.5, max_side: int = 768,
                 timeout: Optional[float] = None, max_repair: int = 1,
                 extra_instruction: str = "", max_tokens: int = 1024,
                 min_window: int = 256, min_side: int = 384) -> None:
        self.config_id = config_id
        #: 供 UI 展示"用哪个模型复核"。build_verifier 会用 provider.model 填上；
        #: 拿不到时（配置被删）留空，不编造。
        self.model_name = ""
        self.context_scale = float(context_scale or 1.5)
        self.max_side = int(max_side or 768)
        #: 小目标的最小上下文窗口（原图像素）与送图最小边长（放大下限）
        self.min_window = int(min_window or 0)
        self.min_side = int(min_side or 0)
        self.timeout = timeout
        self.max_repair = max(0, int(max_repair))
        self.max_tokens = int(max_tokens)
        # 额外要求取自自动标注页同一份配置（additional_prompt），不要求用户重填
        self.extra_instruction = extra_instruction or ""

    # ── 图片准备 ─────────────────────────────────────────────────────────────

    def _images(self, frame: Any, bbox: List[float],
                track_context: Optional[Dict[str, Any]]) -> List[str]:
        """返回要发给 VLM 的图片列表（顺序即时间/参考顺序）。"""
        ctx = track_context or {}
        images: List[str] = []

        # 连续多帧行为理解：优先用调用方准备好的历史裁剪
        history = ctx.get("history") or []
        if history:
            images.extend(history)

        if not images:
            crop = registry.crop_with_context(frame, bbox, self.context_scale,
                                              self.max_side, self.min_window)
            if crop is None:
                return []
            images.append(registry.frame_to_data_url(
                crop, self.max_side, min_side=self.min_side))

        # Visual Prompt：用户给的正样本参考图放在最后，并用文字指明它是参考
        ref = ctx.get("visual_prompt_data_url")
        if ref:
            images.append(ref)
        return images

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def verify(self, frame: Any, bbox: List[float], spec: TargetSpec,
               track_context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        """复核一个候选目标。**任何异常都在这里被吸收**，绝不向上抛。"""
        ctx = track_context or {}
        t_track = ctx.get("track_id")
        base = VerificationResult(config_id=self.config_id)

        # 本地/远程都一样：只发 HTTP，不存在"把模型加载进 RunOS 显存"
        try:
            provider = registry.require(self.config_id)
        except Exception as exc:
            base.error = f"视觉大模型不可用：{exc}"
            LOG.warning("[VLM] %s", base.error)
            return base
        base.model_name = provider.model
        self.model_name = provider.model

        temporal = int(ctx.get("temporal_frames") or 0)
        images = self._images(frame, bbox, ctx)
        if not images:
            base.error = "候选区域裁剪失败（框过小或越界）"
            return base
        base.image_count = len(images)

        attrs = attribute_spec(spec)
        prompt = build_verification_prompt(spec, attrs, self.extra_instruction, temporal)
        if ctx.get("visual_prompt_data_url"):
            prompt += ("\n  · 最后一张图是用户提供的**正样本参考图**，"
                       "请以它的目标外观特征作为判断基准。")

        total_ms = 0.0
        raw = ""
        attempts = 0
        parsed: Optional[Dict[str, Any]] = None
        last_err = ""

        for attempt in range(self.max_repair + 1):
            attempts += 1
            try:
                text, ms = registry.chat(
                    provider, images, prompt,
                    max_tokens=self.max_tokens, timeout=self.timeout, retries=1)
            except Exception as exc:
                total_ms += 0.0
                last_err = str(exc)[:300]
                base.attempts = attempts
                base.latency_ms = total_ms
                base.error = f"调用失败：{last_err}"
                LOG.warning("[VLM] track=%s 调用失败：%s", t_track, last_err)
                return base

            total_ms += ms
            raw = text or ""
            obj = extract_json_object(raw)
            if obj is not None:
                parsed = obj
                break

            # 解析失败 → 有限次修复（把上一次输出回贴给它，要求只吐 JSON）
            if attempt < self.max_repair:
                LOG.info("[VLM] track=%s 第 %d 次输出无法解析，尝试修复", t_track, attempts)
                prompt = prompt + "\n\n" + _REPAIR_PROMPT.format(raw[:1200])

        base.attempts = attempts
        base.latency_ms = total_ms
        base.raw = raw[:2000]

        if parsed is None:
            base.error = "VLM 输出无法解析为 JSON" + (f"（{last_err}）" if last_err else "")
            LOG.warning("[VLM] track=%s %s | raw=%r", t_track, base.error, raw[:160])
            return base

        try:
            result = VerificationResult(
                **{k: v for k, v in parsed.items()
                   if k in ("matched", "confidence", "attributes", "reason")})
        except Exception as exc:
            base.error = f"输出结构校验失败：{exc}"
            LOG.warning("[VLM] track=%s %s | parsed=%s", t_track, base.error, parsed)
            return base

        # 模型没给全属性键时补齐为 false，保证前端展示的键集合稳定
        for key, _ in attrs:
            result.attributes.setdefault(key, False)

        result.verified = True
        result.raw = raw[:2000]
        result.latency_ms = total_ms
        result.attempts = attempts
        result.config_id = self.config_id
        result.model_name = provider.model
        result.image_count = len(images)
        LOG.info("[VLM] track=%s matched=%s conf=%.2f attrs=%s | %.0fms x%d",
                 t_track, result.matched, result.confidence, result.attributes,
                 result.latency_ms, result.attempts)
        return result


class NullVerifier:
    """关闭复核时使用：什么都不做，让规则引擎按纯 YOLOE 结论出事件。"""

    config_id = ""
    model_name = ""

    def verify(self, frame: Any, bbox: List[float], spec: TargetSpec,
               track_context: Optional[Dict[str, Any]] = None) -> VerificationResult:
        return VerificationResult(verified=False, error="未启用视觉大模型复核")


def build_verifier(cfg: Dict[str, Any]) -> Tuple[Any, str]:
    """按配置造 Verifier，返回 (verifier, 不可用原因)。

    配置里的 `additional_prompt`（自动标注页填的补充要求）会自动带入 Prompt，
    这样两个页面共用同一份配置时行为一致。
    """
    if not cfg.get("vlm_enabled"):
        return NullVerifier(), "复核未启用"
    config_id = (cfg.get("vlm_config_id") or "").strip()
    if not config_id:
        return NullVerifier(), "未选择视觉大模型"
    provider = registry.get(config_id)
    if provider is None:
        return NullVerifier(), f"视觉大模型配置不存在：{config_id}"
    extra = provider.extra_instruction or ""
    v = VLMVerifier(
        config_id,
        context_scale=float(cfg.get("vlm_context_scale", 1.5)),
        max_side=int(cfg.get("vlm_max_side", 768)),
        timeout=float(cfg.get("vlm_timeout") or provider.timeout or 60),
        max_repair=int(cfg.get("vlm_max_repair", 1)),
        extra_instruction=extra,
        max_tokens=int(cfg.get("vlm_max_tokens", 1024)),
        min_window=int(cfg.get("vlm_min_window", 256)),
        min_side=int(cfg.get("vlm_min_side", 384)),
    )
    # 任务创建时就能显示"用哪个模型复核"，不必等第一次调用完成
    v.model_name = provider.model or ""
    return v, ""
