"""Online visual-model detector backed by saved visual-provider configuration."""

from __future__ import annotations

import json
import ipaddress
import re
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import vision_registry
from lmm.detectors.base_detector import DetectedBox, DetectorError, FrameResult


def _is_remote_vision_model(provider: Any) -> bool:
    identity = f"{getattr(provider, 'name', '')} {getattr(provider, 'model', '')}".lower()
    base_url = str(getattr(provider, "base_url", ""))
    # VLX-Seek is deliberately not an option in this detector. The selectable
    # models here use the saved OpenAI-compatible VLM provider configuration.
    # SSH providers are explicitly registered through the general model path,
    # so a model name containing "vlx" does not change its transport/protocol.
    if getattr(provider, "transport", "direct") == "ssh":
        return bool(getattr(provider, "node_id", "") and getattr(provider, "port", 0))
    if "vlx" in f"{identity} {base_url.lower()}":
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    if not host or host == "localhost":
        return False
    try:
        address = ipaddress.ip_address(host)
        if address.is_loopback or address.is_unspecified:
            return False
    except ValueError:
        pass
    # A remote compute node may itself run a vLLM/Ollama-compatible endpoint.
    # Its configured hostname, rather than the transport kind, tells us it is remote.
    return True


def online_vision_models() -> list[Dict[str, Any]]:
    """Legacy alias for saved compatible services, including this server's loopback API."""
    return configured_vision_models()


def configured_vision_models() -> list[Dict[str, Any]]:
    """Return all ready saved VLMs, including services running on this machine."""
    providers = []
    for item in vision_registry.list_models():
        provider = vision_registry.get(str(item.get("config_id") or ""))
        if item.get("ready") and provider is not None:
            providers.append({
                "config_id": item["config_id"],
                "display_name": item.get("display_name") or item["config_id"],
                "model": item.get("model") or "",
                "kind_label": item.get("kind_label") or "视觉大模型",
                "supports_confidence_threshold": False,
            })
    return providers


def require_configured_vision_model(provider_id: str):
    """Resolve any saved vision provider; inference routes never accept raw endpoints."""
    try:
        return vision_registry.require(provider_id)
    except LookupError as exc:
        raise DetectorError(str(exc)) from exc


def require_online_vision_model(provider_id: str):
    """The endpoint must be saved; its network location does not change its protocol."""
    return require_configured_vision_model(provider_id)


class VisionModelEngine:
    """Run full-frame semantic detection against a saved remote vision provider."""

    name = "vision-model"

    def __init__(self, provider_id: str, *, max_tokens: int = 1024,
                 allow_local: bool = False) -> None:
        self.provider = (require_configured_vision_model(provider_id) if allow_local
                         else require_online_vision_model(provider_id))
        self.max_tokens = max(128, int(max_tokens))
        self.prompt = ""

    def set_prompt(self, prompt: str) -> None:
        self.prompt = str(prompt or "").strip()

    def health(self) -> Dict[str, Any]:
        """Validate local saved configuration only; deliberately makes no network request."""
        return {"ok": True, "provider": self.provider.name, "model": self.provider.model}

    @staticmethod
    def _parse(text: str, width: int, height: int, prompt: str) -> list[DetectedBox]:
        raw = str(text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise DetectorError("视觉大模型返回内容不是有效 JSON")
            try:
                parsed = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as exc:
                raise DetectorError("视觉大模型返回内容不是有效 JSON") from exc

        records = parsed if isinstance(parsed, list) else parsed.get("detections", []) if isinstance(parsed, dict) else []
        if not isinstance(records, list):
            raise DetectorError("视觉大模型返回缺少 detections 列表")

        boxes = []
        for item in records:
            if not isinstance(item, dict):
                continue
            coords = item.get("bbox") or item.get("box")
            if not isinstance(coords, (list, tuple)) or len(coords) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in coords[:4]]
            except (TypeError, ValueError):
                continue
            # Prompt explicitly requests 0–1000 coordinates to avoid ambiguity.
            x1, x2 = max(0.0, min(width, x1 * width / 1000.0)), max(0.0, min(width, x2 * width / 1000.0))
            y1, y2 = max(0.0, min(height, y1 * height / 1000.0)), max(0.0, min(height, y2 * height / 1000.0))
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(item.get("label") or item.get("category") or prompt or "目标").strip()
            label_zh = str(item.get("label_zh") or label)
            boxes.append(DetectedBox(x1=x1, y1=y1, x2=x2, y2=y2,
                                     confidence=1.0, confidence_source="synthetic",
                                     label=label, label_zh=label_zh))
        return boxes

    def detect(self, frame: Any, frame_index: int = 0,
               request_id: Optional[str] = None) -> FrameResult:
        try:
            height, width = frame.shape[:2]
            image = vision_registry.frame_to_data_url(
                frame, max_side=max(256, int(self.provider.max_side or 1536)), quality=88)
            prompt = (
                "请根据下面的检测要求检查整张图像，只输出 JSON，不要 Markdown 或解释。\n"
                f"检测要求：{self.prompt}\n"
                "格式：{\"detections\":[{\"label\":\"目标名称\",\"label_zh\":\"中文名称\","
                "\"bbox\":[x1,y1,x2,y2]}]}。"
                "bbox 是左上与右下坐标，按图像宽高归一化到 0–1000。"
                "只返回符合检测要求的目标；没有目标时返回 {\"detections\":[]}。"
            )
            text, elapsed_ms = vision_registry.chat(
                self.provider, [image], prompt, max_tokens=self.max_tokens,
                timeout=self.provider.timeout, retries=1)
            boxes = self._parse(text, int(width), int(height), self.prompt)
            result = FrameResult(frame_index=int(frame_index), width=int(width), height=int(height),
                                 boxes=boxes, elapsed_ms=float(elapsed_ms))
            result.timing = {"total_ms": round(float(elapsed_ms), 1),
                             "provider": self.provider.name,
                             "model": self.provider.model}
            return result
        except DetectorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"在线视觉大模型检测失败：{exc}") from exc


# Kept for any older Python integration importing the former Qwen-only name.
QwenVLEngine = VisionModelEngine


def online_qwen_models() -> list[Dict[str, Any]]:
    """Backward-compatible Qwen-only listing for the video inference page."""
    return [item for item in online_vision_models()
            if "qwen" in f"{item['display_name']} {item['model']}".lower()]


def require_online_qwen(provider_id: str):
    provider = require_online_vision_model(provider_id)
    if "qwen" not in f"{provider.name} {provider.model}".lower():
        raise DetectorError("所选视觉大模型不是已配置的在线 Qwen-VL 模型")
    return provider
