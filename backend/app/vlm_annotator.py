"""
VLM auto-annotation adapter — let a vision-language model label images.

Why this module exists
----------------------
Every other model type in ``model_integration._run_inference_unlocked`` is a
*detector*: it returns structured tensors and we just reshape them. A VLM is
different — it returns **text**, which brings four problems a detector never has:

  1. Coordinate space is whatever the model feels like. Qwen2.5-VL emits
     absolute pixels relative to its own internal resize; GPT-4o has no native
     bbox output at all; some models use 0-1 floats. We therefore *dictate* the
     space in the prompt (default 0-1000, like Qwen) and convert back using the
     original image size.
  2. The JSON is unreliable. Models wrap it in ```json fences, add prose, or
     truncate. ``parse_objects`` is deliberately forgiving.
  3. Labels are free-form. ``_save_annotations`` **appends unknown class names
     to data.yaml** (see ``_update_yolo_classes``), so "person" / "Person" /
     "a person" would silently grow the class table on every run. We constrain
     labels to the configured class list and drop anything that doesn't match.
  4. Generated confidence values are not detector scores. Do not request or
     threshold them. Keep 1.0 only for legacy consumers, explicitly marked as
     synthetic; validation still checks geometry, classes and target labels.

Transport uses plain ``requests`` for direct APIs or the saved compute-node
SSH tunnel for server APIs. Both use the OpenAI-compatible ``/chat/completions``
schema supported by GPT-4o / Qwen-VL / vLLM / Ollama / LocalAI. No vendor SDK is required.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import settings

PROVIDERS_FILE = settings.PLATFORM_MODELS_DIR / "vlm_providers.json"

# Coordinate conventions we know how to decode.
COORD_SPACES = ("0-1000", "pixel", "0-1")

# What to do when the model returns a label outside the configured class list.
UNMATCHED_POLICIES = ("drop", "keep")


# ── Provider registry ────────────────────────────────────────────────────────
@dataclass
class VLMProvider:
    """One remote VLM endpoint."""

    id: str
    name: str
    base_url: str = ""                 # direct API; SSH providers use node_id / port / api_path
    model: str = ""                    # e.g. qwen2.5-vl-72b-instruct
    api_key: str = ""
    classes: List[str] = field(default_factory=list)
    coord_space: str = "0-1000"        # one of COORD_SPACES
    max_side: int = 1536               # downscale before upload; normalised coords are unaffected
    timeout: int = 120
    extra_instruction: str = ""
    unmatched: str = "drop"            # one of UNMATCHED_POLICIES
    extra_headers: Dict[str, str] = field(default_factory=dict)
    transport: str = "direct"          # direct | ssh (compute-node loopback tunnel)
    node_id: str = ""
    port: int = 8000
    api_path: str = "/v1"

    def to_dict(self, hide_secret: bool = False) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "classes": list(self.classes),
            "coord_space": self.coord_space,
            "max_side": self.max_side,
            "timeout": self.timeout,
            "extra_instruction": self.extra_instruction,
            "unmatched": self.unmatched,
            "extra_headers": dict(self.extra_headers),
            "transport": self.transport,
            "node_id": self.node_id,
            "port": self.port,
            "api_path": self.api_path,
        }
        # Never echo the full key back to the UI — just say whether one is set.
        d["api_key"] = ("***" if self.api_key else "") if hide_secret else self.api_key
        d["has_api_key"] = bool(self.api_key)
        return d


def load_providers() -> Dict[str, VLMProvider]:
    if not PROVIDERS_FILE.exists():
        return {}
    try:
        raw = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, VLMProvider] = {}
    for pid, cfg in (raw or {}).items():
        try:
            out[pid] = VLMProvider(
                id=pid,
                name=cfg.get("name", pid),
                base_url=cfg.get("base_url", ""),
                model=cfg.get("model", ""),
                api_key=cfg.get("api_key", ""),
                classes=list(cfg.get("classes", [])),
                coord_space=cfg.get("coord_space", "0-1000"),
                max_side=int(cfg.get("max_side", 1536)),
                timeout=int(cfg.get("timeout", 120)),
                extra_instruction=cfg.get("extra_instruction", ""),
                unmatched=cfg.get("unmatched", "drop"),
                extra_headers=dict(cfg.get("extra_headers", {})),
                transport=cfg.get("transport", "direct"),
                node_id=cfg.get("node_id", ""),
                port=int(cfg.get("port", 8000)),
                api_path=cfg.get("api_path", "/v1"),
            )
        except Exception:
            continue
    return out


def save_providers(providers: Dict[str, VLMProvider]) -> None:
    PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        pid: {
            k: v for k, v in p.to_dict().items()
            if k not in ("id", "api_key", "has_api_key")
        } | {"api_key": p.api_key}
        for pid, p in providers.items()
    }
    tmp = PROVIDERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROVIDERS_FILE)


# ── Image encoding ──────────────────────────────────────────────────────────
def encode_image(image_path: Path, max_side: int = 1536) -> Tuple[str, int, int]:
    """Return (data_url, original_width, original_height).

    The image may be downscaled for upload, but we always report the *original*
    dimensions so normalised coordinates are decoded against the right frame.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        if max_side and max(img.size) > max_side:
            scale = max_side / float(max(img.size))
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", orig_w, orig_h


# ── Prompt ──────────────────────────────────────────────────────────────────
def build_prompt(classes: List[str], coord_space: str, extra_instruction: str = "",
                 user_prompt: Optional[str] = None,
                 target_label: Optional[str] = None) -> str:
    if coord_space == "0-1000":
        coord_rule = (
            "Use the Qwen convention: bbox is [x1, y1, x2, y2] with each value an "
            "integer from 0 to 1000, where 0 is the left/top edge and 1000 is the "
            "right/bottom edge of the image."
        )
    elif coord_space == "pixel":
        coord_rule = "bbox is [x1, y1, x2, y2] in absolute pixel coordinates of the image."
    else:
        coord_rule = "bbox is [x1, y1, x2, y2] as floats normalised to 0.0-1.0."

    class_rule = (
        "You MUST choose each label from this exact list: " + ", ".join(classes) + ".\n"
        "Never invent a new label, never change the spelling, never pluralise."
        if classes else
        "Use a short lowercase English noun for each label."
    )

    instructions = []
    if target_label:
        instructions.append(
            f'This request targets only the class "{target_label}". '
            "Ignore every other class, even if it is present in the allowed class list."
        )
    if extra_instruction:
        instructions.append("Provider instruction: " + extra_instruction.strip())
    if user_prompt:
        instructions.append("Current target description and constraints: " + user_prompt.strip())
    target = "\n".join(instructions)

    return (
        "You are a precise object detection labeler. "
        "Locate every visible target instance in the image.\n"
        f"{class_rule}\n"
        f"{coord_rule}\n"
        "x1,y1 is the top-left corner and x2,y2 the bottom-right corner.\n"
        f"{target}\n"
        "Reply with JSON only, no prose, in this exact shape:\n"
        '{"objects": [{"label": "<label>", "bbox": [x1, y1, x2, y2]}]}\n'
        'If there is no target, reply exactly: {"objects": []}'
    )


# ── Forgiving JSON parsing ──────────────────────────────────────────────────
def parse_objects(text: str) -> List[Dict[str, Any]]:
    """Extract the objects array from a model reply that may be messy."""
    if not text:
        return []
    cleaned = text.strip()

    # Strip ```json / ``` fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()

    # Fall back to the outermost JSON object/array in the text.
    if not cleaned.startswith(("{", "[")):
        starts = [m.start() for m in re.finditer(r"[{[]", cleaned)]
        for s in starts:
            for e in range(len(cleaned), s, -1):
                candidate = cleaned[s:e]
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    continue
                return _unwrap(parsed)
        return []

    try:
        return _unwrap(json.loads(cleaned))
    except Exception:
        return _unwrap(_salvage_truncated(cleaned))


def _unwrap(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        for key in ("objects", "detections", "boxes", "results", "annotations"):
            v = parsed.get(key)
            if isinstance(v, list):
                return [o for o in v if isinstance(o, dict)]
        # A single object was returned.
        return [parsed] if "bbox" in parsed or "box" in parsed else []
    if isinstance(parsed, list):
        return [o for o in parsed if isinstance(o, dict)]
    return []


def _salvage_truncated(text: str) -> Any:
    """Best-effort recovery when the reply was cut off mid-array."""
    m = re.search(r"\{\s*\"objects\"\s*:\s*\[(.*)", text, re.S)
    if not m:
        return {}
    body = m.group(1)
    objs = []
    for mm in re.finditer(r"\{[^{}]*\}", body):
        try:
            objs.append(json.loads(mm.group(0)))
        except Exception:
            continue
    return {"objects": objs}


# ── Label normalisation (prevents class-table explosion) ─────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(s).lower())


def match_label(label: str, classes: List[str]) -> Optional[str]:
    """Map a free-form model label onto a configured class name."""
    if not classes:
        return str(label).strip() or None
    raw = str(label).strip()
    if raw in classes:
        return raw
    n = _norm(raw)
    for c in classes:
        if _norm(c) == n:
            return c
    # Substring match either direction — handles "a person" vs "person".
    for c in classes:
        nc = _norm(c)
        if nc and (nc in n or n in nc):
            return c
    return None


# ── Coordinate decoding ─────────────────────────────────────────────────────
def decode_bbox(bbox: Any, coord_space: str, w: int, h: int) -> Optional[List[float]]:
    """Convert a model bbox to the internal [x, y, w, h] pixel form."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        vals = [float(v) for v in bbox[:4]]
    except Exception:
        return None

    if coord_space == "0-1000":
        x1, y1, x2, y2 = [v / 1000.0 * (w if i % 2 == 0 else h) for i, v in enumerate(vals)]
    elif coord_space == "0-1":
        x1, y1, x2, y2 = vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h
    else:  # pixel
        x1, y1, x2, y2 = vals

    # Some models emit cx, cy, w, h or reversed corners — normalise corners.
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # Clamp to the image so a hallucinated box can't produce negative anchors.
    x1 = max(0.0, min(float(w), x1))
    y1 = max(0.0, min(float(h), y1))
    bw = min(bw, float(w) - x1)
    bh = min(bh, float(h) - y1)
    return [x1, y1, bw, bh]


# ── Transport ───────────────────────────────────────────────────────────────
def _endpoint(base_url: str) -> str:
    b = (base_url or "").rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return b + "/chat/completions"


def request_openai_compatible(provider: VLMProvider, method: str, endpoint: str,
                              payload: Optional[Dict[str, Any]] = None,
                              timeout: Optional[float] = None) -> Dict[str, Any]:
    """Call an OpenAI-compatible API directly or through a saved compute-node SSH tunnel."""
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    headers.update(provider.extra_headers or {})
    request_timeout = float(timeout or provider.timeout)

    if provider.transport == "ssh":
        if not provider.node_id or not provider.port:
            raise RuntimeError("服务器接入缺少算力节点或端口")
        api_path = "/" + str(provider.api_path or "/v1").strip("/")
        path = api_path.rstrip("/") + "/" + endpoint.lstrip("/")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        if body is not None:
            headers["Content-Length"] = str(len(body))
        from lmm.tunnel_http import tunnel_http_request_persistent
        status, raw = tunnel_http_request_persistent(
            provider.node_id, int(provider.port), method.upper(), path,
            body, headers, request_timeout,
        )
        if status >= 400:
            raise RuntimeError(f"HTTP {status}: {raw[:400].decode('utf-8', 'replace')}")
        return json.loads(raw.decode("utf-8")) if raw else {}

    base = (provider.base_url or "").rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if not base:
        raise RuntimeError("接口地址不能为空")
    response = requests.request(
        method.upper(), f"{base}/{endpoint.lstrip('/')}",
        headers=headers, json=payload, timeout=request_timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
    return response.json()


def call_vlm(provider: VLMProvider, data_url: str, prompt: str,
             retries: int = 2) -> str:
    payload = {
        "model": provider.model,
        "temperature": 0,
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            data = request_openai_compatible(provider, "POST", "chat/completions", payload)
            return _extract_content(data)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"VLM call failed: {last_err}")


def _extract_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message", {})
        c = msg.get("content")
        if isinstance(c, list):  # some servers return content parts
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""
    # Non-OpenAI shapes
    for key in ("output", "content", "text", "response"):
        v = data.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for k2 in ("text", "content"):
                if isinstance(v.get(k2), str):
                    return v[k2]
    return ""


# ── Main entry ──────────────────────────────────────────────────────────────
def annotate_image(provider: VLMProvider, image_path: Path,
                   confidence_threshold: float = 0.5,
                   user_prompt: Optional[str] = None,
                   target_label: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one image through the VLM and return (annotations, debug)."""
    target_label = (target_label or "").strip() or None
    if target_label and target_label not in provider.classes:
        raise ValueError(
            f'所选类别「{target_label}」不在当前数据集类别列表中；'
            "请先在标注页为当前数据集添加或选择已有类别。"
        )
    data_url, w, h = encode_image(image_path, provider.max_side)
    prompt = build_prompt(provider.classes, provider.coord_space,
                          provider.extra_instruction, user_prompt, target_label)

    t0 = time.time()
    raw = call_vlm(provider, data_url, prompt)
    elapsed = time.time() - t0

    objects = parse_objects(raw)
    annotations: List[Dict[str, Any]] = []
    dropped = 0

    for idx, obj in enumerate(objects):
        label_raw = obj.get("label") or obj.get("name") or obj.get("class") or ""
        bbox_raw = obj.get("bbox") or obj.get("box") or obj.get("bbox_2d")

        box = decode_bbox(bbox_raw, provider.coord_space, w, h)
        if box is None:
            dropped += 1
            continue

        label = match_label(label_raw, provider.classes)
        if label is None:
            if provider.unmatched == "keep":
                label = str(label_raw).strip() or "object"
            else:
                dropped += 1
                continue

        # UI 中统一的「类别」选择是本次标注的硬约束，不能让模型返回同配置里的其他类别。
        if target_label and label != target_label:
            dropped += 1
            continue

        # confidence_threshold is accepted for legacy clients only. Language
        # generated scores, including malformed values, never discard valid boxes.
        class_id = provider.classes.index(label) if label in provider.classes else idx
        annotations.append({
            "type": "bbox",
            "class_id": class_id,
            "class_name": label,
            "confidence": 1.0,
            "confidence_source": "synthetic",
            "bbox": box,
        })

    debug = {
        "raw": raw[:2000],
        "parsed_objects": len(objects),
        "dropped": dropped,
        "below_threshold": 0,
        "supports_confidence_threshold": False,
        "elapsed_s": round(elapsed, 2),
        "image_size": [w, h],
        "prompt": prompt,
    }
    return annotations, debug
