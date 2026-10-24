"""RunOS 通用视觉大模型注册表。

**这是"复用层"，不是"另一套配置系统"。** 配置的读写、HTTP 客户端、图片编码全部沿用
`vlm_annotator.py` 里既有的实现，「大模型自动标注」和「多模态目标检测」共用同一份
`workspace/models/vlm_providers.json`：

    vlm_annotator.load_providers()  ← 唯一真源
    vlm_annotator.call_vlm()        ← 单图调用（本模块单图路径直接转发，零改写）
    vlm_annotator.encode_image()    ← 磁盘图片 → data URL

本模块只补三件自动标注不需要的事：

1. **provider 类别推断**。DashScope / OpenAI / vLLM / Ollama 等都能走 OpenAI 兼容
   协议；服务器端点还可通过已保存的 SSH 算力节点访问，不需要单独的 Provider 类。
2. **多图调用**。`call_vlm` 只发一张图；视觉复核需要「示例参考图 + 候选目标裁剪」，
   连续多帧行为理解需要「同一 track 的 N 张裁剪」。`chat()` 支持 N 张，重试语义与
   `call_vlm` 保持一致。
3. **可选性探测**。`probe()` 用一次极小的文本请求判断这个 Provider 现在能不能用，
   供任务启动前预检（不消耗图片额度）。

⚠️ 本模块**不得**修改 `vlm_annotator.py` 的任何现有行为，否则会破坏自动标注功能。
"""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from vlm_annotator import (  # noqa: F401  (VLMProvider 转出给调用方使用)
    COORD_SPACES,
    PROVIDERS_FILE,
    VLMProvider,
    _endpoint,
    _extract_content,
    call_vlm,
    encode_image,
    load_providers,
    request_openai_compatible,
    save_providers,
)

LOG = logging.getLogger(__name__)

#: base_url 特征 → 类别。顺序敏感：先匹配到的胜出。
_KIND_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("dashscope", ("dashscope.aliyuncs.com", "compatible-mode", "qianwen", "aliyun")),
    ("ollama", (":11434", "ollama")),
    ("vllm", (":8001", "vllm")),
    ("openai", ("api.openai.com",)),
]

KIND_LABEL: Dict[str, str] = {
    "dashscope": "通义千问 Qwen-VL",
    "openai": "OpenAI GPT",
    "vllm": "Local vLLM",
    "ollama": "Local Ollama",
    "openai-compatible": "OpenAI 兼容接口",
    "compute-node": "服务器算力节点",
}

#: 本地部署的类别：不存在"把模型加载进 RunOS 显存"这回事，复用对方常驻服务即可。
LOCAL_KINDS = ("vllm", "ollama")

#: 单图调用时 `call_vlm` 固定的 max_tokens（那段代码不属于本模块，不改）
_TRANSPORT_MAX_TOKENS = 2048


def kind_of(base_url: str) -> str:
    """从 base_url 推断 provider 类别。推断不出也算合法（通用 OpenAI 兼容）。"""
    b = (base_url or "").lower()
    for kind, needles in _KIND_RULES:
        if any(n in b for n in needles):
            return kind
    return "openai-compatible"


def is_local(base_url: str) -> bool:
    """本地部署（vLLM / Ollama）与远程 API 的区别决定要不要考虑连接池/显存。"""
    return kind_of(base_url) in LOCAL_KINDS


# ── 读取现有配置 ─────────────────────────────────────────────────────────────


def describe(p: VLMProvider) -> Dict[str, Any]:
    """给前端下拉用的一条记录。**不含 api_key**（自动标注页也只回 '***'）。"""
    kind = "compute-node" if p.transport == "ssh" else kind_of(p.base_url)
    return {
        "config_id": p.id,
        "display_name": p.name or p.id,
        "kind": kind,
        "kind_label": KIND_LABEL.get(kind, kind),
        "base_url": p.base_url,
        "transport": p.transport,
        "node_id": p.node_id,
        "port": p.port,
        "api_path": p.api_path,
        "model": p.model,
        "is_local": p.transport != "ssh" and is_local(p.base_url),
        "has_api_key": bool(p.api_key),
        # 复核用的默认值直接沿用自动标注页已保存的配置，不要求用户重填
        "default_max_side": p.max_side,
        "default_timeout": p.timeout,
        "coordinate_format": p.coord_space,
        "additional_prompt": p.extra_instruction,
        # 可用的最低门槛：有直接接口或服务器隧道配置，以及模型名
        "ready": bool(p.model and ((p.transport == "ssh" and p.node_id and p.port)
                                   or (p.transport != "ssh" and p.base_url))),
    }


def list_models() -> List[Dict[str, Any]]:
    """全部已保存的视觉大模型配置，按名称排序。"""
    out = [describe(p) for p in load_providers().values()]
    out.sort(key=lambda d: (not d["ready"], d["display_name"]))
    return out


def get(config_id: str) -> Optional[VLMProvider]:
    if not config_id:
        return None
    return load_providers().get(config_id)


def require(config_id: str) -> VLMProvider:
    p = get(config_id)
    if p is None:
        raise LookupError(f"视觉大模型配置不存在：{config_id}")
    if not (p.model and ((p.transport == "ssh" and p.node_id and p.port)
                         or (p.transport != "ssh" and p.base_url))):
        raise LookupError(f"视觉大模型配置不完整（缺服务器连接或模型名）：{config_id}")
    return p


# ── 图片工具 ─────────────────────────────────────────────────────────────────


def frame_to_data_url(frame_bgr: Any, max_side: int = 768, quality: int = 85,
                      min_side: int = 0) -> str:
    """numpy BGR 帧（cv2 读出的格式）→ JPEG data URL，长边压到 max_side。

    `min_side` 用于**放大**过小的裁剪。航拍目标常常只有十几像素，直接发过去模型
    只有一两个 patch 可用，等于让它看马赛克；LANCZOS 放大到几百像素虽然不增加信息，
    但能让模型至少看出轮廓与周边环境。放大是不可逆的信息损失换可用性，默认关闭。
    """
    from PIL import Image

    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR → RGB
    side = max(img.size)
    if max_side and side > max_side:
        scale = max_side / float(side)
        img = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
                         Image.LANCZOS)
    elif min_side and side < min_side:
        scale = min_side / float(max(1, side))
        img = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def crop_with_context(frame_bgr: Any, bbox: List[float], context_scale: float = 1.5,
                      max_side: int = 768, min_window: int = 256) -> Optional[Any]:
    """以候选框为中心扩 `context_scale` 倍再裁剪，返回 BGR 子图。

    为什么必须扩框：判定"是否在水里""是否在道路上"需要环境像素，紧贴目标的裁剪
    里只有目标本身，VLM 只能瞎猜。默认 1.5 表示边长放大 50%。

    为什么还要 `min_window`：**倍数对航拍小目标是失效的**。一个 10×16px 的水面人员，
    扩 1.5 倍也只有 15×24px，窗口内几乎全是目标本身，没有任何环境线索 —— 实测 VLM
    在这种情况下会直接回答"图像极度模糊、证据不足"。所以再加一条绝对下限：窗口长边
    至少 `min_window` 像素（默认 256），不足时以目标为中心向外扩，直到补足。
    """
    if frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except Exception:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    s = max(1.0, float(context_scale or 1.0))
    nw, nh = bw * s, bh * s

    # 小目标补足最小窗口（保持长宽比不断变形，按长边算缩放）
    mw = max(0, int(min_window or 0))
    if mw and max(nw, nh) < mw:
        k = mw / float(max(nw, nh))
        nw, nh = nw * k, nh * k

    nx1 = int(max(0, cx - nw / 2.0)); ny1 = int(max(0, cy - nh / 2.0))
    nx2 = int(min(w, cx + nw / 2.0)); ny2 = int(min(h, cy + nh / 2.0))
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return None
    return frame_bgr[ny1:ny2, nx1:nx2]


# ── 调用 ─────────────────────────────────────────────────────────────────────


def chat(provider: VLMProvider, images: List[str], prompt: str, *,
         max_tokens: int = _TRANSPORT_MAX_TOKENS, timeout: Optional[float] = None,
         retries: int = 1) -> Tuple[str, float]:
    """发一次多图对话，返回 (文本内容, 耗时毫秒)。

    单图且未自定义 token 上限时**直接转发现有 `call_vlm`**（自动标注走的同一条路径）；
    只有需要多图或收紧 token 时才自己拼 payload，重试语义与 `call_vlm` 保持一致。
    """
    if not images:
        raise ValueError("至少需要一张图片")
    if timeout:
        provider = dataclasses.replace(provider, timeout=int(timeout))

    if len(images) == 1 and max_tokens >= _TRANSPORT_MAX_TOKENS:
        t0 = time.time()
        text = call_vlm(provider, images[0], prompt, retries=retries)
        return text, (time.time() - t0) * 1000.0

    content: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": provider.model,
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": content}],
    }

    last_err: Optional[Exception] = None
    t0 = time.time()
    for attempt in range(int(retries) + 1):
        try:
            data = request_openai_compatible(provider, "POST", "chat/completions", payload)
            return _extract_content(data), (time.time() - t0) * 1000.0
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"VLM 调用失败：{last_err}")


def chat_text(provider: VLMProvider, prompt: str, *, max_tokens: int = 1024,
              timeout: Optional[float] = None, retries: int = 1) -> Tuple[str, float]:
    """纯文本调用（不带图）。

    用途：把中文 Prompt 解析成 TargetSpec（阶段二的 AI 解析）。视觉大模型本身也能
    干这件事，没必要再让用户配第二个纯文本模型。
    """
    if timeout:
        provider = dataclasses.replace(provider, timeout=int(timeout))
    payload = {
        "model": provider.model,
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": prompt}],
    }
    last_err: Optional[Exception] = None
    t0 = time.time()
    for attempt in range(int(retries) + 1):
        try:
            data = request_openai_compatible(provider, "POST", "chat/completions", payload)
            return _extract_content(data), (time.time() - t0) * 1000.0
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"VLM 文本调用失败：{last_err}")


def probe(config_id: str, timeout: float = 20.0) -> Dict[str, Any]:
    """可用性预检：一次极小的纯文本请求。不消耗图片额度、不加载任何本地权重。"""
    p = get(config_id)
    if p is None:
        return {"ok": False, "error": "配置不存在", "config_id": config_id}
    connected_config = p.model and (
        (p.transport == "ssh" and p.node_id and p.port)
        or (p.transport != "ssh" and p.base_url)
    )
    if not connected_config:
        return {"ok": False, "error": "配置不完整（缺服务器连接或模型名）", "config_id": config_id}
    t0 = time.time()
    try:
        text, _ = chat(p, [tiny_data_url()], "只回复 ok", max_tokens=16, timeout=timeout)
        return {"ok": True, "config_id": config_id, "model": p.model,
                "kind": "compute-node" if p.transport == "ssh" else kind_of(p.base_url),
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "reply": (text or "").strip()[:40]}
    except Exception as exc:
        return {"ok": False, "config_id": config_id, "error": str(exc)[:300],
                "latency_ms": round((time.time() - t0) * 1000, 1)}


def tiny_data_url() -> str:
    """8×8 纯色 JPEG，用于可用性预检（不占用真实图片额度）。"""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 200, 200)).save(buf, format="JPEG", quality=60)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
