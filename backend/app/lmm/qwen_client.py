# -*- coding: utf-8 -*-
"""STEP 3/4：Qwen2.5-VL-3B-Instruct 服务的客户端（RunOS 侧）。

走与 OSTrack 同一条 SSH 隧道（`lmm.tunnel_http`，按端口分槽位，8023 独立一条），
复用**持久连接**（那 ~115 ms 的建连开销在 tile pass 里会被放大成"每块 tile
都多付一次握手"，必须复用）。

刻意**不在 RunOS 进程里加载 Qwen**：模型常驻在 A6000 GPU1 的独立服务进程里
（用户硬要求：不让多个模型共享同一个 Python 进程）。这里只做 HTTP 调用。

失败语义：与 OSTrack 后端一致 —— **不抛**，返回带 `ok=False` 的结构，
让上层决定 drop / review / 回退，绝不让标注任务因一次调用失败而崩。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .tunnel_http import tunnel_http_request_persistent

LOG = logging.getLogger(__name__)

DEFAULT_PORT = 8023
DEFAULT_TIMEOUT = 180.0


def to_data_url(image: Any, fmt: str = "JPEG", quality: int = 88) -> str:
    """PIL.Image / ndarray(BGR) / bytes → `data:image/jpeg;base64,...`。"""
    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
    elif isinstance(image, str):
        return image if image.startswith("data:") else "data:image/jpeg;base64," + image
    else:
        try:
            from PIL import Image
            if isinstance(image, Image.Image):
                img = image
            else:
                import numpy as np
                arr = np.asarray(image)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    #: 约定：ndarray 是 OpenCV 的 BGR
                    arr = arr[:, :, ::-1]
                img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, fmt, quality=int(quality))
            raw = buf.getvalue()
        except Exception as exc:  # noqa: BLE001
            raise ValueError("无法把图像编码为 data url: %s" % exc) from exc
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


class QwenVLClient:
    """Qwen2.5-VL 服务客户端。"""

    def __init__(self, node_id: str = "", port: int = DEFAULT_PORT,
                 token: str = "", timeout: float = DEFAULT_TIMEOUT) -> None:
        self.node_id = node_id or ""
        self.port = int(port or DEFAULT_PORT)
        self.token = token or ""
        self.timeout = float(timeout)
        self._last_health: Dict[str, Any] = {"ok": False, "error": "未连接"}

    # ── 内部 ──────────────────────────────────────────────────────────────
    def _http(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None,
              timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") \
            if payload is not None else None
        headers = {"Content-Type": "application/json", "x-qwen-token": self.token}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        status, data = tunnel_http_request_persistent(
            self.node_id, self.port, method, path, body, headers,
            self.timeout, timing)
        if status != 200:
            raise ConnectionError("http %d: %s"
                                  % (status, (data or b"")[:300].decode("utf-8", "replace")))
        return json.loads(data.decode("utf-8"))

    # ── 对外 ──────────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        try:
            self._last_health = self._http("GET", "/health")
        except Exception as exc:  # noqa: BLE001
            self._last_health = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        return dict(self._last_health)

    def chat(self, image: Any, prompt: str, max_tokens: int = 256,
             system: str = "") -> Dict[str, Any]:
        """图片 + 文字 → 文本。失败返回 `{ok: False, error: ...}`（不抛）。"""
        t0 = time.perf_counter()
        try:
            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(image)}},
                {"type": "text", "text": prompt},
            ]})
            d = self._http("POST", "/v1/chat/completions", {
                "model": "Qwen2.5-VL-3B-Instruct",
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
            })
            content = ""
            for ch in (d.get("choices") or []):
                content = ((ch.get("message") or {}).get("content") or "")
                break
            return {"ok": True, "text": content,
                    "gen_ms": float(d.get("gen_ms") or 0.0),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "usage": d.get("usage") or {}}
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[Qwen] chat 失败: %s", exc)
            return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                    "text": "", "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}

    def verify(self, image: Any, prompt: str,
               max_tokens: int = 256) -> Dict[str, Any]:
        """要求模型给严格 JSON。解析不出来就是 `ok=False` —— **不猜、不伪造**。"""
        t0 = time.perf_counter()
        try:
            d = self._http("POST", "/verify", {
                "image": to_data_url(image),
                "prompt": prompt,
                "max_new_tokens": int(max_tokens),
            })
            return {"ok": bool(d.get("ok")),
                    "json": d.get("json"),
                    "raw": d.get("raw") or "",
                    "gen_ms": float(d.get("gen_ms") or 0.0),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[Qwen] verify 失败: %s", exc)
            return {"ok": False, "json": None, "raw": "",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
