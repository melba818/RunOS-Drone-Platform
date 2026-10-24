"""VLXAdapter —— 复用 VLXSeekClient，不重新实现协议。

★绝对约束★
    * **不复制** multipart / HTTP / SSH / bbox parsing —— 一律走 `VLXSeekClient.infer_image()`。
    * **必须显式使用 Deployment**：node_id / port / token / timeout / task / lang
      全部来自 Deployment（+ Model 默认值 + request 覆盖），Adapter 自己不猜。
    * SSH 接入必须显式指定 node_id；直连接入必须保存有效 base_url。
      两种方式都不允许隐式选择其他算力节点。
      （P2B 不改那个 fallback，只保证新链路永远不碰到它。）
    * **box.label 永远是 target.label**，绝不能是 prompt_text()。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..target import DetectionTarget
from .base import (
    AdapterError, AdapterResult, COORD_PIXEL_XYXY, InferenceAdapter, to_image_bytes,
)
from ..auth import resolve_auth

__all__ = ["VLXAdapter"]

DEFAULT_PORT = 8021


class VLXAdapter(InferenceAdapter):
    protocol = "vlx_detection"

    def __init__(self, *, secrets=None, client_factory=None):
        """
        secrets        —— LegacySecretResolver（可注入，便于测试）
        client_factory —— 返回具备 `infer_image()` 的对象；默认 VLXSeekClient。
                          **测试注入 fake client 即可，不需要真的连 A6000。**
        """
        self._secrets = secrets
        self._client_factory = client_factory

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def infer(self, *, image, target: DetectionTarget, deployment: Dict[str, Any],
              options: Optional[Dict[str, Any]] = None,
              model: Optional[Dict[str, Any]] = None) -> AdapterResult:
        dep = deployment or {}
        opts = options or {}
        did = str(dep.get("deployment_id") or "")

        # Require a saved direct endpoint, or an explicit SSH node. Never select another node implicitly.
        transport = str((dep.get("metadata") or {}).get("transport") or "ssh").strip().lower()
        node_id = str(dep.get("node_id") or "").strip()
        if transport not in {"direct", "ssh"}:
            raise AdapterError("VLX 传输方式配置无效")
        if transport == "ssh" and not node_id:
            raise AdapterError("VLX 部署实例未配置 node_id")
        if transport == "direct":
            from urllib.parse import urlsplit
            endpoint = urlsplit(str(dep.get("base_url") or ""))
            if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
                raise AdapterError("VLX 直连接口地址无效")

        # ② 凭据：只读解析，拿不到就报错（不会带着空 token 硬连）
        try:
            token = resolve_auth(dep, self._secrets)
        except AdapterError:
            raise
        except Exception as exc:  # pragma: no cover - 兜底，消息已脱敏
            raise AdapterError(f"auth_ref 解析失败：{type(exc).__name__}") from None

        cfg = self._build_client_config(dep, opts)
        if token:
            cfg["token"] = token

        client = self._client(cfg)
        payload = to_image_bytes(image)
        filename = self._filename(image)

        # ③ 给模型的是 prompt_text()（description + extra_prompt，都空才回退 label）
        prompt = target.prompt_text()
        if not prompt:
            raise AdapterError("VLX 需要目标描述：DetectionTarget 的 label/description 均为空")

        t0 = time.time()
        try:
            res = client.infer_image(
                payload, prompt,
                task=opts.get("task") or "detection",
                lang=opts.get("lang") or "zh",
                max_new_tokens=int(opts.get("max_tokens") or opts.get("max_new_tokens") or 256),
                filename=filename,
            )
        except Exception as exc:
            # 只透传异常类型与脱敏后的消息
            raise AdapterError(f"VLX 推理失败：{type(exc).__name__}: {exc}") from None
        latency_ms = (time.time() - t0) * 1000.0

        return AdapterResult(
            boxes=self._parse_boxes(res, target),
            coord_space=COORD_PIXEL_XYXY,
            latency_ms=round(latency_ms, 1),
            deployment_id=did,
            raw={
                "proposal_count": res.get("proposal_count"),
                # ★VLX 不输出置信度：1.0 是占位，不是模型真值★
                "confidence_source": "synthetic",
                "prompt_sent": prompt,
            },
        )

    # ── 客户端 ──────────────────────────────────────────────────────────────
    def _client(self, cfg: Dict[str, Any]):
        if self._client_factory is not None:
            return self._client_factory(cfg)
        from lmm.vlx_client import VLXSeekClient  # 惰性导入，避免拖起重量依赖
        return VLXSeekClient(cfg)

    @staticmethod
    def _build_client_config(dep: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
        """只放非空的键 —— resolve_config 会忽略 None/''，从而正确回落到旧配置。"""
        meta = dep.get("metadata") or {}
        cfg: Dict[str, Any] = {
            "node_id": dep.get("node_id"),
            "transport": str(meta.get("transport") or "ssh").strip().lower(),
            "task": opts.get("task") or meta.get("task") or "detection",
            "lang": opts.get("lang") or meta.get("lang") or "zh",
        }
        if dep.get("port"):
            cfg["port"] = int(dep["port"])
        if dep.get("base_url"):
            cfg["base_url"] = dep["base_url"]
        if opts.get("timeout"):
            cfg["timeout"] = float(opts["timeout"])
        return {k: v for k, v in cfg.items() if v not in (None, "")}

    @staticmethod
    def _filename(image) -> str:
        if isinstance(image, (str, Path)):
            return Path(image).name
        return "frame.jpg"

    # ── 结果解析 ────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_boxes(res: Dict[str, Any], target: DetectionTarget) -> List[Dict[str, Any]]:
        """VLX 返回的是**原图像素 xyxy**，这里原样透传，不做任何归一化。"""
        boxes: List[Dict[str, Any]] = []
        for det in (res or {}).get("detections") or []:
            bb = det.get("bbox_xyxy") or []
            if len(bb) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bb)
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:      # 丢弃倒置 / 退化框
                continue
            model_label = str(det.get("label") or "").strip()
            boxes.append({
                "xyxy": [x1, y1, x2, y2],
                # ★类别永远来自 target.label —— 绝不写 prompt_text()★
                "label": target.label,
                # 展示用：VLX 自己的措辞，兜底到描述再兜底到类别
                "label_zh": model_label or target.description or target.label,
                "confidence": 1.0,
                "confidence_source": "synthetic",   # 不是 VLX 真值，别当分数用
                "raw": det,
            })
        return boxes
