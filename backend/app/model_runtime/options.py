"""参数解析：request options > deployment override > model default。

★为什么 coordinate space 是 default + override★
    同一个逻辑模型未来可能通过不同 server wrapper 暴露不同的坐标格式：
        部署 A 输出 pixel_xyxy
        部署 B 的 OpenAI compatible wrapper 输出 norm_0_1000
    所以坐标格式不能和逻辑模型硬绑定。

    effective_coord_space = deployment.coord_space_override or model.default_coord_space
    —— coord_space 不接受 request 覆盖（它是服务端事实，不是调用方偏好）。

其余参数优先级：
    request options  >  deployment override  >  model default
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["resolve_options", "effective_coord_space", "COORD_SPACES"]

COORD_SPACES = ("pixel_xyxy", "norm_0_1000", "norm_0_1", "yolo_norm")

_PARAMS = (
    # (输出键, request 键, deployment override 键, model default 键, 默认值)
    ("timeout", "timeout", "timeout_override", "default_timeout", None),
    ("max_tokens", "max_tokens", "max_tokens_override", "default_max_tokens", None),
    ("max_new_tokens", "max_new_tokens", "max_tokens_override", "default_max_tokens", None),
    ("task", "task", None, "default_task", None),
    ("lang", "lang", None, "default_lang", None),
)


def _pick(*values):
    for v in values:
        if v is not None:
            return v
    return None


def effective_coord_space(model: Dict[str, Any], deployment: Dict[str, Any]) -> Optional[str]:
    return _pick((deployment or {}).get("coord_space_override"), (model or {}).get("default_coord_space"))


def resolve_options(model: Optional[Dict[str, Any]],
                    deployment: Optional[Dict[str, Any]],
                    request_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """合并三级参数。坐标空间单独处理（只允许 deployment 覆盖 model）。"""
    model = model or {}
    deployment = deployment or {}
    req = request_options or {}

    out: Dict[str, Any] = {}
    for out_key, req_key, dep_key, model_key, fallback in _PARAMS:
        out[out_key] = _pick(
            req.get(req_key),
            deployment.get(dep_key) if dep_key else None,
            model.get(model_key),
            fallback,
        )

    out["coord_space"] = effective_coord_space(model, deployment)
    # 把 request 里其余自定义键原样带过去（不参与优先级判定）
    for k, v in req.items():
        out.setdefault(k, v)
    return out
