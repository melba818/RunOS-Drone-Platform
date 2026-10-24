"""YOLOE 权重与文本编码器的定位、状态检查。

需求文档要求：
- 模型纳入项目现有模型管理机制，启动时检查是否存在；
- **禁止在生产推理过程中偷偷联网下载模型**；
- 模型不存在时前端要能明确提示"YOLOE 模型未安装"并给出安装入口；
- 文本编码器也要提前准备并缓存。

所以这里只做"查得到 / 查不到"的判定，**任何下载都必须由用户显式触发**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

#: YOLOE 权重的专用目录（与 backbone/ reid/ trackers/ 同级，沿用项目分区习惯）
YOLOE_DIR: Path = settings.MODELS_DIR / "yoloe"


def models_dir() -> Path:
    return settings.MODELS_DIR


def weights_dir() -> Path:
    """ultralytics 的 WEIGHTS_DIR。

    注意：本项目里它解析到**项目根目录下的 `weights/`**（即 `D:\\AI\\VisOS\\weights`），
    不是 `backend/workspace/`。文本编码器 `mobileclip2_b.ts` 必须放在这里，
    因为 ultralytics 的 `MobileCLIPTS` 是按文件名去 WEIGHTS_DIR 查的，不接受绝对路径。
    """
    try:
        from ultralytics.utils import WEIGHTS_DIR
        return Path(WEIGHTS_DIR)
    except Exception:
        return Path.cwd() / "weights"


def _search_dirs() -> List[Path]:
    return [YOLOE_DIR, models_dir(), weights_dir(), Path.cwd()]


def resolve_model_path(name: Optional[str] = None) -> Optional[Path]:
    """找到权重文件，找不到返回 None（**不下载**）。"""
    if not name:
        return None
    p = Path(name)
    if p.is_absolute():
        return p if p.is_file() else None
    for d in _search_dirs():
        cand = d / p.name
        if cand.is_file():
            return cand
    return None


def resolve_text_encoder(name: Optional[str] = None) -> Optional[Path]:
    """找到文本编码器（MobileCLIP2 TorchScript）。"""
    return resolve_model_path(name)


def model_status(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """给前端的模型状态。前端据此显示"YOLOE 模型未安装"。"""
    cfg = cfg or {}
    candidates = list(cfg.get("model_candidates") or
                      [cfg.get("model_name") or "yoloe-26s-seg.pt"])
    preferred = cfg.get("model_name") or candidates[0]
    encoder_name = cfg.get("text_encoder") or "mobileclip2_b.ts"

    found: Optional[Path] = None
    used_name = preferred
    for cand in [preferred] + [c for c in candidates if c != preferred]:
        path = resolve_model_path(cand)
        if path is not None:
            found, used_name = path, cand
            break

    enc = resolve_text_encoder(encoder_name)
    installed = found is not None and enc is not None

    if found is None:
        message = f"YOLOE 模型未安装（缺少 {preferred}）"
    elif enc is None:
        message = f"YOLOE 文本编码器未安装（缺少 {encoder_name}）"
    else:
        message = f"{used_name} 已就绪"

    return {
        "installed": installed,
        "ready": installed,
        "message": message,
        "model_name": used_name,
        "preferred_model": preferred,
        "model_path": str(found) if found else None,
        "model_size_mb": round(found.stat().st_size / 1e6, 1) if found else None,
        "text_encoder": encoder_name,
        "text_encoder_path": str(enc) if enc else None,
        "text_encoder_size_mb": round(enc.stat().st_size / 1e6, 1) if enc else None,
        "candidates": candidates,
        "search_dirs": [str(d) for d in _search_dirs()],
        # 下载必须由用户显式触发，前端据此渲染"安装/下载"入口
        "download_hint": (
            f"把 {preferred} 放到 {YOLOE_DIR}，"
            f"把 {encoder_name} 放到 {weights_dir()}（ultralytics 按文件名在此查找）"),
    }


def place_weights_hint() -> Dict[str, str]:
    """交付文档里"模型文件放置位置"一节的数据来源。"""
    return {"model_dir": str(YOLOE_DIR), "text_encoder_dir": str(weights_dir())}
