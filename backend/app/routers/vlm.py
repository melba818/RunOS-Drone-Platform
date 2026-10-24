"""Remote VLM provider configuration for auto-annotation.

VLMs are the only model type with no local weight file, so they cannot be
discovered by scanning workspace/models. They are configured here instead and
then surface through GET /api/models like any other model, which means the
existing auto-annotate panel works with them unchanged.
"""
from typing import Any, Dict, Literal, Optional
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import settings
from vlm_annotator import (
    VLMProvider, load_providers, save_providers, annotate_image,
    COORD_SPACES, UNMATCHED_POLICIES,
)

router = APIRouter()


class ProviderIn(BaseModel):
    id: str
    name: str
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    classes: list = Field(default_factory=list)
    coord_space: str = "0-1000"
    max_side: int = 1536
    timeout: int = 120
    extra_instruction: str = ""
    unmatched: str = "drop"
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    transport: Literal["direct", "ssh"] = "direct"
    node_id: str = ""
    port: int = Field(default=8000, ge=1, le=65535)
    api_path: str = Field(default="/v1", max_length=160)


def validate_connection(body: ProviderIn) -> None:
    if body.transport == "ssh":
        if not body.node_id.strip():
            raise HTTPException(400, "请选择算力节点")
        if not body.api_path.startswith("/") or any(c in body.api_path for c in "?#\r\n"):
            raise HTTPException(400, "API 路径须以 / 开头，且不能包含查询参数")
        from routers.compute_nodes import store
        if not any(str(n.get("id")) == body.node_id for n in store.list()):
            raise HTTPException(400, "所选算力节点不存在，请刷新节点列表")
        return
    parsed = urlsplit(body.base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(400, "接口地址须为有效的 http:// 或 https:// Base URL")


class TestIn(BaseModel):
    provider: Dict[str, Any]
    dataset_id: Optional[str] = None
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    confidence: float = 0.5
    text_prompt: Optional[str] = None


@router.get("/api/vlm/providers")
async def list_providers():
    return {
        "providers": [p.to_dict(hide_secret=True) for p in load_providers().values()],
        "coord_spaces": list(COORD_SPACES),
        "unmatched_policies": list(UNMATCHED_POLICIES),
    }


@router.post("/api/vlm/providers")
async def upsert_provider(body: ProviderIn):
    validate_connection(body)
    if body.coord_space not in COORD_SPACES:
        raise HTTPException(400, f"coord_space must be one of {COORD_SPACES}")
    if body.unmatched not in UNMATCHED_POLICIES:
        raise HTTPException(400, f"unmatched must be one of {UNMATCHED_POLICIES}")
    providers = load_providers()
    existing = providers.get(body.id)
    # An empty api_key on an edit means "keep the stored one".
    api_key = body.api_key or (existing.api_key if existing else "")

    providers[body.id] = VLMProvider(
        id=body.id, name=body.name, base_url=body.base_url, model=body.model,
        api_key=api_key, classes=list(body.classes), coord_space=body.coord_space,
        max_side=body.max_side, timeout=body.timeout,
        extra_instruction=body.extra_instruction, unmatched=body.unmatched,
        extra_headers=dict(body.extra_headers),
        transport=body.transport, node_id=body.node_id.strip(),
        port=body.port, api_path=body.api_path,
    )
    save_providers(providers)

    # Drop any stale in-memory handle so the next use picks up the new config.
    settings.model_manager.loaded_models.pop(body.id, None)
    return {"success": True, "provider": providers[body.id].to_dict(hide_secret=True)}


@router.post("/api/vlm/discover")
async def discover_server_models(body: ProviderIn):
    """Read model ids from a direct API or an SSH-tunneled OpenAI-compatible server."""
    validate_connection(body)
    existing = load_providers().get(body.id)
    provider = VLMProvider(
        id=body.id or "__discover__", name=body.name or "discover",
        base_url=body.base_url, model=body.model,
        api_key=body.api_key or (existing.api_key if existing else ""),
        extra_headers=dict(body.extra_headers), transport=body.transport,
        node_id=body.node_id.strip(), port=body.port, api_path=body.api_path,
        timeout=min(max(int(body.timeout), 5), 60),
    )
    from vlm_annotator import request_openai_compatible
    try:
        data = request_openai_compatible(provider, "GET", "models")
    except Exception as exc:
        raise HTTPException(502, f"读取服务器模型列表失败：{type(exc).__name__}: {exc}") from exc
    rows = data.get("data", []) if isinstance(data, dict) else []
    models = sorted({str(row.get("id") or "").strip() for row in rows
                     if isinstance(row, dict) and row.get("id")})
    return {"models": models, "count": len(models)}


@router.delete("/api/vlm/providers/{pid}")
async def delete_provider(pid: str):
    providers = load_providers()
    if pid not in providers:
        raise HTTPException(404, "Provider not found")
    del providers[pid]
    save_providers(providers)
    settings.model_manager.loaded_models.pop(pid, None)
    return {"success": True}


@router.post("/api/vlm/providers/{pid}/load")
async def load_provider(pid: str):
    """Bind a provider into model_manager so it appears in the annotate dropdown."""
    prov = load_providers().get(pid)
    if prov is None:
        raise HTTPException(404, "Provider not found")
    settings.model_manager.loaded_models[pid] = {
        "id": pid, "name": prov.name or pid, "type": "vlm",
        "pretrained": False, "remote": True,
        "path": prov.base_url or f"ssh://{prov.node_id}:{prov.port}{prov.api_path}",
        "model": prov, "classes": list(prov.classes),
    }
    return {"success": True, "model_id": pid}


def _resolve_image(body: TestIn) -> Path:
    if body.image_path:
        p = Path(body.image_path)
        if p.is_file():
            return p
        raise HTTPException(404, f"image_path not found: {body.image_path}")

    if not body.dataset_id or not body.image_id:
        raise HTTPException(400, "provide either image_path, or dataset_id + image_id")

    dataset_path = settings.DATASETS_DIR / body.dataset_id
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        for found in dataset_path.glob(f"**/{body.image_id}{ext}"):
            return found
    raise HTTPException(404, f"image '{body.image_id}' not found in dataset")


@router.post("/api/vlm/test")
async def test_provider(body: TestIn):
    """Run one image through an unsaved provider config — try before you commit."""
    cfg = dict(body.provider)
    try:
        existing = load_providers().get(str(cfg.get("id") or ""))
        prov = VLMProvider(
            id=cfg.get("id", "__test__"),
            name=cfg.get("name", "test"),
            base_url=cfg.get("base_url", ""),
            model=cfg.get("model", ""),
            api_key=cfg.get("api_key") or (existing.api_key if existing else ""),
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
    except Exception as exc:
        raise HTTPException(400, f"bad provider config: {exc}")

    image = _resolve_image(body)
    if body.dataset_id:
        dataset = settings.active_datasets.get(body.dataset_id) or {}
        dataset_classes = settings.model_manager._get_dataset_classes(
            settings.DATASETS_DIR / body.dataset_id,
            dataset.get("format", "yolo"),
        ) or [str(c) for c in (dataset.get("classes") or [])]
        if dataset_classes:
            prov.classes = dataset_classes
    try:
        annotations, debug = annotate_image(
            prov, image, body.confidence, body.text_prompt
        )
    except Exception as exc:
        # Surface the transport error to the UI instead of a bare 500.
        raise HTTPException(502, f"VLM call failed: {exc}")

    return {
        "success": True,
        "image": str(image),
        "annotations": annotations,
        "debug": debug,
    }


@router.get("/api/vlm/last-debug")
async def last_debug():
    """Raw reply + parse counts from the most recent VLM call."""
    return {"debug": getattr(settings.model_manager, "_last_vlm_debug", None)}
