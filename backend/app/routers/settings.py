import os
import threading
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request

from config import settings
from schemas.common import SettingsConfig
from core.nexushive import read_account_settings, write_account_settings

router = APIRouter()


def _managed_path(value: str | None, fallback: Path, *, shared_models: bool = False) -> Path:
    """Keep account data isolated while allowing the platform's shared models directory."""
    root = settings.WORKSPACE_DIR.resolve()
    candidate = Path(value).expanduser().resolve() if value else fallback.resolve()
    shared = settings.PLATFORM_MODELS_DIR.resolve()
    allowed_shared = shared_models and (candidate == shared or shared in candidate.parents)
    if not allowed_shared and candidate != root and root not in candidate.parents:
        raise ValueError("RunOS paths must stay inside this RunHive account's data directory")
    return candidate


@router.get("/api/settings")
async def get_settings(request: Request):
    try:
        stored = await asyncio.to_thread(read_account_settings, request.state.admin_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="RunHive configuration service is unavailable") from exc

    if stored:
        defaults = {
            "models_path": settings.MODELS_DIR,
            "datasets_path": settings.DATASETS_DIR,
            "output_path": settings.EXPORTS_DIR,
        }
        repaired = dict(stored)
        for key, fallback in defaults.items():
            try:
                repaired[key] = str(_managed_path(str(stored.get(key, "")), fallback, shared_models=key == "models_path"))
            except (OSError, ValueError):
                repaired[key] = str(fallback.resolve())
        settings._app_settings.update(repaired)
        if any(str(stored.get(key, "")) != repaired[key] for key in defaults):
            await asyncio.to_thread(write_account_settings, request.state.admin_id, settings._app_settings)
    return settings._app_settings


@router.post("/api/settings")
async def update_settings(config: SettingsConfig, request: Request):
    try:
        requested = {
            "datasets_path": _managed_path(config.datasets_path, settings.DATASETS_DIR),
            "models_path": _managed_path(config.models_path, settings.MODELS_DIR, shared_models=True),
            "output_path": _managed_path(config.output_path, settings.EXPORTS_DIR),
        }
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if config.datasets_path:
        new_path = requested["datasets_path"]
        new_path.mkdir(parents=True, exist_ok=True)
        settings.DATASETS_DIR = new_path
        settings._app_settings["datasets_path"] = str(new_path)

    if config.models_path:
        new_path = requested["models_path"]
        new_path.mkdir(parents=True, exist_ok=True)
        settings.MODELS_DIR = new_path
        settings._app_settings["models_path"] = str(new_path)

    if config.output_path:
        new_path = requested["output_path"]
        new_path.mkdir(parents=True, exist_ok=True)
        settings.EXPORTS_DIR = new_path
        settings._app_settings["output_path"] = str(new_path)

    settings._app_settings["use_gpu"] = config.use_gpu
    settings._app_settings["gpu_device"] = config.gpu_device

    try:
        saved = await asyncio.to_thread(write_account_settings, request.state.admin_id, settings._app_settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="RunOS settings were not saved to RunHive") from exc

    settings._app_settings.update(saved)
    return {"success": True, "settings": settings._app_settings}


@router.post("/api/restart")
async def restart_server():
    """Trigger a uvicorn --reload by touching main.py."""
    def _touch():
        import time, pathlib
        time.sleep(0.3)
        # Touch this package's parent (backend/main.py)
        p = pathlib.Path(__file__).parent.parent / "main.py"
        p.touch()
    threading.Thread(target=_touch, daemon=True).start()
    return {"success": True, "message": "Reload triggered — uvicorn will restart in ~1 s"}


@router.post("/api/shutdown")
async def shutdown_server():
    import signal as _signal
    def _kill():
        import time
        time.sleep(0.5)
        os.kill(os.getpid(), _signal.SIGTERM)
    threading.Thread(target=_kill, daemon=True).start()
    return {"success": True, "message": "Server shutting down"}
