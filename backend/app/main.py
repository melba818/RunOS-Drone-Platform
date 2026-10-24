"""
CV Dataset Manager - Main FastAPI Application
Thin entry point: app init, middleware, router registration, and lifespan events.
"""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Core startup helpers (must import before routers so state is ready)
from core.startup import (
    _ensure_cuda_torch,
    _ensure_packages,
    _cleanup_temp,
    _restore_jobs,
)
from core.workspace import _restore_datasets

# Routers
from routers import datasets, annotations, training, models, formats, augmentation
from routers import health, snapshots, video, inference, evaluation, camera, compute_nodes
from routers import settings as settings_router
from routers import compute_inference
from routers import vlm as vlm_router
from routers import lmm as lmm_router
from routers import autoannot_v1
import os as _os
_AUTH_MODE = _os.environ.get("RUNOS_AUTH_MODE", "local").strip().lower()
if _AUTH_MODE == "nexushive":
    from routers import nexushive_auth
    from core.nexushive import token_from_request, verify_session
else:
    from routers import local_auth
    from core.local_auth import token_from_request, verify_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("RUNOS_SKIP_RUNTIME_INSTALL") != "1":
        threading.Thread(target=_ensure_cuda_torch, daemon=True).start()
        threading.Thread(
            target=_ensure_packages,
            args=([
                ("huggingface_hub", "huggingface_hub"),
                ("ultralytics",     "ultralytics"),
                ("cv2",             "opencv-python"),
                ("PIL",             "Pillow"),
                ("psutil",          "psutil"),
                ("clip",            "git+https://github.com/ultralytics/CLIP.git"),
                ("ftfy",            "ftfy"),
            ],),
            daemon=True,
        ).start()
    _cleanup_temp()
    _restore_datasets()
    _restore_jobs()
    from compute.remote_training import restore_remote_jobs
    from config import settings as app_settings
    restore_remote_jobs(app_settings.training_manager)
    try:
        yield
    finally:
        camera.service.shutdown()
        from compute.remote_inference import nodes as inference_nodes
        inference_nodes.shutdown()


app = FastAPI(
    title="CV Dataset Manager",
    description="Professional Computer Vision Dataset Management Suite",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("RUNOS_FRONTEND_ORIGIN", "http://127.0.0.1:4000"),
        "http://localhost:4000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_session(request: Request, call_next):
    path = request.url.path
    public_paths = {"/api/health", "/api/auth/exchange", "/api/auth/me", "/api/auth/login"}
    if path.startswith("/api/") and path not in public_paths:
        try:
            claims = verify_session(token_from_request(request))
        except ValueError:
            return JSONResponse({"detail": "RunOS sign-in required" if _AUTH_MODE != "nexushive" else "RunHive sign-in required"}, status_code=401)

        admin_id = int(claims.get("sub", 0))
        tenant_admin_id = int(os.environ.get("RUNOS_OWNER_ADMIN_ID", "1"))
        if admin_id != tenant_admin_id:
            return JSONResponse({"detail": "This RunOS workspace is assigned to a different RunHive account"}, status_code=403)
        request.state.admin_id = admin_id

    return await call_next(request)

# Register routers
app.include_router(datasets.router)
app.include_router(annotations.router)
app.include_router(training.router)
from routers import platform_assistant
app.include_router(platform_assistant.router)
from routers import resource_library
app.include_router(resource_library.router)
app.include_router(models.router)
app.include_router(formats.router)
app.include_router(augmentation.router)
app.include_router(health.router)
app.include_router(snapshots.router)
app.include_router(video.router)
app.include_router(settings_router.router)
app.include_router(inference.router)
app.include_router(camera.router)
app.include_router(compute_nodes.router)
app.include_router(compute_inference.router)
app.include_router(evaluation.router)
app.include_router(vlm_router.router)
# 多模态目标检测（自然语言 → YOLOE → Tracker → RuleEngine → 告警）
app.include_router(lmm_router.router)
# Auto Label V1：VLX + Qwen 双 Teacher 并行检测 / 双向交叉校验 / ACCEPT·REVIEW
app.include_router(autoannot_v1.router)
if _AUTH_MODE == "nexushive":
    app.include_router(nexushive_auth.router)
else:
    app.include_router(local_auth.router)


@app.get("/")
async def root():
    return {
        "status": "running",
        "name": "CV Dataset Manager API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
