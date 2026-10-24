import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, Response

from config import settings
from core.file_access import checked_tree, owned_file_response, owned_path
from schemas.common import TrainingConfig

router = APIRouter()
_weights_metadata_cache = {}

def _weights_metadata(path: Path) -> dict:
    """Read a selected local checkpoint once on CPU; never load catalog weights on GPU."""
    stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
    if key not in _weights_metadata_cache:
        try:
            from ultralytics import YOLO
            model=YOLO(str(path))
            yaml=getattr(model.model,'yaml',{}) or {}
            _weights_metadata_cache[key]={'task':model.task,'architecture':Path(str(yaml.get('yaml_file',''))).stem,'scale':yaml.get('scale','')}
            del model
        except Exception:raise HTTPException(422,'无法读取所选权重，请选择有效的 YOLO 权重文件。') from None
        if len(_weights_metadata_cache)>32:_weights_metadata_cache.pop(next(iter(_weights_metadata_cache)))
    return _weights_metadata_cache[key]


def _resolve_base_model(model_arch: str, model_type: str) -> str:
    """Return the correct model identifier for the chosen arch + task type."""
    arch = model_arch.strip()
    if model_type == "rfdetr":
        return arch
    if model_type == "rtdetrv2":
        return arch
    if model_type == "rtdetr":
        base = arch.replace(".pt", "")
        filename = f"{base}.pt"
    else:
        base = arch.replace("-seg", "").replace("-cls", "").replace(".pt", "")
        if model_type == "segmentation":
            filename = f"{base}-seg.pt"
        elif model_type == "classification":
            filename = f"{base}-cls.pt"
        else:
            filename = f"{base}.pt"
    local_path = settings.MODELS_DIR / filename
    if local_path.exists():
        return str(local_path)
    return filename


def _resolve_initial_weights(config: TrainingConfig) -> str:
    if config.initial_weights == "builtin":
        return _resolve_base_model(config.model_arch, config.model_type)
    if config.model_type not in ("yolo", "segmentation", "classification", "rfdetr"):
        raise HTTPException(status_code=422, detail="Custom weights require a YOLO or RF-DETR task.")
    name = config.custom_weights or ""
    if not name or Path(name).name != name or Path(name).suffix.lower() != (".pth" if config.model_type == "rfdetr" else ".pt"):
        raise HTTPException(status_code=422, detail="Select a matching YOLO .pt or RF-DETR .pth file from Models.")
    models_dir = settings.MODELS_DIR.resolve()
    path = (models_dir / name).resolve()
    if path.parent != models_dir or not path.is_file():
        raise HTTPException(status_code=422, detail="Custom weights are no longer available in VisOS Models. Refresh the list.")
    if config.model_type == "rfdetr":
        from rfdetr_adapter import metadata
        try: metadata(path)
        except Exception as exc: raise HTTPException(422, str(exc))
    else:
        meta=_weights_metadata(path)
        task={'yolo':'detect','segmentation':'segment','classification':'classify'}[config.model_type]
        if meta['task']!=task:
            from training_contract import validation_error
            validation_error('custom_weights','权重任务类型与所选训练任务不匹配。')
        # A custom checkpoint determines the network actually trained. Reject a
        # known mismatch instead of showing an architecture the job will ignore.
        actual=meta['architecture'].lower().replace('yolov','yolo').replace('-seg','').replace('-cls','')
        if meta['scale'] and not actual.endswith(meta['scale']):actual+=meta['scale']
        expected=config.model_arch.lower().replace('yolov','yolo').replace('-seg','').replace('-cls','')
        if actual and actual.startswith('yolo') and actual!=expected:
            from training_contract import validation_error
            validation_error('model_arch','训练架构与所选权重不匹配，请选择该权重对应的架构。')
    return str(path)


@router.get("/api/train/initial-weights")
async def list_initial_weights():
    models_dir = settings.MODELS_DIR.resolve()
    from rfdetr_adapter import metadata
    models = []
    for pt in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if not pt.is_file() or pt.resolve().parent != models_dir: continue
        if pt.suffix.lower() == ".pt":
            models.append({"name": pt.name, "kind": "yolo"})
        elif pt.suffix.lower() == ".pth":
            try: meta = metadata(pt)
            except Exception: continue
            models.append({"name": pt.name, "kind": "rfdetr", "architecture": meta["architecture"]})
    return {"models": models}


@router.post("/api/install-cuda-torch")
async def install_cuda_torch():
    import sys, subprocess
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "--upgrade",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return {"success": True, "message": "CUDA PyTorch installed. Restart the backend server to use GPU."}
        else:
            return {"success": False, "message": result.stderr[-2000:] or result.stdout[-2000:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Install timed out after 5 minutes."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def _authorize_training_paths(training_id: str) -> None:
    """Validate persisted private paths before a parser, loader or resumed worker uses them."""
    job = settings.training_manager.training_jobs.get(training_id, {})
    ident = str(job.get("id", training_id))
    if Path(ident).name != ident or ".." in ident or "\\" in ident:
        raise HTTPException(403, "Training path is outside this account's permitted directory")
    if job.get("dataset_path"):
        checked_tree(job["dataset_path"])
    for key in ("model_path", "last_checkpoint"):
        if job.get(key):
            path = owned_path(job[key])
            if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
                raise HTTPException(403, "Training file access was denied")


def validate_training_config(config: TrainingConfig, request: Request) -> str:
    """Shared preflight for the native form, assistant draft and execution."""
    from training_contract import validate_architecture
    validate_architecture(config)
    if config.compute_node_id:
        from routers.compute_nodes import local_only
        local_only(request)
    if config.dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[config.dataset_id]
    dataset_path = checked_tree(settings.DATASETS_DIR / config.dataset_id)

    ds_format    = (dataset.get("format") or "").lower()
    ds_task_type = (dataset.get("task_type") or "detection").lower()
    mt = config.model_type

    CLASSIFICATION_FORMATS = {"classification-folder", "classification_folder"}
    SEGMENTATION_FORMATS   = {"cityscapes", "ade20k", "yolo-seg"}

    is_cls_dataset = ds_format in CLASSIFICATION_FORMATS or ds_task_type == "classification"
    is_seg_dataset = ds_format in SEGMENTATION_FORMATS or ds_task_type == "segmentation"
    is_det_dataset = not is_cls_dataset and not is_seg_dataset

    detection_model = mt in ("yolo", "rtdetr", "rfdetr")
    seg_model       = mt == "segmentation"
    cls_model       = mt == "classification"

    if is_cls_dataset and not cls_model:
        raise HTTPException(
            status_code=422,
            detail="This dataset is a classification dataset. Please select a Classification model type.",
        )
    if is_seg_dataset and cls_model:
        raise HTTPException(
            status_code=422,
            detail="Segmentation datasets cannot be used to train a classification model. "
                   "Please select an Object Detection or Segmentation model type.",
        )
    if is_det_dataset and seg_model:
        raise HTTPException(
            status_code=422,
            detail="This dataset does not contain polygon segmentation masks. "
                   "Please select an Object Detection model type, or convert your dataset "
                   "to a segmentation format first.",
        )
    if is_det_dataset and cls_model:
        raise HTTPException(
            status_code=422,
            detail="Detection datasets cannot be used for classification training. "
                   "Please use a dataset in classification-folder format.",
        )

    base_model = _resolve_initial_weights(config)
    if config.compute_node_id:
        from routers.compute_nodes import store
        try: node = store.get(config.compute_node_id)
        except KeyError: raise HTTPException(422, '算力节点不存在，请重新选择')
        if node['connection_type'] != 'ssh': raise HTTPException(422, '当前远程训练仅支持 SSH 节点')
        if config.model_type not in ('yolo','rfdetr') or not is_det_dataset:
            raise HTTPException(422, '当前远程训练支持 YOLO / RF-DETR 目标检测')
        if not (dataset_path/'data.yaml').is_file(): raise HTTPException(422, '远程训练需要独立的 YOLO 格式 Split 数据集')
        if not store.public(node)['credential_available']: raise HTTPException(422, '请先在算力节点保存 SSH 凭据')

    return base_model


@router.post("/api/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks, request: Request):
    base_model = validate_training_config(config, request)
    dataset = settings.active_datasets[config.dataset_id]
    dataset_path = settings.DATASETS_DIR / config.dataset_id

    training_id = settings.training_manager.start_training(
        dataset_path,
        dataset["format"],
        config.model_type,
        {
            "compute_node_id": config.compute_node_id,
            "dataset_id": config.dataset_id,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "pretrained": True if config.initial_weights == "custom" else config.pretrained,
            "initial_weights": config.initial_weights,
            "custom_weights": config.custom_weights if config.initial_weights == "custom" else None,
            "device": config.device,
            "save_period": config.save_period,
            "base_model": base_model,
            "models_dir": str(settings.MODELS_DIR),
            "lr0": config.lr0,
            "lrf": config.lrf,
            "optimizer": config.optimizer,
            "patience": config.patience,
            "cos_lr": config.cos_lr,
            "warmup_epochs": config.warmup_epochs,
            "weight_decay": config.weight_decay,
            "mosaic": config.mosaic,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "flipud": config.flipud,
            "fliplr": config.fliplr,
            "amp": config.amp,
            "dropout": config.dropout,
        },
        name=config.name,
    )

    return {"success": True, "training_id": training_id}


@router.get("/api/train/jobs")
async def list_training_jobs():
    return {"jobs": settings.training_manager.list_training_jobs()}


@router.get("/api/train/{training_id}/status")
async def get_training_status(training_id: str, full_logs: bool = False):
    status = settings.training_manager.get_status(training_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training not found")
    if full_logs:
        status['logs'] = list(settings.training_manager.training_jobs[training_id].get('logs', []))
    return status


@router.post("/api/train/{training_id}/stop")
async def stop_training(training_id: str, request: Request):
    job = settings.training_manager.training_jobs.get(training_id, {})
    if job.get('config', {}).get('compute_node_id'):
        from routers.compute_nodes import local_only
        local_only(request)
    success = settings.training_manager.stop_training(training_id)
    return {"success": success}


@router.post("/api/train/{training_id}/pause")
async def pause_training(training_id: str):
    job = settings.training_manager.training_jobs.get(training_id, {})
    if job.get('config', {}).get('compute_node_id'):
        raise HTTPException(422, '远程训练暂不支持暂停续训')
    if job.get('model_type') == 'rfdetr':
        raise HTTPException(422, 'RF-DETR does not support pause/resume; stop and fine-tune from a saved checkpoint instead.')
    success = settings.training_manager.pause_training(training_id)
    if not success:
        raise HTTPException(status_code=400, detail="Training not running or not found")
    return {"success": True, "message": "Pause requested — will stop after current epoch."}


@router.post("/api/train/{training_id}/resume")
async def resume_training(training_id: str):
    _authorize_training_paths(training_id)
    result = settings.training_manager.resume_training(training_id)
    if not result:
        raise HTTPException(status_code=400, detail="Cannot resume — not paused or no checkpoint found")
    return {"success": True, "training_id": result}


@router.post("/api/train/{training_id}/export-format")
async def export_model_format(training_id: str, format: str = "onnx"):
    _authorize_training_paths(training_id)
    export_path = settings.training_manager.export_model_format(training_id, format)
    if not export_path:
        raise HTTPException(status_code=404, detail="Model not found or export failed")
    if not os.path.exists(export_path):
        raise HTTPException(status_code=500, detail="Export produced no output file")
    ext = os.path.splitext(export_path)[1] or f".{format}"
    return owned_file_response(export_path, filename=f"model{ext}")


@router.get("/api/train/{training_id}/export")
async def export_trained_model(training_id: str):
    model_path = settings.training_manager.get_model_path(training_id)
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return owned_file_response(model_path, filename=os.path.basename(model_path))


@router.delete("/api/train/{training_id}")
async def delete_training_job(training_id: str):
    success = settings.training_manager.delete_job(training_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete — job is running or not found")
    return {"success": True}


@router.get("/api/train/{training_id}/checkpoints")
async def list_checkpoints(training_id: str):
    _authorize_training_paths(training_id)
    checkpoints = settings.training_manager.list_checkpoints(training_id)
    return {"checkpoints": checkpoints}


@router.get("/api/train/{training_id}/checkpoint/{filename}")
async def download_checkpoint(training_id: str, filename: str):
    _authorize_training_paths(training_id)
    checkpoints = settings.training_manager.list_checkpoints(training_id)
    for cp in checkpoints:
        if cp["name"] == filename:
            if not os.path.exists(cp["path"]):
                raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")
            return owned_file_response(cp["path"], filename=filename)
    raise HTTPException(status_code=404, detail="Checkpoint not found")


@router.get("/api/train/{training_id}/report")
async def download_training_report(training_id: str):
    import json as _json
    status = settings.training_manager.get_status(training_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training not found")

    job = settings.training_manager.training_jobs.get(training_id, {})
    report = {
        "training_id":   training_id,
        "status":        status["status"],
        "model_type":    job.get("model_type", ""),
        "config":        {k: v for k, v in job.get("config", {}).items() if k not in ("device",)},
        "started_at":    status["started_at"],
        "device_info":   status["device_info"],
        "total_epochs":  status["total_epochs"],
        "current_epoch": status["current_epoch"],
        "model_path":    status["model_path"],
        "final_metrics": status["metrics"],
        "epoch_history": status["epoch_history"],
    }
    content = _json.dumps(report, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="training_report_{training_id}.json"'},
    )
