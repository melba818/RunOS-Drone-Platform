import json as _json
import shutil
import threading
import uuid
from typing import Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response

from config import settings
from core.file_access import owned_path, owned_file_response, owned_file_bytes
from core.startup import _persist_jobs
from core.workspace import _save_dataset_metadata
from tile_pass import TilePassCancelled
import human_review
from core.annotation_lock import annotation_lock

# P2A：统一的目标结构归一化入口（★禁止各 router 自己写一套兼容逻辑★）
from model_runtime.compat import (
    normalize_detection_target,
    MissingLabelError,
    MODE_LEGACY,
    MODE_NONE,
    MODE_TARGET,
    MODE_FIELDS,
)

router = APIRouter()


#: 单图 / 批量都能用的图片定位（双模型链路需要拿到**文件本身**，不能只拿 image_id）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _resolve_image_file(dataset_path: Path, image_id: str,
                        image_path_hint: Optional[str] = None) -> Optional[Path]:
    """先信 hint，再退化到 glob（与 `annotate_single_image` 的顺序保持一致）。"""
    if image_path_hint:
        cand = dataset_path / image_path_hint
        if cand.is_file() and cand.resolve().is_relative_to(dataset_path.resolve()):
            return cand
    for ext in IMAGE_EXTENSIONS:
        for found in dataset_path.glob(f"**/{image_id}{ext}"):
            return found
    return None


def _dual_summary(debug: dict) -> dict:
    """从 `run_chain_dual` 的 debug 里挑出 UI 真正要显示的几个字段。

    刻意**不返回整个 debug** —— 里面有 merged_detail / candidates_detail，
    一张图几十 KB，批量任务里会直接把 status 响应撑爆。
    """
    dec = debug.get("decision") or {}
    return {
        "model_a": debug.get("model_a"),
        "model_b": debug.get("model_b"),
        "accept": int(dec.get("ACCEPT") or 0),
        "review": int(dec.get("REVIEW") or 0),
        "candidates": int(debug.get("candidates_raw") or 0),
        "teachers_down": list(debug.get("teachers_down") or []),
        "teachers_up": list(debug.get("teachers_up") or []),
        "verify": debug.get("dual_verify") or {},
        "class_conflicts": int(debug.get("class_conflicts") or 0),
        "elapsed_ms": debug.get("elapsed_ms"),
        "tile_pass": debug.get("tile_pass") or {},
    }


def _parse_target_param(raw: Optional[str]) -> Optional[dict]:
    """target 以 JSON 字符串传入（query 不方便传对象）。"""
    if raw is None or not str(raw).strip():
        return None
    try:
        obj = _json.loads(raw)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="target 参数不是合法 JSON")
    if not isinstance(obj, dict):
        raise HTTPException(status_code=422, detail="target 参数必须是对象")
    return obj


def _build_target(*, target=None, label=None, description=None,
                  extra_prompt=None, text_prompt=None, confidence=None,
                  allow_runtime_v1=False):
    """归一化 + 执行门禁。

    返回 (DetectionTarget, mode, exec_prompt, exec_uses_target)：
        mode             请求用的是哪种结构（target / fields / legacy / none）
        exec_prompt      真正送给模型的文本
        exec_uses_target True = 按新结构执行（P2B 起才可能为 True）

    ★P2A 语义★
        TARGET_STRUCTURE_EXECUTION=0（当前默认）：
            - 请求侧已完成 DetectionTarget 标准化（可观测、可断言）
            - 执行侧仍用 text_prompt，**写盘语义与 P2A 之前逐字节一致**
            - 若只给了新结构却没给 text_prompt → 409，拒绝半迁移执行
              （否则会把 "description\\nextra_prompt" 整段写成 data.yaml 类别）
        =1（P2B 起）：执行侧改用 tg.prompt_text()，类别改用 tg.label。

    ★P2B 补充★
        409 的本意是「别让**旧**执行逻辑把整句描述写成 data.yaml 类别」。
        若调用方确认本次请求会走 model_runtime_v1（allow_runtime_v1=True，
        目前只有 VLX 单图），类别取自 target.label、描述只送模型，污染前提
        不成立，因此放行并按新结构执行。批量/其他模型不传该参数 → 保持 409。
    """
    try:
        tg, mode = normalize_detection_target(
            target=target, label=label, description=description,
            extra_prompt=extra_prompt, text_prompt=text_prompt,
            confidence=confidence,
        )
    except MissingLabelError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    legacy_text = (text_prompt or "").strip()

    if mode in (MODE_LEGACY, MODE_NONE):
        # 旧路径 / 无提示词：行为与 P2A 之前完全一致
        return tg, mode, legacy_text, False

    if not getattr(settings, "TARGET_STRUCTURE_EXECUTION", 0):
        if not legacy_text:
            # ── P2B：确认走新链路时放行（新链路不会把描述写成类别）──────────
            if allow_runtime_v1:
                try:
                    _legacy_lbl = bool(getattr(settings, "LEGACY_LABEL_FROM_DESCRIPTION", False))
                    _ok = bool(tg.effective_label(_legacy_lbl))
                except Exception:
                    _ok = False
                if _ok:
                    return tg, mode, tg.prompt_text(), True
            raise HTTPException(
                status_code=409,
                detail=(
                    "已识别新目标结构（label/description/extra_prompt），但 P2B 尚未启用；"
                    "当前不会执行，以免沿用旧写盘逻辑把描述写成 data.yaml 类别。"
                    "请同时传 text_prompt，或在 P2B 启用后再提交。"
                ),
            )
        return tg, mode, legacy_text, False

    return tg, mode, tg.prompt_text(), True


@router.post("/api/auto-annotate/{dataset_id}")
async def auto_annotate_dataset(
    dataset_id: str,
    model_id: str,
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id

    results = settings.model_manager.auto_annotate(
        model_id, dataset_path, dataset["format"], confidence_threshold
    )

    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, dataset["format"], dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info

    return {"success": True, "annotated_images": results["annotated_count"]}


@router.post("/api/auto-annotate/{dataset_id}/single/{image_id}")
async def auto_annotate_single_image(
    dataset_id: str,
    image_id: str,
    model_id: str,
    confidence_threshold: float = 0.5,
    point_x: Optional[float] = None,
    point_y: Optional[float] = None,
    text_prompt: Optional[str] = None,
    image_path_hint: Optional[str] = None,
    # ── P2A 新增（可选，与 text_prompt 并存）──
    label: Optional[str] = None,
    description: Optional[str] = None,
    extra_prompt: Optional[str] = None,
    confidence: Optional[float] = None,
    target: Optional[str] = None,
    # ── 双模型（标注页「一个标注 + 一个校验」）──
    verify_model_id: Optional[str] = None,
    tile_pass: bool = False,
    tile_size: int = 640,
    tile_overlap: int = 128,
    tile_batch_size: int = 64,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    prompt_point = (point_x, point_y) if point_x is not None and point_y is not None else None

    if tile_pass:
        from routers.tile_preview import start_preview
        image_file = _resolve_image_file(dataset_path, image_id, image_path_hint)
        if not image_file:
            raise HTTPException(404, "Image file not found")
        classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])
        if not classes:
            classes = [str(c) for c in dataset.get("classes", [])]
        if not classes:
            raise HTTPException(422, "切片标注需要数据集已有类别表")
        tg, mode, exec_prompt, _ = _build_target(target=_parse_target_param(target),
            label=label, description=description, extra_prompt=extra_prompt,
            text_prompt=text_prompt, confidence=confidence if confidence is not None else confidence_threshold)
        if tg.label and tg.label not in classes:
            raise HTTPException(422, "所选类别不在当前数据集类别表中")
        try:
            return start_preview(image_file, classes, dataset_id=dataset_id, image_id=image_id,
                model_id=model_id, verify_model_id=verify_model_id or "", text_prompt=exec_prompt or "",
                label=tg.label or "", confidence=confidence if confidence is not None else confidence_threshold,
                tile_size=tile_size, tile_overlap=tile_overlap, tile_batch_size=tile_batch_size)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    # ── 双模型分支：第二个模型做交叉校验 → 输出 ACCEPT / REVIEW ──────────
    # 放在最前面且**整段 return**，不碰下面任何既有单模型逻辑。
    if verify_model_id:
        from annotation_chain import run_chain_dual

        img_file = _resolve_image_file(dataset_path, image_id, image_path_hint)
        if img_file is None:
            raise HTTPException(status_code=404, detail="Image file not found")
        classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])
        if not classes:
            raise HTTPException(
                status_code=422,
                detail="数据集没有类别表（data.yaml），无法做双模型标注。请先配置类别。",
            )
        try:
            annotations, dd = run_chain_dual(
                img_file,
                text_prompt or "",
                classes,
                model_a=model_id,
                model_b=verify_model_id,
                label=(label or "").strip(),
                dataset_path=dataset_path,
                confidence_threshold=confidence_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        review_job_id = uuid.uuid4().hex
        review = human_review.register_preview(review_job_id, dataset_id, img_file, annotations, model_id=model_id, verify_model_id=verify_model_id)
        return {
            "success": True,
            "annotations": [a for a in annotations if a.get("decision") == "ACCEPT"],
            "review_annotations": [a for a in annotations if a.get("decision") == "REVIEW"],
            "review_job_id": review_job_id,
            "review_summary": review["summary"],
            "target_mode": "dual",
            "dual": _dual_summary(dd),
        }

    # P2B：预检该模型是否会走 model_runtime_v1（只读；查不到一律 False → 保持 409）
    try:
        allow_runtime_v1 = bool(settings.model_manager.will_use_model_runtime_v1(model_id))
    except Exception:
        allow_runtime_v1 = False

    tg, mode, exec_prompt, exec_uses_target = _build_target(
        target=_parse_target_param(target), label=label, description=description,
        extra_prompt=extra_prompt, text_prompt=text_prompt,
        confidence=confidence if confidence is not None else confidence_threshold,
        allow_runtime_v1=allow_runtime_v1,
    )

    try:
        # P2B：把归一化好的 DetectionTarget 往下传（旧调用方不传也照旧工作）。
        # legacy / none 两种模式不带类别，传 None → 新链路不会接管。
        annotations = settings.model_manager.annotate_single_image(
            model_id, dataset_path, dataset["format"], image_id, confidence_threshold,
            prompt_point=prompt_point,
            text_prompt=exec_prompt or None,
            image_path_hint=image_path_hint,
            target=tg if mode in (MODE_TARGET, MODE_FIELDS) else None,
            model_classes=(
                settings.model_manager._get_dataset_classes(dataset_path, dataset.get("format", "yolo"))
                or [str(c) for c in (dataset.get("classes") or [])]
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # P2B：新链路回传的调试信息（不含 auth_ref / token / SSH 凭据）
    runtime_info = getattr(settings.model_manager, "_last_runtime_info", None)
    if runtime_info:
        try:
            settings.model_manager._last_runtime_info = None
        except Exception:
            pass

    return {
        "success": True,
        "annotations": annotations,
        # 仅观测用：告诉调用方本次是怎么归一化的（target / fields / legacy / none）
        "target_mode": mode,
        "target": tg.to_dict(),
        # True = 真的走了 Model → Deployment → Resolver → Adapter 新链路（P2B 起）
        "target_executed": bool(runtime_info),
        "runtime_info": runtime_info,
    }


@router.post("/api/auto-annotate/{dataset_id}/text-batch")
async def auto_annotate_text_batch(
    dataset_id: str,
    model_id: str,
    text_prompt: Optional[str] = None,
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks = None,
    # ── P2A 新增（可选，与 text_prompt 并存）──
    label: Optional[str] = None,
    description: Optional[str] = None,
    extra_prompt: Optional[str] = None,
    confidence: Optional[float] = None,
    target: Optional[str] = None,
    # ── 双模型（标注页「一个标注 + 一个校验」）──
    verify_model_id: Optional[str] = None,
    on_verify_fail: str = "review",
    tile_pass: bool = False,
    tile_size: int = 640,
    tile_overlap: int = 128,
    tile_batch_size: int = 64,
    preserve_existing: bool = False,
):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if tile_pass:
        from tile_pass import validate_options
        try:
            validate_options(tile_size, tile_overlap, tile_batch_size)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    if verify_model_id == model_id:
        raise HTTPException(422, "标注模型和校验模型必须不同")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    job_id = __import__("uuid").uuid4().hex[:8]

    tg, mode, exec_prompt, exec_uses_target = _build_target(
        target=_parse_target_param(target), label=label, description=description,
        extra_prompt=extra_prompt, text_prompt=text_prompt,
        confidence=confidence if confidence is not None else confidence_threshold,
    )
    # 之后所有推理统一使用 exec_prompt；text_prompt 变量不再直接参与推理。
    text_prompt = exec_prompt

    # 批量任务异步执行，先确认标注页选中的类别仍属于当前数据集。
    if (label or "").strip():
        dataset_classes = settings.model_manager._get_dataset_classes(dataset_path, dataset.get("format", "yolo"))
        if not dataset_classes:
            dataset_classes = [str(c) for c in (dataset.get("classes") or [])]
        if dataset_classes and label.strip() not in dataset_classes:
            raise HTTPException(
                status_code=422,
                detail=f'所选类别「{label.strip()}」不在当前数据集类别列表中，请刷新类别后重试。',
            )

    pause_ev = threading.Event(); pause_ev.set()
    stop_ev = threading.Event()

    settings._text_annotate_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "paused": False,
        "progress": 0,
        "total": 0,
        "processed": 0,
        "annotated": 0,
        "failed": 0,
        "total_annotations": 0,
        "dataset_id": dataset_id,
        "model_id": model_id,
        # ★双模型追加字段★：只加不改 —— 历史 job 没有这两个 key 也照常读取
        "verify_model_id": verify_model_id or "",
        "tile_pass": tile_pass, "tile_size": tile_size, "tile_overlap": tile_overlap,
        "tile_batch_size": tile_batch_size, "on_verify_fail": on_verify_fail,
        "preserve_existing": preserve_existing,
        "review_images": [],
        "dual": {"accept": 0, "review": 0},
        "confidence_threshold": confidence_threshold,
        "text_prompt": text_prompt or "",
        # ★P2A 追加字段★：只加不改——历史 job 没有这个 key 也能照常读取，
        # 不做任何迁移 / 改名 / 删除（batch_jobs.json 已有记录不动）。
        "target": tg.to_dict(),
        "target_mode": mode,
        "target_executed": exec_uses_target,
        "started_at": datetime.utcnow().isoformat(),
        "recent_images": [],
        "all_images": [],
        "snapshot": None,
    }
    settings._job_controls[job_id] = {"pause": pause_ev, "stop": stop_ev}
    _persist_jobs()

    def _run_batch(job_id: str):
        try:
            _snap: dict = {"yaml": {}, "labels": {}, "metadata": None}
            for _yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")):
                try:
                    _snap["yaml"][_yf.name] = _yf.read_text(encoding="utf-8")
                except Exception:
                    pass
            _labels_dir = dataset_path / "labels"
            if _labels_dir.exists():
                for _lf in _labels_dir.rglob("*.txt"):
                    try:
                        _snap["labels"][str(_lf.relative_to(dataset_path))] = _lf.read_text(encoding="utf-8")
                    except Exception:
                        pass
            _meta_path = dataset_path / "dataset_metadata.json"
            if _meta_path.exists():
                try:
                    _snap["metadata"] = _meta_path.read_text(encoding="utf-8")
                except Exception:
                    pass
            settings._text_annotate_jobs[job_id]["snapshot"] = _snap

            if dataset.get("format") == "generic-images":
                dataset["format"] = "yolo"
                meta_path = dataset_path / "dataset_metadata.json"
                try:
                    import json as _json
                    meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    meta["format"] = "yolo"
                    meta_path.write_text(_json.dumps(meta, indent=2))
                except Exception as _me:
                    print(f"[batch:{job_id}] failed to persist format upgrade: {_me}")
                _img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                _root_p = settings.dataset_parser._find_dataset_root(dataset_path)
                _imgs_dir = _root_p / "images"
                _imgs_dir.mkdir(exist_ok=True)
                for _f in list(_root_p.iterdir()):
                    if _f.is_file() and _f.suffix.lower() in _img_exts:
                        shutil.move(str(_f), _imgs_dir / _f.name)
                if dataset_id in settings._images_cache:
                    del settings._images_cache[dataset_id]

            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            _seen_paths: set = set()
            image_files: list = []
            for ext in IMAGE_EXTENSIONS:
                for _f in list(dataset_path.rglob(f"*{ext}")) + list(dataset_path.rglob(f"*{ext.upper()}")):
                    if "labels" in _f.parts:
                        continue
                    _rp = _f.resolve()
                    if _rp not in _seen_paths:
                        _seen_paths.add(_rp)
                        image_files.append(_f)

            settings._text_annotate_jobs[job_id]["total"] = len(image_files)
            classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])
            if not classes:
                classes = [str(c) for c in (dataset.get("classes") or [])]
            ctrl = settings._job_controls[job_id]

            for i, image_file in enumerate(image_files):
                if ctrl["stop"].is_set():
                    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                    settings._images_cache.pop(dataset_id, None)
                    return

                if not ctrl["pause"].is_set():
                    settings._text_annotate_jobs[job_id]["paused"] = True
                    settings._text_annotate_jobs[job_id]["status"] = "paused"
                    ctrl["pause"].wait()
                    if ctrl["stop"].is_set():
                        settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                        settings._images_cache.pop(dataset_id, None)
                        return
                    settings._text_annotate_jobs[job_id]["paused"] = False
                    settings._text_annotate_jobs[job_id]["status"] = "running"

                try:
                    if verify_model_id or tile_pass:
                        #: ── 双模型分支：两个模型各自全图 + 裁剪交叉校验 ──
                        from annotation_chain import run_chain_dual
                        annotations, _dd = run_chain_dual(
                            image_file, text_prompt or "", classes,
                            model_a=model_id, model_b=verify_model_id or "",
                            label=(label or "").strip(),
                            dataset_path=dataset_path,
                            confidence_threshold=confidence_threshold,
                            on_verify_fail=on_verify_fail,
                            tile_pass=tile_pass, tile_size=tile_size, tile_overlap=tile_overlap,
                            tile_batch_size=tile_batch_size, control=ctrl,
                            progress=lambda state: _update_tile_progress(job_id, image_file, state),
                        )
                        _d = _dual_summary(_dd)
                        _j = settings._text_annotate_jobs[job_id]
                        _j["dual"]["accept"] += _d["accept"]
                        _j["dual"]["review"] += _d["review"]
                        for _m in _d["teachers_down"]:
                            _j.setdefault("teachers_down", [])
                            if _m not in _j["teachers_down"]:
                                _j["teachers_down"].append(_m)
                    else:
                        model_info = settings.model_manager.loaded_models[model_id]
                        model_type = model_info.get("type")
                        model_classes = classes if model_type == "vlm" and classes else model_info.get("classes") or []
                        annotations = settings.model_manager._run_inference(
                            model_info["model"],
                            model_type,
                            image_file,
                            confidence_threshold,
                            text_prompt=text_prompt or None,
                            model_classes=model_classes,
                            target_label=(tg.label if model_type == "vlm" else None),
                            # 让 vlx/vlm 分支能把「描述」落进 data.yaml 当类别（ensure_class）
                            dataset_path=dataset_path,
                        )
                except TilePassCancelled:
                    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                    _persist_jobs()
                    return
                except Exception as exc:
                    print(f"[batch:{job_id}] inference failed on {image_file.name}: {exc}")
                    settings._text_annotate_jobs[job_id]["failed"] += 1
                    settings._text_annotate_jobs[job_id]["processed"] = i + 1
                    settings._text_annotate_jobs[job_id]["progress"] = int((i + 1) / max(len(image_files), 1) * 100)
                    continue

                if ctrl["stop"].is_set():
                    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                    _persist_jobs()
                    return
                if tile_pass or verify_model_id:
                    _save_tile_review(job_id, image_file, dataset_path, annotations, _dd)
                    annotations = [a for a in annotations if a.get("decision") == "ACCEPT"]
                if annotations:
                    try:
                        if preserve_existing:
                            from assistant_annotation import save_accepted
                            new_count=save_accepted(dataset_path,dataset,image_file,annotations,(label or '').strip())
                        else:
                            settings.model_manager._save_annotations(
                                dataset_path, dataset["format"], image_file, annotations, classes
                            )
                            new_count=len(annotations)
                        settings._text_annotate_jobs[job_id]["annotated"] += int(new_count>0)
                        settings._text_annotate_jobs[job_id]["total_annotations"] += new_count
                        try:
                            rel_path = str(image_file.relative_to(dataset_path))
                        except Exception:
                            rel_path = image_file.name
                        img_entry = {
                            "filename": image_file.name,
                            "path": rel_path,
                            "abs_path": str(image_file),
                            "image_id": image_file.stem,
                            "annotation_count": len(annotations),
                        }
                        recent = settings._text_annotate_jobs[job_id].get("recent_images", [])
                        recent.append(img_entry)
                        settings._text_annotate_jobs[job_id]["recent_images"] = recent[-10:]
                        all_imgs = settings._text_annotate_jobs[job_id].get("all_images", [])
                        all_imgs.append(img_entry)
                        settings._text_annotate_jobs[job_id]["all_images"] = all_imgs
                        settings._images_cache.pop(dataset_id, None)
                    except Exception as exc:
                        print(f"[batch:{job_id}] save failed on {image_file.name}: {exc}")
                        settings._text_annotate_jobs[job_id]["failed"] += 1
                settings._text_annotate_jobs[job_id]["processed"] = i + 1
                settings._text_annotate_jobs[job_id]["progress"] = int((i + 1) / max(len(image_files), 1) * 100)
                if i % 5 == 4:
                    _persist_jobs()

            settings._images_cache.pop(dataset_id, None)
            settings._text_annotate_jobs[job_id]["status"] = "done"
            _persist_jobs()
        except Exception as e:
            settings._text_annotate_jobs[job_id]["status"] = "error"
            settings._text_annotate_jobs[job_id]["error"] = str(e)
            _persist_jobs()

    _lm = settings.model_manager.loaded_models
    for _mid in (model_id, verify_model_id or ""):
        if not _mid:
            continue
        if _mid not in _lm or not _lm[_mid].get("model"):
            settings.model_manager._try_autoload(_mid)
    if model_id not in _lm or not _lm[model_id].get("model"):
        raise HTTPException(status_code=400, detail="Model not loaded or unavailable")
    #: 校验模型起不来**不阻断**任务 —— 双模型会自动退化成"单模型标注 + 如实记
    #: teachers_down"，比整批启动失败有用（用户看不到任何图时会以为功能坏了）。
    if verify_model_id and (verify_model_id not in _lm or not _lm[verify_model_id].get("model")):
        settings._text_annotate_jobs[job_id]["teachers_down"] = [verify_model_id]

    threading.Thread(target=_run_batch, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/api/auto-annotate/text-batch/{job_id}/status")
async def get_text_batch_status(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return human_review.enrich_job(settings._text_annotate_jobs[job_id])


@router.post("/api/auto-annotate/text-batch/{job_id}/pause")
async def pause_text_batch(job_id: str):
    if job_id not in settings._job_controls:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings._text_annotate_jobs[job_id]["status"] not in ("running",):
        raise HTTPException(status_code=400, detail="Job is not running")
    settings._job_controls[job_id]["pause"].clear()
    settings._text_annotate_jobs[job_id]["paused"] = True
    settings._text_annotate_jobs[job_id]["status"] = "paused"
    _persist_jobs()
    return {"status": "paused"}


@router.post("/api/auto-annotate/text-batch/{job_id}/resume")
async def resume_text_batch(job_id: str):
    if job_id not in settings._job_controls:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings._text_annotate_jobs[job_id]["status"] not in ("paused",):
        raise HTTPException(status_code=400, detail="Job is not paused")
    settings._text_annotate_jobs[job_id]["paused"] = False
    settings._text_annotate_jobs[job_id]["status"] = "running"
    settings._job_controls[job_id]["pause"].set()
    _persist_jobs()
    return {"status": "running"}


@router.post("/api/auto-annotate/text-batch/{job_id}/cancel")
async def cancel_text_batch(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    current_status = settings._text_annotate_jobs[job_id]["status"]
    if current_status in ("done", "cancelled", "error"):
        raise HTTPException(status_code=400, detail="Job is already finished")
    if job_id in settings._job_controls:
        ctrl = settings._job_controls[job_id]
        ctrl["stop"].set()
        ctrl["pause"].set()
    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
    _persist_jobs()
    return {"status": "cancelled"}


@router.post("/api/auto-annotate/text-batch/{job_id}/undo")
async def undo_text_batch(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = settings._text_annotate_jobs[job_id]
    if job["status"] not in ("done", "cancelled", "error", "interrupted"):
        raise HTTPException(status_code=400, detail="Job must be finished to undo")

    snapshot = job.get("snapshot")
    if not snapshot:
        raise HTTPException(status_code=400, detail="No snapshot available — cannot undo this job")

    dataset_id = job["dataset_id"]
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = settings.DATASETS_DIR / dataset_id

    with annotation_lock(dataset_path):
        labels_dir = dataset_path / "labels"
        if labels_dir.exists():
            for lf in labels_dir.rglob("*.txt"):
                try:
                    lf.unlink()
                except Exception:
                    pass

        for rel_path, content in snapshot.get("labels", {}).items():
            restore_path = dataset_path / rel_path
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            restore_path.write_text(content, encoding="utf-8")

        current_yamls = set(yf.name for yf in list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml")))
        snap_yamls = snapshot.get("yaml", {})
        for filename, content in snap_yamls.items():
            (dataset_path / filename).write_text(content, encoding="utf-8")
        for yaml_name in current_yamls - set(snap_yamls.keys()):
            try:
                (dataset_path / yaml_name).unlink()
            except Exception:
                pass

        meta_content = snapshot.get("metadata")
        meta_path = dataset_path / "dataset_metadata.json"
        if meta_content:
            meta_path.write_text(meta_content, encoding="utf-8")
            try:
                import json as _jj
                restored_meta = _jj.loads(meta_content)
                settings.active_datasets[dataset_id] = {
                    **settings.active_datasets[dataset_id],
                    "format": restored_meta.get("format", settings.active_datasets[dataset_id]["format"]),
                }
            except Exception:
                pass

        settings._images_cache.pop(dataset_id, None)
        job["status"] = "undone"
        job["snapshot"] = None
        _persist_jobs()
    return {"status": "undone"}


@router.get("/api/auto-annotate/jobs")
async def list_batch_jobs():
    return {"jobs": [human_review.enrich_job(job) for job in settings._text_annotate_jobs.values()]}


@router.delete("/api/auto-annotate/text-batch/{job_id}")
async def delete_batch_job(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if settings._text_annotate_jobs[job_id]["status"] in ("running", "paused"):
        raise HTTPException(status_code=400, detail="Cancel the job before deleting")
    del settings._text_annotate_jobs[job_id]
    settings._job_controls.pop(job_id, None)
    _persist_jobs()
    return {"status": "deleted"}


@router.post("/api/auto-annotate/text-batch/{job_id}/restart")
async def restart_text_batch(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = settings._text_annotate_jobs[job_id]
    if job["status"] not in ("interrupted", "cancelled", "error"):
        raise HTTPException(status_code=400, detail="Job is not restartable")

    dataset_id = job["dataset_id"]
    model_id = job.get("model_id")
    confidence_threshold = job.get("confidence_threshold", 0.5)
    text_prompt = job["text_prompt"]

    if not model_id:
        raise HTTPException(status_code=400, detail="Job has no stored model_id; start a new job instead")
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if model_id not in settings.model_manager.loaded_models or not settings.model_manager.loaded_models[model_id].get("model"):
        settings.model_manager._try_autoload(model_id)
    if model_id not in settings.model_manager.loaded_models or not settings.model_manager.loaded_models[model_id].get("model"):
        raise HTTPException(status_code=400, detail="Model not loaded or unavailable")

    pause_ev = threading.Event(); pause_ev.set()
    stop_ev = threading.Event()
    settings._job_controls[job_id] = {"pause": pause_ev, "stop": stop_ev}
    job["status"] = "running"
    job["paused"] = False
    _persist_jobs()

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    already_done = {img["image_id"] for img in job.get("all_images", [])}

    def _continue_batch(job_id: str):
        try:
            IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
            _seen_paths2: set = set()
            image_files: list = []
            for ext in IMAGE_EXTENSIONS:
                for _f in list(dataset_path.rglob(f"*{ext}")) + list(dataset_path.rglob(f"*{ext.upper()}")):
                    if "labels" in _f.parts:
                        continue
                    _rp = _f.resolve()
                    if _rp not in _seen_paths2:
                        _seen_paths2.add(_rp)
                        image_files.append(_f)

            all_count = len(image_files)
            remaining = [f for f in image_files if f.stem not in already_done]
            settings._text_annotate_jobs[job_id]["total"] = all_count
            classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])
            if not classes:
                classes = [str(c) for c in (dataset.get("classes") or [])]
            ctrl = settings._job_controls[job_id]

            for i, image_file in enumerate(remaining):
                if ctrl["stop"].is_set():
                    settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                    settings._images_cache.pop(dataset_id, None)
                    return
                if not ctrl["pause"].is_set():
                    settings._text_annotate_jobs[job_id]["paused"] = True
                    settings._text_annotate_jobs[job_id]["status"] = "paused"
                    ctrl["pause"].wait()
                    if ctrl["stop"].is_set():
                        settings._text_annotate_jobs[job_id]["status"] = "cancelled"
                        settings._images_cache.pop(dataset_id, None)
                        return
                    settings._text_annotate_jobs[job_id]["paused"] = False
                    settings._text_annotate_jobs[job_id]["status"] = "running"

                try:
                    if job.get("tile_pass") or job.get("verify_model_id"):
                        from annotation_chain import run_chain_dual
                        annotations, _dd = run_chain_dual(image_file, text_prompt or "", classes,
                            model_a=model_id, model_b=job.get("verify_model_id") or "",
                            label=(job.get("target") or {}).get("label") or "",
                            confidence_threshold=confidence_threshold,
                            on_verify_fail=job.get("on_verify_fail", "review"), tile_pass=bool(job.get("tile_pass")),
                            tile_size=job.get("tile_size",640), tile_overlap=job.get("tile_overlap",128),
                            tile_batch_size=job.get("tile_batch_size",64), control=ctrl,
                            progress=lambda state: _update_tile_progress(job_id,image_file,state))
                        _d = _dual_summary(_dd)
                        job.setdefault("dual", {"accept":0,"review":0})
                        job["dual"]["accept"] += _d["accept"]
                        job["dual"]["review"] += _d["review"]
                        _save_tile_review(job_id,image_file,dataset_path,annotations,_dd)
                        annotations = [a for a in annotations if a.get("decision")=="ACCEPT"]
                    else:
                        model_info = settings.model_manager.loaded_models[model_id]
                        model_type = model_info.get("type")
                        model_classes = classes if model_type == "vlm" and classes else model_info.get("classes") or []
                        annotations = settings.model_manager._run_inference(
                            model_info["model"],
                            model_type,
                            image_file,
                            confidence_threshold,
                            text_prompt=text_prompt,
                            model_classes=model_classes,
                            target_label=(
                                ((job.get("target") or {}).get("label") or "").strip() or None
                                if model_type == "vlm"
                                else None
                            ),
                        )
                except TilePassCancelled:
                    job["status"] = "cancelled"
                    _persist_jobs()
                    return
                except Exception as exc:
                    print(f"[batch:{job_id}] inference failed on {image_file.name}: {exc}")
                    settings._text_annotate_jobs[job_id]["failed"] += 1
                    settings._text_annotate_jobs[job_id]["processed"] += 1
                    total_done = len(already_done) + i + 1
                    settings._text_annotate_jobs[job_id]["progress"] = int(total_done / max(all_count, 1) * 100)
                    continue

                if annotations:
                    try:
                        if job.get('preserve_existing'):
                            from assistant_annotation import save_accepted
                            accepted=[a for a in annotations if a.get('decision','ACCEPT')=='ACCEPT']
                            new_count=save_accepted(dataset_path,dataset,image_file,accepted,((job.get('target') or {}).get('label') or '').strip())
                        else:
                            settings.model_manager._save_annotations(
                                dataset_path, dataset["format"], image_file, annotations, classes
                            )
                            new_count=len(annotations)
                        settings._text_annotate_jobs[job_id]["annotated"] += int(new_count>0)
                        settings._text_annotate_jobs[job_id]["total_annotations"] += new_count
                        try:
                            rel_path = str(image_file.relative_to(dataset_path))
                        except Exception:
                            rel_path = image_file.name
                        img_entry = {
                            "filename": image_file.name,
                            "path": rel_path,
                            "abs_path": str(image_file),
                            "image_id": image_file.stem,
                            "annotation_count": len(annotations),
                        }
                        recent = settings._text_annotate_jobs[job_id].get("recent_images", [])
                        recent.append(img_entry)
                        settings._text_annotate_jobs[job_id]["recent_images"] = recent[-10:]
                        all_imgs = settings._text_annotate_jobs[job_id].get("all_images", [])
                        all_imgs.append(img_entry)
                        settings._text_annotate_jobs[job_id]["all_images"] = all_imgs
                        settings._images_cache.pop(dataset_id, None)
                    except Exception as exc:
                        print(f"[batch:{job_id}] save failed on {image_file.name}: {exc}")
                        settings._text_annotate_jobs[job_id]["failed"] += 1
                settings._text_annotate_jobs[job_id]["processed"] += 1
                total_done = len(already_done) + i + 1
                settings._text_annotate_jobs[job_id]["progress"] = int(total_done / max(all_count, 1) * 100)
                if i % 5 == 4:
                    _persist_jobs()

            settings._images_cache.pop(dataset_id, None)
            settings._text_annotate_jobs[job_id]["status"] = "done"
            _persist_jobs()
        except Exception as e:
            settings._text_annotate_jobs[job_id]["status"] = "error"
            settings._text_annotate_jobs[job_id]["error"] = str(e)
            _persist_jobs()

    threading.Thread(target=_continue_batch, args=(job_id,), daemon=True).start()
    return {"status": "running"}


@router.get("/api/auto-annotate/text-batch/{job_id}/preview/{idx}")
async def get_batch_job_preview_image(job_id: str, idx: int):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    recent = settings._text_annotate_jobs[job_id].get("recent_images", [])
    if idx < 0 or idx >= len(recent):
        raise HTTPException(status_code=404, detail="Image index out of range")
    abs_path = recent[idx].get("abs_path")
    if not abs_path or not Path(abs_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return owned_file_response(abs_path)


@router.get("/api/auto-annotate/text-batch/{job_id}/processed-images")
async def get_batch_job_all_images(job_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"images": settings._text_annotate_jobs[job_id].get("all_images", [])}


@router.get("/api/auto-annotate/text-batch/{job_id}/image/{image_id}")
async def get_batch_job_image_by_id(job_id: str, image_id: str):
    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    for img in settings._text_annotate_jobs[job_id].get("all_images", []):
        if img.get("image_id") == image_id:
            abs_path = img.get("abs_path")
            if abs_path and Path(abs_path).exists():
                return owned_file_response(abs_path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/api/auto-annotate/text-batch/{job_id}/annotated/{image_id}")
async def get_batch_job_annotated_image(job_id: str, image_id: str):
    import cv2
    import numpy as np

    if job_id not in settings._text_annotate_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = settings._text_annotate_jobs[job_id]
    dataset_id = job.get("dataset_id")
    if not dataset_id or dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    all_images = job.get("all_images", [])
    img_entry = next((img for img in all_images if img.get("image_id") == image_id), None)
    if not img_entry:
        raise HTTPException(status_code=404, detail="Image not found in job")

    abs_path = img_entry.get("abs_path")
    if not abs_path or not Path(abs_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    img = cv2.imdecode(np.frombuffer(owned_file_bytes(abs_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    h, w = img.shape[:2]
    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    classes = settings.model_manager._get_dataset_classes(dataset_path, dataset["format"])

    def get_class_color(idx):
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        ]
        return colors[idx % len(colors)]

    rel_path = img_entry.get("path", "")
    image_file = Path(abs_path)
    label_file = None

    if dataset["format"] == "yolo":
        search_paths = []
        abs_path_str = str(abs_path)
        if '/images/' in abs_path_str or '\\images\\' in abs_path_str:
            label_from_images = abs_path_str.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
            label_from_images = Path(label_from_images).with_suffix('.txt')
            search_paths.append(label_from_images)
        search_paths.append(image_file.with_suffix('.txt'))
        search_paths.append(image_file.parent / 'labels' / (image_file.stem + '.txt'))
        if rel_path:
            rel_label = Path(rel_path).with_suffix('.txt')
            search_paths.append(dataset_path / 'labels' / rel_label.name)
            for split in ['train', 'valid', 'val', 'test']:
                search_paths.append(dataset_path / 'labels' / split / (image_file.stem + '.txt'))
        search_paths.append(dataset_path / 'labels' / (image_file.stem + '.txt'))

        print(f"[batch-annotate] Searching for label file for {image_file.name}")
        for sp in search_paths:
            print(f"[batch-annotate]   Checking: {sp} -> exists={sp.exists() if sp else False}")
            if sp and sp.exists():
                label_file = sp
                print(f"[batch-annotate]   FOUND: {label_file}")
                break
        if not label_file:
            print(f"[batch-annotate]   No label file found for {image_file.name}")

    if label_file and label_file.exists():
        try:
            from io import StringIO
            with StringIO(owned_file_bytes(label_file).decode('utf-8')) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_idx = int(parts[0])
                    color = get_class_color(class_idx)
                    class_name = classes[class_idx] if class_idx < len(classes) else f"class_{class_idx}"
                    coords = parts[1:]
                    if len(coords) == 4:
                        cx, cy, bw, bh = map(float, coords)
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                        cv2.putText(img, class_name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        polygon_points = []
                        for i_c in range(0, len(coords) - 1, 2):
                            px = int(float(coords[i_c]) * w)
                            py = int(float(coords[i_c + 1]) * h)
                            polygon_points.append([px, py])
                        if len(polygon_points) >= 3:
                            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                            overlay = img.copy()
                            cv2.fillPoly(overlay, [pts], color)
                            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                            M = cv2.moments(pts)
                            if M["m00"] != 0:
                                cx_pt = int(M["m10"] / M["m00"])
                                cy_pt = int(M["m01"] / M["m00"])
                            else:
                                cx_pt, cy_pt = polygon_points[0]
                            (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(img, (cx_pt - 2, cy_pt - th - 4), (cx_pt + tw + 4, cy_pt + 2), color, -1)
                            cv2.putText(img, class_name, (cx_pt, cy_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"[batch] Error drawing annotations: {e}")

    _, buffer = cv2.imencode('.jpg', img)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.delete("/api/datasets/{dataset_id}/annotations")
async def delete_all_annotations(dataset_id: str):
    if dataset_id not in settings.active_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = settings.active_datasets[dataset_id]
    dataset_path = settings.DATASETS_DIR / dataset_id
    fmt = dataset["format"]
    deleted = 0

    if fmt in settings.annotation_manager.YOLO_FORMATS:
        for label_file in dataset_path.rglob("*.txt"):
            if "labels" not in label_file.parts:
                continue
            try:
                label_file.write_text("")
                deleted += 1
            except Exception:
                pass
    elif fmt in ("pascal-voc", "voc"):
        for xml_file in dataset_path.rglob("*.xml"):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall("object"):
                    root.remove(obj)
                tree.write(xml_file)
                deleted += 1
            except Exception:
                pass

    settings._images_cache.pop(dataset_id, None)
    dataset_info = settings.dataset_parser.parse_dataset(dataset_path, fmt, dataset["name"])
    dataset_info["id"] = dataset_id
    settings.active_datasets[dataset_id] = dataset_info
    _save_dataset_metadata(dataset_id, dataset_info)

    return {"success": True, "cleared_files": deleted, "updated_dataset": dataset_info}


# Preview routes share the existing authenticated, per-account router.
from routers.tile_preview import router as tile_preview_router
router.include_router(tile_preview_router)
from routers.reviews import router as reviews_router
router.include_router(reviews_router)

def _update_tile_progress(job_id, image_file, state):
    settings._text_annotate_jobs[job_id]["tile_progress"] = dict(state, image=image_file.name)

def _save_tile_review(job_id, image_file, dataset_path, annotations, debug):
    import hashlib
    relative = str(image_file.relative_to(dataset_path))
    key = hashlib.sha256(relative.encode()).hexdigest()[:20]
    folder = settings.WORKSPACE_DIR / "tile-pass" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    review = [a for a in annotations if a.get("decision") == "REVIEW"]
    human_review.atomic_bytes(folder / (key+".json"), _json.dumps(dict(image=relative,
        annotations=annotations, debug=debug), ensure_ascii=False).encode("utf-8"))
    if review:
        items = settings._text_annotate_jobs[job_id].setdefault("review_images", [])
        items[:] = [item for item in items if item.get("key") != key]
        settings._text_annotate_jobs[job_id]["review_images"].append(
            dict(image=relative, image_id=image_file.stem, count=len(review), key=key))

@router.get("/api/auto-annotate/text-batch/{job_id}/review/{key}")
def read_tile_review(job_id: str, key: str):
    import re
    if job_id not in settings._text_annotate_jobs or not re.fullmatch(r"[a-f0-9]{20}", key):
        raise HTTPException(404, "Review not found")
    path = settings.WORKSPACE_DIR / "tile-pass" / job_id / (key+".json")
    if not path.is_file():
        raise HTTPException(404, "Review not found")
    data = _json.loads(path.read_text())
    queue = human_review.get_job(job_id)
    pending = [c for c in queue["candidates"] if c["image_key"] == key and c["review_status"] == "pending"]
    return {"image": data["image"], "annotations": [dict(type="bbox", class_name=c["class_name"], bbox=c["bbox"], decision="REVIEW", decision_reasons=c["decision_reasons"], evidence=c["evidence"], candidate_id=c["candidate_id"]) for c in pending]}
