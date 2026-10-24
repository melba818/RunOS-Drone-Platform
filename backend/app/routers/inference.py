"""
Inference Router - Run YOLO inference on images, video frames, and webcam streams.
Supports detection, segmentation, pose estimation, and multi-object tracking.
"""
import asyncio
import base64
import json
import logging
import threading
import time
import traceback
import uuid
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, StrictInt

from config import settings

router = APIRouter()

# ── Smart-model types ─────────────────────────────────────────────────────────
_TEXT_PROMPT_TYPES  = {"yoloworld", "groundingdino", "owlvit", "sam", "sam2", "sam3"}
_POINT_PROMPT_TYPES = {"sam", "sam2", "sam3"}
_ZERO_SHOT_PREFIXES = ("sam_", "sam2_", "mobile_sam", "rtdetr", "fastsam")
_ZERO_SHOT_SUBSTRINGS = ("world", "clip", "groundingdino", "owlvit", "owl_vit")

# ── Tracker registry ──────────────────────────────────────────────────────────
# Every entry here ships as a bundled YAML in ultralytics/cfg/trackers/, so no
# extra package (e.g. boxmot) is required. ReID is a native Ultralytics feature:
# BoT-SORT / DeepOCSORT / TrackTrack can extract appearance embeddings from the
# detector backbone ("auto") or from a dedicated ReID checkpoint.
TRACKERS: Dict[str, Dict[str, Any]] = {
    "bytetrack":  {"label": "ByteTrack",               "reid": False},
    "botsort":    {"label": "BoT-SORT",                "reid": True},
    "ocsort":     {"label": "OC-SORT",                 "reid": False},
    "deepocsort": {"label": "Deep OC-SORT",            "reid": True},
    "fasttrack":  {"label": "FastTracker",             "reid": False},
    "tracktrack": {"label": "TrackTrack",              "reid": True},
}
DEFAULT_TRACKER = "bytetrack"

# Tracker knobs that are safe to forward from the UI, with (min, max) guards.
_TRACKER_TUNABLES = {
    "track_high_thresh": (0.0, 1.0),
    "track_low_thresh":  (0.0, 1.0),
    "new_track_thresh":  (0.0, 1.0),
    "match_thresh":      (0.0, 1.0),
    "track_buffer":      (1, 300),
}


def _tracker_type_of(tracker: Optional[str]) -> Optional[str]:
    """Resolve a tracker reference to its algorithm type.

    Accepts a bundled name (`botsort`) or a path to a custom tracker YAML whose
    `tracker_type` field names the algorithm it implements.
    """
    key = (tracker or "").strip()
    if key.lower() in TRACKERS:
        return key.lower()
    p = Path(key)
    if key and p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        t = str(cfg.get("tracker_type", "")).strip().lower() if isinstance(cfg, dict) else ""
        return t if t in TRACKERS else None
    return None


def _normalize_tracker(name: Optional[str]) -> str:
    """Validate a tracker reference: a bundled name or a custom tracker YAML path."""
    key = (name or DEFAULT_TRACKER).strip()
    if not key:
        return DEFAULT_TRACKER
    if _tracker_type_of(key):
        return key.lower() if key.lower() in TRACKERS else key
    raise HTTPException(
        status_code=422,
        detail=f"Unknown tracker '{name}'. Use one of {', '.join(TRACKERS)}, or a tracker YAML path.",
    )


def _tracker_supports_reid(tracker: str) -> bool:
    return bool(TRACKERS.get(_tracker_type_of(tracker) or "", {}).get("reid"))


def _clean_tracker_options(values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only known tracker tunables that fall inside their valid range."""
    out: Dict[str, Any] = {}
    for key, (lo, hi) in _TRACKER_TUNABLES.items():
        raw = (values or {}).get(key)
        if raw is None or raw == "":
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            continue
        if lo <= num <= hi:
            out[key] = int(num) if isinstance(lo, int) and isinstance(hi, int) else round(num, 4)
    return out


_REID_YAML_DIR: Optional[Path] = None
_reid_yaml_lock = threading.Lock()
# keyed by everything the payload depends on -> derived YAML path. Without this the
# config is re-read, re-serialised and re-hashed on *every frame* (predict_frame calls
# in here per frame), which is pure waste at camera frame rates.
_DERIVED_YAML_CACHE: Dict[str, str] = {}


def _resolve_tracker_yaml(tracker: str, with_reid: bool, reid_model: Optional[str],
                          tunables: Optional[Dict[str, Any]]) -> str:
    """Return a tracker YAML path, writing a derived one when overrides are needed.

    `tracker` is either a bundled tracker name or a path to a custom tracker YAML.
    `model.track(tracker=...)` only accepts a YAML name/path, so enabling ReID or
    overriding thresholds requires materialising a config file on disk. Derived
    files are keyed by their content hash, so identical settings are reused.

    A custom config is layered *on top of* the bundled defaults for the algorithm it
    declares, so it only needs to state what it changes rather than every key the
    tracker class reads.
    """
    global _REID_YAML_DIR

    ttype = _tracker_type_of(tracker) or DEFAULT_TRACKER
    is_bundled = tracker.strip().lower() in TRACKERS

    # Nothing to override on a bundled tracker -> use its shipped YAML as-is.
    if is_bundled and not with_reid and not tunables:
        return f"{ttype}.yaml"

    # Everything the derived payload depends on. Imported configs carry their mtime so
    # an edited file is picked up instead of serving a stale derivation.
    stamp = "-"
    if not is_bundled:
        try:
            stamp = str(Path(tracker).stat().st_mtime_ns)
        except OSError:
            pass
    cache_key = "|".join((
        tracker,
        stamp,
        str(int(bool(with_reid))),
        (reid_model or "").strip(),
        json.dumps(tunables or {}, sort_keys=True, ensure_ascii=False),
    ))
    cached = _DERIVED_YAML_CACHE.get(cache_key)
    if cached and Path(cached).is_file():
        return cached

    from ultralytics.utils.checks import check_yaml

    # Layer 1: the algorithm's bundled defaults, which carry every key its class reads.
    try:
        cfg = dict(yaml.safe_load(Path(check_yaml(f"{ttype}.yaml")).read_text(encoding="utf-8")))
    except Exception:
        cfg = {"tracker_type": ttype}

    # Layer 2: the imported config, which only declares what it changes.
    if not is_bundled:
        try:
            cfg.update(dict(yaml.safe_load(Path(check_yaml(tracker)).read_text(encoding="utf-8"))))
        except Exception:
            pass

    cfg["tracker_type"] = ttype

    if with_reid and _tracker_supports_reid(tracker):
        cfg["with_reid"] = True
        # "auto" extracts embeddings from the detector backbone (no extra download).
        cfg["model"] = reid_model.strip() if reid_model and reid_model.strip() else "auto"
    else:
        cfg["with_reid"] = False

    cfg.update(tunables or {})

    import hashlib

    # Serialize with plain PyYAML. ultralytics' YAML class exposes only load/__call__
    # in this version (no .dump), so it cannot be used as a serializer here.
    payload = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    with _reid_yaml_lock:
        if _REID_YAML_DIR is None:
            _REID_YAML_DIR = Path(settings.TEMP_DIR) / "trackers"
        # Re-created on every call, not only the first. `workspace/temp` is emptied by
        # the startup cleaner and by disk clean-ups, and a cached path would otherwise
        # stay stale for the whole life of the process: every tracking call would then
        # die with FileNotFoundError: [Errno 2] ... temp/trackers/<algo>-<hash>.yaml.
        try:
            _REID_YAML_DIR.mkdir(parents=True, exist_ok=True)
            # Name by algorithm + content hash: `tracker` may be an absolute path, which
            # cannot be used verbatim as a file name.
            target = _REID_YAML_DIR / f"{ttype}-{digest}.yaml"
            if not target.exists():
                target.write_text(payload, encoding="utf-8")
        except OSError as exc:
            # Surface an actionable message instead of a bare `[Errno 2] ...` in the UI.
            raise HTTPException(
                status_code=500,
                detail=f"无法写入跟踪器配置（目录 {_REID_YAML_DIR} 不可用）：{exc}",
            )
        _DERIVED_YAML_CACHE[cache_key] = str(target)
    return str(target)



def _is_plain_yolo(name: str) -> bool:
    """True for YOLO .pt files that work with raw YOLO predict."""
    lower = name.lower()
    if any(lower.startswith(p) for p in _ZERO_SHOT_PREFIXES):
        return False
    if any(s in lower for s in _ZERO_SHOT_SUBSTRINGS):
        return False
    return True


def _normalize_classes(value) -> Optional[List[int]]:
    """Missing/null/empty selection means all classes; reject malformed IDs."""
    if value is None:
        return None
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise HTTPException(status_code=422, detail="Classes must be a list of non-negative integer IDs")
    return sorted(set(value)) or None


def _parse_classes(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not value.strip():
        return None
    try:
        return _normalize_classes(json.loads(value))
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="Classes must be a JSON array of integer IDs")


def _model_classes(model) -> List[Dict[str, Any]]:
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return sorted([{"id": int(i), "name": str(name)} for i, name in items], key=lambda c: c["id"])


def _validate_model_classes(model, classes: Optional[List[int]]) -> None:
    if classes:
        unknown = sorted(set(classes) - {c["id"] for c in _model_classes(model)})
        if unknown:
            raise HTTPException(status_code=422, detail=f"Class IDs not present in this model: {unknown}")


@router.get("/api/infer/classes")
def list_infer_classes(model_path: str):
    """Read the selected model's own names, including custom trained classes."""
    if not Path(model_path).is_file():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    try:
        with settings.model_manager.runtime_lock:
            model = _get_or_create_session("classes", model_path, "detect", "bytetrack")["model"]
            return {"classes": _model_classes(model)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read model classes: {e}")


def _render_annotations(image_path: Path, annotations: List[Dict[str, Any]]) -> tuple:
    """Draw bbox/polygon annotations, return (jpeg_bytes, detections_list)."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
    ]
    detections: List[Dict[str, Any]] = []

    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        ann_type = ann.get("type", "bbox")
        cls_name = ann.get("class_name", "object")
        conf = float(ann.get("confidence") or 0.0)
        confidence_source = ann.get("confidence_source", "detector")
        label = f"{cls_name} {conf:.2f}" if conf > 0 and confidence_source == "detector" else cls_name

        if ann_type == "bbox":
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            detections.append({
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "confidence_source": confidence_source,
                "bbox": [round(x, 1), round(y, 1), round(x + bw, 1), round(y + bh, 1)],
            })

        elif ann_type == "polygon":
            pts_flat = ann.get("points") or []
            if not pts_flat and "segmentation" in ann:
                seg = ann["segmentation"]
                pts_flat = seg[0] if (seg and isinstance(seg[0], list)) else seg
            if len(pts_flat) >= 6:
                pts = (np.array(pts_flat, dtype=np.float32)
                         .reshape(-1, 2).astype(np.int32))
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                cv2.polylines(img, [pts], True, color, 2)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                cv2.putText(img, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "confidence_source": confidence_source,
                    "mask_points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                                    for p in pts[:30].tolist()],
                })

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf), detections


# ── Session manager (keeps model instances alive for tracking continuity) ─────
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 300  # seconds of inactivity before cleanup


def _cleanup_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        stale = [sid for sid, s in _sessions.items() if now - s["last_used"] > SESSION_TTL]
        for sid in stale:
            _sessions.pop(sid, None)


def _get_or_create_session(session_id: str, model_path: str, task: str, tracker: str,
                           classes: Optional[List[int]] = None, device=None) -> Dict:
    canonical = str(Path(model_path).resolve())
    is_rf = Path(canonical).suffix.lower() == ".pth"
    if is_rf:
        from rfdetr_adapter import metadata
        metadata(canonical)
        import torch
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    owner = "camera" if session_id.startswith("camera:") else session_id
    cache_key = f"track:{owner}:{tracker}:{canonical}" if task == "track" else f"predict:{canonical}"
    if device is not None:
        cache_key += f":device:{device}"
    with settings.model_manager.runtime_lock, _sessions_lock:
        existing = _sessions.get(cache_key)
        stamp = Path(canonical).stat().st_mtime_ns
        if existing and existing["stamp"] == stamp:
            if task == "track" and (existing.get("stream_id") != session_id or existing.get("classes") != classes):
                # Reuse weights on camera restart, but never reuse IDs from the previous stream/classes.
                for active_tracker in getattr(getattr(existing["model"], "predictor", None), "trackers", []):
                    active_tracker.reset()
                existing.update(stream_id=session_id, classes=classes)
            existing["last_used"] = time.time()
            return existing
        model_id = next((mid for mid, item in settings.model_manager.loaded_models.items()
                         if is_rf and item.get("type") == "rfdetr" and item.get("device") == device and item.get("model") is not None
                         and item.get("path") == canonical and item.get("_rf_stamp", stamp) == stamp), None)
        if model_id is None:
            model_id = settings.model_manager.load_model(canonical, "rfdetr" if is_rf else "yolo", **({"device": device} if device is not None else {}))
        info = settings.model_manager.loaded_models[model_id]
        if info.get("model") is None or info.get("error"):
            settings.model_manager.loaded_models.pop(model_id, None)
            raise RuntimeError(info.get("error", "Model could not be loaded"))
        if is_rf:
            from rfdetr_adapter import RFDetectorAdapter
            info["_rf_stamp"] = stamp
            instance = RFDetectorAdapter(info["model"], canonical)
        else:
            instance = info["model"]
        record = {"model": instance, "model_id": model_id, "stamp": stamp,
                  "model_path": canonical, "last_used": time.time(), "stream_id": session_id, "classes": classes}
        if existing:
            settings.model_manager.loaded_models.pop(existing["model_id"], None)
        _sessions[cache_key] = record
        while len(_sessions) > 2:
            oldest = min((k for k in _sessions if k != cache_key), key=lambda k: _sessions[k]["last_used"])
            old = _sessions.pop(oldest)
            settings.model_manager.loaded_models.pop(old["model_id"], None)
        return record


def predict_frame(frame, model_path: str, task="detect", confidence=0.25, iou=0.45,
                  imgsz=640, tracker="bytetrack", classes=None, session_id="", device=None,
                  with_reid=False, reid_model=None, track_options=None):
    """One ndarray prediction path for images, file video, browser and local camera."""
    if not Path(model_path).is_file():
        raise HTTPException(status_code=404, detail="Model file no longer exists")
    if Path(model_path).suffix.lower() == ".pth" and task not in ("detect", "track"):
        raise HTTPException(422, "当前 RF-DETR 标签页支持目标检测和多目标跟踪。")
    # A tracker builds its appearance encoder from `args.model`, so the eager route has to
    # be in place before the first track call. Idempotent and cheap after the first time.
    from reid_encoders import install_eager_reid_hook
    install_eager_reid_hook()
    selected = _normalize_classes(classes)
    tracker = _normalize_tracker(tracker)
    tunables = _clean_tracker_options(track_options)

    # ReID is only meaningful for appearance-aware trackers; silently ignore it elsewhere.
    use_reid = bool(with_reid) and _tracker_supports_reid(tracker)
    # The encoder is part of the session key: trackers are created once per session and
    # `persist=True` keeps them, so a changed encoder must land in a fresh session.
    encoder = (reid_model or "").strip() if use_reid else ""
    settings_key = f"{tracker}|reid={int(use_reid)}|{encoder}|{tunables}"

    with settings.model_manager.runtime_lock:
        model = _get_or_create_session(
            session_id, model_path, task, settings_key, selected, device=device
        )["model"]
        _validate_model_classes(model, selected)
        options = dict(conf=confidence, iou=iou, imgsz=imgsz, verbose=False, classes=selected)
        if device is not None:
            options["device"] = device
        if task == "track":
            tracker_yaml = _resolve_tracker_yaml(tracker, use_reid, reid_model, tunables)
            return model.track(frame, tracker=tracker_yaml, persist=True, **options)
        return model.predict(frame, **options)


def infer_numpy(frame, model_path: str, timings=None, compute_node_id=None, **options):
    if compute_node_id:
        from compute.remote_inference import nodes
        return nodes.frame(compute_node_id, frame, model_path, timings, options)
    with settings.model_manager.runtime_lock:
        started = time.perf_counter()
        results = predict_frame(frame, model_path, **options)
        if timings is not None:
            timings['inference_ms'] = (time.perf_counter()-started)*1000
            # New Ultralytics versions copy the CPU checkpoint into the predictor backend.
            # Result tensor placement reflects actual execution, unlike YOLO.model.device.
            data = getattr(getattr(results[0], 'boxes', None), 'data', None)
            timings['device'] = str(getattr(data, 'device', 'unknown'))
        return _build_result(results, options.get("task", "detect"), options.get("task") == "track")


# ── Video processing jobs ─────────────────────────────────────────────────────
_video_jobs: Dict[str, Dict[str, Any]] = {}


# ── Result annotation helper ──────────────────────────────────────────────────

def _build_result(results, task: str, is_tracking: bool):
    """Return (jpeg_bytes, detections_list) from ultralytics Results."""
    import cv2

    result = results[0]
    annotated = result.plot()
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_bytes = bytes(buf)

    detections: List[Dict[str, Any]] = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            d: Dict[str, Any] = {
                "class_id":   int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [round(float(v), 1) for v in box.xyxy[0].tolist()],
            }
            if is_tracking and box.id is not None:
                d["track_id"] = int(box.id[0])
            detections.append(d)

    if task == "segment" and result.masks is not None:
        for i, poly in enumerate(result.masks.xy):
            if i < len(detections):
                detections[i]["mask_points"] = [
                    [round(float(p[0]), 1), round(float(p[1]), 1)]
                    for p in poly[:30]
                ]

    if task == "pose" and result.keypoints is not None:
        for i, kp in enumerate(result.keypoints.xy):
            if i < len(detections):
                detections[i]["keypoints"] = [
                    [round(float(p[0]), 1), round(float(p[1]), 1)]
                    for p in kp.tolist()
                ]

    return img_bytes, detections


# ── Frame endpoint (webcam streaming) ─────────────────────────────────────────

class FrameRequest(BaseModel):
    model_path: str
    frame_b64: str          # base64-encoded JPEG
    task: str = "detect"   # detect | segment | pose | track
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    tracker: str = DEFAULT_TRACKER   # bytetrack | botsort | ocsort | deepocsort | fasttrack | tracktrack
    session_id: str = ""
    # ── Tracking / ReID ──
    with_reid: bool = False          # enable appearance embeddings (ReID-aware trackers only)
    reid_model: Optional[str] = None # "auto" = detector backbone features; or a .pt / exported model path
    track_high_thresh: Optional[float] = None
    track_low_thresh: Optional[float] = None
    new_track_thresh: Optional[float] = None
    match_thresh: Optional[float] = None
    track_buffer: Optional[int] = None
    classes: Optional[List[StrictInt]] = None

    def tracker_kwargs(self) -> Dict[str, Any]:
        """Tracking-only options, isolated so prediction calls stay unaffected."""
        return {
            "with_reid": self.with_reid,
            "reid_model": self.reid_model,
            "track_options": {
                "track_high_thresh": self.track_high_thresh,
                "track_low_thresh": self.track_low_thresh,
                "new_track_thresh": self.new_track_thresh,
                "match_thresh": self.match_thresh,
                "track_buffer": self.track_buffer,
            },
        }

    def predict_kwargs(self) -> Dict[str, Any]:
        """Exactly the keyword arguments `infer_numpy` accepts, ready to splat.

        `model_path` is included because callers splat this straight into `infer_numpy`
        (`infer_numpy(frame, **options)`); `infer_numpy` consumes it and forwards the rest
        to `predict_frame`. `session_id` is deliberately **not** included — callers pass it
        explicitly alongside the splat (the camera uses `camera:<id>`), and having it in
        both places raises "got multiple values for keyword argument 'session_id'".

        Build this instead of `model_dump()`-ing the model: the five tracker tunables are
        declared flat here for the HTTP API, but `predict_frame` takes them as a single
        `track_options` dict. Dumping the model hands them over flat and the call dies with
        `predict_frame() got an unexpected keyword argument 'track_high_thresh'`.
        """
        return {
            "model_path": self.model_path,
            "task": self.task, "confidence": self.confidence, "iou": self.iou,
            "imgsz": self.imgsz, "tracker": self.tracker,
            "classes": _normalize_classes(self.classes),
            **self.tracker_kwargs(),
        }


@router.post("/api/infer/frame")
async def infer_frame(req: FrameRequest):
    if not req.model_path or not Path(req.model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {req.model_path}")
    classes = _normalize_classes(req.classes)
    try:
        import cv2
        import numpy as np

        img_data = base64.b64decode(req.frame_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame")

        session_id = req.session_id or str(uuid.uuid4())[:8]
        options = req.predict_kwargs()   # carries model_path as well
        options["session_id"] = session_id
        options["classes"] = classes
        img_bytes, detections = await asyncio.to_thread(infer_numpy, frame, **options)
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/infer/session/{session_id}")
async def close_session(session_id: str):
    with settings.model_manager.runtime_lock, _sessions_lock:
        for key in list(_sessions):
            if key.startswith(f"track:{session_id}:"):
                info = _sessions.pop(key)
                settings.model_manager.loaded_models.pop(info["model_id"], None)
    return {"success": True}


# ── Single-image endpoint ──────────────────────────────────────────────────────

@router.post("/api/infer/image")
async def infer_image(
    model_path: str = Form(...),
    task: str = Form("detect"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    image: UploadFile = File(...),
    classes: Optional[str] = Form(None),
    tracker: str = Form(DEFAULT_TRACKER),
    with_reid: bool = Form(False),
    reid_model: Optional[str] = Form(None),
    track_high_thresh: Optional[float] = Form(None),
    track_low_thresh: Optional[float] = Form(None),
    new_track_thresh: Optional[float] = Form(None),
    match_thresh: Optional[float] = Form(None),
    track_buffer: Optional[int] = Form(None),
):
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    selected_classes = _parse_classes(classes)
    _normalize_tracker(tracker)  # reject unknown trackers before decoding the upload
    try:
        import cv2
        import numpy as np

        img_data = await image.read()
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        img_bytes, detections = await asyncio.to_thread(
            infer_numpy, frame, model_path, task=task, confidence=confidence, iou=iou,
            imgsz=imgsz, tracker=tracker, classes=selected_classes, session_id=str(uuid.uuid4()),
            with_reid=with_reid, reid_model=reid_model,
            track_options={
                "track_high_thresh": track_high_thresh,
                "track_low_thresh": track_low_thresh,
                "new_track_thresh": new_track_thresh,
                "match_thresh": match_thresh,
                "track_buffer": track_buffer,
            })
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Video processing ───────────────────────────────────────────────────────────

def _write_track_files(job_id: str, out_path: Path, fps: float, width: int, height: int,
                       trajectories: Dict[int, List[Dict[str, Any]]], class_names: Dict[int, str]) -> Dict[str, str]:
    """Persist MOTChallenge .txt and a JSON trajectory summary next to the video."""
    mot_path = out_path.with_name(f"{job_id}_tracks.txt")
    json_path = out_path.with_name(f"{job_id}_tracks.json")

    rows: List[str] = []
    for track_id, points in trajectories.items():
        for p in points:
            x, y, w, h = p["bbox"]
            rows.append(
                f"{p['frame']},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{p['confidence']:.4f},{p['class_id']},-1"
            )
    # MOTChallenge convention: sorted by frame, then by track id.
    rows.sort(key=lambda line: (int(line.split(",")[0]), int(line.split(",")[1])))
    mot_path.write_text("\n".join(rows), encoding="utf-8")

    summary = {
        "job_id": job_id,
        "video": {"fps": round(fps, 3), "width": width, "height": height},
        "track_count": len(trajectories),
        "total_detections": sum(len(v) for v in trajectories.values()),
        "tracks": [
            {
                "track_id": track_id,
                "class_id": points[0]["class_id"],
                "class_name": class_names.get(points[0]["class_id"], str(points[0]["class_id"])),
                "first_frame": points[0]["frame"],
                "last_frame": points[-1]["frame"],
                "length": len(points),
                "frames": [p["frame"] for p in points],
                "bboxes": [p["bbox"] for p in points],
                "confidences": [p["confidence"] for p in points],
            }
            for track_id, points in sorted(trajectories.items())
        ],
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"mot_path": str(mot_path), "json_path": str(json_path)}


def _video_log(job, message):
    lines=job.setdefault('logs',[])
    lines.append(time.strftime('%H:%M:%S')+' '+message)
    del lines[:-200]


def _run_video_job(job_id: str, model_path: str, video_path: str, task: str,
                   confidence: float, iou: float, imgsz: int, tracker: str,
                   classes: Optional[List[int]] = None,
                   with_reid: bool = False, reid_model: Optional[str] = None,
                   track_options: Optional[Dict[str, Any]] = None) -> None:
    job = _video_jobs[job_id]
    cap = out = None
    try:
        import cv2

        job["status"] = "running"
        job["message"] = "加载模型…"
        _video_log(job, '输入已接收，准备模型：'+Path(model_path).name)
        job["progress"] = 0.0
        job["processed"] = 0
        job["eta"] = None
        job["proc_fps"] = None
        t0 = time.time()
        last_tick = 0.0
        is_tracking = task == "track"
        trajectories: Dict[int, List[Dict[str, Any]]] = {}
        class_names: Dict[int, str] = {}

        # Announce up front whether appearance matching will really happen, so a run that
        # silently ignores the encoder is visible before the frames are processed.
        job.update(_reid_verdict(with_reid, tracker, reid_model))
        if job["reid_enabled"]:
            from reid_encoders import reset_reid_tally
            reset_reid_tally()
            logging.info("job %s: ReID 已启用（%s）", job_id, job.get("reid_label", ""))
        else:
            logging.info("job %s: ReID 未启用（%s）", job_id, job.get("reid_reason", ""))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError("Could not open input video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        job["total_frames"] = total
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _video_log(job, f'视频 {w}×{h}，{total} 帧；开始推理')
        last_log = 0.0

        out_path = Path(video_path).parent / f"{job_id}_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not out.isOpened(): raise RuntimeError("Could not create output video")

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            with settings.model_manager.runtime_lock:
                results = predict_frame(frame, model_path, task, confidence, iou, imgsz,
                                        tracker, classes, session_id=job_id,
                                        with_reid=with_reid, reid_model=reid_model,
                                        track_options=track_options)
                out.write(results[0].plot())

                # Collect per-frame trajectories (frames are 1-indexed per MOTChallenge).
                if is_tracking and results[0].boxes is not None:
                    names = results[0].names
                    for box in results[0].boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id[0])
                        cid = int(box.cls[0])
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                        trajectories.setdefault(tid, []).append({
                            "frame": idx + 1,
                            "class_id": cid,
                            "confidence": round(float(box.conf[0]), 4),
                            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                        })
                        class_names[cid] = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)

            idx += 1
            # Plain frame ratio: strictly non-decreasing, so the bar can never rewind.
            job["progress"] = round(idx / total * 100, 1)
            job["processed"] = idx
            now = time.time()
            # Refresh the human-readable line ~2x/s. Writing it on every frame would
            # be pointless churn, but leaving it frozen reads as a hung job.
            if now - last_tick >= 0.5:
                last_tick = now
                elapsed = now - t0
                rate = idx / elapsed if elapsed > 0 else 0.0
                job["proc_fps"] = round(rate, 2)
                job["eta"] = round((total - idx) / rate, 1) if rate > 0 else None
                job["message"] = f"处理中 {idx}/{total} 帧"
                if now-last_log >= 3:
                    last_log=now
                    _video_log(job, f'已处理 {idx}/{total} 帧 | {rate:.2f} FPS | {job["progress"]:.1f}%')

        cap.release()
        cap = None
        out.release()
        out = None

        _video_log(job, '逐帧推理完成，正在整理结果文件')
        job["result_path"] = str(out_path)
        job["frames"] = idx
        job["message"] = f"Processed {idx} frames"

        if is_tracking and trajectories:
            files = _write_track_files(job_id, out_path, fps, w, h, trajectories, class_names)
            job["track_count"] = len(trajectories)
            job["total_detections"] = sum(len(v) for v in trajectories.values())
            job["tracks_mot_path"] = files["mot_path"]
            job["tracks_json_path"] = files["json_path"]
            job["message"] += f" · {len(trajectories)} tracks"
        elif is_tracking:
            job["track_count"] = 0
            job["total_detections"] = 0
            job["message"] += " · no tracks found"

        # Report what the encoder actually did, not just what was configured: zero calls
        # with detections present means appearance matching never ran.
        if job.get("reid_enabled") and not job.get("reid_native"):
            from reid_encoders import reid_tally
            tally = reid_tally()
            job["reid_calls"] = tally["calls"]
            job["reid_crops"] = tally["crops"]
            job["message"] += f" · ReID {tally['calls']} 次 / {tally['crops']} crop"
            logging.info("job %s: ReID 实际编码 %s 个 crop（%s 次调用）",
                         job_id, tally["crops"], tally["calls"])
        job['progress']=100
        job['status']='completed'
        _video_log(job, f'完成：{idx} 帧，结果视频可下载')
    except Exception as e:
        # The full traceback (TorchScript included) is the only thing that actually localises
        # these failures, so it goes to the log; the UI gets one actionable line.
        logging.exception("video job %s failed", job_id)
        job["status"] = "failed"
        job["error"] = traceback.format_exc()
        job["message"] = f"Failed: {_job_error_summary(e)}"
        _video_log(job, job['message'])
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        # A finished job must not keep holding VRAM: the detector and the appearance
        # encoder stay resident in the model cache, so back-to-back jobs on a 2 GB card
        # otherwise run out of memory part-way through the second or third one.
        _release_gpu(job_id)



@router.post("/api/infer/video")
async def infer_video(
    model_path: str = Form(...),
    task: str = Form("detect"),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    tracker: str = Form(DEFAULT_TRACKER),
    video: UploadFile = File(...),
    classes: Optional[str] = Form(None),
    with_reid: bool = Form(False),
    reid_model: Optional[str] = Form(None),
    track_high_thresh: Optional[float] = Form(None),
    track_low_thresh: Optional[float] = Form(None),
    new_track_thresh: Optional[float] = Form(None),
    match_thresh: Optional[float] = Form(None),
    track_buffer: Optional[int] = Form(None),
):
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    selected_classes = _parse_classes(classes)
    _normalize_tracker(tracker)  # fail fast with 422 instead of failing inside the job thread
    _preflight_reid(reid_model, with_reid)  # refuse an impossible backbone/head pairing up front

    job_id = str(uuid.uuid4())[:8]
    save_dir = settings.TEMP_DIR / "inference"
    save_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    video_path = save_dir / f"{job_id}_input{suffix}"
    video_path.write_bytes(await video.read())

    _video_jobs[job_id] = {
        "id": job_id, "status": "starting", "progress": 0,
        "message": "Starting…", "result_path": None, "error": None,
    }
    threading.Thread(
        target=_run_video_job,
        args=(job_id, model_path, str(video_path), task, confidence, iou, imgsz, tracker, selected_classes),
        kwargs={
            "with_reid": with_reid,
            "reid_model": reid_model,
            "track_options": {
                "track_high_thresh": track_high_thresh,
                "track_low_thresh": track_low_thresh,
                "new_track_thresh": new_track_thresh,
                "match_thresh": match_thresh,
                "track_buffer": track_buffer,
            },
        },
        daemon=True,
    ).start()
    return {"job_id": job_id}


@router.get("/api/infer/video/{job_id}/status")
async def video_status(job_id: str):
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _video_jobs[job_id]
    return {"id": job_id, "status": j["status"], "progress": j["progress"],
            "message": j.get("message", ""), "error": j.get("error"),
            "logs": list(j.get("logs", [])),
            "frames": j.get("frames"), "track_count": j.get("track_count"),
            "processed": j.get("processed"), "total_frames": j.get("total_frames"),
            "eta": j.get("eta"), "proc_fps": j.get("proc_fps"),
            "total_detections": j.get("total_detections"),
            "has_tracks": bool(j.get("tracks_mot_path")),
            "reid_enabled": j.get("reid_enabled"), "reid_reason": j.get("reid_reason"),
            "reid_label": j.get("reid_label"), "reid_native": j.get("reid_native"),
            "reid_calls": j.get("reid_calls"), "reid_crops": j.get("reid_crops")}


@router.get("/api/infer/video/{job_id}/result")
async def video_result(job_id: str):
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _video_jobs[job_id]
    if j["status"] != "completed" or not j.get("result_path"):
        raise HTTPException(status_code=400, detail="Video not ready")
    p = j["result_path"]
    if not Path(p).exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(p, filename=f"inference_{job_id}.mp4", media_type="video/mp4")


@router.get("/api/infer/video/{job_id}/tracks/{fmt}")
async def video_tracks(job_id: str, fmt: str):
    """Download exported trajectories: fmt = mot (MOTChallenge .txt) | json."""
    if job_id not in _video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _video_jobs[job_id]
    if j["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not finished")
    if fmt == "mot":
        p, name, media = j.get("tracks_mot_path"), f"tracks_{job_id}.txt", "text/plain"
    elif fmt == "json":
        p, name, media = j.get("tracks_json_path"), f"tracks_{job_id}.json", "application/json"
    else:
        raise HTTPException(status_code=422, detail="Format must be 'mot' or 'json'")
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="No tracks were recorded for this job")
    return FileResponse(p, filename=name, media_type=media)


@router.get("/api/infer/trackers")
async def list_trackers():
    """Tracker catalogue for the UI, so options always match what the backend supports."""
    return {"default": DEFAULT_TRACKER,
            "trackers": [{"value": k, **v} for k, v in TRACKERS.items()]}


# ── ReID model catalogue + download ───────────────────────────────────────────
# Encoders usable by ultralytics' native ReID (trackers/utils/reid.py):
#   *.pt classification checkpoints -> YOLO() embed path (torch only, no extras)
#   *.onnx ReID exports            -> AutoBackend (requires onnxruntime)
# All ten files below are official ultralytics/assets release assets.

_REID_OFFICIAL: List[Dict[str, str]] = (
    [{"name": f"yolo26{k}-cls.pt", "kind": "pt"} for k in "nsmlx"]
    + [{"name": f"yolo26{k}-reid.onnx", "kind": "onnx"} for k in "nsmlx"]
)
_REID_ASSET_NAMES = frozenset(spec["name"] for spec in _REID_OFFICIAL)
_REID_DL_STATUS: Dict[str, Dict[str, Any]] = {}


def _reid_dir() -> Path:
    """Directory where official ReID encoders are stored (created on demand)."""
    d = Path(settings.MODELS_DIR) / "reid"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reid_local_candidates() -> List[Dict[str, Any]]:
    """ReID-usable local checkpoints (.pt/.onnx with reid/cls names) on disk."""
    out: List[Dict[str, Any]] = []
    seen = set()
    roots = [Path(settings.MODELS_DIR)]
    try:
        from ultralytics.utils import SETTINGS as UL_SETTINGS
        wd = Path(str(UL_SETTINGS.get("weights_dir", "")))
        if str(wd) and wd not in roots:
            roots.append(wd)  # catches assets auto-downloaded by ultralytics earlier
    except Exception:
        pass
    official_dir = _reid_dir()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if not p.is_file() or p.suffix.lower() not in (".pt", ".onnx"):
                    continue
                low = p.name.lower()
                if "reid" not in low and "-cls" not in low:
                    continue
                if p.name in _REID_ASSET_NAMES and p.parent == official_dir:
                    continue  # already shown in the official section
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "name": p.name,
                    "path": key,
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                    "official": p.name in _REID_ASSET_NAMES,
                })
            except OSError:
                continue
    return sorted(out, key=lambda m: m["name"].lower())


@router.get("/api/infer/reid-models")
async def list_reid_models():
    """ReID encoder catalogue: official downloadable assets + local candidates."""
    reid_dir = _reid_dir()
    try:
        import importlib.util
        ort = importlib.util.find_spec("onnxruntime") is not None
    except Exception:
        ort = False
    official = []
    for spec in _REID_OFFICIAL:
        name = spec["name"]
        target = reid_dir / name
        st = _REID_DL_STATUS.get(name, {})
        if target.exists() and target.stat().st_size > 1e5:
            status = "downloaded"
        elif st.get("status") == "downloading":
            status = "downloading"
        elif st.get("status") == "error":
            status = "error"
        else:
            status = "not_downloaded"
        official.append({
            **spec,
            "path": str(target),
            "status": status,
            "progress": st.get("progress", 0),
            "size_mb": round(target.stat().st_size / 1e6, 1) if target.exists() else None,
            "error": st.get("error"),
        })
    return {"onnxruntime": ort, "official": official, "local": _reid_local_candidates()}


class ReidDownloadRequest(BaseModel):
    name: str


@router.post("/api/infer/reid-models/download")
async def download_reid_model(req: ReidDownloadRequest, background_tasks: BackgroundTasks):
    """Download an official ReID encoder into workspace/models/reid (background)."""
    name = Path(req.name.strip()).name  # basename only — no path traversal
    if name not in _REID_ASSET_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown ReID asset '{req.name}'. Downloadable: {', '.join(sorted(_REID_ASSET_NAMES))}",
        )
    target = _reid_dir() / name
    if target.exists() and target.stat().st_size > 1e5:
        return {"status": "already_downloaded", "path": str(target)}
    if _REID_DL_STATUS.get(name, {}).get("status") == "downloading":
        return {"status": "already_downloading"}
    _REID_DL_STATUS[name] = {"status": "downloading", "progress": 5, "error": None}
    background_tasks.add_task(_do, name, target)
    return {"status": "started", "path": str(target)}


_REID_MIRROR_PREFIXES = (
    "https://ghfast.top/",
    "https://gh-proxy.com/",
    "https://mirror.ghproxy.com/",
)


def _download_reid_chunked(url: str, tgt: Path, nm: str) -> None:
    """Plain-urllib chunked download (bypasses curl_cffi) with real progress.

    Writes to `<name>.part` and renames into place only when complete.
    """
    from urllib.request import Request, urlopen

    part = tgt.parent / (tgt.name + ".part")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                _REID_DL_STATUS[nm]["progress"] = min(99, int(got * 100 / total))
    if part.stat().st_size < 1e5:
        raise RuntimeError(f"文件过小（{part.stat().st_size} 字节），下载不完整")
    part.replace(tgt)


def _do(nm: str, tgt: Path):
    stop = threading.Event()

    def _tick():
        # attempt_download_asset has no progress callback; fake a slow climb
        # so the UI shows movement while the real transfer runs.
        prog = 30
        while not stop.wait(2.0) and prog < 90:
            prog = min(90, prog + 5)
            st = _REID_DL_STATUS.get(nm)
            if st and st.get("status") == "downloading":
                st["progress"] = prog

    threading.Thread(target=_tick, daemon=True).start()
    try:
        import shutil
        from ultralytics.utils.downloads import attempt_download_asset

        got = Path(attempt_download_asset(str(tgt)))
        if got.exists() and got.stat().st_size > 1e5:
            if got.resolve() != tgt.resolve():
                # Resolved elsewhere (e.g. weights_dir) — copy into the reid dir
                # so the catalogue entry stays authoritative.
                shutil.copyfile(got, tgt)
        else:
            # GitHub releases are unreliable on some networks — fall back to
            # mirror proxies, then a direct urllib transfer, until one works.
            stop.set()  # hand progress over to real Content-Length updates
            gh = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{nm}"
            urls = [p + gh for p in _REID_MIRROR_PREFIXES] + [gh]
            errors = []
            for url in urls:
                try:
                    _download_reid_chunked(url, tgt, nm)
                    break
                except Exception as e:
                    errors.append(f"{url.split('/')[2]}: {e}")
                    (tgt.parent / (tgt.name + ".part")).unlink(missing_ok=True)
            else:
                raise RuntimeError("；".join(errors))
        _REID_DL_STATUS[nm] = {"status": "done", "progress": 100, "error": None}
    except Exception as e:
        _REID_DL_STATUS[nm] = {"status": "error", "progress": 0, "error": str(e)}
        try:
            tgt.unlink(missing_ok=True)  # drop partial download
        except OSError:
            pass
    finally:
        stop.set()


# ── Model zones: one list per role the pipeline can consume ────────────────────
# The pipeline is  detector -> tracker -> (optional appearance module) -> tracks.
# Each role keeps its own list, so a detector and an appearance encoder can be
# loaded at the same time. Files land in workspace/models (detector / other /
# tracker) or workspace/models/reid (reid / head); external paths can be
# registered without copying.
#   detector : object-detection / segmentation / pose weights (task inferred)
#   tracker  : a tracker YAML + the algorithm type it implements
#   backbone : the feature extractor that produces the appearance embedding.
#              Any trunk works (OSNet, DINOv2/DINOv3, CLIP, timm…); what matters
#              is its output width, which is what a projection head must match.
#   reid     : legacy alias of `backbone`, kept so older imports stay reachable
#   head     : optional projection head stacked on top of a `backbone`
#   other    : anything else, kept for reference and never executed

_IMPORT_ZONES = ("detector", "tracker", "backbone", "reid", "head", "other")
_REID_EXPORT_SUFFIX = ".torchscript"


def _registry_path() -> Path:
    return Path(settings.MODELS_DIR) / "_imported.json"


def _load_registry() -> List[Dict[str, Any]]:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("entries", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _save_registry(entries: List[Dict[str, Any]]) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")


def _zone_dir(zone: str) -> Path:
    if zone == "backbone":
        d = Path(settings.MODELS_DIR) / "backbone"
    elif zone in ("reid", "head"):
        d = _reid_dir()
    elif zone == "tracker":
        d = Path(settings.MODELS_DIR) / "trackers"
    else:
        d = Path(settings.MODELS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _composed_dir() -> Path:
    """Derived encoder+head artifacts.

    Kept in a sub-directory so the reid zone's file scan (which only looks at direct
    children) does not list generated modules as if the user had imported them.
    """
    d = _reid_dir() / "composed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inside_models(p: Path) -> bool:
    """True when a path lives inside our own models dir (safe to delete on re-import)."""
    try:
        Path(p).resolve().relative_to(Path(settings.MODELS_DIR).resolve())
        return True
    except (ValueError, OSError):
        return False


def _head_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Optional sidecar describing how this head was trained, e.g.

        {"type": "ProjectionHead", "backbone": "dinov3_vits16",
         "feature_strategy": "last4_cls", "input_dim": 1536, "output_dim": 128,
         "normalize": true}

    Looked up next to the weights (`<name>.json` / `<stem>.head.json`) so nothing has to
    be edited inside the .pth. When present, the UI can preselect that strategy; when
    absent the choice stays with the user and dims are matched instead.
    """
    import json
    cands = [path.with_suffix(".json"), path.with_name(f"{path.stem}.head.json")]
    for c in cands:
        if not c.is_file():
            continue
        try:
            raw = json.loads(c.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(raw, dict):
            keep = {k: raw[k] for k in
                    ("type", "backbone", "feature_strategy", "input_dim", "output_dim",
                     "normalize") if k in raw}
            if keep:
                return {**keep, "source": c.name}
    return None


def _preflight_reid(reid_model: Optional[str], with_reid: bool) -> None:
    """Refuse an impossible appearance configuration before a single frame is processed.

    The pairing is decided on the *aggregated* width — the vector the head actually sees —
    so `cls` (768d for ViT-B) against a 1536d head is caught here rather than surfacing as
    a crash halfway through a video.
    """
    if not with_reid or not reid_model:
        return
    # A cached artifact that no longer exists (it was rebuilt under a different route, or
    # pruned) must say so here, not as a confusing failure inside the job thread.
    from reid_encoders import BUILTIN_PREFIX
    if not str(reid_model).startswith(BUILTIN_PREFIX) and not Path(reid_model).exists():
        raise HTTPException(
            status_code=400,
            detail=(f"所选的外观模型产物已不存在：{reid_model}。"
                    f"请重新选择 Backbone（必要时重新串接 Projection Head）后再处理。"))
    try:
        from reid_encoders import read_eager_recipe
        recipe = read_eager_recipe(reid_model)
    except Exception:
        return
    if not recipe or not recipe.get("head"):
        return
    agg, head_in = recipe.get("agg_dim"), recipe.get("head_in")
    if isinstance(agg, int) and isinstance(head_in, int) and agg != head_in:
        raise HTTPException(
            status_code=400,
            detail=(f"维度不兼容：当前特征聚合方式（{recipe.get('feature_strategy')}）输出 "
                    f"{agg} 维，而 Projection Head 需要输入 {head_in} 维，无法串接。"
                    f"请改选能让维度对上的聚合方式后再处理。"))


def _reid_verdict(with_reid: bool, tracker: str, reid_model: Optional[str]) -> Dict[str, Any]:
    """What appearance matching will actually do for this job, and why.

    A tracker that ignores appearance silently drops the encoder, so the enabled flag
    alone is misleading — the reason has to travel with it.
    """
    supported = _tracker_supports_reid(tracker)
    use = bool(with_reid) and supported
    if not use:
        reason = ("未启用外观模型（ReID）" if not with_reid else
                  f"{TRACKERS.get(_tracker_type_of(tracker) or '', {}).get('label', tracker)} "
                  f"不使用外观特征，请换用 BoT-SORT / Deep OC-SORT / TrackTrack")
        return {"reid_enabled": False, "reid_reason": reason}
    raw = (reid_model or "").strip()
    # "auto" never loads an encoder: ultralytics reuses the detector's own backbone
    # features. That is real appearance matching, so it must not be reported as
    # "enabled but unused" just because our encoder counter stays at zero.
    if not raw or raw == "auto":
        return {"reid_enabled": True, "reid_reason": "",
                "reid_label": "检测器原生特征（auto，不额外加载编码器）",
                "reid_native": True}
    label = Path(raw).name if not raw.startswith("builtin:") else raw
    recipe = None
    try:
        from reid_encoders import read_eager_recipe
        recipe = read_eager_recipe(raw or "auto")
    except Exception:
        recipe = None
    if recipe:
        bits = [b for b in (recipe.get("label"), recipe.get("feature_strategy")) if b]
        if recipe.get("out_dim"):
            bits.append(f"{recipe['out_dim']} 维")
        label = " · ".join(bits) or label
    return {"reid_enabled": True, "reid_reason": "", "reid_label": label}


def _release_gpu(session_id: Optional[str] = None) -> None:
    """Drop everything this session pinned on the GPU and return the memory.

    Sessions keep their tracker (and therefore the appearance encoder) alive until they
    are closed, so a job has to close its own session explicitly — otherwise the next job
    starts with several hundred MB less to work with.
    """
    if session_id:
        # Session keys are `track:<session_id>:...` / `predict:<path>`, so match the
        # middle segment rather than the head of the key.
        prefix = f"track:{session_id}:"
        with settings.model_manager.runtime_lock, _sessions_lock:
            for key in [k for k in _sessions if k.startswith(prefix)]:
                info = _sessions.pop(key)
                settings.model_manager.loaded_models.pop(info.get("model_id"), None)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _job_error_summary(exc: BaseException) -> str:
    """One actionable line for the UI; the full traceback goes to the log instead."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith(("RuntimeError", "ValueError", "TypeError", "OSError",
                          "KeyError", "AttributeError", "IndexError", "MemoryError")):
            return ln
    return lines[-1] if lines else type(exc).__name__


def _prepare_head_entry(path: Path) -> Dict[str, Any]:
    """Inspect a projection head: report the dims it needs so it can be paired."""
    from reid_encoders import inspect_weights

    info = inspect_weights(str(path))
    entry = {
        "kind": info.get("kind", "unknown"),
        "arch": info.get("arch"),
        "in_dim": info.get("in_dim"),
        "out_dim": info.get("out_dim"),
        "status": "ready",
        "message": info.get("message", ""),
    }
    if info.get("kind") != "head":
        entry.update(status="unsupported",
                     message=info.get("message") or "不是可识别的 Projection Head 权重")
        return entry
    meta = _head_metadata(path)
    if meta:
        entry["meta"] = meta
        strat = meta.get("feature_strategy")
        entry["message"] = (f"{entry['message']}；训练配置：{strat}" if strat
                            else entry["message"])
    return entry


def _prepare_tracker_entry(path: Path, declared_type: Optional[str]) -> Dict[str, Any]:
    """Inspect a tracker config: the algorithm type must be known and unambiguous.

    The resolved type is written back into the config, so the file on disk always
    declares which algorithm it implements and can be handed straight to ultralytics.
    """
    entry: Dict[str, Any] = {"kind": "tracker", "status": "ready"}
    if path.suffix.lower() not in (".yaml", ".yml"):
        entry.update(status="unsupported",
                     message="跟踪器需要 YAML 配置（.yaml/.yml）")
        return entry
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        entry.update(status="unsupported", message=f"无法解析 YAML：{e}")
        return entry
    if not isinstance(cfg, dict):
        entry.update(status="unsupported", message="跟踪器配置必须是一个映射（mapping）")
        return entry

    declared = (declared_type or "").strip().lower()
    if declared and declared not in TRACKERS:
        entry.update(status="unsupported",
                     message=f"未知的跟踪器类型 '{declared_type}'")
        return entry

    in_file = str(cfg.get("tracker_type", "")).strip().lower()
    ttype = declared or (in_file if in_file in TRACKERS else "")
    if not ttype:
        entry.update(status="needs_type",
                     message="配置里没有可识别的 tracker_type，导入时请指定跟踪器类型")
        return entry

    entry["tracker_type"] = ttype
    entry["arch"] = TRACKERS[ttype]["label"]
    keys = [k for k in cfg if k != "tracker_type"]
    extra = f"，含 {len(keys)} 项参数" if keys else ""
    entry["message"] = (f"自定义跟踪器配置（类型：{TRACKERS[ttype]['label']}{extra}）；"
                        f"选中后取代同名内置配置")
    if declared and in_file and declared != in_file:
        entry["message"] += f"（文件原写 {in_file}，已按导入时指定的 {declared} 处理）"
    # Persist the resolved type so the artifact is a self-describing tracker config.
    cfg["tracker_type"] = ttype
    entry["_normalized"] = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    return entry


def _prepare_reid_entry(path: Path, name: str, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect a feature-extraction backbone; export a TorchScript module when needed.

    This serves both the `backbone` zone and the legacy `reid` zone — they are the
    same role (produce an embedding for a crop). The pipeline feeds crops to this
    module and gets embeddings back, so anything ultralytics cannot consume
    directly is converted here.
    """
    from reid_encoders import inspect_weights, export_encoder

    dest_dir = Path(out_dir) if out_dir is not None else _reid_dir()
    info = inspect_weights(str(path))
    entry = {
        "kind": info.get("kind", "unknown"),
        "arch": info.get("arch"),
        "out_dim": info.get("out_dim"),
        "in_dim": info.get("in_dim"),
        # Whether multi-layer aggregation can be applied to this trunk: it needs a
        # raw transformer we rebuild, not an already-exported module.
        "aggregatable": info.get("kind") in ("dinov3", "vit"),
        "status": "ready" if info.get("loadable") else "needs_module",
        "message": info.get("message", ""),
    }
    if info.get("kind") == "head":
        # A head is not an encoder; route the user to the projection-head slot.
        entry.update(status="needs_encoder",
                     message=(info.get("message") or "") +
                             "；请放到「Projection Head」位置，与 Backbone 配套使用")
        return entry
    if info.get("kind") == "vit":
        # Width is known (so pairing can be checked) but a bare state dict cannot run.
        return entry
    if info.get("kind") == "dinov3":
        # A raw DINOv3 checkpoint: rebuild the official trunk around these weights
        # and export it, so any .pth/.safetensors the user has is directly usable.
        try:
            from reid_encoders import export_dinov3_encoder
            out, built = export_dinov3_encoder(str(path), str(dest_dir))
            entry["artifacts"] = [str(path), out]
            entry.update(path=out, status="ready",
                         out_dim=built.get("out_dim") or info.get("out_dim"),
                         # The artifact is a small recipe; the bytes that matter are the
                         # checkpoint it rebuilds from, so report that size.
                         size_mb=round(path.stat().st_size / 1e6, 2),
                         message=f"已按官方结构重建（推理走 eager PyTorch，"
                                 f"{Path(out).name}），"
                                 f"输出 {built.get('out_dim') or info.get('out_dim')} 维特征")
        except Exception as e:
            entry.update(status="error", message=f"DINOv3 转换失败：{e}")
        return entry
    if info.get("kind") in ("osnet", "unknown"):
        try:
            out = export_encoder(str(path), str(dest_dir), size=(256, 128))
            if out:
                entry["artifacts"] = [str(path), out]   # both belong to this entry
                entry.update(path=out, status="ready",
                             size_mb=round(Path(out).stat().st_size / 1e6, 2),
                             message=f"已转换为 TorchScript（{Path(out).name}），"
                                     f"输出 {info.get('out_dim')} 维特征")
            elif info.get("kind") == "unknown":
                entry.update(status="unsupported",
                             message=info.get("message") or "无法识别的权重格式")
        except Exception as e:
            entry.update(status="error", message=f"转换失败：{e}")
    elif info.get("kind") == "onnx":
        entry.update(status="ready", message=info.get("message") or "ONNX 模型")
    return entry


class ImportResponse(BaseModel):
    ok: bool
    entry: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/api/infer/models/import")
async def import_model(
    zone: str = Form(...),
    file: Optional[UploadFile] = File(None),
    path: Optional[str] = Form(None),
    tracker_type: Optional[str] = Form(None),
):
    """Import a file (upload) or register an existing path into a model zone.

    `tracker_type` is required for the `tracker` zone: a tracker config file cannot
    always declare which algorithm it implements, so the UI asks for it explicitly.
    """
    zone = (zone or "").strip().lower()
    if zone not in _IMPORT_ZONES:
        raise HTTPException(status_code=422,
                            detail=f"Unknown zone '{zone}'. Use: {', '.join(_IMPORT_ZONES)}")
    if not file and not (path and path.strip()):
        raise HTTPException(status_code=422, detail="Provide either a file upload or a path")

    from reid_encoders import ascii_safe_name  # local import: pulls in torch

    target_dir = _zone_dir(zone)
    if file:
        original = Path(file.filename or "model.bin").name
        stem = Path(original).stem
        suffix = Path(original).suffix.lower()
        # ASCII-only target name: torch/ultralytics C++ loaders mangle non-ASCII paths.
        safe = ascii_safe_name(stem, suffix)
        dest = target_dir / safe
        n = 1
        while dest.exists():
            dest = target_dir / f"{Path(safe).stem}-{n}{suffix}"
            n += 1
        dest.write_bytes(await file.read())
        src_path, source = dest, "upload"
    else:
        src_path = Path(path.strip())
        if not src_path.exists():
            raise HTTPException(status_code=404, detail=f"路径不存在：{src_path}")
        original, source = src_path.name, "path"

    entry: Dict[str, Any] = {
        "id": str(uuid.uuid4())[:8],
        "zone": zone,
        "name": original,
        "display_name": original,
        "path": str(src_path),
        "size_mb": round(src_path.stat().st_size / 1e6, 2),
        "source": source,
        "added_at": time.time(),
        "official": False,
        "status": "ready",
    }

    if zone in ("backbone", "reid"):
        # Same role (feature extractor); `reid` is kept for entries imported before
        # the backbone zone existed. Artifacts land in the zone's own directory.
        entry.update(_prepare_reid_entry(src_path, original, out_dir=target_dir))
    elif zone == "head":
        entry.update(_prepare_head_entry(src_path))
    elif zone == "tracker":
        entry.update(_prepare_tracker_entry(src_path, tracker_type))
    elif zone == "detector" and src_path.suffix.lower() == ".pth":
        from rfdetr_adapter import metadata
        try:
            meta = metadata(src_path)
            entry.update(kind="rfdetr", arch=meta["architecture"])
        except Exception as exc:
            raise HTTPException(422, str(exc))
    else:  # detector / other
        entry.update(kind="unknown", arch=None)

    entry.setdefault("artifacts", [str(src_path)])

    # A tracker config is only useful once it names its algorithm. Uploads are already
    # ours so the type is written in place; an external file is copied into the tracker
    # dir so the user's own file is never modified.
    if zone == "tracker" and entry.get("_normalized") and entry.get("status") == "ready":
        payload = entry.pop("_normalized")
        if source == "upload":
            src_path.write_text(payload, encoding="utf-8")
        else:
            dest = _zone_dir("tracker") / ascii_safe_name(src_path.stem, ".yaml")
            dest.write_text(payload, encoding="utf-8")
            entry["artifacts"] = [str(src_path), str(dest)]
            entry["path"] = str(dest)
        entry["size_mb"] = round(Path(entry["path"]).stat().st_size / 1e6, 2)
    entry.pop("_normalized", None)

    # Re-importing the same file replaces the old entry and drops its generated copies
    # (never touching files outside workspace/models, e.g. user-registered paths). A
    # failed import must not destroy a previously working artifact, so only a ready
    # entry is allowed to clean up its predecessor's files.
    entries = _load_registry()
    if entry["status"] == "ready":
        for old in entries:
            if old.get("zone") == zone and old.get("name") == original:
                for art in (old.get("artifacts") or [old.get("path")]):
                    if not art:
                        continue
                    ap = Path(art)
                    if ap.exists() and ap != src_path and _inside_models(ap):
                        try:
                            ap.unlink()
                        except OSError:
                            pass
    # An upload whose target name is taken lands on `-1`, `-2`, … (see the collision loop
    # above). Those copies are only ever produced here and only the newest one is
    # referenced, so a successful import sweeps its own numbered siblings as well —
    # otherwise every re-import would leave another stray entry in the zone listing.
    if entry["status"] == "ready" and source == "upload":
        stem, suffix = src_path.stem, src_path.suffix
        head, _, tail = stem.rpartition("-")
        base = head if (head and tail.isdigit()) else stem
        for sib in src_path.parent.glob(f"{base}*{suffix}"):
            if sib != src_path and sib.is_file() and _inside_models(sib):
                try:
                    sib.unlink()
                except OSError:
                    pass
    entries = [e for e in entries
               if e.get("path") != entry["path"]
               and not (e.get("zone") == zone and e.get("name") == original)]
    entries.append(entry)
    _save_registry(entries)
    return {"ok": True, "entry": entry, "error": None}


@router.delete("/api/infer/models/{entry_id}")
async def remove_imported_model(entry_id: str):
    entries = _load_registry()
    keep, removed = [], None
    for e in entries:
        if e.get("id") == entry_id:
            removed = e
        else:
            keep.append(e)
    if removed is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    _save_registry(keep)
    return {"ok": True, "removed": entry_id}


@router.get("/api/infer/models/catalog")
async def models_catalog():
    """Everything the inference panel's model zones need, in one round trip."""
    # ── detector: existing YOLO models (training runs + workspace) ──
    detectors: List[Dict[str, Any]] = []
    for job in settings.training_manager.list_training_jobs():
        mp = job.get("model_path")
        if mp and Path(mp).exists() and _is_plain_yolo(Path(mp).name):
            detectors.append({"id": job["id"], "name": job["name"], "path": mp,
                              "kind": "yolo", "status": "ready", "source": "training",
                              "size_mb": round(Path(mp).stat().st_size / 1e6, 2)})
    for pt in sorted(Path(settings.MODELS_DIR).glob("*.pt")):
        low = pt.name.lower()
        # ReID encoders are also .pt files — keep them out of the detector list.
        if any(s in low for s in ("reid", "-cls", "osnet")):
            continue
        if _is_plain_yolo(pt.name):
            detectors.append({"id": pt.stem, "name": pt.name, "path": str(pt),
                              "kind": "yolo", "status": "ready", "source": "workspace",
                              "size_mb": round(pt.stat().st_size / 1e6, 2)})

    from rfdetr_adapter import metadata as rf_metadata
    for checkpoint in sorted(Path(settings.MODELS_DIR).glob("*.pth")):
        try:
            meta = rf_metadata(checkpoint)
        except Exception:
            continue
        detectors.append({"id": checkpoint.stem, "name": checkpoint.name, "path": str(checkpoint),
                          "kind": "rfdetr", "arch": meta["architecture"], "status": "ready",
                          "source": "workspace", "size_mb": round(checkpoint.stat().st_size / 1e6, 2)})

    # ── reid: official assets (downloaded) ──
    reid: List[Dict[str, Any]] = []
    reid_dir = _reid_dir()
    for spec in _REID_OFFICIAL:
        target = reid_dir / spec["name"]
        if target.exists() and target.stat().st_size > 1e5:
            reid.append({"id": f"official:{spec['name']}", "name": spec["name"],
                         "path": str(target), "kind": spec["kind"], "official": True,
                         "status": "ready", "size_mb": round(target.stat().st_size / 1e6, 2)})

    # ── split registry entries into their zones ──
    trackers: List[Dict[str, Any]] = []
    backbones: List[Dict[str, Any]] = []
    heads: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    entries = _load_registry()
    migrated = False
    for raw in entries:
        e = raw
        if not Path(e.get("path", "")).exists():
            e = dict(e, status="missing", message="文件已不存在")
        z = e.get("zone")
        if z == "head" and e.get("status") == "needs_backbone":
            # Legacy status from before projection heads could be paired with an encoder.
            try:
                fresh = _prepare_head_entry(Path(e["path"])) if Path(e.get("path", "")).exists() else {}
                raw.update(fresh)
                e = raw
                migrated = True
            except Exception:
                pass
        if z == "detector":
            detectors.append(e)
        elif z == "backbone":
            backbones.append(e)
        elif z == "reid":
            reid.append(e)
            # Entries imported before the backbone zone existed are encoders by role.
            # Head weights still belong to the head list; the encoders are mirrored into
            # the backbone list after the loop, where the same model registered in both
            # zones can be collapsed (see below).
            if e.get("kind") == "head":
                heads.append(e)
        elif z == "head":
            heads.append(e)
        elif z == "tracker":
            trackers.append(e)
        else:
            others.append(e)
    if migrated:
        _save_registry(entries)

    # Legacy `reid`-zone encoders stay reachable from the backbone list, but the same
    # weights registered in both zones (an import made before and after the backbone zone
    # existed) must not appear twice in the same dropdown. Only an exact name + kind +
    # size match is treated as the same model, so a genuinely different file still shows.
    carried = {(b.get("name"), b.get("kind"), b.get("size_mb")) for b in backbones}
    for e in reid:
        if e.get("kind") == "head":
            continue
        key = (e.get("name"), e.get("kind"), e.get("size_mb"))
        if key in carried:
            continue
        carried.add(key)
        backbones.append(e)

    # The same weights can be reachable both as a training run and as a registered
    # path; list each physical file once.
    seen_paths: set = set()
    deduped: List[Dict[str, Any]] = []
    for d in detectors:
        try:
            key = str(Path(d.get("path", "")).resolve())
        except OSError:
            key = d.get("path", "")
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped.append(d)
    detectors = deduped

    # local ReID files in the reid dir that are not already accounted for
    known = {e.get("path") for e in reid}
    for e in _load_registry():
        known.update(e.get("artifacts") or [])   # original + converted copies
    for p in sorted(reid_dir.glob("*")):
        if not p.is_file() or p.suffix.lower() in (".json", ".part") or str(p) in known:
            continue
        # Inspect rather than assume: a file dropped here can be a head, a classifier or
        # an encoder, and only the real kind lets the UI route it to the right slot
        # (a head must never be offered as a Backbone).
        try:
            from reid_encoders import inspect_weights
            bi = inspect_weights(str(p))
        except Exception:
            bi = {}
        reid.append({"id": f"local:{p.name}", "name": p.name, "path": str(p),
                     "kind": bi.get("kind") or ("onnx" if p.suffix.lower() == ".onnx"
                                                else "unknown"),
                     "arch": bi.get("arch"),
                     "out_dim": bi.get("out_dim"),
                     "aggregatable": bi.get("kind") in ("dinov3", "vit"),
                     "status": "ready" if bi.get("loadable") else "needs_module",
                     "message": bi.get("message", ""),
                     "size_mb": round(p.stat().st_size / 1e6, 2)})

    # Unregistered files dropped straight into the backbone dir. Anything whose
    # width we can determine is worth listing, so probe it rather than assume.
    bknown: set = set()
    for e in _load_registry():
        bknown.update([e.get("path")] + (e.get("artifacts") or []))
    for p in sorted(_zone_dir("backbone").glob("*")):
        if not p.is_file() or p.suffix.lower() in (".json", ".part") or str(p) in bknown:
            continue
        try:
            from reid_encoders import inspect_weights
            bi = inspect_weights(str(p))
        except Exception:
            bi = {}
        backbones.append({"id": f"local:{p.name}", "name": p.name, "path": str(p),
                          "kind": bi.get("kind", "unknown"), "arch": bi.get("arch"),
                          "out_dim": bi.get("out_dim"),
                          "aggregatable": bi.get("kind") in ("dinov3", "vit"),
                          "status": "ready" if bi.get("loadable") else "needs_module",
                          "message": bi.get("message", ""),
                          "size_mb": round(p.stat().st_size / 1e6, 2)})

    try:
        import importlib.util
        ort = importlib.util.find_spec("onnxruntime") is not None
    except Exception:
        ort = False

    official = []
    for spec in _REID_OFFICIAL:
        target = reid_dir / spec["name"]
        st = _REID_DL_STATUS.get(spec["name"], {})
        status = ("downloaded" if target.exists() and target.stat().st_size > 1e5
                  else st.get("status", "not_downloaded"))
        official.append({**spec, "path": str(target), "status": status,
                         "progress": st.get("progress", 0), "error": st.get("error")})

    # Built-in backbones: standard trunks offered directly, grouped so the UI can
    # present "OSNet" and "Backbone" as separate families the way the panel does.
    try:
        from reid_encoders import (BUILTIN_BACKBONES, BUILTIN_PREFIX,
                                   builtin_backbone_status)
        builtins = []
        for spec in BUILTIN_BACKBONES:
            status, why = builtin_backbone_status(spec)
            builtins.append({"value": BUILTIN_PREFIX + spec["id"], "label": spec["label"],
                             "family": spec["family"], "out_dim": spec["out_dim"],
                             "note": spec.get("note", ""),
                             "status": status, "message": why})
    except Exception as e:  # noqa: BLE001
        builtins = []

    # The aggregation strategies are declared once, server-side; the UI multiplies the
    # backbone width by `mult` so both sides always quote the same dimensions.
    strategies: List[Dict[str, Any]] = []
    supports: Dict[str, bool] = {}
    try:
        from reid_encoders import (DEFAULT_FEATURE_STRATEGY as _DEF,
                                   FEATURE_STRATEGIES as _STRATS)
        strategies = [{"value": s["id"], "label": s["label"], "mult": s["mult"],
                       "note": s["note"]} for s in _STRATS]
        for spec in BUILTIN_BACKBONES:
            # Only transformer trunks expose the intermediate-layer read-out.
            # (DINOv3 ships as a local file, so `vit` is the marker, not `hub`.)
            supports[BUILTIN_PREFIX + spec["id"]] = bool(spec.get("vit")) \
                or spec.get("source") == "hub"
    except Exception:  # noqa: BLE001
        _DEF = "cls"

    return {"onnxruntime": ort, "official": official,
            "trackers": [{"value": k, "label": v["label"], "reid": v["reid"]}
                         for k, v in TRACKERS.items()],
            "backbones": builtins,
            "feature_strategies": {"default": _DEF, "options": strategies,
                                   "supports": supports},
            "zones": {"detector": detectors, "tracker": trackers,
                      "backbone": backbones, "reid": reid,
                      "head": heads, "other": others}}


class ComposeRequest(BaseModel):
    encoder: str                  # path of a `backbone`/`reid` entry, or `builtin:<id>`
    head: Optional[str] = None    # path of a `head` entry
    feature_strategy: Optional[str] = None   # see reid_encoders.FEATURE_STRATEGIES


def _original_checkpoint(enc_path: Path) -> Optional[Path]:
    """The state dict a composed artifact was built from, if we still have it.

    Importing a DINOv3 `.pth` converts it to TorchScript right away, and a finished
    module cannot be reopened into raw transformer blocks. Multi-layer strategies need
    those blocks, so fall back to the original checkpoint recorded in `artifacts`.
    """
    try:
        target = str(enc_path.resolve())
    except OSError:
        target = str(enc_path)
    from reid_encoders import _TS_SUFFIXES, inspect_weights
    for e in _load_registry():
        arts = [str(a) for a in (e.get("artifacts") or [])] + [str(e.get("path") or "")]
        if not any(a and Path(a).exists() and str(Path(a).resolve()) == target for a in arts):
            continue
        for a in arts:
            if not a:
                continue
            p = Path(a)
            if not p.is_file() or p.suffix.lower() in _TS_SUFFIXES:
                continue
            # Only a raw trunk we can rebuild (DINOv3 today) supports layer reads.
            if inspect_weights(str(p)).get("kind") in ("dinov3", "vit"):
                return p
    return None


@router.post("/api/infer/models/compose")
async def compose_reid(req: ComposeRequest):
    """Fold a backbone + projection head into one loadable embedding module.

    The pair must agree on the dimension where they meet; the composed artifact is
    cached and can be handed to the tracker as its appearance model. `encoder` may
    name a built-in backbone (`builtin:resnet50`) instead of a file.
    """
    from reid_encoders import (BUILTIN_PREFIX, DEFAULT_FEATURE_STRATEGY,
                               builtin_backbone_spec, export_builtin_encoder,
                               export_dinov3_encoder, export_encoder, inspect_weights)

    enc_raw = (req.encoder or "").strip()
    strategy = (req.feature_strategy or DEFAULT_FEATURE_STRATEGY).strip() or DEFAULT_FEATURE_STRATEGY

    head = None
    if req.head and req.head.strip():
        head = Path(req.head.strip())
        if not head.is_file():
            raise HTTPException(status_code=404, detail=f"Projection Head 不存在：{head}")

    if enc_raw.startswith(BUILTIN_PREFIX) or builtin_backbone_spec(enc_raw):
        spec = builtin_backbone_spec(enc_raw)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"未知的 Backbone：{enc_raw}")
        try:
            out, built = export_builtin_encoder(spec["id"], str(_composed_dir()),
                                                size=(256, 128),
                                                head_ckpt=str(head) if head else None,
                                                feature_strategy=strategy)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not out:
            raise HTTPException(status_code=400, detail="该 Backbone 无法转换为可加载的编码器")
        dim = (inspect_weights(str(head)).get("out_dim") if head else built.get("out_dim"))
        return {"ok": True, "path": out, "out_dim": dim,
                "message": (f"{spec['label']} + Projection Head 已串接，输出 {dim} 维" if head
                            else f"{spec['label']} 已就绪，输出 {dim} 维"),
                "head": bool(head),
                "strategy": built.get("strategy", strategy),
                "backbone_dim": built.get("backbone_dim"),
                "agg_dim": built.get("agg_dim")}

    enc = Path(enc_raw)
    if not enc.is_file():
        raise HTTPException(status_code=404, detail=f"编码器不存在：{enc}")

    # Anything past the plain read-out needs the trunk's own blocks, so rebuild from
    # the checkpoint the artifact came from instead of wrapping the finished module.
    agg_info: Dict[str, Any] = {}
    if strategy != DEFAULT_FEATURE_STRATEGY:
        src = _original_checkpoint(enc)
        if src is None:
            raise HTTPException(
                status_code=400,
                detail=("无法对该 Backbone 做多层特征聚合：它已经是导出好的模块，"
                        "找不到可重建的原始权重，请重新导入原始 .pth 后再选择聚合方式"))
        try:
            out, agg_info = export_dinov3_encoder(str(src), str(_composed_dir()),
                                                  size=(256, 128),
                                                  head_ckpt=str(head) if head else None,
                                                  feature_strategy=strategy)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        dim = agg_info.get("out_dim")
        if not out:
            raise HTTPException(status_code=400, detail="该编码器无法转换为可加载的编码器")
        return {"ok": True, "path": out, "out_dim": dim,
                "message": (f"特征链路 {agg_info.get('backbone_dim')}d → "
                            f"{agg_info.get('agg_dim')}d → {dim}d"
                            if head else
                            f"特征链路 {agg_info.get('backbone_dim')}d → "
                            f"{agg_info.get('agg_dim')}d"),
                "head": bool(head),
                "strategy": agg_info.get("strategy", strategy),
                "backbone_dim": agg_info.get("backbone_dim"),
                "agg_dim": agg_info.get("agg_dim")}

    enc_info = inspect_weights(str(enc))
    # A raw transformer checkpoint goes the same eager route even with the default
    # read-out: tracing it is what breaks it, not the aggregation choice.
    if enc_info.get("kind") in ("dinov3", "vit"):
        from reid_encoders import build_dinov3, export_eager_encoder
        try:
            net = build_dinov3(str(enc))
            stem = Path(enc).stem + f"-{strategy}" + (f"+{head.stem}" if head else "")
            out, agg_info = export_eager_encoder(net, stem, str(_composed_dir()),
                                                 size=(256, 128),
                                                 head_ckpt=str(head) if head else None,
                                                 feature_strategy=strategy,
                                                 label="DINOv3", backbone=str(enc))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        dim = agg_info.get("out_dim")
        return {"ok": True, "path": out, "out_dim": dim,
                "message": (f"特征链路 {agg_info.get('backbone_dim')}d → "
                            f"{agg_info.get('agg_dim')}d → {dim}d" if head else
                            f"特征链路 {agg_info.get('backbone_dim')}d → "
                            f"{agg_info.get('agg_dim')}d"),
                "head": bool(head),
                "strategy": agg_info.get("strategy", strategy),
                "backbone_dim": agg_info.get("backbone_dim"),
                "agg_dim": agg_info.get("agg_dim")}

    try:
        if head is None:
            out = export_encoder(str(enc), str(_composed_dir()), size=(256, 128), tag="-enc")
            dim = enc_info.get("out_dim")
        else:
            head_info = inspect_weights(str(head))
            out = export_encoder(str(enc), str(_composed_dir()), size=(256, 128),
                                 head_ckpt=str(head), tag="-chained")
            dim = head_info.get("out_dim")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not out:
        raise HTTPException(status_code=400, detail="该编码器无法转换为可加载的 TorchScript")
    return {"ok": True, "path": out, "out_dim": dim,
            "message": (f"编码器 + Projection Head 已串接，输出 {dim} 维" if head
                        else f"编码器已就绪，输出 {dim} 维"),
            "head": bool(head),
            "strategy": strategy,
            "backbone_dim": enc_info.get("out_dim"),
            "agg_dim": enc_info.get("out_dim")}


# ── List YOLO models (training runs + workspace .pt files) ─────────────────────

@router.get("/api/infer/models")
async def list_infer_models():
    models = []
    for job in settings.training_manager.list_training_jobs():
        mp = job.get("model_path")
        if mp and Path(mp).exists() and _is_plain_yolo(Path(mp).name):
            models.append({
                "id": job["id"],
                "name": job["name"],
                "path": mp,
                "model_type": job["model_type"],
                "status": job["status"],
                "started_at": job["started_at"],
            })
    for pt in sorted(settings.MODELS_DIR.glob("*.pt")):
        if _is_plain_yolo(pt.name):
            models.append({
                "id": pt.stem,
                "name": pt.name,
                "path": str(pt),
                "model_type": "unknown",
                "status": "available",
                "started_at": "",
            })
    return {"models": models}


# ── Smart-model inference (SAM / YOLOWorld / GroundingDINO / OWL-ViT / …) ──────

class SmartFrameRequest(BaseModel):
    model_id: str
    frame_b64: str
    confidence: float = 0.5
    text_prompt: Optional[str] = None
    point_x: Optional[float] = None
    point_y: Optional[float] = None


@router.get("/api/infer/smart/models")
async def list_smart_models():
    """All loaded smart models ready for inference."""
    out = []
    for model_id, info in settings.model_manager.loaded_models.items():
        if info.get("model") is None:
            continue
        mtype = info.get("type", "unknown")
        out.append({
            "id":             model_id,
            "name":           info.get("name", model_id),
            "type":           mtype,
            "supports_text":  mtype in _TEXT_PROMPT_TYPES,
            "supports_point": mtype in _POINT_PROMPT_TYPES,
        })
    return {"models": out}


def _smart_infer_sync(model_id: str, tmp: Path, confidence: float,
                      text_prompt: Optional[str], point_x: Optional[float],
                      point_y: Optional[float]):
    """Blocking inference using model_manager. Returns (img_bytes, detections)."""
    model_info = settings.model_manager.loaded_models.get(model_id)
    if not model_info or model_info.get("model") is None:
        raise ValueError(f"Model '{model_id}' not loaded. Load it in the Models tab first.")
    model = model_info["model"]
    model_type = model_info.get("type", "unknown")
    prompt_point = (point_x, point_y) if point_x is not None and point_y is not None else None
    with settings.model_manager.runtime_lock:
        annotations = settings.model_manager._run_inference(
            model, model_type, tmp, confidence, prompt_point, text_prompt or None
        )
    return _render_annotations(tmp, annotations)


@router.post("/api/infer/smart/image")
async def infer_smart_image(
    model_id: str = Form(...),
    confidence: float = Form(0.5),
    text_prompt: Optional[str] = Form(None),
    point_x: Optional[float] = Form(None),
    point_y: Optional[float] = Form(None),
    image: UploadFile = File(...),
):
    save_dir = settings.TEMP_DIR / "inference"
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "img.jpg").suffix or ".jpg"
    tmp = save_dir / f"{str(uuid.uuid4())[:8]}_smart{suffix}"
    tmp.write_bytes(await image.read())
    try:
        img_bytes, detections = await asyncio.to_thread(_smart_infer_sync,
            model_id, tmp, confidence, text_prompt, point_x, point_y
        )
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
        }
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/api/infer/smart/frame")
async def infer_smart_frame(req: SmartFrameRequest):
    try:
        import cv2, numpy as np
        img_data = base64.b64decode(req.frame_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame")

        save_dir = settings.TEMP_DIR / "inference"
        save_dir.mkdir(parents=True, exist_ok=True)
        tmp = save_dir / f"{str(uuid.uuid4())[:8]}_sframe.jpg"
        cv2.imwrite(str(tmp), frame)

        img_bytes, detections = await asyncio.to_thread(_smart_infer_sync,
            req.model_id, tmp, req.confidence,
            req.text_prompt, req.point_x, req.point_y
        )
        return {
            "annotated_b64": base64.b64encode(img_bytes).decode(),
            "detections": detections,
        }
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "tmp" in dir() and tmp.exists():
            tmp.unlink(missing_ok=True)


# ── Smart-model catalog + inline load/download ────────────────────────────────

# All smart models surfaced in the inference tab
_SMART_CATALOG: List[Dict[str, Any]] = [
    {"id": "sam_vit_b",           "name": "SAM Base",              "type": "sam"},
    {"id": "sam_vit_l",           "name": "SAM Large",             "type": "sam"},
    {"id": "sam2_tiny",           "name": "SAM 2 Tiny",            "type": "sam2"},
    {"id": "sam2_small",          "name": "SAM 2 Small",           "type": "sam2"},
    {"id": "sam2_base",           "name": "SAM 2 Base+",           "type": "sam2"},
    {"id": "sam2_large",          "name": "SAM 2 Large",           "type": "sam2"},
    {"id": "sam21_tiny",          "name": "SAM 2.1 Tiny",          "type": "sam2"},
    {"id": "sam21_small",         "name": "SAM 2.1 Small",         "type": "sam2"},
    {"id": "sam21_base",          "name": "SAM 2.1 Base+",         "type": "sam2"},
    {"id": "sam21_large",         "name": "SAM 2.1 Large",         "type": "sam2"},
    {"id": "sam3",                "name": "SAM 3",                  "type": "sam3"},
    {"id": "yoloworld_s",         "name": "YOLO-World S",          "type": "yoloworld"},
    {"id": "yoloworld_m",         "name": "YOLO-World M",          "type": "yoloworld"},
    {"id": "yoloworld_l",         "name": "YOLO-World L",          "type": "yoloworld"},
    {"id": "groundingdino_t",     "name": "GroundingDINO Tiny",    "type": "groundingdino"},
    {"id": "groundingdino_b",     "name": "GroundingDINO Base",    "type": "groundingdino"},
    {"id": "owlvit_base_patch32", "name": "OWL-ViT Base/32",       "type": "owlvit"},
    {"id": "owlvit_base_patch16", "name": "OWL-ViT Base/16",       "type": "owlvit"},
    {"id": "owlvit_large_patch14","name": "OWL-ViT Large/14",      "type": "owlvit"},
]


@router.get("/api/infer/smart/catalog")
async def smart_catalog():
    """Full smart-model catalog with live download/load status."""
    # Build a lookup from list_models() which already checks disk + HF cache
    all_models = {m["id"]: m for m in settings.model_manager.list_models()}

    out = []
    local_rf = [
        {"id": f.stem, "name": f.name, "type": "rfdetr"}
        for f in sorted(settings.MODELS_DIR.glob("*.pth"))
        if settings.model_manager._detect_model_type(f) == "rfdetr"
        and f.stem not in {e["id"] for e in _SMART_CATALOG}
    ]
    for entry in [*_SMART_CATALOG, *local_rf]:
        mid   = entry["id"]
        mtype = entry["type"]
        info  = settings.model_manager.loaded_models.get(mid, {})
        dl    = settings._download_status.get(mid, {})
        disk  = all_models.get(mid, {})

        if info.get("model") is not None:
            status = "loaded"
        elif dl.get("status") == "downloading":
            status = "downloading"
        elif disk.get("downloaded"):
            status = "downloaded"
        else:
            status = "not_downloaded"

        out.append({
            "id":               mid,
            "name":             entry["name"],
            "type":             mtype,
            "status":           status,
            "download_progress": dl.get("progress", 0),
            "error":            dl.get("error") or info.get("error"),
            "supports_text":    mtype in _TEXT_PROMPT_TYPES,
            "supports_point":   mtype in _POINT_PROMPT_TYPES,
        })
    return {"models": out}


@router.post("/api/infer/smart/load")
async def load_smart_model(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    model_type: str = Form(...),
    hf_token: Optional[str] = Form(None),
):
    """Download (if needed) and load a smart model into memory."""
    info = settings.model_manager.loaded_models.get(model_id, {})
    if info.get("model") is not None:
        return {"status": "already_loaded"}
    if settings._download_status.get(model_id, {}).get("status") == "downloading":
        return {"status": "already_loading"}

    settings._download_status[model_id] = {
        "status": "downloading", "progress": 5,
        "phase": "starting", "message": f"Loading {model_id}…",
    }

    def _do_load(mtype: str, mname: str, token: Optional[str]):
        try:
            def _phase(phase: str):
                if phase == "loading_memory":
                    settings._download_status[mname] = {
                        "status": "downloading", "progress": 80,
                        "phase": "loading_memory",
                        "message": "Loading into memory…",
                    }
            local_path = settings.MODELS_DIR / f"{mname}.pth"
            if (mtype == "rfdetr" and Path(mname).name == mname and local_path.is_file()
                    and local_path.resolve().parent == settings.MODELS_DIR.resolve()):
                _phase("loading_memory")
                with settings.model_manager.runtime_lock:
                    temporary_id = settings.model_manager.load_model(str(local_path), "rfdetr")
                    record = settings.model_manager.loaded_models.pop(temporary_id)
                    record["id"] = mname
                    settings.model_manager.loaded_models[mname] = record
            else:
                settings.model_manager.load_pretrained(mtype, mname, hf_token=token, _status_hook=_phase)
            loaded = settings.model_manager.loaded_models.get(mname, {})
            if loaded.get("error") or loaded.get("model") is None:
                settings._download_status[mname] = {
                    "status": "error", "progress": 0, "error": loaded["error"]
                }
            else:
                settings._download_status[mname] = {"status": "done", "progress": 100}
        except Exception as e:
            settings._download_status[mname] = {
                "status": "error", "progress": 0, "error": str(e)
            }

    background_tasks.add_task(_do_load, model_type, model_id, hf_token)
    return {"status": "started"}


@router.get("/api/infer/smart/load-status/{model_id}")
async def smart_load_status(model_id: str):
    return settings._download_status.get(model_id, {"status": "idle", "progress": 0})
