# -*- coding: utf-8 -*-
"""★独立任务体系★：Auto Label V1 的 job / queue / status / 输出目录。

================================================================================
与既有批量任务的边界
================================================================================
既有 `routers/annotations.py` 的 `_text_annotate_jobs` 是 `settings` 上的
**全局共享字典**，服务于推理页的批量标注。Auto Label V1 **不碰它**：

    |               | 既有 text-batch          | Auto Label V1            |
    |---------------|--------------------------|--------------------------|
    | 状态存放      | settings._text_annotate_jobs | workspace/autoannot/<id>/job.json |
    | 日志          | print                    | log.jsonl（结构化）      |
    | 输出          | 直接写回数据集 labels/   | 先写 job 自己的 labels/，显式 apply 才落盘 |
    | 证据          | 无                       | raw/*.json 全量留档      |

这样自动标注跑崩了、跑错了，都不会动到实时链路和数据集。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import evidence as ev

LOG = logging.getLogger(__name__)

#: 任务控制量。★模块级★，不进 settings —— 与实时链路彻底分开。
_CONTROLS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

TERMINAL = ("done", "cancelled", "failed")


def _now() -> float:
    return time.time()


def _job_file(job_id: str) -> Path:
    return ev.job_dir(job_id) / "job.json"


def _index_file() -> Path:
    return ev.root_dir() / "index.json"


def _read_index() -> List[Dict[str, Any]]:
    fp = _index_file()
    if not fp.exists():
        return []
    try:
        return json.loads(fp.read_text(encoding="utf-8")) or []
    except ValueError:
        return []


def _write_index(items: List[Dict[str, Any]]) -> None:
    _index_file().write_text(json.dumps(items, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    fp = _job_file(job_id)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _save_job(job: Dict[str, Any]) -> None:
    fp = _job_file(job["job_id"])
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    #: 索引里同步一份摘要
    with _LOCK:
        items = _read_index()
        items = [x for x in items if x.get("job_id") != job["job_id"]]
        items.append({
            "job_id": job["job_id"], "dataset_id": job.get("dataset_id"),
            "status": job.get("status"), "created_at": job.get("created_at"),
            "finished_at": job.get("finished_at"),
            "progress": job.get("progress"),
            "text_prompt": job.get("text_prompt", ""),
        })
        items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
        _write_index(items)


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    return _read_index()[: int(limit)]


# ── 类别表 ─────────────────────────────────────────────────────────────────
def read_class_table(dataset_path: Path) -> List[str]:
    """从 `data.yaml` 的 `names` 读类别表（COCO 走 `categories`）。

    ★与 `annotation_chain` 的硬约束一致★：类别表**只**取数据集已有的，
    自动标注绝不往里加新类别。读不到就返回空列表 —— 上层必须报错，
    不能静默产出 0 框。
    """
    root = Path(dataset_path)
    try:
        import yaml  # type: ignore
    except Exception:  # noqa: BLE001
        yaml = None  # type: ignore

    if yaml is not None:
        for yf in list(root.glob("*.yaml")) + list(root.glob("*.yml")):
            try:
                cfg = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
                names = cfg.get("names")
                if isinstance(names, dict):
                    return [str(v) for v in names.values()]
                if isinstance(names, list):
                    return [str(v) for v in names]
            except Exception:  # noqa: BLE001
                continue

    for jf in list(root.glob("*.json")) + list(root.glob("annotations/*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            cats = data.get("categories")
            if isinstance(cats, list):
                return [str(c.get("name")) for c in cats if c.get("name")]
        except Exception:  # noqa: BLE001
            continue
    return []


# ── 创建 / 控制 ────────────────────────────────────────────────────────────
def create_job(dataset_id: str, dataset_path: str, text_prompt: str,
               params: Optional[Dict[str, Any]] = None,
               image_limit: int = 0) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    jd = ev.job_dir(job_id)
    ev.ensure_job_dirs(jd)

    classes = read_class_table(Path(dataset_path))
    images = ev.list_images(Path(dataset_path))
    if image_limit and image_limit > 0:
        images = images[: int(image_limit)]

    job: Dict[str, Any] = {
        "job_id": job_id,
        "dataset_id": str(dataset_id),
        "dataset_path": str(dataset_path),
        "text_prompt": str(text_prompt or ""),
        "status": "pending",
        "params": dict(params or {}),
        "class_table": classes,
        "output_dir": str(jd),
        "progress": {"total": len(images), "processed": 0, "accept": 0,
                     "review": 0, "failed": 0, "annotations": 0},
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    _save_job(job)
    ev.append_log(jd, "info", "任务已创建", total=len(images), classes=len(classes))
    if not classes:
        ev.append_log(jd, "error", "类别表为空 —— 拒绝启动（禁止自动扩张类别）")
        job["status"] = "failed"
        job["error"] = "数据集中读不到类别表（data.yaml names），请先配置类别"
        _save_job(job)
        return job

    with _LOCK:
        _CONTROLS[job_id] = {"pause": threading.Event(), "stop": threading.Event()}
        _CONTROLS[job_id]["pause"].set()

    t = threading.Thread(target=_run_job, args=(job_id, images),
                         name="autolabel-%s" % job_id, daemon=True)
    t.start()
    return job


def _wait_if_paused(job_id: str) -> bool:
    """暂停则阻塞；要求停止返回 True。"""
    ctl = _CONTROLS.get(job_id)
    if not ctl:
        return True
    if ctl["stop"].is_set():
        return True
    ctl["pause"].wait()          #: pause event 被 clear 就在这等着
    return ctl["stop"].is_set()


def set_pause(job_id: str, paused: bool) -> bool:
    ctl = _CONTROLS.get(job_id)
    if not ctl:
        return False
    if paused:
        ctl["pause"].clear()
    else:
        ctl["pause"].set()
    return True


def request_stop(job_id: str) -> bool:
    ctl = _CONTROLS.get(job_id)
    if not ctl:
        return False
    ctl["stop"].set()
    ctl["pause"].set()           #: 先放开暂停，让它走到停止检查
    return True


# ── 工作线程 ───────────────────────────────────────────────────────────────
def _run_job(job_id: str, images: List[Path]) -> None:
    from annotation_chain import run_chain_v1

    job = load_job(job_id)
    if job is None:
        return
    jd = ev.job_dir(job_id)
    job["status"] = "running"
    job["started_at"] = _now()
    _save_job(job)
    ev.append_log(jd, "info", "任务开始", images=len(images))

    params = dict(job.get("params") or {})
    classes = list(job.get("class_table") or [])
    ds_path = Path(job["dataset_path"])

    for idx, img_path in enumerate(images):
        if _wait_if_paused(job_id):
            break

        key = str(img_path.relative_to(ds_path)) if _is_relative(img_path, ds_path) \
            else img_path.name
        t0 = time.time()
        try:
            annotations, debug = run_chain_v1(
                img_path, job.get("text_prompt") or "", classes, **params)
        except Exception as exc:  # noqa: BLE001
            job["progress"]["failed"] += 1
            ev.append_log(jd, "error", "图片失败: %s: %s" % (type(exc).__name__, exc),
                          image=key)
            _save_job(job)
            continue

        n_acc = sum(1 for a in annotations if a.get("decision") == "ACCEPT")
        n_rev = len(annotations) - n_acc
        job["progress"]["accept"] += n_acc
        job["progress"]["review"] += n_rev
        job["progress"]["annotations"] += len(annotations)
        job["progress"]["processed"] += 1

        #: ★第三原则★：原始结果全量落盘
        ev.write_raw(jd, key, {
            "image": key,
            "image_abs": str(img_path),
            "text_prompt": job.get("text_prompt") or "",
            "class_table": classes,
            "params": params,
            "vlx_raw": debug.get("vlx") or {},
            "qwen_raw": debug.get("qwen") or {},
            "qwen_verify_vlx": debug.get("qwen_verify_vlx") or {},
            "vlx_verify_qwen": debug.get("vlx_verify_qwen") or {},
            "candidates_before_merge": debug.get("candidates_detail") or [],
            "merged_after": debug.get("merged_detail") or [],
            "class_conflicts": debug.get("class_conflicts") or 0,
            "annotations": annotations,
            "decision_summary": debug.get("decision") or {},
            "elapsed_ms": debug.get("elapsed_ms"),
            "detect_parallel_ms": debug.get("detect_parallel_ms"),
        })

        #: ACCEPT → YOLO（★只写 job 自己的目录★）
        w, h = (debug.get("image_size") or [0, 0])[:2]
        lines = ev.to_yolo_lines(annotations, int(w or 0), int(h or 0),
                                 only_accept=True)
        ev.write_labels(jd, key, lines)

        #: REVIEW 进人工复核清单
        if n_rev:
            ev.write_review(jd, [{
                "image": key,
                "items": [{"class_name": a.get("class_name"),
                           "bbox": a.get("bbox"),
                           "decision_reasons": a.get("decision_reasons"),
                           "evidence": a.get("evidence")}
                          for a in annotations if a.get("decision") != "ACCEPT"],
            }])

        ev.append_log(jd, "info", "完成 %d/%d" % (idx + 1, len(images)),
                      image=key, accept=n_acc, review=n_rev,
                      ms=round((time.time() - t0) * 1000.0, 1))
        if idx % 5 == 0 or idx + 1 == len(images):
            _save_job(job)

    ctl = _CONTROLS.get(job_id)
    stopped = bool(ctl and ctl["stop"].is_set())
    job["status"] = "cancelled" if stopped else "done"
    job["finished_at"] = _now()
    _save_job(job)
    ev.append_log(jd, "info", "任务结束 status=%s" % job["status"],
                  **job["progress"])
    with _LOCK:
        _CONTROLS.pop(job_id, None)


def _is_relative(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


# ── 落盘到数据集（显式 apply）────────────────────────────────────────────────
def apply_to_dataset(job_id: str) -> Dict[str, Any]:
    """把 job 里 ACCEPT 的 YOLO 标签**显式**写回数据集。

    镜像规则与既有 `_save_annotations` 一致：
        images/x.jpg      → labels/x.txt
        train/images/x.jpg → train/labels/x.txt
    """
    job = load_job(job_id)
    if job is None:
        return {"ok": False, "error": "任务不存在"}
    jd = ev.job_dir(job_id)
    ds = Path(job["dataset_path"])
    written, skipped = 0, 0
    for txt in sorted((jd / "labels").glob("*.txt")):
        stem = txt.stem
        #: 在数据集里找同名图片，决定 labels 该放哪
        hits = [p for p in ds.rglob(stem + ".*")
                if p.suffix.lower() in ev.IMAGE_SUFFIXES]
        if not hits:
            skipped += 1
            continue
        img = hits[0]
        parts = list(img.relative_to(ds).parts)
        lab_parts = ["labels" if x.lower() in ("images", "imgs") else x
                     for x in parts[:-1]]
        out = ds.joinpath(*lab_parts) if lab_parts else ds / "labels"
        out = out / "labels" if not lab_parts else out
        out.mkdir(parents=True, exist_ok=True)
        (out / (stem + ".txt")).write_text(txt.read_text(encoding="utf-8"),
                                           encoding="utf-8")
        written += 1
    ev.append_log(jd, "info", "apply 到数据集", written=written, skipped=skipped)
    return {"ok": True, "written": written, "skipped": skipped}


def delete_job(job_id: str) -> bool:
    import shutil
    jd = ev.job_dir(job_id)
    if jd.exists():
        shutil.rmtree(jd, ignore_errors=True)
    items = [x for x in _read_index() if x.get("job_id") != job_id]
    _write_index(items)
    return True
