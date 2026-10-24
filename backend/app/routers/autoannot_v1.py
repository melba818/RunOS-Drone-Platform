# -*- coding: utf-8 -*-
"""Auto Label V1 的独立路由。

★独立命名空间 `/api/autoannot/v1/*`★ —— 与既有 `/api/auto-annotate/*`
（推理页批量标注）**完全不重叠**，互不干扰、互不迁移。

编排层在 `annotation_chain.run_chain_v1`，任务与留档在 `autoannot.jobs`，
本文件只做 HTTP 层：参数校验 + 转发 + 读取产物。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from autoannot import evidence as ev
from autoannot import jobs as autojobs
from config import settings

LOG = logging.getLogger(__name__)

router = APIRouter(tags=["autoannot-v1"])


def _resolve_dataset(dataset_id: str, dataset_path: str = "") -> Path:
    """数据集路径解析：**服务端算，不信前端传**。

    前端手里只有 `dataset_id`（没有 workspace 的绝对位置），所以路径一律由
    `settings.DATASETS_DIR / dataset_id` 推出；显式传 `dataset_path` 仅作兜底
    （且必须真的存在才认）。
    """
    ds = Path(dataset_path) if dataset_path else \
        Path(settings.DATASETS_DIR) / str(dataset_id)
    if not ds.exists():
        ds = Path(settings.DATASETS_DIR) / str(dataset_id)
    return ds


class CreateJobBody(BaseModel):
    dataset_id: str
    dataset_path: str = ""
    text_prompt: str = ""
    image_limit: int = 0
    params: Dict[str, Any] = Field(default_factory=dict)


class PreviewBody(BaseModel):
    dataset_id: str
    #: 数据集内的相对路径（`/api/datasets/{id}/images` 返回的 `path`）
    image_path: str = ""
    dataset_path: str = ""
    class_table: list = Field(default_factory=list)
    text_prompt: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


@router.post("/api/autoannot/v1/jobs")
async def create_job(body: CreateJobBody):
    ds = _resolve_dataset(body.dataset_id, body.dataset_path)
    if not ds.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")
    job = autojobs.create_job(body.dataset_id, str(ds),
                              body.text_prompt, body.params, body.image_limit)
    if job.get("status") == "failed":
        raise HTTPException(status_code=422, detail=job.get("error"))
    return job


@router.get("/api/autoannot/v1/jobs")
async def list_jobs(limit: int = Query(50, ge=1, le=500)):
    return {"jobs": autojobs.list_jobs(limit)}


@router.get("/api/autoannot/v1/jobs/{job_id}")
async def get_job(job_id: str):
    job = autojobs.load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@router.get("/api/autoannot/v1/jobs/{job_id}/log")
async def get_log(job_id: str, limit: int = Query(300, ge=1, le=5000)):
    jd = ev.job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"log": ev.read_log(jd, limit)}


@router.get("/api/autoannot/v1/jobs/{job_id}/raw")
async def list_raw(job_id: str):
    jd = ev.job_dir(job_id)
    if not jd.exists():
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"files": ev.list_raw(jd)}


@router.get("/api/autoannot/v1/jobs/{job_id}/raw/{name}")
async def read_raw(job_id: str, name: str):
    jd = ev.job_dir(job_id)
    data = ev.read_raw(jd, name)
    if data is None:
        raise HTTPException(status_code=404, detail="证据文件不存在")
    return data


@router.get("/api/autoannot/v1/jobs/{job_id}/review")
async def get_review(job_id: str, limit: int = Query(500, ge=1, le=10000)):
    jd = ev.job_dir(job_id)
    fp = jd / "review.jsonl"
    if not fp.exists():
        return {"items": []}
    import json
    out = []
    for ln in fp.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return {"items": out}


@router.post("/api/autoannot/v1/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    if not autojobs.set_pause(job_id, True):
        raise HTTPException(status_code=409, detail="任务已结束或不存在")
    return {"ok": True}


@router.post("/api/autoannot/v1/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    if not autojobs.set_pause(job_id, False):
        raise HTTPException(status_code=409, detail="任务已结束或不存在")
    return {"ok": True}


@router.post("/api/autoannot/v1/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    if not autojobs.request_stop(job_id):
        raise HTTPException(status_code=409, detail="任务已结束或不存在")
    return {"ok": True}


@router.post("/api/autoannot/v1/jobs/{job_id}/apply")
async def apply_job(job_id: str):
    return autojobs.apply_to_dataset(job_id)


@router.delete("/api/autoannot/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    autojobs.delete_job(job_id)
    return {"ok": True}


@router.get("/api/autoannot/v1/dataset-classes/{dataset_id}")
async def dataset_classes(dataset_id: str):
    """只读：数据集的类别表 —— 前端开跑前用它确认"不会凭空造类别"。"""
    ds = _resolve_dataset(dataset_id)
    if not ds.exists():
        raise HTTPException(status_code=404, detail="数据集不存在")
    classes = autojobs.read_class_table(ds)
    return {"classes": classes,
            "images": len(ev.list_images(ds)),
            "empty": not classes}


@router.post("/api/autoannot/v1/preview")
async def preview_single(body: PreviewBody):
    """单图试跑（不建任务、不落盘）—— 用来快速验证 prompt 与阈值。

    返回完整 debug，含 ACCEPT/REVIEW 与证据，方便前端直接渲染对比。
    """
    from annotation_chain import run_chain_v1

    ds = _resolve_dataset(body.dataset_id, body.dataset_path)
    rel = str(body.image_path or "").lstrip("/\\")
    p = (ds / rel) if rel else None
    if p is None or not p.exists():
        #: 兜底：按文件名在数据集里找
        hits = [q for q in ds.rglob(Path(rel).name)
                ] if rel else []
        if not hits:
            raise HTTPException(status_code=404, detail="图片不存在: %s" % rel)
        p = hits[0]

    classes = [str(c) for c in (body.class_table or [])]
    if not classes:
        classes = autojobs.read_class_table(ds)
    if not classes:
        raise HTTPException(status_code=422, detail="类别表为空，拒绝执行")
    try:
        annotations, debug = run_chain_v1(p, body.text_prompt, classes,
                                          **(body.params or {}))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"annotations": annotations, "debug": debug}
