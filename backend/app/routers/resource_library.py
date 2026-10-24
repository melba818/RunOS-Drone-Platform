import asyncio, json, mimetypes, os, re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from config import settings
from resource_store import ResourceStore, EXTENSIONS
from core.file_access import owned_file_response
from media_upload import save_upload
from resource_analysis import smart_search, pair_change, mission_change

router = APIRouter(prefix="/api/assistant/resources")
store = ResourceStore(settings.WORKSPACE_DIR)
tasks = set()
slots = asyncio.Semaphore(2)


def owner(request):
    value = getattr(request.state, "admin_id", None)
    if type(value) is not int or value < 1:
        raise HTTPException(401, "请重新登录。")
    return value


class Folder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=180)
    parent: str = ""


@router.get("/catalog")
async def catalog(request: Request):
    owner(request)
    from vlm_annotator import load_providers

    return {
        "vision_models": [
            {"id": p.id, "name": p.name, "model": p.model}
            for p in load_providers().values()
        ],
        "algorithm_models": [
            {k: v for k, v in model.items() if k in {"id", "name", "type", "loaded", "downloaded", "remote"}}
            for model in settings.model_manager.list_models()
            if model.get("type") != "vlm"
        ],
        "file_limit_mb": 512,
        "extensions": {k: sorted(v) for k, v in EXTENSIONS.items()},
    }


@router.get("")
async def list_assets(
    request: Request,
    kind: str = "all",
    parent: str | None = None,
    q: str = "",
    page: int = Query(1, ge=1, le=10000),
    limit: int = Query(60, ge=1, le=500),
    route_id: str | None = None,
    mission_id: str | None = None,
):
    return store.list(
        owner(request), kind, parent, q, page, limit, route_id, mission_id
    )


@router.post("/folders")
async def folder(body: Folder, request: Request):
    return store.add(owner(request), body.name, "folder", body.parent)


@router.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    parent: str = Form(""),
    metadata: str = Form("{}"),
):
    oid = owner(request)
    store.parent(oid, parent)
    try:
        meta = json.loads(metadata)
    except ValueError:
        raise HTTPException(422, "资源关联信息无效。") from None
    if not isinstance(meta, dict) or len(metadata) > 4000:
        raise HTTPException(422, "资源关联信息过长。")
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    kind = next((k for k, v in EXTENSIONS.items() if suffix in v), None)
    if not kind:
        raise HTTPException(422, "文件类型不在资源库支持范围内。")
    path, size = await save_upload(file, store.files, set().union(*EXTENSIONS.values()))
    try:
        asset = store.add(oid, name, kind, parent, path.name, size)
        if meta:
            asset = store.update(oid, asset["id"], {"metadata": meta})
        return asset
    except BaseException:
        if "asset" in locals():
            with store.db() as c:
                c.execute(
                    "DELETE FROM assets WHERE id=? AND owner=?", (asset["id"], oid)
                )
        path.unlink(missing_ok=True)
        raise


@router.patch("/{ident}")
async def update(ident: str, body: dict, request: Request):
    return store.update(owner(request), ident, body)


@router.get("/{ident}/file")
async def asset_file(ident: str, request: Request, download: bool = False):
    row = store.get(owner(request), ident)
    return owned_file_response(
        store.path(row),
        filename=row["name"] if download else None,
        media_type=mimetypes.guess_type(row["name"])[0] or "application/octet-stream",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        },
    )


@router.get("/{ident}/thumbnail")
async def thumbnail(ident: str, request: Request):
    row = store.get(owner(request), ident)
    path = store.path(row)
    if row["kind"] not in {"image", "video"}:
        raise HTTPException(422, "此资源没有缩略图。")
    preview = store.previews / (ident + ".jpg")
    if not preview.exists():

        def make():
            from resource_analysis import frames

            _, image = next(frames(path, max_frames=1))
            image.thumbnail((400, 300))
            image.save(preview, format="JPEG", quality=85)

        try:
            await asyncio.to_thread(make)
        except Exception:
            raise HTTPException(422, "资源预览暂不可用。") from None
    return FileResponse(
        preview, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"}
    )


async def run_analysis(oid, job, body):
    async with slots:
        store.finish_job(oid, job["id"], "running")
        try:
            fn = {
                "smart_search": smart_search,
                "change_pair": pair_change,
                "change_mission": mission_change,
            }[job["kind"]]
            if job["kind"] == "smart_search":
                result = await asyncio.to_thread(
                    fn,
                    store,
                    oid,
                    body,
                    lambda partial: store.finish_job(
                        oid, job["id"], "running", partial
                    ),
                )
            else:
                result = await asyncio.to_thread(fn, store, oid, body)
            store.finish_job(oid, job["id"], "succeeded", result)
        except asyncio.CancelledError:
            store.finish_job(
                oid,
                job["id"],
                "interrupted",
                error="服务重启，分析已中断，请手动重试。",
            )
            raise
        except Exception as exc:
            message = (
                str(exc.detail)[:500]
                if isinstance(exc, HTTPException)
                else "分析未完成，请检查文件与视觉模型连接。"
            )
            store.finish_job(oid, job["id"], "failed", error=message)


@router.post("/analyses/{kind}")
async def analyse(kind: str, body: dict, request: Request):
    oid = owner(request)
    if kind not in {"smart_search", "change_pair", "change_mission"}:
        raise HTTPException(422, "分析类型无效。")
    if len(json.dumps(body)) > 16000:
        raise HTTPException(413, "分析参数过长。")
    allowed = (
        {
            "resource_ids",
            "query",
            "provider_id",
            "route_id",
            "mission_id",
            "interval_seconds",
        }
        if kind == "smart_search"
        else (
            {"before_id", "after_id", "assume_aligned", "sensitivity", "provider_id"}
            if kind == "change_pair"
            else {
                "route_id",
                "baseline_mission",
                "current_mission",
                "assume_aligned",
                "sensitivity",
                "provider_id",
            }
        )
    )
    allowed.add("request_id")
    if set(body) - allowed:
        raise HTTPException(422, "分析字段无效。")
    request_id = body.get("request_id", "")
    if request_id and (
        not isinstance(request_id, str) or not re.fullmatch("[a-f0-9]{32}", request_id)
    ):
        raise HTTPException(422, "分析请求标识无效。")
    if request_id:
        with store.db() as c:
            existing = c.execute(
                "SELECT id,input FROM analyses WHERE owner=? AND kind=? AND request_key=?",
                (oid, kind, request_id),
            ).fetchone()
        if existing:
            if json.loads(existing["input"]) != body:
                raise HTTPException(409, "分析标识与原参数不一致。")
            return store.job(oid, existing["id"])
    with store.db() as c:
        if (
            c.execute(
                "SELECT count(*) FROM analyses WHERE owner=? AND status IN ('queued','running')",
                (oid,),
            ).fetchone()[0]
            >= 3
        ):
            raise HTTPException(429, "当前已有分析任务，请等待完成。")
    # Validate ownership and type before enqueuing expensive inference.
    ids = body.get("resource_ids", [])
    if (
        not isinstance(ids, list)
        or len(ids) > 50
        or any(not isinstance(i, str) for i in ids)
    ):
        raise HTTPException(422, "资源列表无效。")
    for ident in ids + [body[k] for k in ["before_id", "after_id"] if k in body]:
        store.get(oid, ident)
    job = store.create_job(oid, kind, body)
    if job.pop("created_now", False):
        task = asyncio.create_task(run_analysis(oid, job, body))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    return job


@router.get("/analyses/jobs/{ident}")
async def analysis_job(ident: str, request: Request):
    return store.job(owner(request), ident)


@router.get("/analyses/history")
async def history(request: Request):
    oid = owner(request)
    with store.db() as c:
        rows = c.execute(
            "SELECT id FROM analyses WHERE owner=? ORDER BY created DESC LIMIT 60",
            (oid,),
        ).fetchall()
    return {"items": [store.job(oid, r["id"]) for r in rows]}


def recover():
    with store.db() as c:
        c.execute(
            "UPDATE analyses SET status='interrupted',error='服务重启，请检查结果后手动重试。' WHERE status IN ('queued','running')"
        )


recover()
