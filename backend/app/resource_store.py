"""Durable account-local assets. IDs, not user-supplied filesystem paths, cross APIs."""

import json, os, sqlite3, time, uuid
from pathlib import Path
from contextlib import contextmanager
from fastapi import HTTPException
from core.file_access import owned_path

KINDS = {"folder", "image", "video", "archive", "model", "design"}
EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "video": {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts"},
    "archive": {".zip", ".tar", ".gz", ".7z"},
    "model": {".pt", ".pth", ".onnx", ".safetensors", ".glb", ".gltf", ".obj", ".ply"},
    "design": {".kmz", ".kml", ".geojson", ".json", ".yaml", ".yml", ".pdf", ".dxf"},
}


def clean_name(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 180
        or any(ord(c) < 32 for c in value)
        or "/" in value
        or "\\" in value
    ):
        raise HTTPException(422, "名称不能为空，且不能包含路径或控制字符。")
    return value.strip()


class ResourceStore:
    def __init__(self, root):
        self.root = Path(root).resolve() / "resources"
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = self.root / "files"
        self.files.mkdir(exist_ok=True)
        self.previews = self.root / "previews"
        self.previews.mkdir(exist_ok=True)
        self.database = self.root / "library.sqlite3"
        with self.db() as c:
            c.executescript(
                """
            CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,owner INTEGER NOT NULL,name TEXT NOT NULL,
            kind TEXT NOT NULL,parent TEXT NOT NULL,path TEXT NOT NULL,size INTEGER NOT NULL,metadata TEXT NOT NULL,
            trashed INTEGER NOT NULL DEFAULT 0,created REAL NOT NULL,updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS asset_scope ON assets(owner,trashed,parent,created);
            CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY,owner INTEGER NOT NULL,kind TEXT NOT NULL,
            status TEXT NOT NULL,input TEXT NOT NULL,result TEXT,error TEXT,created REAL NOT NULL,updated REAL NOT NULL);
            """
            )
            if "trash_group" not in {
                r[1] for r in c.execute("PRAGMA table_info(assets)")
            }:
                c.execute(
                    "ALTER TABLE assets ADD COLUMN trash_group TEXT NOT NULL DEFAULT ''"
                )
            if "request_key" not in {
                r[1] for r in c.execute("PRAGMA table_info(analyses)")
            }:
                c.execute(
                    "ALTER TABLE analyses ADD COLUMN request_key TEXT NOT NULL DEFAULT ''"
                )
            c.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS analysis_request ON analyses(owner,kind,request_key) WHERE request_key!=''"
            )
        os.chmod(self.database, 0o600)

    @contextmanager
    def db(self):
        c = sqlite3.connect(self.database, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            with c:
                yield c
        finally:
            c.close()

    def get(self, owner, ident, trash=False):
        with self.db() as c:
            r = c.execute(
                "SELECT * FROM assets WHERE owner=? AND id=?", (owner, ident)
            ).fetchone()
        if not r or (r["trashed"] and not trash):
            raise HTTPException(404, "资源不存在或已移入回收站。")
        return dict(r)

    def path(self, row):
        p = owned_path(self.files / row["path"], root=self.files, exists=True)
        if (
            not row["path"]
            or p.parent != self.files
            or not p.is_file()
            or p.is_symlink()
        ):
            raise HTTPException(404, "资源文件不可用。")
        return p

    def parent(self, owner, ident):
        if ident:
            row = self.get(owner, ident)
            if row["kind"] != "folder":
                raise HTTPException(422, "目标必须是文件夹。")
        return ident

    def public(self, r):
        return {
            k: r[k]
            for k in [
                "id",
                "name",
                "kind",
                "parent",
                "size",
                "trashed",
                "created",
                "updated",
            ]
        } | {"metadata": json.loads(r["metadata"])}

    def add(self, owner, name, kind, parent="", path="", size=0, metadata=None):
        name = clean_name(name)
        if kind not in KINDS:
            raise HTTPException(422, "资源类型无效。")
        self.parent(owner, parent)
        ident = uuid.uuid4().hex
        now = time.time()
        with self.db() as c:
            c.execute(
                "INSERT INTO assets(id,owner,name,kind,parent,path,size,metadata,trashed,created,updated) VALUES(?,?,?,?,?,?,?,?,0,?,?)",
                (
                    ident,
                    owner,
                    name,
                    kind,
                    parent,
                    path,
                    size,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.public(self.get(owner, ident))

    def list(
        self,
        owner,
        kind="all",
        parent=None,
        query="",
        page=1,
        limit=60,
        route_id=None,
        mission_id=None,
    ):
        if kind not in KINDS | {"all", "media", "trash"}:
            raise HTTPException(422, "资源分类无效。")
        where = ["owner=?", "trashed=?"]
        args = [owner, int(kind == "trash")]
        if parent is not None:
            where.append("parent=?")
            args.append(parent)
        if kind in KINDS:
            where.append("kind=?")
            args.append(kind)
        elif kind == "media":
            where.append("kind IN ('image','video','archive','folder')")
        if query:
            where.append("name LIKE ? ESCAPE '\\'")
            args.append(
                "%"
                + query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")[:180]
                + "%"
            )
        if route_id:
            where.append("json_extract(metadata,'$.route_id')=?")
            args.append(str(route_id))
        if mission_id:
            where.append("json_extract(metadata,'$.mission_id')=?")
            args.append(str(mission_id))
        clause = " AND ".join(where)
        with self.db() as c:
            total = c.execute(
                "SELECT count(*) FROM assets WHERE " + clause, args
            ).fetchone()[0]
            rows = c.execute(
                "SELECT * FROM assets WHERE "
                + clause
                + " ORDER BY kind='folder' DESC,created DESC LIMIT ? OFFSET ?",
                args + [limit, (page - 1) * limit],
            ).fetchall()
            counts = {
                r["kind"]: {"count": r["n"], "bytes": r["bytes"]}
                for r in c.execute(
                    "SELECT kind,count(*) n,coalesce(sum(size),0) bytes FROM assets WHERE owner=? AND trashed=0 GROUP BY kind",
                    (owner,),
                )
            }
        return {
            "items": [self.public(dict(r)) for r in rows],
            "total": total,
            "counts": counts,
            "page": page,
        }

    def update(self, owner, ident, body):
        row = self.get(owner, ident, True)
        if set(body) - {"name", "parent", "metadata", "trash"}:
            raise HTTPException(422, "资源字段无效。")
        name = clean_name(body.get("name", row["name"]))
        parent = body.get("parent", row["parent"])
        trash = body.get("trash", bool(row["trashed"]))
        if type(trash) is not bool:
            raise HTTPException(422, "回收站状态无效。")
        if not trash:
            if parent:
                try:
                    self.parent(owner, parent)
                except HTTPException:
                    if row["trashed"] and "parent" not in body:
                        parent = ""
                    else:
                        raise
            cursor = parent
            for _ in range(100):
                if not cursor:
                    break
                if cursor == ident:
                    raise HTTPException(422, "不能将文件夹移动到自身或子文件夹。")
                cursor = self.get(owner, cursor)["parent"]
            else:
                raise HTTPException(422, "文件夹层级过深。")
        metadata = json.loads(row["metadata"])
        if "metadata" in body:
            value = body["metadata"]
            if not isinstance(value, dict) or set(value) - {
                "tags",
                "route_id",
                "mission_id",
                "waypoint",
                "captured_at",
                "longitude",
                "latitude",
                "panorama",
                "description",
                "source_media_id",
            }:
                raise HTTPException(422, "资源标签字段无效。")
            if len(json.dumps(value, ensure_ascii=False)) > 4000:
                raise HTTPException(422, "标签内容过长。")
            if "tags" in value and (
                not isinstance(value["tags"], list)
                or len(value["tags"]) > 30
                or any(not isinstance(t, str) or len(t) > 80 for t in value["tags"])
            ):
                raise HTTPException(422, "标签无效。")
            for key in [
                "route_id",
                "mission_id",
                "waypoint",
                "captured_at",
                "description",
                "source_media_id",
            ]:
                if key in value and (
                    not isinstance(value[key], (str, int, float))
                    or len(str(value[key])) > 600
                ):
                    raise HTTPException(422, "资源关联字段无效。")
            for key, low, high in [("longitude", -180, 180), ("latitude", -90, 90)]:
                if key in value and (
                    type(value[key]) not in [int, float]
                    or not low <= value[key] <= high
                ):
                    raise HTTPException(422, "资源坐标无效。")
            metadata.update(value)
        with self.db() as c:
            c.execute(
                "UPDATE assets SET name=?,parent=?,metadata=?,trashed=?,updated=? WHERE id=? AND owner=?",
                (
                    name,
                    parent,
                    json.dumps(metadata, ensure_ascii=False),
                    int(trash),
                    time.time(),
                    ident,
                    owner,
                ),
            )
            # Keep folder children together in the recycle bin, and restore only that subtree.
            if row["kind"] == "folder" and bool(row["trashed"]) != trash:
                base = "WITH RECURSIVE descendants(id) AS (SELECT id FROM assets WHERE parent=? AND owner=? UNION ALL SELECT a.id FROM assets a JOIN descendants d ON a.parent=d.id WHERE a.owner=?) "
                if trash:
                    c.execute(
                        base
                        + "UPDATE assets SET trashed=1,trash_group=?,updated=? WHERE owner=? AND trashed=0 AND id IN (SELECT id FROM descendants)",
                        (ident, owner, owner, ident, time.time(), owner),
                    )
                else:
                    c.execute(
                        base
                        + "UPDATE assets SET trashed=0,trash_group='',updated=? WHERE owner=? AND trash_group=? AND id IN (SELECT id FROM descendants)",
                        (ident, owner, owner, time.time(), owner, ident),
                    )
        return self.public(self.get(owner, ident, True))

    def create_job(self, owner, kind, body):
        ident = uuid.uuid4().hex
        now = time.time()
        key = body.get("request_id", "")
        serialized = json.dumps(body, ensure_ascii=False, sort_keys=True)
        with self.db() as c:
            c.execute("BEGIN IMMEDIATE")
            if key:
                old = c.execute(
                    "SELECT * FROM analyses WHERE owner=? AND kind=? AND request_key=?",
                    (owner, kind, key),
                ).fetchone()
                if old:
                    if json.loads(old["input"]) != body:
                        raise HTTPException(409, "分析请求标识已用于其他参数。")
                    return {**self.job(owner, old["id"]), "created_now": False}
            if (
                c.execute(
                    "SELECT count(*) FROM analyses WHERE owner=? AND status IN ('queued','running')",
                    (owner,),
                ).fetchone()[0]
                >= 3
            ):
                raise HTTPException(429, "当前已有分析任务，请等待完成。")
            c.execute(
                "INSERT INTO analyses(id,owner,kind,status,input,result,error,created,updated,request_key) VALUES(?,?,?,'queued',?,NULL,NULL,?,?,?)",
                (ident, owner, kind, serialized, now, now, key),
            )
        return {**self.job(owner, ident), "created_now": True}

    def job(self, owner, ident):
        with self.db() as c:
            r = c.execute(
                "SELECT * FROM analyses WHERE owner=? AND id=?", (owner, ident)
            ).fetchone()
        if not r:
            raise HTTPException(404, "分析记录不存在。")
        return {
            k: r[k] for k in ["id", "kind", "status", "error", "created", "updated"]
        } | {
            "result": json.loads(r["result"]) if r["result"] else None,
            "input": json.loads(r["input"]),
        }

    def finish_job(self, owner, ident, status, result=None, error=None):
        with self.db() as c:
            c.execute(
                "UPDATE analyses SET status=?,result=?,error=?,updated=? WHERE owner=? AND id=?",
                (
                    status,
                    (
                        json.dumps(result, ensure_ascii=False)
                        if result is not None
                        else None
                    ),
                    error,
                    time.time(),
                    owner,
                    ident,
                ),
            )
