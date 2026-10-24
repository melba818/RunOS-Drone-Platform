"""推理页视频任务的**持久**索引（`temp/inference/lmm_video_tasks.json`）。

为什么不能只用内存里的 `_TASKS`：

后端一重启（开发期一天重启十几次），「视频推理记录」就空了、日志没了、下载按钮
变 404 —— 用户刚说过"记录跟操作对不上"，再给一个会凭空消失的列表就是同一个问题
换了个形状。而结果视频本身就落在 `temp/inference/` 且**启动时不会被清理**
（`startup._cleanup_temp()` 刻意保留该目录），所以索引和产物同生共死，是自洽的。

只记**推理页上传的视频任务**（`save_video` 的那批）。多模态页的实时任务不进这里：
它们没有结果文件、也没有终点，放进「视频推理记录」就是干扰项。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

_LOCK = threading.RLock()
#: 进程内缓存（首次访问时从磁盘读）
_ROWS: Optional[List[Dict[str, Any]]] = None
_MAX_ROWS = 50
_MAX_LOGS = 200
_FILE = "lmm_video_tasks.json"


def _dir() -> Path:
    from .render import output_dir
    return output_dir()


def _path() -> Path:
    return _dir() / _FILE


def _load() -> List[Dict[str, Any]]:
    global _ROWS
    if _ROWS is not None:
        return _ROWS
    _ROWS = []
    try:
        p = _path()
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _ROWS = [r for r in data if isinstance(r, dict)]
    except Exception:  # noqa: BLE001 —— 索引坏了不该让识别跑不起来
        LOG.warning("[LMM] 视频任务索引读取失败，按空处理", exc_info=True)
        _ROWS = []
    return _ROWS


def _flush() -> None:
    """整份重写 + 原子替换。

    行数上限 50、每行日志上限 200，整份最多几百 KB —— 不值得为它做增量写。
    用 `os.replace` 是为了"写到一半崩溃"时不会留下半个 JSON（下次启动读不动）。
    """
    rows = _load()
    try:
        _dir().mkdir(parents=True, exist_ok=True)
        p = _path()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows[-_MAX_ROWS:], ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001
        LOG.warning("[LMM] 视频任务索引写入失败", exc_info=True)


def upsert(task_id: str, **fields: Any) -> None:
    """新增或更新一行。**只有「建任务」时该用它** —— 它会凭空造出记录。"""
    if not task_id:
        return
    with _LOCK:
        rows = _load()
        for r in rows:
            if r.get("id") == task_id:
                r.update(fields)
                break
        else:
            rows.append({"id": task_id, **fields})
        _flush()


def update(task_id: str, **fields: Any) -> None:
    """只更新**已存在**的行；不存在就什么都不做。

    任务生命周期里的所有回写都走这个：多模态页的实时任务（没有结果视频、
    本来就不该进索引）会在这里被自然过滤掉，不会因为一次 `_set` 就被记进来。
    """
    if not task_id:
        return
    with _LOCK:
        for r in _load():
            if r.get("id") == task_id:
                r.update(fields)
                _flush()
                return


def add_log(task_id: str, line: Dict[str, Any]) -> None:
    """把一条日志追加进对应行。**行不存在就什么都不做** —— 这样多模态页那些
    不落盘的实时任务不会把索引撑爆。"""
    if not task_id:
        return
    with _LOCK:
        for r in _load():
            if r.get("id") == task_id:
                logs = r.setdefault("logs", [])
                logs.append(line)
                if len(logs) > _MAX_LOGS:
                    del logs[: len(logs) - _MAX_LOGS]
                _flush()
                return


def remove(task_id: str) -> None:
    """删掉一行（任务被显式删除时）。产物文件本身不在这里删 —— 那是另一件事。"""
    with _LOCK:
        rows = _load()
        keep = [r for r in rows if r.get("id") != task_id]
        if len(keep) != len(rows):
            rows[:] = keep
            _flush()


def rows() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(r) for r in _load()]


def get(task_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        for r in _load():
            if r.get("id") == task_id:
                return dict(r)
    return None


def logs(task_id: str) -> List[Dict[str, Any]]:
    r = get(task_id) or {}
    return [x for x in (r.get("logs") or []) if isinstance(x, dict)]
