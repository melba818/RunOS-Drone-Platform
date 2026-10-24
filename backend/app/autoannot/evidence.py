# -*- coding: utf-8 -*-
"""★第三原则：模型原始结果必须全部保存★

为什么要存：融合规则一定会反复调（阈值、冲突策略、`on_verify_fail`…）。
**只要原始结果在，改规则就是"重放 JSON"而不是"重跑两个模型"** ——
一次 VLX 全图 + 几十块 Qwen tile + 双向校验，一张图就是几分钟。

落盘布局（`workspace/autoannot/<job_id>/`）：

    <job_id>/
    ├── job.json              任务状态
    ├── log.jsonl             追加式日志
    ├── raw/<image>.json      ★逐图原始证据★（本模块负责）
    ├── labels/*.txt          ACCEPT 的 YOLO 导出（apply 前只在这里）
    └── review.jsonl          REVIEW 清单（人工复核用）

`raw/<image>.json` 的内容严格覆盖要求的六项：
    VLX 原始 / Qwen Tile 原始 / Qwen 校验 VLX / VLX 校验 Qwen /
    Merge 前候选 / Merge 后结果 / 最终 ACCEPT|REVIEW
"""
from __future__ import annotations

import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger(__name__)

#: 输出根目录。★刻意与 datasets 分开★：自动标注的产物不直接污染数据集。
AUTOANNOT_ROOT = Path(__file__).resolve().parent.parent / "workspace" / "autoannot"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def root_dir() -> Path:
    AUTOANNOT_ROOT.mkdir(parents=True, exist_ok=True)
    return AUTOANNOT_ROOT


def job_dir(job_id: str) -> Path:
    return root_dir() / str(job_id)


def ensure_job_dirs(jid: Path) -> None:
    for sub in ("raw", "labels"):
        (jid / sub).mkdir(parents=True, exist_ok=True)


def write_raw(jid: Path, image_key: str, payload: Dict[str, Any]) -> Path:
    """写一份逐图原始证据，返回文件路径。

    `image_key` 用相对路径（去掉盘符/分隔符）做文件名，避免不同数据集的
    同名图片互相覆盖时丢失信息 —— 仍然保留原名可读。

    ★为什么尾部改成短哈希而不是 `safe[-60:]`★：截断相对路径会切出一个
    词被劈开的怪名字（`..._423__ages_frames_..._423.jpg.json`，那个 `ages`
    是 `images` 被截掉前两个字母），既难读又看不出它代表什么。用全量 key
    的 md5 前 6 位做去重后缀，唯一性与 `[-60:]` 等价（更强），但文件名干净。
    """
    stem = Path(image_key).stem
    digest = hashlib.md5(image_key.encode("utf-8")).hexdigest()[:6]
    fp = jid / "raw" / ("%s__%s.json" % (stem[:64], digest))
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def append_log(jid: Path, level: str, message: str, **extra: Any) -> None:
    """追加式日志（JSONL）。★不用 print★ —— 自动标注任务要能被事后检索。"""
    rec = {"ts": time.time(), "level": str(level), "msg": str(message)}
    if extra:
        rec.update(extra)
    with (jid / "log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_log(jid: Path, limit: int = 500) -> List[Dict[str, Any]]:
    fp = jid / "log.jsonl"
    if not fp.exists():
        return []
    lines = fp.read_text(encoding="utf-8").splitlines()[-int(limit):]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def list_raw(jid: Path) -> List[str]:
    d = jid / "raw"
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.json"))


def read_raw(jid: Path, name: str) -> Optional[Dict[str, Any]]:
    fp = jid / "raw" / name
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except ValueError:
        return None


# ── 导出 ────────────────────────────────────────────────────────────────────
def to_yolo_lines(annotations: Iterable[Dict[str, Any]], width: int,
                  height: int, only_accept: bool = True) -> List[str]:
    """annotation dict → YOLO 行（`class_id cx cy w h`，归一化 0~1）。

    `annotations` 里 `bbox` 是 `[x, y, w, h]` **绝对像素**（与既有
    `_save_annotations` 的约定一致），这里做归一化。
    """
    lines: List[str] = []
    w = max(1.0, float(width or 1))
    h = max(1.0, float(height or 1))
    for a in annotations or []:
        if only_accept and str(a.get("decision")) != "ACCEPT":
            continue
        try:
            cid = int(a.get("class_id"))
            x, y, bw, bh = (float(v) for v in (a.get("bbox") or [])[:4])
        except (TypeError, ValueError):
            continue
        if bw <= 0 or bh <= 0:
            continue
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        lines.append("%d %.6f %.6f %.6f %.6f"
                     % (cid, cx, cy, bw / w, bh / h))
    return lines


def write_labels(jid: Path, image_key: str, lines: List[str]) -> Optional[Path]:
    """把一张图的 YOLO 行写进 **job 自己的** labels/ 目录。

    ★不直接写回数据集★ —— 那是 `apply` 的事。自动标注先产出、后确认。
    """
    if not lines:
        return None
    stem = Path(image_key).stem
    fp = jid / "labels" / ("%s.txt" % stem)
    fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fp


def write_review(jid: Path, items: List[Dict[str, Any]]) -> Path:
    """REVIEW 清单（JSONL）—— 人工复核的入口。"""
    fp = jid / "review.jsonl"
    with fp.open("a", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return fp


def list_images(dataset_path: Path) -> List[Path]:
    """列出数据集里的图片。

    只走 `images/` 这类目录，**跳过 `labels/`**（里面可能有同名 txt，
    也可能被用户塞进图片做预览，但那不是要标注的对象）。
    """
    root = Path(dataset_path)
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part.lower() in ("labels", "label", "annotations")
               for part in p.relative_to(root).parts[:-1]):
            continue
        if p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out
