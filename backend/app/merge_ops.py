# -*- coding: utf-8 -*-
"""STEP 4：Candidate Merge —— Full Image(VLX) 与 Tile(Qwen) 两路候选的几何融合。

================================================================================
★ 最重要的一条：本模块**绝不使用 VLX 的 confidence 做 NMS** ★
================================================================================
VLX-Seek 不给真实置信度（`vlx_engine.py` 里硬编码 1.0，
`adapters/vlx.py` 标注 `confidence_source: "synthetic"`）。拿 1.0 去做 NMS，
等于让"谁先来谁留下"，结果完全不可解释 —— 而且会把真目标压掉。

所以第一版**刻意不上 Weighted Box Fusion**：WBF 的前提是各模型的分数可比较，
这里两个模型（VLX / Qwen）根本没有可比较的真实 detector confidence。

改用 **class-aware geometric merge**：
    同类别
    + IoU 阈值
    + 中心距离
    + tile 边界规则（贴边框 = 目标被切了一半，不能当完整目标）
分数只作为**原样记录**（provenance），不参与任何排序 / 阈值 / 抑制。

Qwen Verify 之后可以保存 verification 结果，但**绝不把语言模型自己编的数字
伪装成 detector confidence** —— 那是 VLM 的自述，不是检测器的概率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from lmm.tracking.types import center_dist, iou_xyxy
except Exception:  # noqa: BLE001  —— 独立使用时自带一份，避免强依赖
    def iou_xyxy(a, b):  # type: ignore[misc]
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / ua

    def center_dist(a, b):  # type: ignore[misc]
        acx = (a[0] + a[2]) * 0.5
        acy = (a[1] + a[3]) * 0.5
        bcx = (b[0] + b[2]) * 0.5
        bcy = (b[1] + b[3]) * 0.5
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

Box = Tuple[float, float, float, float]

SOURCE_VLX_FULL = "vlx_full"
SOURCE_QWEN_TILE = "qwen_tile"

#: 通用双模型链路（标注页「一个标注 + 一个校验」）用的来源名。
#: 与上面两个是**并列关系**，不是替代 —— V1 的 VLX+Qwen 链路继续用原来的名字。
SOURCE_MODEL_A = "model_a"      #: 标注模型（主）
SOURCE_MODEL_B = "model_b"      #: 校验模型


@dataclass
class Candidate:
    """一路来源给出的一个候选框。"""

    bbox: Box
    label: str
    source: str                       # vlx_full / qwen_tile
    score: Optional[float] = None     #: **只记录，不参与融合**（见文件头）
    tile_index: Optional[int] = None
    edge: bool = False                #: 贴 tile 边界 → 可能是半截目标
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergedCandidate:
    bbox: Box
    label: str
    #: ★provenance★：这个框是从哪来的、原始框分别是什么、为什么合并
    sources: List[str]
    raw_boxes: List[Box]
    merge_reason: str
    n_full: int = 0
    n_tile: int = 0
    edge_only: bool = False
    verification: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "label": self.label,
            "sources": list(self.sources),
            "raw_boxes": [[round(float(v), 2) for v in b] for b in self.raw_boxes],
            "merge_reason": self.merge_reason,
            "n_full": self.n_full,
            "n_tile": self.n_tile,
            "edge_only": self.edge_only,
            "verification": self.verification,
        }


def _median_box(boxes: Sequence[Box]) -> Box:
    """各坐标取中位数 —— 不被某个"半截框"带偏，也不会像并集那样膨胀。"""
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    cols = list(zip(*boxes))
    out = []
    for col in cols:
        s = sorted(float(v) for v in col)
        n = len(s)
        mid = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) * 0.5
        out.append(float(mid))
    return (out[0], out[1], out[2], out[3])


def _diag(box: Box) -> float:
    return max(1e-6, ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5)


def class_aware_merge(candidates: Sequence[Candidate],
                      iou_threshold: float = 0.5,
                      center_ratio: float = 0.5,
                      prefer_full: bool = True,
                      ) -> List[MergedCandidate]:
    """同类别 + IoU/中心距 + tile 边界规则的几何融合。

    参数：
        iou_threshold   IoU 达到多少算同一个目标
        center_ratio    中心距 <= ratio × 平均对角线也算同一个目标
        prefer_full     合并后的主框优先取 **Full Image(VLX)** 那条 ——
                        VLX 看的是整图，给的是完整目标；tile 里很可能是半截。

    返回 `MergedCandidate`，每条都带 `sources` / `raw_boxes` / `merge_reason`。
    """
    by_label: Dict[str, List[Candidate]] = {}
    for c in candidates or []:
        by_label.setdefault(str(c.label or ""), []).append(c)

    merged_all: List[MergedCandidate] = []
    for label, group in by_label.items():
        used = [False] * len(group)
        clusters: List[List[int]] = []
        for i, ci in enumerate(group):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, cj in enumerate(group):
                    if used[j]:
                        continue
                    hit = False
                    reason = ""
                    for k in cluster:
                        ck = group[k]
                        ov = iou_xyxy(ck.bbox, cj.bbox)
                        thr_dist = center_ratio * ((_diag(ck.bbox) + _diag(cj.bbox)) * 0.5)
                        d = center_dist(ck.bbox, cj.bbox)
                        if ov >= iou_threshold:
                            hit, reason = True, "iou=%.2f" % ov
                        elif d <= thr_dist:
                            hit, reason = True, "center=%.0fpx<=%.0f" % (d, thr_dist)
                        if hit:
                            break
                    if hit:
                        cluster.append(j)
                        used[j] = True
                        changed = True
            clusters.append(cluster)

        for cl in clusters:
            items = [group[k] for k in cl]
            boxes = [it.bbox for it in items]
            fulls = [it for it in items if it.source == SOURCE_VLX_FULL]
            tiles = [it for it in items if it.source == SOURCE_QWEN_TILE]
            if len(items) == 1:
                reason = "single:%s" % items[0].source
            else:
                a, b = items[0].bbox, items[1].bbox
                ov = iou_xyxy(a, b)
                reason = ("merged%d:iou=%.2f,center=%.0fpx"
                          % (len(items), ov, center_dist(a, b)))
            if prefer_full and fulls:
                main = fulls[0].bbox
                reason += "|main=vlx_full"
            else:
                main = _median_box(boxes)
                reason += "|main=median"
            edge_only = bool(tiles) and not fulls and all(t.edge for t in tiles)
            if edge_only:
                reason += "|edge_only"
            merged_all.append(MergedCandidate(
                bbox=tuple(float(v) for v in main),  # type: ignore[arg-type]
                label=label,
                sources=sorted({it.source for it in items}),
                raw_boxes=[tuple(float(v) for v in b) for b in boxes],  # type: ignore[misc]
                merge_reason=reason,
                n_full=len(fulls), n_tile=len(tiles), edge_only=edge_only,
                extra={"scores": [it.score for it in items
                                  if it.score is not None] or None},
            ))
    return merged_all


def dedupe_boxes(merged: Sequence[MergedCandidate],
                 iou_threshold: float = 0.7) -> List[MergedCandidate]:
    """融合之后再扫一遍：同类别、IoU 仍很高的两条再并成一条（几何去重）。

    与 `class_aware_merge` 的区别：那一步是"不同来源的同一目标"，这一步是
    "融合算法本身的遗留重复"（例如 A-B 命中、B-C 命中但 A-C 没命中时）。
    """
    out: List[MergedCandidate] = []
    for m in merged or []:
        hit = None
        for o in out:
            if o.label != m.label:
                continue
            if iou_xyxy(o.bbox, m.bbox) >= iou_threshold:
                hit = o
                break
        if hit is None:
            out.append(m)
            continue
        hit.raw_boxes = list(hit.raw_boxes) + list(m.raw_boxes)
        hit.sources = sorted(set(hit.sources) | set(m.sources))
        hit.n_full += m.n_full
        hit.n_tile += m.n_tile
        hit.merge_reason += "|dedupe_iou=%.2f" % iou_xyxy(hit.bbox, m.bbox)
    return out


# ── 类别归一化 ─────────────────────────────────────────────────────────────
def normalize_class(name: str, class_table: Sequence[str]) -> Optional[str]:
    """把模型自由生成的类别映射到**当前 dataset 已有的类别表**。

    ★硬性要求★：映射不上就返回 None —— 由调用方 drop 或进 review，
    **绝不自动往 data.yaml 里加新类别**（那会让数据集的类别体系失控）。

    匹配顺序：精确 → 大小写/空格归一 → 中文别名表里的同义。
    """
    if not name:
        return None
    table = [str(c) for c in (class_table or [])]
    raw = str(name).strip()
    if not table:
        return None
    if raw in table:
        return raw
    low = raw.lower().replace(" ", "").replace("_", "")
    for c in table:
        if c.lower().replace(" ", "").replace("_", "") == low:
            return c
    return None


#: 常见中英文别名 → 标准类别名（只在**类别表里已有该标准名**时才生效，
#: 绝不凭空造类别）
CLASS_ALIASES = {
    "人": "person", "人员": "person", "行人": "person", "human": "person",
    "people": "person", "man": "person", "woman": "person",
    "车": "car", "汽车": "car", "轿车": "car", "vehicle": "car",
    "船": "boat", "船只": "boat", "ship": "boat", "vessel": "boat",
    "无人机": "drone", "uav": "drone",
    "游泳者": "swimmer", "swimer": "swimmer",
}


def normalize_class_with_alias(name: str, class_table: Sequence[str]) -> Optional[str]:
    """先按别名归一到标准名，再走 `normalize_class`。"""
    direct = normalize_class(name, class_table)
    if direct:
        return direct
    raw = str(name or "").strip()
    alias = CLASS_ALIASES.get(raw) or CLASS_ALIASES.get(raw.lower())
    if alias:
        return normalize_class(alias, class_table)
    return None
