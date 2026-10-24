# -*- coding: utf-8 -*-
"""STEP 4：自动标注增强链路（编排层）。

链路（用户拍定，不得改序）：
    原图 ─┬─ Full Image ─→ VLX ─────────────┐
          └─ Tile Pass ──→ Qwen2.5-VL-3B ───┴→ Candidate Merge
                                                    ↓
                                              几何去重
                                                    ↓
                                        Qwen2.5-VL-3B Verify
                                                    ↓
                                              类别归一化 → 输出

=====================================================================
三条硬约束（用户逐条拍定，本文件必须始终满足）
=====================================================================
1. **VLX 没有真实 confidence**。
   `vlx_engine.py` 里那个 1.0 是硬编码的 synthetic 值。因此：
   - 绝不用 VLX 的置信度做 NMS / 阈值过滤 / 排序；
   - 融合只能依据：class 归一化、IoU、center distance、Full/Tile 来源、tile 边界。
2. **第一版不上 Weighted Box Fusion**。
   两个模型没有可比较的真实 detector confidence，WBF 的"加权"没有物理意义。
   用 `merge_ops.class_aware_merge`（同类别 + 几何 + 边界规则）。
3. **绝不自动扩张类别表**。
   所有 label 必须经 `merge_ops.normalize_class_with_alias` 映射到
   **当前 dataset 已有类别**；映射不上 = drop（或 review），
   **禁止**把模型自由生成的名字写进 data.yaml。

=====================================================================
复用而非重建（用户明确要求）
=====================================================================
- 复用 `vlm_annotator.build_prompt / parse_objects / decode_bbox`：
  那份宽容 JSON 解析与 0-1000 坐标解码已经过实战，不重写一份。
- 复用 `model_integration.auto_annotate()` 的批量循环与 `_save_annotations`
  writer：本模块只产出与 `_run_inference_unlocked` 完全同形的 annotation
  dict，由既有 batch job（pause/resume/cancel/undo/persist）驱动。
- **不碰** `annotation_tools` writers、WebRTC、MediaMTX、实时 capture、
  source 生命周期。

失败语义：与 `RemoteOSTrackBackend` / `QwenVLClient` 一致 —— 单张图内任何
一步失败都只记 debug，**不让批量任务因一张图崩掉**。
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from merge_ops import (
    SOURCE_MODEL_A,
    SOURCE_MODEL_B,
    SOURCE_QWEN_TILE,
    SOURCE_VLX_FULL,
    Candidate,
    MergedCandidate,
    class_aware_merge,
    dedupe_boxes,
    normalize_class_with_alias,
)
from tiling import (
    clamp_box,
    crop_with_context,
    from_tile,
    generate_tiles,
    near_tile_edge,
)

try:
    #: `merge_ops` 的两条路径（import 成功 / 自带 fallback）都会绑定这个名字
    from merge_ops import iou_xyxy as _iou_xyxy
except Exception:  # pragma: no cover —— 再兜一层，保证本模块永远可 import
    def _iou_xyxy(a, b):  # type: ignore[misc]
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / ua

LOG = logging.getLogger(__name__)

#: `on_verify_fail` 的合法取值
VERIFY_FAIL_POLICIES = ("keep", "drop")

DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 128
DEFAULT_MAX_TILES = 64           #: 安全阀：超大图切片数量的硬上限
DEFAULT_QWEN_PORT = 8023


# ── Qwen 客户端装配 ──────────────────────────────────────────────────────────
def _make_qwen_client(node_id: str = "", port: int = 0,
                      token: str = "", timeout: float = 180.0):
    """按 lmm 配置装配 Qwen 客户端。

    Qwen 服务与 OSTrack 在同一台 A6000 上，所以 node_id 缺省回落到
    `lmm_ostrack_node_id`；端口缺省 8023。
    """
    from lmm import config as _cfg
    from lmm.qwen_client import QwenVLClient

    cfg = {}
    try:
        cfg = _cfg.load() or {}
    except Exception:  # noqa: BLE001
        cfg = {}

    nid = node_id or cfg.get("lmm_qwen_node_id") or cfg.get("lmm_ostrack_node_id") or ""
    p = int(port or cfg.get("lmm_qwen_port") or DEFAULT_QWEN_PORT)
    tok = token or cfg.get("lmm_qwen_token") or ""
    return QwenVLClient(node_id=nid, port=p, token=tok, timeout=float(timeout))


# ── 各阶段 ──────────────────────────────────────────────────────────────────
def _vlx_full_candidates(image_path: Path, text_prompt: str,
                         class_table: Sequence[str],
                         debug: Dict[str, Any],
                         label_override: Optional[str] = None) -> List[Candidate]:
    """Full Image → VLX。

    VLX 的返回框已经是原图像素 xyxy；它**不返回置信度**（那 1.0 是合成的），
    所以这里 `score=None` —— 不编、不用。

    `label_override`：通用双模型链路用。VLX 是开放词汇模型，写进类别表的名字
    应该由调用方显式指定（标注页那个「类别」框），而不是拿整句目标描述去归一化。
    不传时行为与 V1 完全一致（仍用 `text_prompt` 归一化）。
    """
    from lmm import config as _cfg
    from lmm.vlx_client import VLXSeekClient

    out: List[Candidate] = []
    prompt = (text_prompt or "").strip()
    if not prompt:
        debug["vlx_error"] = "未提供目标描述，跳过 Full Image 阶段"
        return out

    try:
        cfg = _cfg.load() or {}
        res = VLXSeekClient().infer_image(
            image_path.read_bytes(), prompt,
            task=cfg.get("vlx_seek_task") or "detection",
            lang=cfg.get("vlx_seek_lang") or "zh",
            max_new_tokens=int(cfg.get("vlx_seek_max_new_tokens") or 256),
            filename=image_path.name,
        )
    except Exception as exc:  # noqa: BLE001
        debug["vlx_error"] = "%s: %s" % (type(exc).__name__, exc)
        LOG.warning("[chain] VLX full 失败: %s", exc)
        return out

    dets = (res or {}).get("detections") or []
    debug["vlx_raw"] = len(dets)
    for det in dets:
        xyxy = det.get("bbox_xyxy") or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if x2 <= x1 or y2 <= y1:
            continue
        label = normalize_class_with_alias(label_override or prompt, class_table)
        if label is None:
            debug["unmatched_prompt"] = debug.get("unmatched_prompt", 0) + 1
            continue
        out.append(Candidate(bbox=(x1, y1, x2, y2), label=label,
                             source=SOURCE_VLX_FULL, score=None,
                             tile_index=None, edge=False,
                             raw={"prompt": prompt}))
    debug["vlx_candidates"] = len(out)
    return out


def _qwen_tile_candidates(img, text_prompt: str,
                          class_table: Sequence[str],
                          qwen, tile_size: int, overlap: int,
                          min_tile: int, max_tiles: int,
                          debug: Dict[str, Any]) -> List[Candidate]:
    """Tile Pass → Qwen2.5-VL-3B。

    坐标链路：tile 内 0-1000 → tile 像素 → `from_tile` 回源图坐标。
    `edge`（贴 tile 边界）会被记下来，融合时用于判断"半截目标"。

    Qwen 可能会回一个 `confidence` —— 我们**记录但绝不用于 NMS/融合**
    （见文件头约束 1）。
    """
    from vlm_annotator import build_prompt, decode_bbox, parse_objects

    out: List[Candidate] = []
    w, h = img.size
    tiles = generate_tiles(int(w), int(h), int(tile_size), int(overlap),
                           min_tile=int(min_tile))
    if len(tiles) > int(max_tiles):
        tiles = tiles[: int(max_tiles)]
        debug["tiles_truncated"] = True
    debug["tiles"] = len(tiles)

    prompt = build_prompt(list(class_table), "0-1000",
                          extra_instruction=(text_prompt or "").strip())
    debug["tile_prompt"] = prompt

    tile_ms: List[float] = []
    n_err = 0
    last_err = ""
    consec_fail = 0
    for t in tiles:
        tw, th = t.width, t.height
        crop = img.crop(t.as_tuple())
        t0 = time.perf_counter()
        r = qwen.chat(crop, prompt, max_tokens=512)
        tile_ms.append(round((time.perf_counter() - t0) * 1000.0, 1))
        if not r.get("ok"):
            n_err += 1
            consec_fail += 1
            last_err = str(r.get("error") or "tile 调用失败")
            #: ★服务整体不在时别把整轮 tile 都跑完★ —— 每块都要等一次连接失败
            #: （隧道侧实测 ~480ms/块），8 块就是 4s 纯浪费；一张图如此，一批
            #: 几百张就是几十分钟。连续失败即判定"不可用"并终止本轮。
            if consec_fail >= QWEN_ABORT_AFTER and not out:
                debug["qwen_aborted"] = True
                break
            continue
        consec_fail = 0
        for obj in parse_objects(r.get("text") or ""):
            raw_box = obj.get("bbox") or obj.get("box") or obj.get("bbox_2d")
            xywh = decode_bbox(raw_box, "0-1000", tw, th)
            if xywh is None:
                continue
            x, y, bw, bh = xywh
            tile_box = (x, y, x + bw, y + bh)          #: tile 内 xyxy
            full_box = from_tile(tile_box, t, w, h)     #: 回源图并 clamp

            raw_label = (obj.get("label") or obj.get("name")
                         or obj.get("class") or "")
            label = normalize_class_with_alias(raw_label, class_table)
            if label is None:
                debug["unmatched_tile"] = debug.get("unmatched_tile", 0) + 1
                continue

            conf = obj.get("confidence")
            try:
                conf = float(conf) if conf is not None else None
            except Exception:  # noqa: BLE001
                conf = None

            out.append(Candidate(bbox=full_box, label=label,
                                 source=SOURCE_QWEN_TILE,
                                 score=conf,              #: 只记录
                                 tile_index=t.index,
                                 edge=near_tile_edge(tile_box, t),
                                 raw={"tile": t.index, "raw_label": str(raw_label)}))

    if tile_ms:
        tile_ms.sort()
        debug["tile_ms"] = {"avg": round(sum(tile_ms) / len(tile_ms), 1),
                            "p50": tile_ms[len(tile_ms) // 2],
                            "max": tile_ms[-1]}
    if n_err:
        debug["tile_errors"] = n_err
    #: ★Teacher 缺席判定：每一块 tile 都调不通 = 服务整体不可用，不是"这张图没目标"。
    if tiles and n_err >= len(tiles):
        debug["qwen_unavailable"] = True
        debug["qwen_last_error"] = last_err[:300]
    elif debug.get("qwen_aborted"):
        debug["qwen_unavailable"] = True
        debug["qwen_last_error"] = last_err[:300]
    debug["qwen_tile_candidates"] = len(out)
    return out


def _verify_candidates(img, merged: List[MergedCandidate],
                       class_table: Sequence[str], qwen,
                       context: float, on_verify_fail: str,
                       debug: Dict[str, Any]) -> List[MergedCandidate]:
    """Qwen2.5-VL-3B Verify：只问"是不是指定类别"，要严格 JSON。

    - `valid=true` → 类别再过一次 `normalize_class_with_alias` 才落盘
      （模型自由生成的名字**不直接信任**）；
    - `valid=false` → drop；
    - 服务调用失败 → 按 `on_verify_fail` 策略（默认 keep，避免一次抖动
      就把整批清空；但会如实记进 debug）。
    """
    w, h = img.size
    kept: List[MergedCandidate] = []
    n_valid = n_invalid = n_unmatched = n_fail = 0

    for m in merged:
        cbox = crop_with_context(m.bbox, int(w), int(h), context=float(context))
        crop = img.crop(tuple(int(round(v)) for v in cbox))
        r = qwen.verify(crop, _verify_prompt(m.label))
        if not r.get("ok"):
            n_fail += 1
            m.verification = {"ok": False, "error": r.get("error", "verify 调用失败")}
            if on_verify_fail == "drop":
                continue
            kept.append(m)          #: fail-open，但 verification 已如实标记
            continue

        js = r.get("json") or {}
        if not isinstance(js, dict) or not bool(js.get("valid")):
            n_invalid += 1
            m.verification = {"ok": True, "valid": False,
                              "reason": str(js.get("reason", ""))[:200]}
            continue

        cls = normalize_class_with_alias(js.get("class") or m.label, class_table)
        if cls is None:
            n_unmatched += 1
            m.verification = {"ok": True, "valid": True,
                              "unmatched_class": str(js.get("class"))}
            continue                #: ★映射不上就 drop，绝不扩张类别表★

        n_valid += 1
        m.label = cls
        m.verification = {"ok": True, "valid": True, "class": cls,
                          "reason": str(js.get("reason", ""))[:200]}
        kept.append(m)

    debug["verify"] = {"valid": n_valid, "invalid": n_invalid,
                       "unmatched": n_unmatched, "failed": n_fail,
                       "policy_on_fail": on_verify_fail}
    return kept


def _verify_prompt(label: str) -> str:
    """Verify 的 prompt —— 只要一个严格 JSON，不要推理过程，不要新类别。"""
    return (
        "Look at the object inside the highlighted region of this image crop.\n"
        "Decide whether it is a '%s'.\n"
        "Reply with JSON only, no prose, exactly:\n"
        '{"valid": true, "class": "%s", "reason": "<short reason>"}\n'
        "If it is NOT a '%s', reply:\n"
        '{"valid": false, "class": "unknown", "reason": "<short reason>"}\n'
        "You MUST NOT invent a new class name."
        % (label, label, label)
    )


# ── 主入口 ──────────────────────────────────────────────────────────────────
def run_chain(
    image_path: Path,
    text_prompt: str,
    class_table: Sequence[str],
    *,
    vlx: bool = True,
    qwen_tiles: bool = True,
    verify: bool = True,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_tile: int = 64,
    max_tiles: int = DEFAULT_MAX_TILES,
    iou_threshold: float = 0.5,
    center_ratio: float = 0.5,
    dedupe_iou: float = 0.7,
    verify_context: float = 1.5,
    on_verify_fail: str = "keep",
    node_id: str = "",
    qwen_port: int = 0,
    qwen_token: str = "",
    qwen_timeout: float = 180.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """跑完整增强链路，返回 `(annotations, debug)`。

    `annotations` 与 `model_integration._run_inference_unlocked` 的返回同形，
    可直接交给既有的 `_save_annotations` writer。
    """
    from PIL import Image

    t0 = time.perf_counter()
    debug: Dict[str, Any] = {"image": str(image_path)}
    classes = [str(c) for c in (class_table or [])]
    debug["class_table"] = list(classes)

    if on_verify_fail not in VERIFY_FAIL_POLICIES:
        raise ValueError("on_verify_fail 必须是 %s 之一" % (VERIFY_FAIL_POLICIES,))

    # 类别表为空时**不要静默产出 0 框** —— 那是配置错误，必须让人看见。
    if not classes:
        raise RuntimeError(
            "自动标注增强链路需要 dataset 已有的类别表；当前为空。"
            "请先在数据集里配置类别（data.yaml），再跑增强标注。"
        )

    with Image.open(image_path) as im:
        img = im.convert("RGB")
    w, h = img.size
    debug["image_size"] = [w, h]

    cands: List[Candidate] = []

    # 1) Full Image → VLX
    if vlx:
        cands += _vlx_full_candidates(Path(image_path), text_prompt or "",
                                      classes, debug)

    # 2) Tile Pass → Qwen2.5-VL-3B
    if qwen_tiles:
        try:
            qwen = _make_qwen_client(node_id, qwen_port, qwen_token, qwen_timeout)
        except Exception as exc:  # noqa: BLE001
            debug["qwen_error"] = "%s: %s" % (type(exc).__name__, exc)
            LOG.warning("[chain] Qwen 客户端装配失败: %s", exc)
            qwen = None
        if qwen is not None:
            cands += _qwen_tile_candidates(img, text_prompt or "", classes, qwen,
                                           tile_size, overlap, min_tile,
                                           max_tiles, debug)

    debug["candidates_raw"] = len(cands)
    debug["sources"] = {
        SOURCE_VLX_FULL: sum(1 for c in cands if c.source == SOURCE_VLX_FULL),
        SOURCE_QWEN_TILE: sum(1 for c in cands if c.source == SOURCE_QWEN_TILE),
    }

    # 3) Candidate Merge（同类别 + 几何 + 边界；★不用 VLX 置信度★）
    merged = class_aware_merge(cands, iou_threshold=float(iou_threshold),
                               center_ratio=float(center_ratio),
                               prefer_full=True)
    debug["merged"] = len(merged)

    # 4) 几何去重
    merged = dedupe_boxes(merged, iou_threshold=float(dedupe_iou))
    debug["after_dedupe"] = len(merged)

    # 5) Verify
    if verify and merged:
        try:
            qwen = _make_qwen_client(node_id, qwen_port, qwen_token, qwen_timeout)
        except Exception as exc:  # noqa: BLE001
            debug["verify_error"] = "%s: %s" % (type(exc).__name__, exc)
            qwen = None
        if qwen is not None:
            merged = _verify_candidates(img, merged, classes, qwen,
                                        verify_context, on_verify_fail, debug)

    # 6) 类别归一化 → 输出
    annotations: List[Dict[str, Any]] = []
    for m in merged:
        label = normalize_class_with_alias(m.label, classes)
        if label is None:
            debug["unmatched_final"] = debug.get("unmatched_final", 0) + 1
            continue
        x1, y1, x2, y2 = clamp_box(m.bbox, int(w), int(h))
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            cls_id = classes.index(label)
        except ValueError:  # pragma: no cover —— normalize 已保证命中
            debug["unmatched_final"] = debug.get("unmatched_final", 0) + 1
            continue
        annotations.append({
            "type": "bbox",
            "class_id": int(cls_id),
            "class_name": label,
            # ★占位值，不是 detector 置信度★：VLX 的 1.0 是合成的，
            # Qwen 自报的数字按用户要求**不得**伪装成传统置信度。
            "confidence": 1.0,
            "confidence_source": "synthetic",
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            # provenance（writer 不消费，但 debug/预览可以看到）
            "sources": list(m.sources),
            "merge_reason": m.merge_reason,
            "verification": m.verification,
        })

    debug["annotations"] = len(annotations)
    debug["merged_detail"] = [m.to_dict() for m in merged]
    debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return annotations, debug


# ══════════════════════════════════════════════════════════════════════════
# Auto Label V1 —— 双 Teacher 并行检测 + 双向交叉校验 + ACCEPT / REVIEW
# ══════════════════════════════════════════════════════════════════════════
#
# 与 V0（上面那个 `run_chain`）的关系：
#   * V0 = 顺序检测 + merge 之后单向 Qwen Verify。**继续保留** ——
#     `model_integration.py` 的 `model_type == "vlx_qwen"` 分支仍在用它，
#     既有契约不破坏；
#   * V1 = Auto Label 正式链路，按拍定的主链路重排：
#
#       IMAGE ─┬─ VLX Full Image ──┐   ← ★并行★（不是 VLX 跑完 Qwen 才开始）
#              └─ Qwen Tile Pass ──┤
#                                  ↓
#              Qwen 验 VLX    /    VLX 验 Qwen        ← ★双向交叉校验★
#                                  ↓
#                            Candidate Merge
#                                  ↓
#              Class Conflict / Spatial Dedupe / Evidence Build
#                                  ↓
#                            Quality Decision
#                                  ↓
#                          ACCEPT ──┴── REVIEW
#
# ★★ 两条 Teacher 关系铁律（用户明确拍定）★★
#   1. VLX 与 Qwen 是**两个 Teacher**，不是"主模型 + 辅助模型"：
#      两者都负责发现目标，也都负责校验对方的候选。
#   2. 仍然**不用** WBF / 复杂置信度系统（见文件头约束 1、2）。
#      交叉校验的结果只用于 **ACCEPT/REVIEW 判定**与**证据留档**，
#      **绝不参与 NMS、抑制与排序**。
# ══════════════════════════════════════════════════════════════════════════

#: 质量决策的两种结果
DECISION_ACCEPT = "ACCEPT"
DECISION_REVIEW = "REVIEW"

#: ★反向校验阈值★：VLX 在裁剪区里检出的框与原候选框的 IoU 达到多少算"确认"。
#: 为什么比 merge 的 0.5 松：校验只问"这块里有没有这个东西"，VLX 在放大的
#: 裁剪图上给出的框通常比 Qwen 在 tile 里给的更贴合目标本体，框会整体偏移一点，
#: 用 0.5 会把大量真目标误判成"VLX 没看见"。
V1_VLX_VERIFY_IOU = 0.30

#: ★类别冲突阈值★：两个**不同类别**的框 IoU 超过多少，判定为"两个 Teacher
#: 对同一块区域给了不同答案" —— 这种必须进 REVIEW 让人看，机器不该替人选。
V1_CLASS_CONFLICT_IOU = 0.50

#: ★Qwen 整体不可用时的止损阈值★：tile pass 里连续失败这么多块就直接终止本轮
#: （前提是当前还没出过任何候选）。见 `_qwen_tile_candidates` 里的注释。
QWEN_ABORT_AFTER = 2


def _vlx_infer_crop(jpeg_bytes: bytes, prompt: str) -> List[List[float]]:
    """对一张**裁剪图**做 VLX 检测，返回 crop 像素 xyxy 列表。

    反向校验用：把 Qwen 给的候选框裁出来，问 VLX「这块里有没有 <类别>」。
    返回的坐标是 **crop 内**的像素，调用方负责加偏移回源图。

    失败返回 `[]`（不抛）—— 与 `QwenVLClient` 一致的失败语义：
    一次调用抖动不该让整张图的标注崩掉。
    """
    from lmm import config as _cfg
    from lmm.vlx_client import VLXSeekClient

    try:
        cfg = _cfg.load() or {}
        res = VLXSeekClient().infer_image(
            jpeg_bytes, prompt,
            task=cfg.get("vlx_seek_task") or "detection",
            lang=cfg.get("vlx_seek_lang") or "zh",
            max_new_tokens=int(cfg.get("vlx_seek_max_new_tokens") or 256),
            filename="verify_crop.jpg",
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[v1] VLX 反向校验调用失败: %s", exc)
        return []

    out: List[List[float]] = []
    for det in (res or {}).get("detections") or []:
        xyxy = det.get("bbox_xyxy") or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _vlx_verify_qwen_candidates(img, cands: List[Candidate],
                                iou_threshold: float = V1_VLX_VERIFY_IOU,
                                context: float = 1.5,
                                debug: Optional[Dict[str, Any]] = None
                                ) -> None:
    """★反向校验★：VLX 校验 Qwen 候选（就地写 `c.raw["xverify"]`）。

    Qwen 在 tile 上容易把水面反光、岸边石块看成目标；但反过来 VLX 看整图时
    又会漏掉小目标。让 VLX **只盯着这一小块**重检一次，正好补上彼此的短板 ——
    这就是"两个 Teacher 互相校验"的意义，而不是主模型带辅助模型。

    判定：crop 里 VLX 检出的框与原候选框 IoU >= 阈值 → `ok=True`。

    ★只标记，不删候选★：删与不删交给后面的 Quality Decision，
    这样"VLX 说没看见"这件事本身会**完整留在证据里**，事后调规则不用重跑模型。
    """
    dbg = debug if debug is not None else {}
    w, h = img.size
    n_ok = n_deny = n_fail = 0
    lat: List[float] = []

    for c in cands:
        if c.source != SOURCE_QWEN_TILE:
            continue
        t0 = time.perf_counter()
        cbox = crop_with_context(c.bbox, int(w), int(h), context=float(context))
        x0, y0, x1, y1 = (int(round(v)) for v in cbox)
        if x1 <= x0 or y1 <= y0:
            c.raw["xverify"] = {"by": "vlx", "ok": False, "error": "crop 退化"}
            n_fail += 1
            continue
        crop = img.crop((x0, y0, x1, y1))
        buf = BytesIO()
        try:
            crop.convert("RGB").save(buf, "JPEG", quality=92)
        except Exception as exc:  # noqa: BLE001
            c.raw["xverify"] = {"by": "vlx", "ok": False,
                                "error": "%s: %s" % (type(exc).__name__, exc)}
            n_fail += 1
            continue

        boxes = _vlx_infer_crop(buf.getvalue(), str(c.label))
        #: crop 坐标 → 源图坐标
        best = 0.0
        for b in boxes:
            ov = _iou_xyxy(c.bbox, (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0))
            if ov > best:
                best = ov
        ok = bool(boxes) and best >= float(iou_threshold)
        c.raw["xverify"] = {
            "by": "vlx",
            "ok": ok,
            "iou": round(best, 4),
            "n_det": len(boxes),
            "crop": [x0, y0, x1, y1],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        lat.append(round((time.perf_counter() - t0) * 1000.0, 1))
        if ok:
            n_ok += 1
        else:
            n_deny += 1

    dbg["vlx_verify_qwen"] = {"confirmed": n_ok, "denied": n_deny, "failed": n_fail,
                              "iou_threshold": float(iou_threshold)}
    if lat:
        lat.sort()
        dbg["vlx_verify_qwen"]["ms"] = {"p50": lat[len(lat) // 2], "max": lat[-1]}


def _qwen_verify_vlx_candidates(img, cands: List[Candidate],
                                class_table: Sequence[str], qwen,
                                context: float = 1.5,
                                debug: Optional[Dict[str, Any]] = None
                                ) -> None:
    """★正向校验★：Qwen 校验 VLX 候选（就地写 `c.raw["xverify"]`）。

    与 V0 的 `_verify_candidates` 的关键区别：**这里只标记不删除**。
    V0 在 merge 之后 verify，invalid 直接 drop；V1 要把"Qwen 说这不是"这件事
    留进证据 —— 否则事后想放宽规则，只能重跑一遍 Qwen。

    服务调用失败 → `ok=None`（未知），由 Quality Decision 按 `on_verify_fail`
    策略处理，绝不静默当成"通过"。
    """
    dbg = debug if debug is not None else {}
    w, h = img.size
    n_ok = n_deny = n_unmatched = n_fail = 0

    for c in cands:
        if c.source != SOURCE_VLX_FULL:
            continue
        cbox = crop_with_context(c.bbox, int(w), int(h), context=float(context))
        crop = img.crop(tuple(int(round(v)) for v in cbox))
        r = qwen.verify(crop, _verify_prompt(str(c.label)))

        if not r.get("ok"):
            n_fail += 1
            c.raw["xverify"] = {"by": "qwen", "ok": None,
                                "error": str(r.get("error", "verify 调用失败"))[:200]}
            continue

        js = r.get("json") or {}
        if not isinstance(js, dict) or not bool(js.get("valid")):
            n_deny += 1
            c.raw["xverify"] = {"by": "qwen", "ok": False,
                                "reason": str(js.get("reason", ""))[:200]}
            continue

        #: 模型自由生成的类别名**不直接信任**，必须再归一化一次
        cls = normalize_class_with_alias(js.get("class") or c.label, class_table)
        if cls is None:
            n_unmatched += 1
            c.raw["xverify"] = {"by": "qwen", "ok": False,
                                "unmatched_class": str(js.get("class"))[:120]}
            continue

        n_ok += 1
        c.raw["xverify"] = {"by": "qwen", "ok": True, "class": cls,
                            "reason": str(js.get("reason", ""))[:200]}
        if cls != c.label:
            #: 校验方给出了**不同但合法**的类别 → 记下来，决策阶段做冲突判定
            c.raw["xverify"]["class_differs"] = True

    dbg["qwen_verify_vlx"] = {"confirmed": n_ok, "denied": n_deny,
                              "unmatched": n_unmatched, "failed": n_fail}


def _detect_parallel(image_path: Path, img, text_prompt: str,
                     class_table: Sequence[str],
                     use_vlx: bool, use_qwen: bool,
                     tile_size: int, overlap: int, min_tile: int,
                     max_tiles: int,
                     node_id: str, qwen_port: int, qwen_token: str,
                     qwen_timeout: float,
                     debug: Dict[str, Any]) -> List[Candidate]:
    """★第一阶段：VLX Full Image 与 Qwen Tile Pass **真正并行**★。

    为什么必须并行（而不是 VLX 跑完再跑 Qwen）：
      * VLX 一次全图推理 ≈1.4 s，Qwen 一个 tile ≈数百 ms × 几十块 —— 两者都是
        **跨机网络等待**，CPU 基本空闲。串行等于把两份等待时间相加；
      * 两者**互不依赖**，没有任何数据依赖要求先算谁。

    为什么用线程池而不是 asyncio：本工程的模型客户端（`VLXSeekClient` /
    `QwenVLClient`）都是 **同步阻塞** 的（走 `tunnel_http`），没有 async 版本。
    用 `ThreadPoolExecutor(max_workers=2)` 让两条阻塞链路各占一个线程，
    效果与 `asyncio.gather` 完全一致，且不改动任何既有客户端。

    两个任务各自用独立的 debug 子字典，回来再合并 —— 避免两个线程同时写
    同一个 dict 造成键互相覆盖。
    """
    dbg_vlx: Dict[str, Any] = {}
    dbg_qwen: Dict[str, Any] = {}
    futures: Dict[str, Any] = {}

    qwen = None
    if use_qwen:
        try:
            qwen = _make_qwen_client(node_id, qwen_port, qwen_token, qwen_timeout)
        except Exception as exc:  # noqa: BLE001
            debug["qwen_error"] = "%s: %s" % (type(exc).__name__, exc)
            LOG.warning("[v1] Qwen 客户端装配失败: %s", exc)
            use_qwen = False

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="autolabel") as ex:
        if use_vlx:
            futures["vlx"] = ex.submit(_vlx_full_candidates, Path(image_path),
                                       text_prompt or "", class_table, dbg_vlx)
        if use_qwen and qwen is not None:
            futures["qwen"] = ex.submit(_qwen_tile_candidates, img,
                                        text_prompt or "", class_table, qwen,
                                        tile_size, overlap, min_tile, max_tiles,
                                        dbg_qwen)

        vlx_cands: List[Candidate] = []
        qwen_cands: List[Candidate] = []
        if "vlx" in futures:
            try:
                vlx_cands = futures["vlx"].result() or []
            except Exception as exc:  # noqa: BLE001
                debug["vlx_task_error"] = "%s: %s" % (type(exc).__name__, exc)
                LOG.warning("[v1] VLX 并行任务异常: %s", exc)
        if "qwen" in futures:
            try:
                qwen_cands = futures["qwen"].result() or []
            except Exception as exc:  # noqa: BLE001
                debug["qwen_task_error"] = "%s: %s" % (type(exc).__name__, exc)
                LOG.warning("[v1] Qwen 并行任务异常: %s", exc)

    debug["vlx"] = dbg_vlx
    debug["qwen"] = dbg_qwen
    debug["detect_parallel_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    debug["sources"] = {
        SOURCE_VLX_FULL: sum(1 for c in vlx_cands),
        SOURCE_QWEN_TILE: sum(1 for c in qwen_cands),
    }
    return list(vlx_cands) + list(qwen_cands)


def _find_class_conflicts(merged: List[MergedCandidate],
                          iou_threshold: float = V1_CLASS_CONFLICT_IOU
                          ) -> Dict[int, List[str]]:
    """★类别冲突检测★：两个**不同类别**的框高度重叠 → 两个 Teacher 对同一块
    区域给了不同答案。

    `class_aware_merge` 只合并**同类别**的候选，所以"VLX 说是 person、Qwen
    说是 boat"这种情况不会在 merge 里被消掉，而是留下两个重叠框。这里把它
    显式标出来：**机器不该替人选**，一律进 REVIEW。

    返回 `{merged 下标: [冲突的另一个类别, ...]}`，写进证据。
    """
    conflicts: Dict[int, List[str]] = {}
    n = len(merged)
    for i in range(n):
        for j in range(i + 1, n):
            if merged[i].label == merged[j].label:
                continue
            if _iou_xyxy(merged[i].bbox, merged[j].bbox) < float(iou_threshold):
                continue
            conflicts.setdefault(i, []).append(str(merged[j].label))
            conflicts.setdefault(j, []).append(str(merged[i].label))
    return conflicts


def _build_evidence(m: MergedCandidate, cands: List[Candidate],
                    conflicts: List[str]) -> Dict[str, Any]:
    """★证据构建★：把这一条框的"身世"完整记下来。

    `merge_ops.class_aware_merge` 会把参与合并的候选框原样存进
    `MergedCandidate.raw_boxes`（就是那些 Candidate 的 `bbox` 对象本身），
    所以这里可以用**精确相等**把校验结果捞回来，不用再去猜 IoU 配对。
    """
    by_box = {c.bbox: c for c in cands}
    teachers: List[str] = []
    xverify: List[Dict[str, Any]] = []
    for box in m.raw_boxes:
        c = by_box.get(box)
        if c is None:
            continue
        teachers.append("vlx" if c.source == SOURCE_VLX_FULL else "qwen")
        xv = c.raw.get("xverify")
        if isinstance(xv, dict):
            xverify.append(dict(xv))
    return {
        "sources": list(m.sources),
        "teachers": sorted(set(teachers)),
        "raw_boxes": [list(b) for b in m.raw_boxes],
        "merge_reason": m.merge_reason,
        "n_full": m.n_full,
        "n_tile": m.n_tile,
        "edge_only": bool(m.edge_only),
        "cross_verify": xverify,
        "class_conflicts": list(conflicts),
    }


def _decide(m: MergedCandidate, ev: Dict[str, Any],
            on_verify_fail: str = "review",
            teachers_down: Sequence[str] = ()) -> Tuple[str, List[str]]:
    """★质量决策★：ACCEPT 还是 REVIEW。

    规则（初版，全部可解释，不含任何加权分数）：

      * 两个 Teacher 都发现，且无类别冲突        → ACCEPT（互相印证）
      * 单个 Teacher 发现，对方校验**通过**      → ACCEPT
      * 单个 Teacher 发现，对方校验**否认**      → REVIEW
      * 单个 Teacher 发现，对方校验**失败/未知** → 按 `on_verify_fail`：
            `"review"`（默认，保守）/ `"accept"`（fail-open）
      * 存在类别冲突                            → REVIEW（机器不替人选）
      * `edge_only`（贴 tile 边界的半截目标）    → REVIEW

    ★刻意不做的事★：不把"两个 Teacher 都命中"折算成一个更高的分数再去排序 ——
    那正是用户明令禁止的"复杂置信度系统"。决策只产出**两档类别 + 理由列表**。
    """
    reasons: List[str] = []
    has_vlx = SOURCE_VLX_FULL in m.sources
    has_qwen = SOURCE_QWEN_TILE in m.sources

    if has_vlx and has_qwen:
        #: 两个 Teacher 都发现了它 —— 互相印证，最强的一种证据
        reasons.append("dual_teacher")
        decision = DECISION_ACCEPT

    else:
        #: 单 Teacher：看对方有没有把它校验通过
        oks = [bool(x.get("ok")) for x in (ev.get("cross_verify") or [])
               if isinstance(x, dict)]
        unknowns = [x for x in (ev.get("cross_verify") or [])
                    if isinstance(x, dict) and x.get("ok") is None]

        if any(oks):
            reasons.append("single_teacher+cross_confirmed")
            decision = DECISION_ACCEPT
        elif unknowns:
            #: 校验服务没给答案 —— 不等于"通过"，按策略保守处理。
            #: ★区分「对方服务整体不可用」与「单次调用抖动」★：前者是配置/运维问题，
            #: 理由必须写实（否则日志里只会看到一片 verify_unknown，看不出根因）。
            down = set(teachers_down or ())
            #: 本框由 vlx 发现 → 校验方是 qwen；反之亦然
            peer = "qwen" if has_vlx else "vlx"
            if peer in down:
                reasons.append("single_teacher+teacher_down:%s" % peer)
            else:
                reasons.append("single_teacher+verify_unknown")
            decision = DECISION_ACCEPT if on_verify_fail == "accept" else DECISION_REVIEW
        elif oks:
            #: 有校验结果，但**全部否认**
            reasons.append("single_teacher+cross_denied")
            decision = DECISION_REVIEW
        else:
            #: 根本没做校验（关掉了 cross_verify / 没有对方客户端）
            reasons.append("single_teacher+no_verify")
            decision = DECISION_ACCEPT if on_verify_fail == "accept" else DECISION_REVIEW

    # ── ★覆盖规则★：这两类一律压到 REVIEW，前面判成 ACCEPT 也不作数 ──────
    #: ① 两个 Teacher 对同一区域给了**不同类别** —— 机器不替人选
    if ev.get("class_conflicts"):
        reasons.append("class_conflict:%s" % ",".join(ev["class_conflicts"]))
        decision = DECISION_REVIEW
    #: ② 贴 tile 边界的半截目标 —— 框很可能是目标的一部分，不是完整目标
    if m.edge_only:
        reasons.append("edge_only")
        decision = DECISION_REVIEW

    return decision, reasons


def run_chain_v1(
    image_path: Path,
    text_prompt: str,
    class_table: Sequence[str],
    *,
    vlx: bool = True,
    qwen_tiles: bool = True,
    cross_verify: bool = True,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_tile: int = 64,
    max_tiles: int = DEFAULT_MAX_TILES,
    iou_threshold: float = 0.5,
    center_ratio: float = 0.5,
    dedupe_iou: float = 0.7,
    verify_context: float = 1.5,
    vlx_verify_iou: float = V1_VLX_VERIFY_IOU,
    class_conflict_iou: float = V1_CLASS_CONFLICT_IOU,
    on_verify_fail: str = "review",
    node_id: str = "",
    qwen_port: int = 0,
    qwen_token: str = "",
    qwen_timeout: float = 180.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """★Auto Label V1 主入口★。

    返回 `(annotations, debug)`。`annotations` 与
    `model_integration._run_inference_unlocked` 的返回同形（可直接交给既有
    `_save_annotations` writer），**额外**多了 `decision` 与 `evidence` 两个键：
    writer 不消费它们，但导出/留档会用到。

    ★★ 与 V0 `run_chain` 的三点实质差别 ★★
      1. 检测阶段**并行**（V0 串行）；
      2. 校验**双向**且**在 merge 之前**（V0 是 merge 之后单向 Qwen verify）；
      3. 输出**带 ACCEPT/REVIEW 决策与完整证据**（V0 只有 keep/drop）。
    """
    from PIL import Image

    t0 = time.perf_counter()
    debug: Dict[str, Any] = {"image": str(image_path), "version": "v1"}
    classes = [str(c) for c in (class_table or [])]
    debug["class_table"] = list(classes)

    if on_verify_fail not in ("review", "accept"):
        raise ValueError("on_verify_fail 必须是 'review' 或 'accept'")

    #: 类别表为空 = 配置错误，绝不静默产出 0 框
    if not classes:
        raise RuntimeError(
            "Auto Label V1 需要 dataset 已有的类别表；当前为空。"
            "请先在数据集里配置类别（data.yaml），再跑自动标注。"
        )

    with Image.open(image_path) as im:
        img = im.convert("RGB")
    w, h = img.size
    debug["image_size"] = [w, h]

    # ── 1) 并行检测 ─────────────────────────────────────────────────────
    cands = _detect_parallel(Path(image_path), img, text_prompt or "", classes,
                             bool(vlx), bool(qwen_tiles),
                             tile_size, overlap, min_tile, max_tiles,
                             node_id, qwen_port, qwen_token, qwen_timeout, debug)
    debug["candidates_raw"] = len(cands)

    # ── 2) 双向交叉校验（★merge 之前★）───────────────────────────────────
    if cross_verify and cands:
        #: 2a) Qwen 校验 VLX
        if any(c.source == SOURCE_VLX_FULL for c in cands):
            try:
                qwen = _make_qwen_client(node_id, qwen_port, qwen_token, qwen_timeout)
            except Exception as exc:  # noqa: BLE001
                debug["qwen_verify_error"] = "%s: %s" % (type(exc).__name__, exc)
                qwen = None
            if qwen is not None:
                _qwen_verify_vlx_candidates(img, cands, classes, qwen,
                                            float(verify_context), debug)
        #: 2b) VLX 校验 Qwen
        if any(c.source == SOURCE_QWEN_TILE for c in cands):
            _vlx_verify_qwen_candidates(img, cands,
                                        iou_threshold=float(vlx_verify_iou),
                                        context=float(verify_context), debug=debug)

    # ── 2c) Teacher 可用性总结 ──────────────────────────────────────────
    #: "这张图没目标"与"这个模型压根没在跑"必须能区分 —— 后者会让
    #: 单 Teacher 产出全部掉进 REVIEW，需要在 UI 上一次说清，而不是让人
    #: 逐框去读 `ChannelException`。
    down: List[str] = []
    #: 注意：两条模型的诊断写在**各自的 debug 子字典**里（`_detect_parallel` 刻意
    #: 让两个线程各写一份再合并，避免并发键覆盖），所以两个层级都要查。
    if debug.get("vlx_task_error") or (debug.get("vlx") or {}).get("vlx_error"):
        down.append("vlx")
    _qd = debug.get("qwen") or {}
    if (debug.get("qwen_error") or debug.get("qwen_task_error")
            or _qd.get("qwen_error") or _qd.get("qwen_unavailable")):
        down.append("qwen")
        #: 抬到顶层一份，前端只读 `debug` 的第一层就能给出提示
        if _qd.get("qwen_unavailable"):
            debug["qwen_unavailable"] = True
            debug["qwen_last_error"] = _qd.get("qwen_last_error", "")
    debug["teachers_down"] = down
    debug["teachers_up"] = [t for t in ("vlx", "qwen") if t not in down]

    # ── 3) Merge ────────────────────────────────────────────────────────
    merged = class_aware_merge(cands, iou_threshold=float(iou_threshold),
                               center_ratio=float(center_ratio), prefer_full=True)
    debug["merged"] = len(merged)

    # ── 4) 空间去重（同类别）────────────────────────────────────────────
    merged = dedupe_boxes(merged, iou_threshold=float(dedupe_iou))
    debug["after_dedupe"] = len(merged)

    # ── 5) 类别冲突 + 证据构建 + 质量决策 ────────────────────────────────
    conflicts = _find_class_conflicts(merged, float(class_conflict_iou))
    debug["class_conflicts"] = len(conflicts)

    annotations: List[Dict[str, Any]] = []
    n_accept = n_review = 0
    for idx, m in enumerate(merged):
        label = normalize_class_with_alias(m.label, classes)
        if label is None:
            debug["unmatched_final"] = debug.get("unmatched_final", 0) + 1
            continue
        x1, y1, x2, y2 = clamp_box(m.bbox, int(w), int(h))
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            cls_id = classes.index(label)
        except ValueError:  # pragma: no cover —— normalize 已保证命中
            debug["unmatched_final"] = debug.get("unmatched_final", 0) + 1
            continue

        ev = _build_evidence(m, cands, conflicts.get(idx) or [])
        decision, reasons = _decide(m, ev, on_verify_fail,
                                    debug.get("teachers_down") or ())
        if decision == DECISION_ACCEPT:
            n_accept += 1
        else:
            n_review += 1

        annotations.append({
            "type": "bbox",
            "class_id": int(cls_id),
            "class_name": label,
            # ★占位值，不是 detector 置信度★（见文件头约束 1）
            "confidence": 1.0,
            "confidence_source": "synthetic",
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            # ── V1 新增 ──
            "decision": decision,
            "decision_reasons": reasons,
            "evidence": ev,
        })

    debug["decision"] = {DECISION_ACCEPT: n_accept, DECISION_REVIEW: n_review}
    debug["annotations"] = len(annotations)
    debug["merged_detail"] = [m.to_dict() for m in merged]
    debug["candidates_detail"] = [
        {"bbox": [round(float(v), 2) for v in c.bbox], "label": c.label,
         "source": c.source, "edge": c.edge, "raw": c.raw}
        for c in cands
    ]
    debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return annotations, debug


# ══════════════════════════════════════════════════════════════════════════
# 通用双模型链路（标注页「一个模型标注 + 一个模型校验」）
# ══════════════════════════════════════════════════════════════════════════
#
# 与上面 V1（VLX + Qwen）**并列**，不是替代：
#   * V1 写死了两个 Teacher（VLX 全图 / Qwen tile），依赖 A6000 上的 Qwen 8023；
#   * 这一条接收**任意两个 model_id**（本机 yolo / rfdetr / yoloworld…，或 VLX），
#     由标注页自己选。用户明确要求「暂时不要再用 A6000 这台服务器」，所以
#     默认组合就是本机两个权重互检，VLX 只是**可选**而非必需。
#
# 链路（用户拍定：两步都做）：
#   两个模型各自跑**全图** → 类别感知 merge + 几何去重
#        → 只对**单方命中**的框，交给另一个模型在**裁剪图**上复核
#        → ACCEPT / REVIEW + 完整证据
#
# 复用原则：全部走 `model_manager._run_inference`（既有推理通道，带 runtime_lock），
# 只有 VLX 走 `lmm.vlx_client`，**不重建**任何推理代码。
# ══════════════════════════════════════════════════════════════════════════

#: ★裁剪复核阈值★：对方在裁剪图里检出的框与原候选框 IoU 达到多少算"确认"。
#: 比 merge 的 0.5 松 —— 复核只问"这块里有没有东西"，裁剪放大后框会整体偏移一点。
DUAL_VERIFY_IOU = 0.30

#: autoload 的互斥锁：两个线程同时 `_try_autoload` 会并发写 `loaded_models`。
_DUAL_LOAD_LOCK = threading.Lock()


def _teacher_error(dbg: Dict[str, Any]) -> str:
    """从 Teacher 的 debug 子字典里取出"这条模型整体不可用"的原因。

    不同后端把错误写在不同的键上（本地模型 `error`、VLX `vlx_error`、
    Qwen `qwen_error`），这里统一成一个出口，调用方不用逐个猜。
    """
    for k in ("error", "vlx_error", "qwen_error", "vlx_task_error", "qwen_task_error"):
        v = (dbg or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _resolve_teacher(model_id: str, debug: Dict[str, Any]) -> Dict[str, Any]:
    """把一个 model_id 准备到"可推理"状态（幂等 autoload），返回摘要。

    失败不抛 —— 与 `QwenVLClient` 一致的失败语义：一个 Teacher 挂了不该让
    整张图/整批崩掉，而是把原因记进证据里让 UI 如实显示。
    """
    info: Dict[str, Any] = {"id": str(model_id or ""), "ok": False, "type": "", "error": ""}
    if not model_id:
        info["error"] = "未指定模型"
        return info
    try:
        from config import settings
        mm = settings.model_manager
    except Exception as exc:  # noqa: BLE001
        info["error"] = "model_manager 不可用: %s" % exc
        return info

    try:
        with _DUAL_LOAD_LOCK:
            if model_id not in mm.loaded_models or not mm.loaded_models[model_id].get("model"):
                mm._try_autoload(model_id)
            loaded = (mm.loaded_models or {}).get(model_id) or {}
    except Exception as exc:  # noqa: BLE001
        info["error"] = "autoload 失败: %s: %s" % (type(exc).__name__, exc)
        return info

    info["type"] = str(loaded.get("type") or "")
    if not loaded.get("model"):
        info["error"] = "模型未加载: %s" % (loaded.get("error") or "unknown")
        return info
    info["ok"] = True
    return info


def _teacher_detect(model_id: str, image_file: Path, *,
                    text_prompt: str, label: str,
                    class_table: Sequence[str], confidence: float,
                    dataset_path: Optional[Path],
                    source: str, debug: Dict[str, Any]) -> List[Candidate]:
    """用**任意一个已注册模型**对一张图片做检测，返回 Candidate 列表。

    ★为什么不新建推理通道★
        本机 yolo / rfdetr / yoloworld / sam 统一走 `model_manager._run_inference`
        （它内部带 `runtime_lock`，与视频/批量推理共用同一把锁，不会并发踩
        predictor）；只有 VLX 走 `lmm.vlx_client`。两边返回的 annotation dict
        都归一成 xyxy + 归一化类别。

    失败一律返回 `[]` 并把原因写进 `debug`（见 `_teacher_error`）。
    """
    from config import settings

    out: List[Candidate] = []
    mm = settings.model_manager
    info = (mm.loaded_models or {}).get(model_id) or {}
    mtype = str(info.get("type") or "")

    #: VLX：开放词汇，类别名由 `label` 决定（不传才退回整句描述）
    if mtype == "vlx":
        try:
            got = _vlx_full_candidates(Path(image_file), text_prompt or "",
                                       class_table, debug,
                                       label_override=(label or "").strip() or None)
        except Exception as exc:  # noqa: BLE001
            debug["vlx_error"] = "%s: %s" % (type(exc).__name__, exc)
            return []
        for c in got:
            c.source = source
            c.raw["model_id"] = model_id
            c.raw["source"] = source
        return list(got)

    try:
        raw = mm._run_inference(
            info.get("model"), mtype, Path(image_file), float(confidence),
            text_prompt=(text_prompt or None),
            model_classes=(list(class_table) if mtype == "vlm" else info.get("classes") or []),
            target_label=(label or None) if mtype == "vlm" else None,
            dataset_path=None,
        )
    except Exception as exc:  # noqa: BLE001
        debug["error"] = "%s: %s" % (type(exc).__name__, exc)
        return []

    debug["raw"] = len(raw or [])
    for a in raw or []:
        if not isinstance(a, dict):
            continue
        xyxy = _ann_to_xyxy(a)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        #: ★模型自由生成的类别名不直接信任★ —— 映射不上就丢，绝不扩张类别表
        cls = normalize_class_with_alias(a.get("class_name") or a.get("label") or "",
                                         class_table)
        if cls is None:
            debug["unmatched"] = debug.get("unmatched", 0) + 1
            continue
        score = a.get("confidence")
        try:
            score = float(score) if score is not None else None
        except Exception:  # noqa: BLE001
            score = None
        out.append(Candidate(bbox=(x1, y1, x2, y2), label=cls, source=source,
                             score=score,   #: 只记录，绝不参与 NMS / 排序
                             raw={"model_id": model_id, "raw_class": str(a.get("class_name") or "")}))
    debug["candidates"] = len(out)
    return out


def _ann_to_xyxy(a: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """把 `_run_inference_unlocked` 的 annotation dict 统一成 xyxy。

    三种形态都要覆盖：bbox(x,y,w,h) / segmentation(多边形) / points(多边形)。
    """
    b = a.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            x, y, bw, bh = (float(v) for v in b)
        except Exception:  # noqa: BLE001
            return None
        return (x, y, x + bw, y + bh)

    pts = a.get("points")
    if pts is None:
        seg = a.get("segmentation")
        if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)):
            pts = seg[0]
    if isinstance(pts, (list, tuple)) and len(pts) >= 4:
        try:
            xs = [float(pts[i]) for i in range(0, len(pts) - 1, 2)]
            ys = [float(pts[i]) for i in range(1, len(pts), 2)]
        except Exception:  # noqa: BLE001
            return None
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _detect_dual_parallel(image_path: Path, *, model_a: str, model_b: str,
                          text_prompt: str, label: str,
                          class_table: Sequence[str], confidence: float,
                          dataset_path: Optional[Path],
                          debug: Dict[str, Any]) -> List[Candidate]:
    """两个模型**并行**各跑一次全图（互不依赖，都是等待型负载）。

    与 V1 `_detect_parallel` 同样的理由：模型客户端是同步阻塞的，用
    `ThreadPoolExecutor(max_workers=2)` 让两条链路各占一个线程。
    本机模型内部还有 `runtime_lock`，会自然串行 —— 那正是我们要的（保护
    predictor），远程模型（VLX）则真并行。
    """
    from PIL import Image

    dbg_a: Dict[str, Any] = {}
    dbg_b: Dict[str, Any] = {}
    futures: Dict[str, Any] = {}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dualdet") as ex:
        futures["a"] = ex.submit(_teacher_detect, model_a, Path(image_path),
                                 text_prompt=text_prompt, label=label,
                                 class_table=class_table, confidence=confidence,
                                 dataset_path=dataset_path,
                                 source=SOURCE_MODEL_A, debug=dbg_a)
        if model_b:
            futures["b"] = ex.submit(_teacher_detect, model_b, Path(image_path),
                                     text_prompt=text_prompt, label=label,
                                     class_table=class_table, confidence=confidence,
                                     dataset_path=dataset_path,
                                     source=SOURCE_MODEL_B, debug=dbg_b)

        cands: List[Candidate] = []
        for key, src in (("a", SOURCE_MODEL_A), ("b", SOURCE_MODEL_B)):
            if key not in futures:
                continue
            try:
                cands.extend(futures[key].result() or [])
            except Exception as exc:  # noqa: BLE001
                (dbg_a if key == "a" else dbg_b)["error"] = \
                    "%s: %s" % (type(exc).__name__, exc)
                LOG.warning("[dual] %s 并行任务异常: %s", key, exc)

    #: ★键名不能叫 model_a / model_b★ —— 那两个是**模型 id**（UI 要直接显示），
    #: 这里放的是各自的 debug 子字典，用 detect_* 区分，否则会把 id 覆盖掉。
    debug["detect_a"] = dbg_a
    debug["detect_b"] = dbg_b
    debug["detect_parallel_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    debug["sources"] = {
        SOURCE_MODEL_A: sum(1 for c in cands if c.source == SOURCE_MODEL_A),
        SOURCE_MODEL_B: sum(1 for c in cands if c.source == SOURCE_MODEL_B),
    }
    return cands


def _dual_verify(img, cands: List[Candidate], *,
                 model_a: str, model_b: str, label: str,
                 class_table: Sequence[str], confidence: float,
                 dataset_path: Optional[Path],
                 iou_threshold: float, context: float,
                 pair_iou: float = 0.5,
                 text_prompt: str = "",
                 debug: Dict[str, Any], control=None, progress=None) -> None:
    """★裁剪复核★：只被**单方**命中的框，交给另一个模型在裁剪图上再看一次。

    ★为什么双方都命中的框要跳过★
        两个模型在同一位置都出了框，merge 之后必然是 `dual_teacher` → ACCEPT，
        再花一次跨机/跨进程推理去问"你看见了吗"纯属浪费（批量几百张时这一条
        就是翻倍）。用 `pair_iou` 先做一次"对面有没有对上的框"的判断，命中就跳过。

    就地写 `c.raw["xverify"]`：
      * `ok=True`  对方在同一位置也看到了；
      * `ok=False` 对方看到了别处 / 什么都没看到；
      * `ok=None`  对方**服务不可用** —— 不等于否认，决策阶段单独处理。

    ★只标记，不删候选★：删与不删交给 Quality Decision，这样"对方说没看见"
    这件事本身会完整留在证据里，事后调规则不用重跑模型。
    """
    w, h = img.size
    n_ok = n_deny = n_fail = 0
    n_skip = 0
    lat: List[float] = []
    from_a = [c for c in cands if c.source == SOURCE_MODEL_A]
    from_b = [c for c in cands if c.source == SOURCE_MODEL_B]

    for index, c in enumerate(cands):
        from tile_pass import checkpoint
        checkpoint(control)
        if progress:
            progress({"phase": "verify", "completed": index, "total": len(cands)})
        #: 这个候选来自 A → 让 B 复核；来自 B → 让 A 复核
        peer = model_b if c.source == SOURCE_MODEL_A else model_a
        if not peer:
            continue
        #: ★对面已经在同一位置出过框 → 不必再问★
        others = from_b if c.source == SOURCE_MODEL_A else from_a
        if any(c.label == o.label and _iou_xyxy(c.bbox, o.bbox) >= float(pair_iou) for o in others):
            n_skip += 1
            continue

        t0 = time.perf_counter()
        cbox = crop_with_context(c.bbox, int(w), int(h), context=float(context))
        x0, y0, x1, y1 = (int(round(v)) for v in cbox)
        if x1 <= x0 or y1 <= y0:
            c.raw["xverify"] = {"by": peer, "ok": False, "error": "裁剪退化"}
            n_fail += 1
            continue

        crop = img.crop((x0, y0, x1, y1))
        tmp: Optional[Path] = None
        dbg: Dict[str, Any] = {}
        try:
            fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            tmp = Path(tmp_name)
            crop.convert("RGB").save(str(tmp), "JPEG", quality=92)
            # Keep the original target description, including attribute constraints, during crop verification.
            verification_prompt = (text_prompt or label or str(c.label)).strip()
            got = _teacher_detect(peer, tmp,
                                  text_prompt=verification_prompt,
                                  label=str(c.label),
                                  class_table=class_table, confidence=confidence,
                                  dataset_path=dataset_path,
                                  source="verify", debug=dbg)
        except Exception as exc:  # noqa: BLE001
            c.raw["xverify"] = {"by": peer, "ok": None,
                                "error": "%s: %s" % (type(exc).__name__, exc)}
            n_fail += 1
            continue
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except Exception:  # noqa: BLE001
                    pass

        err = _teacher_error(dbg)
        if err:
            #: ★"服务挂了"与"对方没看见"必须分开★ —— 前者不该被当成否认
            c.raw["xverify"] = {"by": peer, "ok": None, "error": err[:200]}
            n_fail += 1
            continue

        best, best_label, matched_box = 0.0, None, None
        for g in got:
            global_box = (g.bbox[0] + x0, g.bbox[1] + y0, g.bbox[2] + x0, g.bbox[3] + y0)
            ov = _iou_xyxy(c.bbox, global_box)
            if ov > best:
                best, best_label, matched_box = ov, g.label, global_box
        ok = bool(got) and best >= float(iou_threshold) and best_label == c.label
        xv: Dict[str, Any] = {"by": peer, "ok": ok, "iou": round(best, 4),
                              "n_det": len(got), "crop": [x0, y0, x1, y1],
                              "target_prompt": verification_prompt,
                              "matched_bbox": [round(float(v), 2) for v in matched_box] if matched_box else None,
                              "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        if best_label is not None and best_label != c.label:
            #: 复核方给了**不同但合法**的类别 → 记下来（不自动改，机器不替人选）
            xv["class_differs"] = str(best_label)
        c.raw["xverify"] = xv
        lat.append(xv["latency_ms"])
        if ok:
            n_ok += 1
        else:
            n_deny += 1

    debug["dual_verify"] = {"confirmed": n_ok, "denied": n_deny, "failed": n_fail,
                            "skipped_pair": n_skip,
                            "iou_threshold": float(iou_threshold)}
    if lat:
        lat.sort()
        debug["dual_verify"]["ms"] = {"p50": lat[len(lat) // 2], "max": lat[-1]}


def _build_evidence_dual(m: MergedCandidate, cands: List[Candidate],
                         conflicts: List[str],
                         model_a: str, model_b: str) -> Dict[str, Any]:
    """证据构建：把这一条框的"身世"完整记下来（谁发现的、对方怎么说）。

    `teachers` 直接记 **model_id**（比 V1 的 "vlx"/"qwen" 更具体）—— 标注页
    能选几十个模型，只写角色名事后根本查不出是哪两个权重在互相校验。
    """
    from tile_pass import observation
    members = m.extra.get("members") or [c for c in cands if c.label == m.label and c.bbox in m.raw_boxes]
    observations = []
    xverify = []
    for c in members:
        observations.extend(c.raw.get("observations") or [observation(c)])
        if isinstance(c.raw.get("xverify"), dict):
            xverify.append(dict(c.raw["xverify"]))
    return {
        "sources": list(m.sources),
        "teachers": sorted({str(c.raw.get("model_id") or c.source) for c in members}),
        "model_a": model_a, "model_b": model_b,
        "raw_boxes": [list(b) for b in m.raw_boxes],
        "merge_reason": m.merge_reason,
        "cross_verify": xverify, "class_conflicts": list(conflicts),
        "observations": observations,
        "passes": sorted({o["pass_type"] for o in observations}),
    }



def _decide_dual(m: MergedCandidate, ev: Dict[str, Any],
                 on_verify_fail: str = "review",
                 teachers_down: Sequence[str] = (),
                 verifier_present: bool = True) -> Tuple[str, List[str]]:
    """质量决策：ACCEPT 还是 REVIEW（与 V1 同构，只是角色名换成了 model_id）。

      * 两个模型都发现                    → ACCEPT（互相印证）
      * 单个模型发现，对方复核**通过**    → ACCEPT
      * 单个模型发现，对方复核**否认**    → REVIEW
      * 对方复核**未知/失败**             → 按 `on_verify_fail`
      * 压根没配校验模型                  → ACCEPT（就是普通单模型标注）
      * 存在类别冲突                      → REVIEW（机器不替人选）
    """
    reasons: List[str] = []
    has_a = SOURCE_MODEL_A in m.sources
    has_b = SOURCE_MODEL_B in m.sources

    if has_a and has_b:
        reasons.append("dual_teacher")
        decision = DECISION_ACCEPT
    else:
        oks = [bool(x.get("ok")) for x in (ev.get("cross_verify") or [])
               if isinstance(x, dict)]
        unknowns = [x for x in (ev.get("cross_verify") or [])
                    if isinstance(x, dict) and x.get("ok") is None]

        if any(oks):
            reasons.append("single_teacher+cross_confirmed")
            decision = DECISION_ACCEPT
        elif unknowns:
            #: ★区分「对方服务整体不可用」与「单次调用抖动」★
            peer_id = ev.get("model_b") if has_a else ev.get("model_a")
            if str(peer_id) in set(teachers_down or ()):
                reasons.append("single_teacher+teacher_down:%s" % peer_id)
            else:
                reasons.append("single_teacher+verify_unknown")
            decision = DECISION_ACCEPT if on_verify_fail == "accept" else DECISION_REVIEW
        elif oks:
            #: 有复核结果，但**全部否认**
            reasons.append("single_teacher+cross_denied")
            decision = DECISION_REVIEW
        else:
            #: 根本没做复核（没配校验模型 / 关掉了 cross_verify / 对方压根起不来）
            if not verifier_present:
                reasons.append("single_model_no_verifier")
                decision = DECISION_ACCEPT
            else:
                #: ★对方是"整体不可用"还是"这次没跑复核"必须分得清★ ——
                #: 前者是配置/运维问题，日志里必须能一眼看出来。
                peer_id = ev.get("model_b") if has_a else ev.get("model_a")
                if str(peer_id) in set(teachers_down or ()):
                    reasons.append("single_teacher+teacher_down:%s" % peer_id)
                else:
                    reasons.append("single_teacher+no_verify")
                decision = DECISION_ACCEPT if on_verify_fail == "accept" else DECISION_REVIEW

    # ── ★覆盖规则★：这一类一律压到 REVIEW，前面判成 ACCEPT 也不作数 ──────
    if ev.get("class_conflicts"):
        reasons.append("class_conflict:%s" % ",".join(str(x) for x in ev["class_conflicts"]))
        decision = DECISION_REVIEW

    return decision, reasons


def run_chain_dual(
    image_path: Path,
    text_prompt: str,
    class_table: Sequence[str],
    *,
    model_a: str,
    model_b: str = "",
    label: str = "",
    dataset_path: Optional[Path] = None,
    confidence_threshold: float = 0.5,
    cross_verify: bool = True,
    iou_threshold: float = 0.5,
    center_ratio: float = 0.5,
    dedupe_iou: float = 0.7,
    verify_context: float = 1.5,
    verify_iou: float = DUAL_VERIFY_IOU,
    class_conflict_iou: float = V1_CLASS_CONFLICT_IOU,
    on_verify_fail: str = "review",
    tile_pass: bool = False,
    tile_size: int = 640,
    tile_overlap: int = 128,
    tile_batch_size: int = 64,
    control=None, progress=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """★通用双模型主入口★（标注页「一个标注 + 一个校验」）。

    `model_a` = 标注模型（必填），`model_b` = 校验模型（空 = 不做交叉校验，
    等价于普通单模型标注）。

    返回 `(annotations, debug)`；`annotations` 与
    `model_integration._run_inference_unlocked` 同形（可直接交给既有
    `_save_annotations` / 批量任务），额外带 `decision` / `decision_reasons` /
    `evidence` 三个键（writer 不消费，留档与 UI 用）。

    失败语义：任何一步挂掉都只记 debug，不让调用方崩。
    唯一会抛的是**类别表为空**（配置错误，静默产出 0 框更糟）。
    """
    from PIL import Image

    from tile_pass import validate_options, checkpoint, collect_tiles, consolidate, strict_merge, review_overlapping_tile_boxes
    if tile_pass:
        validate_options(tile_size, tile_overlap, tile_batch_size)
    if model_b and model_a == model_b:
        raise ValueError("标注模型和校验模型必须不同；同一个模型不能当作两个独立 Teacher")
    checkpoint(control)
    t0 = time.perf_counter()
    debug: Dict[str, Any] = {"image": str(image_path), "version": "dual-tile-v1" if tile_pass else "dual",
                             "model_a": str(model_a or ""), "model_b": str(model_b or "")}
    classes = [str(c) for c in (class_table or [])]
    debug["class_table"] = list(classes)

    if on_verify_fail not in ("review", "accept"):
        raise ValueError("on_verify_fail 必须是 'review' 或 'accept'")

    #: 类别表为空 = 配置错误，绝不静默产出 0 框
    if not classes:
        raise RuntimeError(
            "双模型标注需要数据集已有的类别表；当前为空。"
            "请先在数据集里配置类别（data.yaml），再跑自动标注。"
        )
    if not str(model_a or "").strip():
        raise RuntimeError("双模型标注需要指定一个标注模型（model_a）。")

    #: ── 0) 两个 Teacher 各就各位（串行 autoload，避免并发写 loaded_models）──
    ta = _resolve_teacher(str(model_a), debug)
    tb = _resolve_teacher(str(model_b), debug) if str(model_b or "").strip() else \
        {"id": "", "ok": False, "type": "", "error": "未配置校验模型"}
    debug["teachers"] = {str(model_a): ta, str(model_b): tb}
    verifier_present = bool(str(model_b or "").strip())

    with Image.open(image_path) as im:
        img = im.convert("RGB")
    w, h = img.size
    debug["image_size"] = [w, h]

    # ── 1) 两个模型并行各跑一次全图 ────────────────────────────────────
    cands: List[Candidate] = []
    if ta.get("ok"):
        cands = _detect_dual_parallel(
            Path(image_path), model_a=str(model_a), model_b=str(model_b or ""),
            text_prompt=text_prompt or "", label=str(label or ""),
            class_table=classes, confidence=float(confidence_threshold),
            dataset_path=dataset_path, debug=debug)
    else:
        debug.setdefault("detect_a", {})["error"] = ta.get("error") or "标注模型不可用"
    for candidate in cands:
        candidate.raw["pass_type"] = "full"
    if tile_pass and ta.get("ok"):
        def detect_tile(path):
            dbg = {}
            got = _teacher_detect(str(model_a), path, text_prompt=text_prompt or "",
                label=str(label or ""), class_table=classes, confidence=float(confidence_threshold),
                dataset_path=None, source=SOURCE_MODEL_A, debug=dbg)
            return got, _teacher_error(dbg)
        def tile_progress(state):
            if progress:
                progress(dict(state, phase="tile"))
        tiled, td = collect_tiles(img, detect_tile, tile_size=tile_size,
            tile_overlap=tile_overlap, tile_batch_size=tile_batch_size,
            control=control, progress=tile_progress)
        cands.extend(tiled)
        debug["tile_pass"] = td
    elif tile_pass:
        debug["tile_pass"] = {"enabled": True, "coverage_complete": False,
            "error": ta.get("error"), "total": 0, "completed": 0}
    debug["candidates_raw"] = len(cands)
    if tile_pass:
        cands = consolidate(cands)
        debug["candidates_consolidated"] = len(cands)
    checkpoint(control)


    # ── 2) 对单方命中的框做裁剪复核（★merge 之前★）─────────────────────
    if cross_verify and verifier_present and cands and ta.get("ok") and tb.get("ok"):
        _dual_verify(img, cands, model_a=str(model_a), model_b=str(model_b),
                     label=str(label or ""), text_prompt=text_prompt or "", class_table=classes,
                     confidence=float(confidence_threshold),
                     dataset_path=dataset_path,
                     iou_threshold=float(verify_iou),
                     context=float(verify_context), debug=debug, control=control, progress=progress)

    # ── 3) Teacher 可用性总结 ──────────────────────────────────────────
    #: "这张图没目标"与"这个模型压根没在跑"必须能区分 —— 后者会让单 Teacher
    #: 产出全部掉进 REVIEW，需要在 UI 上一次说清，而不是让人逐框去读异常。
    down: List[str] = []
    if not ta.get("ok") or _teacher_error(debug.get("detect_a") or {}):
        down.append(str(model_a))
    if verifier_present and (not tb.get("ok") or _teacher_error(debug.get("detect_b") or {})):
        down.append(str(model_b))
    debug["teachers_down"] = down
    debug["teachers_up"] = [t for t in (str(model_a), str(model_b))
                            if t and t not in down]

    # ── 4) Merge + 几何去重 ────────────────────────────────────────────
    merged = (strict_merge(cands, float(iou_threshold)) if tile_pass else
              class_aware_merge(cands, iou_threshold=float(iou_threshold),
                                center_ratio=float(center_ratio), prefer_full=True))
    debug["merged"] = len(merged)
    if not tile_pass:
        merged = dedupe_boxes(merged, iou_threshold=float(dedupe_iou))
    debug["after_dedupe"] = len(merged)

    # ── 5) 类别冲突 + 证据 + 决策 ──────────────────────────────────────
    conflicts = _find_class_conflicts(merged, float(class_conflict_iou))
    debug["class_conflicts"] = len(conflicts)

    annotations: List[Dict[str, Any]] = []
    n_accept = n_review = 0
    for idx, m in enumerate(merged):
        lb = normalize_class_with_alias(m.label, classes)
        if lb is None:
            debug["unmatched_final"] = debug.get("unmatched_final", 0) + 1
            continue
        x1, y1, x2, y2 = clamp_box(m.bbox, int(w), int(h))
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            cls_id = classes.index(lb)
        except ValueError:  # pragma: no cover —— normalize 已保证命中
            debug["unmatched_final"] = debug.get("unmatched_final", 0) + 1
            continue

        ev = _build_evidence_dual(m, cands, conflicts.get(idx) or [],
                                  str(model_a), str(model_b))
        decision, reasons = _decide_dual(
            m, ev, on_verify_fail, debug.get("teachers_down") or (), verifier_present)
        if tile_pass and "full" not in ev["passes"]:
            confirmed = len(ev["teachers"]) > 1 or any(x.get("ok") is True for x in ev["cross_verify"])
            if not confirmed:
                decision = DECISION_REVIEW
                reasons.append("tile_only_requires_independent_confirmation")
            if m.edge_only:
                decision = DECISION_REVIEW
                reasons.append("tile_edge_only")
        if decision == DECISION_ACCEPT:
            n_accept += 1
        else:
            n_review += 1

        annotations.append({
            "type": "bbox",
            "class_id": int(cls_id),
            "class_name": lb,
            #: ★占位值★：两个模型的置信度不可比，不参与任何排序/阈值（见文件头约束 1）
            "confidence": 1.0,
            "confidence_source": "synthetic",
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "decision": decision,
            "decision_reasons": reasons,
            "evidence": ev,
        })

    if tile_pass:
        annotations, conflicts = review_overlapping_tile_boxes(annotations)
        debug["tile_overlap_review"] = conflicts
        n_accept = sum(a["decision"] == DECISION_ACCEPT for a in annotations)
        n_review = sum(a["decision"] == DECISION_REVIEW for a in annotations)
    debug["decision"] = {DECISION_ACCEPT: n_accept, DECISION_REVIEW: n_review}
    debug["annotations"] = len(annotations)
    debug["merged_detail"] = [m.to_dict() for m in merged]
    debug["candidates_detail"] = [
        {"bbox": [round(float(v), 2) for v in c.bbox], "label": c.label,
         "source": c.source, "model_id": (c.raw or {}).get("model_id"),
         "raw": c.raw}
        for c in cands
    ]
    debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    checkpoint(control)
    return annotations, debug
