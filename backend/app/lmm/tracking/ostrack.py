# -*- coding: utf-8 -*-
"""R4A：OSTrack-256 后端（`TrackerBackend` 的第一个实现）。

三条硬约束（第 31/32/33 节）：

  1. **模型只加载一次**。多目标 = 多个 template/state，不是多个模型；
     否则每来一个目标就多占 ~350 MB 显存，MX250 上第一个目标就会 OOM。
  2. **进出必须换算回 source pixel_xyxy**。内部是 128 模板 / 256 搜索的 crop，
     前端与事件统计永远不该看到这些比例。
  3. **异常不许外泄**。CUDA OOM / 权重缺失 / 数值异常一律转成
     `TrackerResult(ok=False)`，让上层退回"只有 VLX"的模式。

关于 timm：OSTrack 源码 import 了 timm 的几个工具函数，后端 venv 里没有 timm，
而 pip 装它有可能连带动 torch（会伤到 ultralytics 那条线），所以 `_vendor/timm`
提供了一份只含这几个符号的最小替身。见 `_vendor/timm/__init__.py` 的说明。
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import TrackerBackend, bbox_clip
from .types import TrackerResult

LOG = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "_vendor")


def _ensure_vendor_path() -> None:
    """把 vendor 目录临时放进 sys.path（**只在加载模型时**）。

    OSTrack 源码用的是 `from lib.models...` 这种绝对导入，不加路径就 import 不到。
    但**绝不能在模块导入期就加**：vendor 里有两个极其通用的包名（`lib` / `timm`），
    一旦插到 `sys.path[0]`，整个后端后续任何 `import lib` / `import timm` 都会
    拿到我们的替身 —— 而 `lmm.tracking` 是被 task_manager 在启动时 import 的，
    等于进程一起就污染了全局。所以只在真正要建模型的前一刻加。
    """
    if _VENDOR not in sys.path:
        sys.path.append(_VENDOR)

DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "workspace", "models", "ostrack", "vitb_256_mae_ce_32x4_ep300.pth.tar")


def _detect_sizes(state_dict: Dict[str, Any]) -> Tuple[int, int]:
    """从权重里反推 template / search 尺寸（不靠文件名）。

    `pos_embed_z` = 模板的 patch 网格（+可能的 cls），`pos_embed_x` = 搜索区的。
    网格边长 × patch(16) 就是像素尺寸 —— 写死 256 的话，哪天换成 384 的权重
    就会静默跑出一个"输入尺寸不匹配"的错，而错误信息完全指不到这里。
    """
    try:
        z = int(state_dict["backbone.pos_embed_z"].shape[1])
        x = int(state_dict["backbone.pos_embed_x"].shape[1])
        gz = int(round(math.sqrt(z)))
        gx = int(round(math.sqrt(x)))
        if gz * gz == z and gx * gx == x:
            return gz * 16, gx * 16
    except Exception:  # noqa: BLE001
        pass
    return 128, 256


def _build_cfg(template_size: int, search_size: int,
               template_factor: float = 2.0,
               search_factor: float = 4.0) -> SimpleNamespace:
    """OSTrack 的配置对象。原本是 yacs CfgNode，这里用 SimpleNamespace 顶替 ——
    我们只用到属性访问，不装 yacs 就少一个依赖。数值与
    `experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml` 逐项对齐。

    ★STEP4★ `template_factor` / `search_factor` 默认 **2.0 / 4.0**（= 官方 yaml），
    不给参数时与改造前逐字一致。之所以要能调：这两个值决定模板与搜索区**各带多少
    上下文**，而航拍小目标（实测 33×25 px）与官方评测集的目标尺度差一个量级 ——
    上下文带多了，模板里大部分是背景，跟踪器的框会被背景"撑大"（画面上就是
    "框巨大、不贴合"）。A/B 实测见 `.workbuddy/STEP4-*报告`。
    """
    return SimpleNamespace(
        DATA=SimpleNamespace(
            MEAN=[0.485, 0.456, 0.406], STD=[0.229, 0.224, 0.225],
            SEARCH=SimpleNamespace(SIZE=search_size, FACTOR=4.0,
                                   CENTER_JITTER=3, SCALE_JITTER=0.25),
            TEMPLATE=SimpleNamespace(SIZE=template_size, FACTOR=2.0,
                                     CENTER_JITTER=0, SCALE_JITTER=0),
        ),
        MODEL=SimpleNamespace(
            PRETRAIN_FILE="", EXTRA_MERGER=False, RETURN_INTER=False,
            RETURN_STAGES=[0],
            BACKBONE=SimpleNamespace(
                TYPE="vit_base_patch16_224_ce", STRIDE=16,
                CE_LOC=[3, 6, 9], CE_KEEP_RATIO=[0.7, 0.7, 0.7],
                CE_TEMPLATE_RANGE="CTR_POINT", CAT_MODE="direct", SEP_SEG=False),
            HEAD=SimpleNamespace(TYPE="CENTER", NUM_CHANNELS=256),
            HIDDEN_DIM=256,
        ),
        TRAIN=SimpleNamespace(DROP_PATH_RATE=0.1),
        TEST=SimpleNamespace(
            TEMPLATE_SIZE=template_size, SEARCH_SIZE=search_size,
            #: （`DATA.*.FACTOR` 是训练期 jitter，推理用不到，保持 4.0/2.0 不动）
            TEMPLATE_FACTOR=float(template_factor),
            SEARCH_FACTOR=float(search_factor), EPOCH=300),
    )


class OSTrackBackend(TrackerBackend):
    """OSTrack-256（ViT-Base + candidate elimination）。"""

    name = "ostrack-256"

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None,
                 fp16: bool = True, template_size: Optional[int] = None,
                 search_size: Optional[int] = None,
                 template_factor: Optional[float] = None,
                 search_factor: Optional[float] = None) -> None:
        self.checkpoint = checkpoint or DEFAULT_CKPT
        self.fp16 = bool(fp16)
        self.device = device or ("cuda" if self._cuda_ok() else "cpu")
        self._lock = threading.RLock()
        self._states: Dict[int, Dict[str, Any]] = {}
        self._load_error: Optional[str] = None
        self._load_s: float = 0.0
        self._updates = 0
        self._lat_sum = 0.0
        self.model = None
        self.cfg = None
        #: ★STEP4★ 不传就是官方值（2.0 / 4.0），行为与改造前逐字一致
        self.template_factor = float(template_factor or 2.0)
        self.search_factor = float(search_factor or 4.0)
        self.template_size = int(template_size or 0)
        self.search_size = int(search_size or 0)
        self._load()

    # ── 加载 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    def _load(self) -> None:
        t0 = time.time()
        _ensure_vendor_path()
        try:
            import torch
            from lib.models.layers.head import build_box_head
            from lib.models.ostrack.ostrack import OSTrack
            from lib.models.ostrack.vit_ce import vit_base_patch16_224_ce
            from lib.test.utils.hann import hann2d
            from lib.test.tracker.data_utils import Preprocessor
            from lib.train.data.processing_utils import sample_target
            from lib.utils.box_ops import clip_box
            from lib.utils.ce_utils import generate_mask_cond
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"import:{type(exc).__name__}:{exc}"
            LOG.exception("[OSTrack] 依赖导入失败")
            return

        try:
            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(self.checkpoint)
            ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            net = ckpt.get("net") if isinstance(ckpt, dict) else ckpt
            if net is None:
                raise ValueError("权重文件里没有 'net' 键")
            tsz, ssz = _detect_sizes(net)
            self.template_size = int(self.template_size or tsz)
            self.search_size = int(self.search_size or ssz)
            cfg = _build_cfg(self.template_size, self.search_size,
                             self.template_factor, self.search_factor)

            backbone = vit_base_patch16_224_ce(
                pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO)
            backbone.finetune_track(cfg=cfg, patch_start_index=1)
            box_head = build_box_head(cfg, backbone.embed_dim)
            model = OSTrack(backbone, box_head, aux_loss=False, head_type=cfg.MODEL.HEAD.TYPE)
            missing, unexpected = model.load_state_dict(net, strict=True)
            if missing or unexpected:
                LOG.warning("[OSTrack] 权重加载有差异 missing=%d unexpected=%d",
                            len(missing), len(unexpected))
            model.eval()
            if self.device == "cuda":
                model = model.cuda()
                if self.fp16:
                    model = model.half()
            self.model = model

            feat_sz = self.search_size // cfg.MODEL.BACKBONE.STRIDE
            self.feat_sz = int(feat_sz)
            self.output_window = hann2d(
                torch.tensor([feat_sz, feat_sz]).long(), centered=True)
            if self.device == "cuda":
                self.output_window = self.output_window.cuda()
            if self.fp16 and self.device == "cuda":
                self.output_window = self.output_window.half()

            self.cfg = cfg
            self._sample_target = sample_target
            self._preprocessor = Preprocessor()
            self._clip_box = clip_box
            self._generate_mask_cond = generate_mask_cond
            self._torch = torch
            self._warmup()
            self._load_s = round(time.time() - t0, 3)
            LOG.info("[OSTrack] 已加载 %s（template=%d search=%d device=%s fp16=%s，%.2fs）",
                     os.path.basename(self.checkpoint), self.template_size,
                     self.search_size, self.device, self.fp16, self._load_s)
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            LOG.exception("[OSTrack] 模型加载失败")

    def _warmup(self) -> None:
        """烧掉 CUDA / cuDNN 的首次调用开销。

        实测（MX250，`.workbuddy/_r4a_ostrack_bench_out.txt`）：**第一次 update
        要 1.5~1.7 s**，之后稳定在 130 ms 左右。这个尖峰来自 cuDNN 卷积算法搜索
        与 kernel 首次加载，跟输入内容无关。不预热的话，第一个目标的头几帧会
        直接撞上前端的 stale 超时、看起来像"跟踪丢了"。
        """
        if self.model is None:
            return
        try:
            side = self.search_size
            dummy = np.zeros((side, side, 3), dtype=np.uint8)
            box = (side * 0.3, side * 0.3, side * 0.6, side * 0.6)
            tid = -1
            r = self.initialize(tid, dummy, box)
            if r.ok:
                self.update(tid, dummy)
                self.update(tid, dummy)
            self.reset(tid)
        except Exception as exc:  # noqa: BLE001
            #: 预热失败不算加载失败——顶多第一次 update 慢一点
            LOG.debug("[OSTrack] 预热跳过：%s", exc)

    # ── 工具 ──────────────────────────────────────────────────────────────
    def _to_rgb(self, frame: Any) -> Any:
        """BGR → RGB。OSTrack 的归一化常量是 ImageNet（RGB 顺序）。"""
        if frame is None:
            return None
        if getattr(frame, "ndim", 3) == 3 and frame.shape[2] >= 3:
            return frame[:, :, ::-1]
        return frame

    def _tensorize(self, patch: Any, amask: Any):
        t = self._preprocessor.process(patch, amask)
        return t.tensors

    def _dtype(self):
        return self._torch.float16 if (self.fp16 and self.device == "cuda") else self._torch.float32

    def _pref(self, x):
        torch = self._torch
        return x.cuda().to(self._dtype()) if self.device == "cuda" else x

    # ── TrackerBackend ────────────────────────────────────────────────────
    def initialize(self, track_id: int, frame: Any, bbox_xyxy: Sequence[float]) -> TrackerResult:
        t0 = time.time()
        if self.model is None:
            return TrackerResult(track_id, ok=False, latency_ms=(time.time() - t0) * 1000,
                                 reason=self._load_error or "OSTrack 未加载")
        try:
            im = self._to_rgb(frame)
            H, W = im.shape[0], im.shape[1]
            x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
            box_xywh = [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]
            z_patch, resize_factor, z_amask = self._sample_target(
                im, box_xywh, self.cfg.TEST.TEMPLATE_FACTOR, output_sz=self.template_size)
            z = self._pref(self._tensorize(z_patch, z_amask))

            box_mask_z = None
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                torch = self._torch
                tb = self._xyxy_to_crop_tensor(box_xywh, box_xywh, resize_factor,
                                               self.template_size, torch)
                box_mask_z = self._generate_mask_cond(self.cfg, 1, z.device, tb)
            with self._lock:
                self._states[track_id] = {
                    "z": z, "box_mask_z": box_mask_z, "state": box_xywh,
                    "size": (H, W), "frame_id": 0,
                }
            return TrackerResult(track_id, bbox_xyxy=(x1, y1, x2, y2), score=1.0, ok=True,
                                 latency_ms=(time.time() - t0) * 1000.0,
                                 extra={"template_size": self.template_size,
                                        "resize_factor": resize_factor})
        except Exception as exc:  # noqa: BLE001
            LOG.debug("[OSTrack] initialize 失败 track=%s: %s", track_id, exc)
            return TrackerResult(track_id, ok=False, latency_ms=(time.time() - t0) * 1000,
                                 reason=f"{type(exc).__name__}: {exc}")

    def update(self, track_id: int, frame: Any) -> TrackerResult:
        t0 = time.time()
        st = self._states.get(track_id) if self.model is not None else None
        if self.model is None or st is None:
            return TrackerResult(track_id, ok=False, latency_ms=(time.time() - t0) * 1000,
                                 reason=self._load_error or "track 未初始化")
        try:
            torch = self._torch
            im = self._to_rgb(frame)
            H, W = im.shape[0], im.shape[1]
            x_patch, resize_factor, x_amask = self._sample_target(
                im, st["state"], self.cfg.TEST.SEARCH_FACTOR, output_sz=self.search_size)
            x = self._pref(self._tensorize(x_patch, x_amask))
            #: dtype 必须在**进模型前**统一：fp16 模型 + float 输入会报
            #: "Input type (float) and bias type (Half)"，而错误信息完全指不到
            #: 是模板还是搜索区，所以三样（模板 / 搜索 / CE mask）逐个对齐。
            z = st["z"]
            if hasattr(z, "to") and z.dtype != x.dtype:
                z = z.to(x.dtype)
                st["z"] = z
            #: CE 模板掩码**绝不能跟着转 dtype**：CandidateElimination 里是
            #: `attn_t[box_mask_z]` 张量索引，只接受 bool/整型，转 fp16 会直接
            #: "tensors used as indices must be long, int, byte or bool tensors"。
            mask = st["box_mask_z"]
            if mask is not None and hasattr(mask, "dtype") and mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            with torch.no_grad():
                out = self.model(template=z, search=x, ce_template_mask=mask)
            score_map = out["score_map"]
            response = self.output_window * score_map
            pred_boxes = self.model.box_head.cal_bbox(
                response, out["size_map"], out["offset_map"]).view(-1, 4)
            pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
            new_state = self._map_box_back(pred_box, st["state"], resize_factor)
            new_state = self._clip_box(list(new_state), H, W, margin=10)
            st["state"] = new_state
            st["size"] = (H, W)
            bbox = (new_state[0], new_state[1],
                    new_state[0] + new_state[2], new_state[1] + new_state[3])
            score = float(response.max().item()) if hasattr(response, "max") else 0.0
            with self._lock:
                self._updates += 1
                self._lat_sum += (time.time() - t0) * 1000.0
            return TrackerResult(track_id, bbox_xyxy=bbox, score=score, ok=True,
                                 latency_ms=(time.time() - t0) * 1000.0,
                                 extra={"resize_factor": resize_factor,
                                        "feat_sz": self.feat_sz})
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[OSTrack] update 失败 track=%s: %s", track_id, exc, exc_info=True)
            return TrackerResult(track_id, ok=False, latency_ms=(time.time() - t0) * 1000,
                                 reason=f"{type(exc).__name__}: {exc}")

    def reset(self, track_id: int) -> None:
        with self._lock:
            self._states.pop(track_id, None)

    def remove(self, track_id: int) -> None:
        self.reset(track_id)

    def shift_state(self, track_id: int, dx: float, dy: float) -> bool:
        """把这条 track 的内部状态（上一帧框）在当前图坐标系里平移 (dx, dy)。

        ★STEP 2.1★ 为什么需要它：远程服务为了省上传带宽，客户端只传**目标周围
        的 crop** 而不是 1080p 全图（745 KB → 29 KB，upload 2428 ms → 1 ms）。
        但 tracker 的状态是"上一帧框"，它记的是**上一次那张图**的坐标系；本次
        crop 的原点变了，状态必须先跟着平移，否则 OSTrack 会拿错位置去采样搜索
        区 —— 表现是"框瞬间跳到画面角落然后丢失"。

        约定：调用方保证 tracker 状态当前在 `prev` 坐标系，目标是平移到
        `prev + (dx, dy)` 对应的坐标系；**宽度高度不变**，只动左上角。
        返回 False 表示这条 track 不存在（调用方应视为"需要先 initialize"）。
        """
        with self._lock:
            st = self._states.get(track_id)
            if st is None:
                return False
            s = st.get("state")
            if not s or len(s) < 4:
                return False
            st["state"] = [float(s[0]) - float(dx), float(s[1]) - float(dy),
                           float(s[2]), float(s[3])]
            return True

    def health(self) -> Dict[str, Any]:
        h: Dict[str, Any] = {
            "ok": self.model is not None,
            "name": self.name,
            "device": self.device,
            "fp16": self.fp16,
            "checkpoint": os.path.basename(self.checkpoint) if self.checkpoint else "",
            "template_size": self.template_size,
            "search_size": self.search_size,
            #: ★STEP4★ 让 `GET /health` 能直接看出当前跑的是哪一组 factor
            "template_factor": self.template_factor,
            "search_factor": self.search_factor,
            "load_seconds": self._load_s,
            "tracks": len(self._states),
            "error": self._load_error,
        }
        with self._lock:
            if self._updates:
                h["updates"] = self._updates
                h["avg_update_ms"] = round(self._lat_sum / self._updates, 2)
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                h["vram_free_mb"] = round(free / 1048576.0, 1)
                h["vram_total_mb"] = round(total / 1048576.0, 1)
        except Exception:  # noqa: BLE001
            pass
        return h

    def track_ids(self) -> List[int]:
        with self._lock:
            return list(self._states.keys())

    def close(self) -> None:
        with self._lock:
            self._states.clear()
        # 模型权重是共享的，**不在这里释放**：下一条任务还要用；真正要释放时
        # 由进程退出或显存保护逻辑处理（第 31 节：不重复加载就已经够了）。

    # ── 坐标换算（source ↔ crop）────────────────────────────────────────
    def _xyxy_to_crop_tensor(self, box_xywh, box_extract, resize_factor, crop_sz, torch):
        """把框从原图换到 crop 内（OSTrack `transform_bbox_to_crop` 的等价实现）。"""
        ex_cx = box_extract[0] + 0.5 * box_extract[2]
        ex_cy = box_extract[1] + 0.5 * box_extract[3]
        cx = box_xywh[0] + 0.5 * box_xywh[2]
        cy = box_xywh[1] + 0.5 * box_xywh[3]
        out_cx = (crop_sz - 1) / 2 + (cx - ex_cx) * resize_factor
        out_cy = (crop_sz - 1) / 2 + (cy - ex_cy) * resize_factor
        w = box_xywh[2] * resize_factor
        h = box_xywh[3] * resize_factor
        t = torch.tensor([[out_cx - 0.5 * w, out_cy - 0.5 * h, w, h]], dtype=torch.float32)
        return t

    def _map_box_back(self, pred_box, state, resize_factor):
        """crop 坐标 → 原图 xywh（与 OSTrack 的 `map_box_back` 一字不差）。"""
        cx_prev = state[0] + 0.5 * state[2]
        cy_prev = state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
