"""Custom ReID encoders for models Ultralytics cannot load on its own.

Ultralytics' native ReID (``ultralytics/trackers/utils/reid.py``) only understands:
  * ``*.pt``  -> a YOLO checkpoint (loaded via ``YOLO()``, embeddings from ``embed=[...]``)
  * other ext -> ``AutoBackend`` (ONNX / TorchScript / TensorRT / OpenVINO …), needs matching runtimes

Real-world ReID weights are often neither: torchreid's **OSNet** ships as a bare
``state_dict``, and many pipelines keep a small **projection head** (e.g. on top of DINOv3)
as a separate ``.pth``. This module rebuilds those architectures from the state dict and
exports them as **TorchScript**, so Ultralytics can load them through ``AutoBackend``
without patching anything inside the package.

Public API
    inspect_weights(path)          -> dict describing what a checkpoint is
    build_encoder_from_sd(sd, ...) -> nn.Module (embedding extractor) or None
    export_encoder(ckpt, out)      -> path to the .torchscript file (or None)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Built-in backbones ───────────────────────────────────────────────────────
# The appearance slot is Backbone + Projection Head. The head is always imported (it is
# trained on the user's own data), but a backbone is usually a standard trunk, so
# the common ones are offered here instead of asking the user to hunt for files.
#
# `source` says where the weights come from:
#   torchvision : built and downloaded by torchvision (no extra dependency)
#   hub         : torch.hub repo (needs GitHub access the first time)
#   torchreid   : torchreid's model zoo (needs the `torchreid` package)
# `out_dim` is only a label; the real width is always measured after building.

BUILTIN_BACKBONES: List[Dict[str, Any]] = [
    {"id": "osnet_x0_25", "family": "OSNet", "label": "OSNet x0.25",
     "source": "torchreid", "model": "osnet_x0_25", "out_dim": 512,
     "note": "轻量行人 ReID Backbone，256×128 输入"},
    {"id": "osnet_ain_x0_25", "family": "OSNet", "label": "OSNet-AIN x0.25",
     "source": "torchreid", "model": "osnet_ain_x0_25", "out_dim": 512,
     "note": "带注意力（AIN）的 OSNet 变体"},
    {"id": "mobilenetv3_small", "family": "Backbone", "label": "MobileNetV3-Small",
     "source": "torchvision", "fn": "mobilenet_v3_small", "out_dim": 576,
     "note": "最快，适合 CPU / 边缘设备"},
    {"id": "resnet50", "family": "Backbone", "label": "ResNet50",
     "source": "torchvision", "fn": "resnet50", "out_dim": 2048,
     "note": "通用基线，特征维度高"},
    {"id": "convnext_tiny", "family": "Backbone", "label": "ConvNeXt-Tiny",
     "source": "torchvision", "fn": "convnext_tiny", "out_dim": 768,
     "note": "卷积版 ViT，精度/速度均衡"},
    # DINOv3 trunks ship as local files: the weights were downloaded once and are
    # referenced by path, so they behave like any other built-in (no license step).
    {"id": "dinov3_vits16", "family": "Backbone", "label": "DINOv3 ViT-S/16",
     "source": "local", "model": "dinov3_vits16",
     "weights": r"D:\AI\models\DINOv3_ViT-S16.safetensors",
     "out_dim": 384, "vit": True, "note": "自监督 ViT，小模型（本地权重）"},
    {"id": "dinov3_vitb16", "family": "Backbone", "label": "DINOv3 ViT-B/16",
     "source": "local", "model": "dinov3_vitb16",
     "weights": r"D:\AI\models\vit_base_patch16_dinov3.lvd1689m.safetensors",
     "out_dim": 768, "vit": True, "note": "自监督 ViT，基础模型（本地权重）"},
]

BUILTIN_PREFIX = "builtin:"


def builtin_backbone_spec(spec_id: str) -> Optional[Dict[str, Any]]:
    sid = (spec_id or "").strip()
    if sid.startswith(BUILTIN_PREFIX):
        sid = sid[len(BUILTIN_PREFIX):]
    return next((s for s in BUILTIN_BACKBONES if s["id"] == sid), None)


_GITHUB_OK: Optional[bool] = None


def _github_reachable() -> bool:
    """One-shot, process-cached reachability probe (torch.hub clones from GitHub)."""
    global _GITHUB_OK
    if _GITHUB_OK is None:
        try:
            import urllib.request
            urllib.request.urlopen("https://github.com", timeout=4)
            _GITHUB_OK = True
        except Exception:
            _GITHUB_OK = False
    return _GITHUB_OK


_DRIVE_OK: Optional[bool] = None


def _drive_reachable() -> bool:
    """torchreid keeps its OSNet weights on Google Drive; probe once per process."""
    global _DRIVE_OK
    if _DRIVE_OK is None:
        try:
            import urllib.request
            urllib.request.urlopen("https://drive.google.com", timeout=5)
            _DRIVE_OK = True
        except Exception:
            _DRIVE_OK = False
    return _DRIVE_OK


def builtin_backbone_available(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """Whether a built-in can actually be built here, and why not when it can't."""
    src = spec.get("source")
    if src == "local":
        w = spec.get("weights") or ""
        if not w or not Path(w).is_file():
            return False, f"权重文件不存在：{w or '(未配置)'}"
        return True, f"本地权重：{Path(w).name}"
    if src == "torchvision":
        try:
            import importlib.util
            if importlib.util.find_spec("torchvision") is None:
                return False, "未安装 torchvision"
        except Exception:
            return False, "未安装 torchvision"
        return True, "首次使用会自动下载预训练权重"
    if src == "torchreid":
        try:
            import importlib
            importlib.import_module("torchreid")
        except Exception:
            return False, "未安装 torchreid；请手动导入 OSNet 权重文件"
        if not _drive_reachable():
            return False, "OSNet 权重托管在 Google Drive，当前网络访问不到；请手动导入权重文件"
        return True, "权重来自 torchreid 模型库"
    return False, f"未知的构建方式：{src}"


def _strip_classifier(net: nn.Module) -> nn.Module:
    """Drop a torchvision trunk's classification head so it returns embeddings.

    The whole head is replaced, not just its last Linear: MobileNetV3's head is
    `Linear(576→1024) → Hardswish → Dropout → Linear(1024→1000)`, so removing only
    the last layer would leave a 1024-d vector that is neither the embedding nor a
    logit. Whatever precedes the head (GAP, flatten, LayerNorm2d) still runs, so the
    adapter's `_as_2d()` finishes the job when the result is still a feature map.
    """
    for attr in ("fc", "head", "classifier"):
        if hasattr(net, attr):
            setattr(net, attr, nn.Identity())
            break
    return net


def build_builtin_backbone(spec_id: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate a built-in backbone with pretrained weights; measure its width."""
    spec = builtin_backbone_spec(spec_id)
    if spec is None:
        raise RuntimeError(f"未知的 Backbone：{spec_id}")
    ok, why = builtin_backbone_available(spec)
    if not ok:
        raise RuntimeError(f"{spec['label']} 不可用：{why}")

    src = spec["source"]
    if src == "torchvision":
        from torchvision import models as tvm
        net = getattr(tvm, spec["fn"])(weights="DEFAULT")
        net = _strip_classifier(net)
    elif src == "local":
        # Weights are already on disk: build the matching trunk and load them in,
        # so nothing is downloaded and no license step is involved.
        net = build_dinov3(spec.get("weights") or "", spec.get("out_dim"))
    elif src == "hub":
        net = torch.hub.load(spec["repo"], spec["model"], pretrained=True, verbose=False)
    elif src == "torchreid":
        import torchreid
        net = torchreid.models.build_model(name=spec["model"], num_classes=1000,
                                           pretrained=True)
    else:
        raise RuntimeError(f"未知的构建方式：{src}")

    net.eval()
    dim = probe_out_dim(BackboneAdapter(net))
    if dim:
        spec = dict(spec, out_dim=dim)
    return net, spec


def builtin_backbone_status(spec: Dict[str, Any]) -> Tuple[str, str]:
    """Catalogue state of a built-in: `ready` | `unavailable`.

    `ready` covers both trunks whose weights are already on disk (DINOv3) and ones
    that can be fetched on first use; `unavailable` means the trunk cannot be built
    here at all, with `why` naming the concrete blocker.
    """
    src = spec.get("source")
    if src == "local":
        # Weights already on disk: nothing to download, nothing to authorise.
        w = spec.get("weights") or ""
        if not w or not Path(w).is_file():
            return "unavailable", f"权重文件不存在：{w or '(未配置)'}"
        return "ready", f"本地权重：{Path(w).name}"
    if src == "torchreid":
        # Import for real: torchreid pulls in gdown / tensorboard lazily, so merely
        # finding the package is not enough to know a trunk can be built.
        try:
            import importlib
            importlib.import_module("torchreid")
        except Exception as e:
            return "unavailable", (f"torchreid 不可用（{type(e).__name__}）；"
                                   f"也可直接导入 OSNet 权重文件（.pth），"
                                   f"程序会按权重自动重建网络并导出 TorchScript")
        if not _drive_reachable():
            return "unavailable", ("权重托管在 Google Drive，当前网络访问不到；"
                                   "请直接导入 OSNet 权重文件（.pth），"
                                   "程序会按权重自动重建网络并导出 TorchScript")
        return "ready", "首次使用会从 torchreid 模型库下载预训练权重"
    ok, why = builtin_backbone_available(spec)
    return ("ready" if ok else "unavailable"), why


# ── DINOv3 (local weights) ──────────────────────────────────────────────────
# The two DINOv3 trunks are shipped as local files: the weights were downloaded
# once and are referenced by path, so the built-in entries behave like any other
# built-in — no download and no license step. The trunk itself is still the
# official facebookresearch/dinov3 architecture, built from the repo and then
# loaded with the tensors found in the file (.pth or .safetensors alike).

DINO3_REPO = "facebookresearch/dinov3"
# Embedding width -> official hub model name.
DINO3_DIM_TO_MODEL = {
    384: "dinov3_vits16",
    768: "dinov3_vitb16",
    1024: "dinov3_vitl16",
    1536: "dinov3_vith16plus",
    4096: "dinov3_vit7b16",
}


def dinov3_repo_dir() -> Optional[str]:
    """Cached facebookresearch/dinov3 clone, if torch.hub already fetched one."""
    try:
        hub = Path(torch.hub.get_dir())
    except Exception:
        hub = Path.home() / ".cache" / "torch" / "hub"
    try:
        for cand in sorted(hub.glob("facebookresearch_dinov3*")):
            if (cand / "hubconf.py").exists():
                return str(cand)
    except OSError:
        pass
    return None


def dinov3_buildable() -> Tuple[bool, str]:
    """Can a DINOv3 trunk be constructed on this machine right now?"""
    if dinov3_repo_dir():
        return True, "本地已缓存 DINOv3 仓库"
    if _github_reachable():
        return True, "将从 GitHub 拉取 facebookresearch/dinov3"
    return False, ("无法访问 GitHub，构建不了 DINOv3 骨架；"
                   "请把权重导出成 TorchScript（torch.jit.script(model).save(...)）后再导入")


def _is_dinov3_sd(sd: Dict[str, Any]) -> bool:
    """DINOv3 trunks: RoPE instead of a learned pos_embed, plus storage tokens.

    Storage/register tokens appear under several names depending on how the file was
    exported (`storage_tokens`, `reg_token`, `register_tokens`).
    """
    keys = set(sd.keys())
    if keys & {"storage_tokens", "reg_token"}:
        return True
    return ("register_tokens" in keys or "mask_token" in keys) \
        and "pos_embed" not in keys and any(k.startswith("blocks.") for k in keys)


def read_checkpoint(path: str) -> Any:
    """Load .safetensors / .pth / TorchScript into an object (dict or nn.Module)."""
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file      # extra dependency, imported lazily
        return load_file(path)
    return _safe_load(path)[0]


def normalise_dinov3_sd(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a DINOv3 checkpoint's keys to the module's own names.

    Two exports are in circulation:
      * the repo's own `.pth`      -> `storage_tokens`, `blocks.N.ls1.gamma`
      * the HF/timm `.safetensors` -> `reg_token`,      `blocks.N.gamma_1/gamma_2`
    Both describe the same tensors, so the older naming is rewritten here rather than
    silently skipped (an unmapped layer-scale leaves every block near-identity and the
    features collapse to a constant).
    """
    out = dict(sd)
    if "reg_token" in out and "storage_tokens" not in out:
        out["storage_tokens"] = out.pop("reg_token")
    if not any(k.endswith("ls1.gamma") for k in out):
        for k in list(out):
            if k.endswith("gamma_1"):
                out[k[: -len("gamma_1")] + "ls1.gamma"] = out.pop(k)
            elif k.endswith("gamma_2"):
                out[k[: -len("gamma_2")] + "ls2.gamma"] = out.pop(k)
    return out


def load_dinov3_weights(net: nn.Module, ckpt: str) -> None:
    """Load a DINOv3 checkpoint into a freshly built trunk, with key normalisation.

    A few tensors are legitimately absent from exported files (mask_token, the qkv
    bias buffers, the RoPE periods buffer) — they either stay at their deterministic
    init or are never used. Anything else missing means the architecture does not
    match, which must surface as an error instead of a silently broken model.
    """
    sd = read_checkpoint(ckpt)
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    if not isinstance(sd, dict):
        raise RuntimeError(f"无法识别的权重对象：{type(sd).__name__}")
    sd = normalise_dinov3_sd(sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    tolerated = ("mask_token", "rope_embed.periods", "qkv.bias", "qkv.bias_mask")
    core = [k for k in missing if not any(k.endswith(t) for t in tolerated)]
    if core:
        raise RuntimeError(
            f"权重与 DINOv3 结构不匹配，缺少 {len(core)} 个必要参数（如 {', '.join(core[:3])}）")
    if unexpected:
        raise RuntimeError(
            f"权重中有 {len(unexpected)} 个参数用不上（如 {', '.join(unexpected[:3])}）")


def build_dinov3_trunk(name: str) -> nn.Module:
    """Construct an uninitialised official DINOv3 trunk (`dinov3_vits16`, ...)."""
    repo = dinov3_repo_dir()
    if repo:
        return torch.hub.load(repo, name, source="local", pretrained=False, verbose=False)
    return torch.hub.load(DINO3_REPO, name, pretrained=False, verbose=False,
                          trust_repo=True)


def build_dinov3(ckpt: str, dim: Optional[int] = None) -> nn.Module:
    """Rebuild the official DINOv3 trunk that matches `ckpt` and load the weights."""
    obj = read_checkpoint(ckpt)
    if obj is None:
        raise RuntimeError(f"无法读取权重：{ckpt}")
    if isinstance(obj, nn.Module):
        return obj
    if not isinstance(obj, dict):
        raise RuntimeError(f"无法识别的权重对象：{type(obj).__name__}")
    if dim is None:
        vit = _vit_spec(obj)
        dim = vit[1] if vit else None
    name = DINO3_DIM_TO_MODEL.get(dim or -1)
    if not name:
        raise RuntimeError(f"无法按特征维度 {dim} 判定 DINOv3 规格（支持 384/768/1024/1536/4096 维）")
    # Build the trunk ourselves and load the tensors: that way .pth and .safetensors
    # are handled identically, and the key naming of either export is normalised.
    net = build_dinov3_trunk(name)
    load_dinov3_weights(net, ckpt)
    net.eval()
    return net


# ── Feature aggregation: how a ViT trunk's layers/read-outs become one vector ──
# A projection head is often trained on something wider than the trunk's own width:
# DINOv3 ViT-S/16 emits 384-d, while a common recipe feeds it `cat(last 4 CLS)` = 1536-d.
# The width therefore has to be a product of real tensors from real blocks — never a
# repeat/pad/projection invented to satisfy a shape check.
#
# `mult` = how many copies of embed_dim the strategy concatenates, `layers` = how many
# trailing transformer blocks it reads. Both live here so the UI, the exporter and the
# compatibility check all derive their numbers from the same table.

AGG_LAYERS = 4                      # how many trailing blocks "last4*" strategies read

FEATURE_STRATEGIES: List[Dict[str, Any]] = [
    {"id": "cls", "label": "CLS Token", "mult": 1, "layers": 1,
     "note": "最后一层 CLS token"},
    {"id": "mean_patch", "label": "Mean Patch", "mult": 1, "layers": 1,
     "note": "最后一层 patch token 均值池化"},
    {"id": "cls_mean", "label": "CLS + Mean Patch", "mult": 2, "layers": 1,
     "note": "最后一层 CLS 与 patch 均值拼接"},
    {"id": "last4_cls", "label": "Last 4 CLS Concat", "mult": AGG_LAYERS, "layers": AGG_LAYERS,
     "note": f"最后 {AGG_LAYERS} 个 block 的 CLS 拼接"},
    {"id": "last4_mean", "label": f"Last {AGG_LAYERS} Mean Patch Concat",
     "mult": AGG_LAYERS, "layers": AGG_LAYERS,
     "note": f"最后 {AGG_LAYERS} 个 block 的 patch 均值拼接"},
]

DEFAULT_FEATURE_STRATEGY = "cls"

# Artifact produced for a transformer trunk: not a graph, but the recipe needed to
# rebuild the module eagerly (see the "Eager PyTorch route" section below).
EAGER_ARTIFACT_SUFFIX = ".reid.json"


def aggregate_out_dim(base_dim: Optional[int], strategy: Optional[str]) -> Optional[int]:
    """Width after aggregation = trunk width × read-out multiplier."""
    if base_dim is None:
        return None
    spec = feature_strategy_spec(strategy)
    return int(base_dim) * int(spec["mult"]) if spec else None


def feature_strategy_spec(strategy: Optional[str]) -> Optional[Dict[str, Any]]:
    sid = (strategy or "").strip() or DEFAULT_FEATURE_STRATEGY
    return next((s for s in FEATURE_STRATEGIES if s["id"] == sid), None)


def supports_feature_aggregation(net: Any) -> bool:
    """True when the trunk exposes the DINOv2/DINOv3 intermediate-layer API."""
    return callable(getattr(net, "get_intermediate_layers", None))


class DINOv3Aggregator(nn.Module):
    """Turn a ViT trunk into a single feature vector using one of the strategies.

    Relies on the model's own `get_intermediate_layers(x, n, return_class_token=True)`,
    so the tensors really come from the requested blocks — no hooks, and nothing is
    padded, repeated or projected to reach a target width. Each requested layer
    yields `(patch_tokens, cls_token)`; every strategy below is a genuine read of
    those tensors.
    """

    def __init__(self, model: nn.Module, strategy: str = DEFAULT_FEATURE_STRATEGY):
        super().__init__()
        self.model = model
        self.strategy = strategy
        spec = feature_strategy_spec(strategy)
        if spec is None:
            raise RuntimeError(f"未知的特征聚合方式：{strategy}")
        # Resolved once so forward() stays TorchScript-friendly (no dict lookups).
        self.layers_needed: int = int(spec["layers"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = self.model.get_intermediate_layers(x, n=self.layers_needed,
                                                    return_class_token=True, norm=True)
        last = layers[-1]
        patch_tokens, cls_token = last[0], last[1]
        if self.strategy == "cls":
            return cls_token
        if self.strategy == "mean_patch":
            return patch_tokens.mean(dim=1)
        if self.strategy == "cls_mean":
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        if self.strategy == "last4_cls":
            return torch.cat([layer[1] for layer in layers], dim=-1)
        if self.strategy == "last4_mean":
            return torch.cat([layer[0].mean(dim=1) for layer in layers], dim=-1)
        return cls_token


def build_feature_module(net: nn.Module, strategy: Optional[str]
                         ) -> Tuple[nn.Module, int, bool]:
    """Wrap a trunk with the read-out `strategy` implies.

    Returns (module, measured_width, aggregated) — the width is measured with a real
    forward pass rather than derived arithmetically, so what gets reported is what
    the module actually produces.
    """
    spec = feature_strategy_spec(strategy)
    if spec is None:
        raise RuntimeError(f"未知的特征聚合方式：{strategy}")
    if supports_feature_aggregation(net):
        module: nn.Module = DINOv3Aggregator(net, spec["id"])
    else:
        if spec["id"] != DEFAULT_FEATURE_STRATEGY:
            raise RuntimeError(
                f"该 Backbone 不支持多层特征聚合（没有 get_intermediate_layers 接口），"
                f"无法使用「{spec['label']}」")
        module = BackboneAdapter(net)
    dim = probe_out_dim(module)
    if not dim:
        raise RuntimeError("无法测出该 Backbone 输出的特征维度")
    return module, int(dim), spec["id"] != DEFAULT_FEATURE_STRATEGY


def load_head_module(head_ckpt: Optional[str]) -> Tuple[Optional[nn.Module], Optional[int], Optional[int]]:
    """Rebuild a projection head -> (module, in_dim, out_dim). None when unused."""
    if not head_ckpt:
        return None, None, None
    hobj, herr = _safe_load(head_ckpt)
    if hobj is None:
        raise RuntimeError(f"无法读取 Projection Head 权重：{herr}")
    head_sd = hobj.state_dict() if isinstance(hobj, nn.Module) else hobj
    if not isinstance(head_sd, dict):
        raise RuntimeError(f"无法识别的 Projection Head 权重：{type(head_sd).__name__}")
    head = build_head_from_sd(head_sd)
    if head is None:
        raise RuntimeError("Projection Head 权重无法重建为可运行的模块")
    dims = head_dim(head_sd)
    return head, (dims[0] if dims else None), (dims[1] if dims else None)


def build_reid_module(net: nn.Module, size: Tuple[int, int] = (256, 128),
                      head_ckpt: Optional[str] = None,
                      feature_strategy: Optional[str] = None,
                      label: str = "") -> Tuple[nn.Module, Dict[str, Any]]:
    """Assemble trunk -> optional aggregation -> optional head as a live module.

    Returns (module, info). The module is a plain eager `nn.Module` — exporting it is a
    separate step, because a DINOv3 trunk cannot survive `torch.jit.trace` (see
    `install_eager_reid_hook`), so the same assembly has to serve both routes.

    The dimension check happens on the *aggregated* width, which is what the head is
    actually fed. `ReIDWrapper` already applies the single L2 normalisation at the end,
    so nothing here normalises twice.
    """
    base_dim = probe_out_dim(BackboneAdapter(net))
    feature_module, agg_dim, aggregated = build_feature_module(net, feature_strategy)
    head, head_in, head_out = load_head_module(head_ckpt)
    # What actually reaches the head is the aggregated vector, so that is what has to
    # equal its input width — matching the trunk's own width is neither necessary nor
    # sufficient once multi-layer aggregation is involved. Checking here means an
    # incompatible pairing is refused before any frame is processed, with both widths
    # named, instead of failing later inside the graph.
    if head is not None and head_in and agg_dim and head_in != agg_dim:
        raise RuntimeError(
            f"Projection Head 需要输入 {head_in} 维特征，当前特征聚合方式"
            f"（{feature_strategy or DEFAULT_FEATURE_STRATEGY}）输出 {agg_dim} 维，"
            f"两者不匹配，无法串接")

    enc = ReIDWrapper(feature_module, size=size, head=head)
    enc.eval()
    spec = feature_strategy_spec(feature_strategy)
    info: Dict[str, Any] = {
        "label": label,
        "backbone_dim": base_dim,
        "agg_dim": agg_dim,
        "aggregated": aggregated,
        "strategy": spec["id"] if spec else DEFAULT_FEATURE_STRATEGY,
        "strategy_label": spec["label"] if spec else "",
        "head_in": head_in, "head_out": head_out,
        "out_dim": head_out or agg_dim,
    }
    return enc, info


def export_reid_module(net: nn.Module, stem: str, out_dir: str,
                       size: Tuple[int, int] = (256, 128),
                       head_ckpt: Optional[str] = None,
                       feature_strategy: Optional[str] = None,
                       label: str = "") -> Tuple[str, Dict[str, Any]]:
    """Assemble the module and serialize it to TorchScript (the OSNet/CNN route)."""
    enc, info = build_reid_module(net, size=size, head_ckpt=head_ckpt,
                                  feature_strategy=feature_strategy, label=label)
    out = Path(out_dir) / ascii_safe_name(stem, ".torchscript")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        scripted = torch.jit.script(enc)
    except Exception:
        scripted = torch.jit.trace(enc, torch.zeros(1, 3, *size), strict=False)
    scripted.save(str(out))
    return str(out), info


def export_eager_encoder(net: nn.Module, stem: str, out_dir: str,
                         size: Tuple[int, int] = (256, 128),
                         head_ckpt: Optional[str] = None,
                         feature_strategy: Optional[str] = None,
                         label: str = "", backbone: Optional[str] = None
                         ) -> Tuple[str, Dict[str, Any]]:
    """Assemble the module and write a self-describing recipe artifact — no TorchScript.

    Used for trunks that cannot be traced (DINOv3). The artifact is a small JSON that
    names the backbone, the read-out strategy and the head; `build_eager_encoder()`
    rebuilds exactly that assembly as a live module when the tracker starts. Nothing is
    compiled, so no trace-time constant can leak into the graph.
    """
    enc, info = build_reid_module(net, size=size, head_ckpt=head_ckpt,
                                  feature_strategy=feature_strategy, label=label)
    out = Path(out_dir) / ascii_safe_name(stem, EAGER_ARTIFACT_SUFFIX)
    out.parent.mkdir(parents=True, exist_ok=True)
    recipe = {
        "route": "eager",
        "artifact": str(out),
        "backbone": backbone or "",
        "head": str(head_ckpt) if head_ckpt else None,
        "feature_strategy": info.get("strategy") or DEFAULT_FEATURE_STRATEGY,
        "size": [int(size[0]), int(size[1])],
        "agg_dim": info.get("agg_dim"),
        "head_in": info.get("head_in"),
        "out_dim": info.get("out_dim"),
        "label": label,
    }
    out.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out), info


def export_dinov3_encoder(ckpt: str, out_dir: str, size: Tuple[int, int] = (256, 128),
                          head_ckpt: Optional[str] = None,
                          feature_strategy: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Build a DINOv3 trunk from a checkpoint and hand back a runnable artifact.

    `feature_strategy` selects how the trunk's intermediate layers are aggregated
    before the optional projection head; see FEATURE_STRATEGIES.
    """
    net = build_dinov3(ckpt)
    strat = (feature_strategy or DEFAULT_FEATURE_STRATEGY).strip() or DEFAULT_FEATURE_STRATEGY
    stem = (f"{Path(ckpt).stem}-{strat}"
            + (f"+{Path(head_ckpt).stem}" if head_ckpt else ""))
    out, info = export_eager_encoder(net, stem, out_dir, size=size, head_ckpt=head_ckpt,
                                     feature_strategy=strat, label="DINOv3",
                                     backbone=str(ckpt))
    if info.get("backbone_dim"):
        info["arch"] = f"DINOv3 输出 {info['backbone_dim']} 维"
    return out, info


# ── Eager PyTorch route (DINOv3 and friends) ─────────────────────────────────
# A DINOv3 trunk CANNOT be turned into a portable TorchScript graph. Its rope module does
# `torch.arange(0.5, H, device=self.periods.device)`; when that runs under `torch.jit.trace`
# the device is a trace-time constant, so the tracer constant-folds a **CPU** tensor into
# the graph. Loading the artifact with `map_location="cuda"` moves every buffer and
# parameter to the GPU but cannot move graph constants, so the first real crop arriving
# from a GPU detector dies with
#     RuntimeError: Expected all tensors to be on the same device, but found at least two
#     devices, cuda:0 and cpu!
# (the failing op is the `angles = ... / periods` division in rope_position_encoding.py).
# Tracing also emits TracerWarnings ("converting a tensor to a Python integer ... the trace
# might not generalize"), and there is no reliable way to patch the frozen constants out of
# a saved graph.
#
# So the DINOv3 route runs **eager PyTorch**: the composed artifact carries a small recipe
# sidecar, and `build_eager_encoder()` rebuilds the same trunk + aggregator + head as a live
# `nn.Module`, moved to whatever device the tracker uses. OSNet and the other plain trunks
# keep the TorchScript route unchanged — their graphs are device-portable, so there is no
# reason to disturb something that works.

EAGER_SIDECAR_SUFFIX = ".reid.json"
_EAGER_HOOK_INSTALLED = False
_LOG = logging.getLogger("reid")


def eager_sidecar_path(artifact: str) -> Optional[Path]:
    """Sidecar describing how to run this artifact eagerly, if one exists."""
    p = Path(str(artifact))
    for cand in (Path(f"{p}{EAGER_SIDECAR_SUFFIX}"), p.with_suffix(EAGER_SIDECAR_SUFFIX)):
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def write_eager_recipe(artifact: str, recipe: Dict[str, Any]) -> str:
    """Attach an eager recipe to an exported artifact (kept next to it on disk)."""
    path = Path(f"{artifact}{EAGER_SIDECAR_SUFFIX}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(recipe, route="eager", artifact=str(artifact))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def read_eager_recipe(model: Any) -> Optional[Dict[str, Any]]:
    """The eager recipe for a tracker's `model` value, or None to use the normal path.

    Four ways in: a recipe artifact written by `export_eager_encoder`, a TorchScript
    artifact carrying a `.reid.json` sidecar, a built-in DINOv3 asked for by id, or the
    built-in prefix form. Plain files (OSNet exports, ONNX, YOLO `.pt`) return None and
    keep the existing TorchScript / AutoBackend route.
    """
    if model is None:
        return None
    raw = str(model).strip()
    if not raw or raw == "auto":
        return None
    if raw.lower().endswith(EAGER_ARTIFACT_SUFFIX):
        try:
            data = json.loads(Path(raw).read_text(encoding="utf-8"))
        except Exception:  # a corrupt recipe must not break inference
            return None
        if isinstance(data, dict) and data.get("route") == "eager":
            data.setdefault("artifact", raw)
            return data
        return None
    side = eager_sidecar_path(raw)
    if side is not None:
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
        except Exception:  # a corrupt sidecar must not break inference
            return None
        if isinstance(data, dict) and data.get("route") == "eager":
            data.setdefault("artifact", raw)
            return data
        return None
    spec = builtin_backbone_spec(raw)
    if spec is not None and spec.get("vit"):
        # DINOv3 by id: rebuild it eagerly rather than tracing it.
        return {"route": "eager", "backbone": raw,
                "feature_strategy": DEFAULT_FEATURE_STRATEGY}
    return None


def probe_reid_stages(net: nn.Module, head_ckpt: Optional[str] = None,
                      feature_strategy: Optional[str] = None,
                      size: Tuple[int, int] = (256, 128)) -> Dict[str, Any]:
    """Run one dummy crop through every stage and report the real tensor shapes.

    Diagnostics only: the widths that gate inference come from `build_reid_module`, which
    measures them with a real forward pass. This merely makes the pipeline visible — one
    line per stage — so a shape/dtype/device problem can be read off instead of guessed.
    """
    stages: Dict[str, Any] = {}
    spec = feature_strategy_spec(feature_strategy) or feature_strategy_spec(DEFAULT_FEATURE_STRATEGY)
    x = torch.zeros(1, 3, *size)
    stages["crop input"] = tuple(x.shape)
    try:
        with torch.no_grad():
            if supports_feature_aggregation(net):
                layers = net.get_intermediate_layers(x, n=int(spec["layers"]),
                                                    return_class_token=True, norm=True)
                patch, cls = layers[-1][0], layers[-1][1]
                stages["backbone CLS"] = tuple(cls.shape)
                stages["mean patch"] = tuple(patch.shape[:1] + patch.mean(dim=1).shape[1:])
                if spec["id"] == "last4_cls":
                    agg = torch.cat([layer[1] for layer in layers], dim=-1)
                elif spec["id"] == "last4_mean":
                    agg = torch.cat([layer[0].mean(dim=1) for layer in layers], dim=-1)
                elif spec["id"] == "cls_mean":
                    agg = torch.cat([cls, patch.mean(dim=1)], dim=-1)
                elif spec["id"] == "mean_patch":
                    agg = patch.mean(dim=1)
                else:
                    agg = cls
                stages["aggregator output"] = tuple(agg.shape)
            else:
                feature_module, agg_dim, _ = build_feature_module(net, feature_strategy)
                with torch.no_grad():
                    agg = feature_module(x)
                stages["aggregator output"] = tuple(agg.shape)
            stages["(passed to head)"] = tuple(agg.shape)
        if head_ckpt:
            head, head_in, head_out = load_head_module(head_ckpt)
            if head is not None:
                stages["head input"] = tuple(agg.shape)
                with torch.no_grad():
                    out = head(agg)
                stages["head output"] = tuple(out.shape)
                stages["head dims"] = (head_in, head_out)
    except Exception as e:  # diagnostics must never break a working build
        stages["probe error"] = f"{type(e).__name__}: {e}"
    return stages


def build_eager_encoder(recipe: Dict[str, Any], device: Any = None
                        ) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a composed encoder as a live PyTorch module (no TorchScript anywhere).

    Assembles exactly what the recipe names — trunk, feature aggregation, projection head —
    and moves the finished module to the tracker's device. The dimension pairing is
    validated while assembling, so an incompatible head/aggregation combination is refused
    with both widths in the message instead of failing inside a graph.
    """
    backbone = str(recipe.get("backbone") or "").strip()
    if not backbone:
        raise RuntimeError("ReID 配方缺少 backbone，无法构建编码器")
    strategy = (recipe.get("feature_strategy") or DEFAULT_FEATURE_STRATEGY).strip() \
        or DEFAULT_FEATURE_STRATEGY
    head_ckpt = recipe.get("head") or None
    size = tuple(recipe.get("size") or (256, 128))

    spec = builtin_backbone_spec(backbone)
    if spec is not None:
        net, spec = build_builtin_backbone(backbone)
        label = spec.get("label") or backbone
    else:
        net = build_dinov3(backbone)
        label = Path(backbone).stem
    net.eval()

    with torch.no_grad():
        enc, info = build_reid_module(net, size=size, head_ckpt=head_ckpt,
                                      feature_strategy=strategy, label=label)
        # Probe on the device the module currently sits on (CPU) — the forward pass is the
        # only way to get honest shapes, and a small card should not hold a second copy.
        stages = probe_reid_stages(net, head_ckpt=head_ckpt,
                                   feature_strategy=strategy, size=size)
    info = dict(info, stages=stages, backbone=backbone, label=label)

    if device is not None:
        try:
            enc = enc.to(device)
            # `.to()` can succeed on a card that cannot actually run the trunk: activations
            # are only allocated on the first forward, so probe it before the job starts.
            with torch.no_grad():
                enc(torch.zeros(1, 3, *size, device=next(enc.parameters()).device))
        except RuntimeError as e:  # a 2 GB card can hold the detector or the trunk, not both
            if "out of memory" not in str(e).lower():
                raise
            _LOG.warning("ReID 编码器显存不足，回退到 CPU：%s", e)
            device = torch.device("cpu")
            enc = enc.to(device)
    enc.eval()
    _LOG.info(
        "ReID eager encoder ready | backbone=%s aggregator=%s head=%s | trunk=%sd agg=%sd "
        "out=%sd | device=%s",
        label, info.get("strategy"), Path(head_ckpt).name if head_ckpt else "(none)",
        info.get("backbone_dim"), info.get("agg_dim"), info.get("out_dim"),
        next(enc.parameters()).device if any(True for _ in enc.parameters()) else device,
    )
    for name, shape in stages.items():
        _LOG.info("  ReID stage %-18s %s", name, shape)
    return enc, info


# ── Observability: did the appearance encoder actually run? ──────────────────
# "ReID is enabled" (a config flag) and "ReID was used" (the encoder saw crops) are
# different claims, and only the second one means the tracker really matched on
# appearance. The counter below is attached to whatever encoder the tracker ends up
# with — eager or TorchScript — so a job can report crops actually embedded.
_REID_TALLY: Dict[str, int] = {"calls": 0, "crops": 0}


def reset_reid_tally() -> None:
    _REID_TALLY["calls"] = 0
    _REID_TALLY["crops"] = 0


def reid_tally() -> Dict[str, int]:
    return dict(_REID_TALLY)


class _ReidCallCounter(nn.Module):
    """Count encoder invocations and how many crops went through them."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _REID_TALLY["calls"] += 1
        _REID_TALLY["crops"] += int(x.shape[0]) if x.dim() >= 1 else 0
        return self.inner(x)


class _EagerProbe(nn.Module):
    """Logs the first crop tensor that actually reaches the encoder, then forwards.

    The pipeline resizes and normalises crops before the trunk sees them, so the only
    honest way to report what the backbone receives is to observe it on the way in.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._logged = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._logged:
            self._logged = True
            _LOG.info("ReID 首个 crop 张量：shape=%s dtype=%s device=%s",
                      tuple(x.shape), x.dtype, x.device)
        return self.inner(x)


def install_eager_reid_hook() -> bool:
    """Teach Ultralytics' `ReID` to honour eager recipes. Idempotent.

    `build_encoder()` resolves `ReID` from its module globals at call time, so replacing the
    class attribute once is enough for every tracker (BoT-SORT, Deep OC-SORT, TrackTrack).
    The original `__init__` is still used for every artifact without a recipe, which keeps
    OSNet and the other TorchScript encoders on exactly the path they already used.
    """
    global _EAGER_HOOK_INSTALLED
    if _EAGER_HOOK_INSTALLED:
        return True
    try:
        from ultralytics.trackers.utils import reid as reid_mod
    except Exception as e:  # pragma: no cover - ultralytics missing
        _LOG.warning("无法安装 eager ReID 钩子：%s", e)
        return False
    original_init = reid_mod.ReID.__init__

    def __init__(self, model, imgsz: int = 224, device=None, fp16: bool = False):  # noqa: N807
        recipe = read_eager_recipe(model)
        if recipe is None:
            original_init(self, model, imgsz=imgsz, device=device, fp16=fp16)
            # Count crops on the TorchScript route too, so "was ReID used?" has one
            # answer regardless of which encoder is loaded. Counting wraps the call and
            # changes nothing about how the model runs.
            if not self.is_pt:
                self.model = _ReidCallCounter(self.model)
            return
        torch_dev = (torch.device(device) if device is not None
                     else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        enc, info = build_eager_encoder(recipe, device=torch_dev)
        # The encoder may have fallen back to CPU if the GPU could not hold it.
        torch_dev = next(enc.parameters()).device
        # `ReID.__call__` crops the detections and calls `self.model(batch)`; reusing that
        # machinery means the encoder contract stays identical to the TorchScript route.
        self.imgsz = int(recipe.get("imgsz") or imgsz or 224)
        self.batch_size = None
        self.device = torch_dev
        self.is_pt = False
        self.fp16 = False
        # Wrapped so the first crop batch is logged and every batch counted; the module
        # contract is unchanged.
        self.model = _ReidCallCounter(_EagerProbe(enc))
        self.eager_info = info

    reid_mod.ReID.__init__ = __init__
    _EAGER_HOOK_INSTALLED = True
    _LOG.info("eager ReID 钩子已安装（DINOv3 走 eager，OSNet 保持 TorchScript）")
    return True


# ── Weight inspection ────────────────────────────────────────────────────────

_YOLO_KEYS = {"model", "ema", "train_args", "updates", "optimizer"}


_TS_SUFFIXES = (".torchscript", ".ts", ".jit", ".ptl")


def _safe_load(path: str) -> Tuple[Any, Optional[str]]:
    """Load a checkpoint, returning (object, error). Tries weights_only first.

    A TorchScript archive is preferred over a pickled module whenever the suffix
    says so: `torch.save(model)` bakes in the defining module's import path, so a
    custom trunk class cannot be unpickled anywhere else. TorchScript carries the
    graph itself, which is why it is the recommended way to bring in a backbone.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _TS_SUFFIXES:
        try:
            return torch.jit.load(path, map_location="cpu"), None
        except Exception as e:
            ts_err = f"{type(e).__name__}: {e}"
        try:
            return torch.load(path, map_location="cpu", weights_only=False), None
        except Exception as e:
            return None, f"{ts_err} | {type(e).__name__}: {e}"
    try:
        return torch.load(path, map_location="cpu", weights_only=True), None
    except Exception as e:
        first = f"{type(e).__name__}: {e}"
    try:
        return torch.load(path, map_location="cpu", weights_only=False), None
    except Exception as e:  # pragma: no cover - genuinely unreadable file
        return None, f"{first} | {type(e).__name__}: {e}"


def inspect_weights(path: str) -> Dict[str, Any]:
    """Describe a checkpoint: YOLO / OSNet / projection head / unknown."""
    p = Path(path)
    info: Dict[str, Any] = {
        "path": str(p),
        "name": p.name,
        "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None,
        "kind": "unknown",
        "arch": None,
        "out_dim": None,
        "in_dim": None,
        "message": "",
        "loadable": False,
    }
    if not p.exists():
        info["message"] = "文件不存在"
        return info

    obj, err = _safe_load(str(p))
    if obj is None:
        info["message"] = f"无法读取权重：{err}"
        return info

    # YOLO / Ultralytics checkpoint?
    if isinstance(obj, dict) and _YOLO_KEYS & set(obj.keys()):
        info["kind"] = "yolo"
        info["arch"] = "YOLO"
        info["loadable"] = True
        info["message"] = "Ultralytics 检查点，原生可用"
        return _yolo_task_hint(obj, info)

    if isinstance(obj, nn.Module):
        info["kind"] = "module"
        info["arch"] = obj.__class__.__name__
        info["loadable"] = True
        # A complete module can simply be run, so its width is measured instead of
        # guessed — this is what makes an arbitrary backbone (DINOv3, CLIP, timm…)
        # pairable with a projection head.
        dim = probe_out_dim(obj)
        if dim:
            info["out_dim"] = dim
            info["message"] = f"完整模型（{obj.__class__.__name__}），实测输出 {dim} 维特征"
        else:
            info["message"] = "完整 nn.Module，可导出为 TorchScript"
        return info

    if not isinstance(obj, dict):
        info["message"] = f"无法识别的对象类型：{type(obj).__name__}"
        return info

    sd = obj
    keys = list(sd.keys())

    # torchreid OSNet: conv1 / conv2..conv5 / gate / fc / classifier
    if any(k.startswith("conv1.") for k in keys) and any(k.startswith("conv2.") for k in keys):
        arch, out_dim = _osnet_spec(sd)
        if out_dim:
            info["kind"] = "osnet"
            info["arch"] = arch
            info["out_dim"] = out_dim
            info["loadable"] = True
            info["message"] = f"OSNet Backbone（{arch}），可导出 TorchScript，输出 {out_dim} 维特征"
            return info

    # DINOv3: recognised separately because we can rebuild the trunk ourselves
    # from the official repo, so an imported checkpoint becomes runnable here.
    if _is_dinov3_sd(sd):
        vit = _vit_spec(sd)
        arch, dim = vit if vit else ("ViT", None)
        ok, why = dinov3_buildable()
        info["kind"] = "dinov3"
        info["arch"] = f"DINOv3 {arch}"
        info["out_dim"] = dim
        info["loadable"] = ok
        info["message"] = (f"DINOv3 Backbone（{arch}），输出 {dim} 维特征。"
                           f"{'将自动构建骨架并导出 TorchScript' if ok else why}")
        return info

    # ViT-family backbone (DINOv2/DINOv3/CLIP/DeiT …): a trunk, not a head.
    vit = _vit_spec(sd)
    if vit:
        arch, dim = vit
        info["kind"] = "vit"
        info["arch"] = arch
        info["out_dim"] = dim
        info["loadable"] = False   # a bare state dict is not runnable on its own
        info["message"] = (f"ViT Backbone（{arch}），输出 {dim} 维特征。"
                           f"裸 state_dict 无法直接加载：请提供完整模型或 TorchScript 导出"
                           f"（torch.jit.script(model).save(...)），即可与 Projection Head 串接")
        return info

    # Projection head: a short Linear/BN stack, first Linear takes a big feature vector
    head = _head_spec(sd)
    if head:
        info["kind"] = "head"
        info["arch"] = "ProjectionHead"
        info["in_dim"], info["out_dim"] = head
        info["loadable"] = True
        info["message"] = (f"Projection Head：输入 {head[0]} 维 → 输出 {head[1]} 维。"
                           f"可与输出 {head[0]} 维的编码器串接使用")
        return info

    info["message"] = f"无法识别的权重结构（{len(keys)} 个参数：{', '.join(keys[:3])}…）"
    return info


def _yolo_task_hint(obj: dict, info: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort task/format hint for a YOLO checkpoint."""
    try:
        args = obj.get("train_args") or {}
        if isinstance(args, dict) and args.get("task"):
            info["arch"] = f"YOLO-{args['task']}"
    except Exception:
        pass
    return info


def _osnet_spec(sd: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """Return (arch label, embedding dim) for an OSNet state dict.

    Widths are read straight from the checkpoint instead of guessing a named variant —
    many repos ship OSNet variants whose channel layout differs from torchreid's defaults.
    """
    fc_w = sd.get("fc.0.weight")
    dim = int(fc_w.shape[0]) if fc_w is not None and fc_w.ndim == 2 else None
    ch = _osnet_channels(sd)
    arch = f"OSNet {'→'.join(str(c) for c in ch)}" if ch else "OSNet"
    if dim:
        arch += f" ({dim}d)"
    return arch, dim


# ── Generic backbone recognition ─────────────────────────────────────────────
# A projection head is only usable together with the backbone it was trained on,
# so the backbone slot must accept *any* feature extractor — DINOv2/DINOv3, CLIP,
# timm trunks, a TorchScript export, whatever. What the pipeline needs to know is
# just its output width, and that can be read from the state dict (ViT-style keys)
# or measured with a dummy forward pass on a complete module.

_VIT_TRUNK_PREFIXES = ("backbone.", "model.backbone.", "student.backbone.",
                       "teacher.backbone.", "module.backbone.", "visual.",
                       "model.", "module.", "student.", "teacher.", "encoder.")


def _trunk_of(sd: Dict[str, Any]) -> Dict[str, Any]:
    """DINOv*/HF checkpoints wrap the trunk in a prefix; unwrap to the trunk itself."""
    keys = list(sd.keys())
    for pref in _VIT_TRUNK_PREFIXES:
        if not any(k.startswith(pref) for k in keys):
            continue
        inner = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        if any(k.startswith(("cls_token", "pos_embed", "patch_embed", "blocks.",
                             "layers.", "ln_post", "stem.")) for k in inner):
            return inner
    return sd


def _vit_spec(sd: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Recognise a ViT-family trunk from its state dict -> (arch label, embed dim).

    Works for DINOv2 / DINOv3 / DeiT / MAE / CLIP-ViT style checkpoints without
    hard-coding any single repo's naming: the embedding width is the last axis of
    `cls_token` / `pos_embed`, or the output channels of `patch_embed.proj`.
    """
    trunk = _trunk_of(sd)
    dim: Optional[int] = None
    for key in ("cls_token", "pos_embed", "patch_embed.proj.weight", "mask_token"):
        t = trunk.get(key)
        if t is None or not hasattr(t, "shape"):
            continue
        dim = int(t.shape[0]) if key == "patch_embed.proj.weight" else int(t.shape[-1])
        if dim:
            break
    if not dim:
        for k in sorted(trunk.keys()):
            if k.endswith(("norm.weight", "fc_norm.weight", "ln_post.weight")):
                v = trunk[k]
                if hasattr(v, "shape") and v.shape:
                    dim = int(v.shape[0])
                    break
    if not dim:
        return None

    parts = ["ViT"]
    depth = {k.split(".")[1] for k in trunk
             if k.split(".")[0] in ("blocks", "layers") and k.split(".")[1].isdigit()}
    if depth:
        parts.append(f"{len(depth)}层")
    pe = trunk.get("patch_embed.proj.weight")
    if pe is not None and len(pe.shape) == 4:
        parts.append(f"patch{int(pe.shape[-1])}")
    parts.append(f"{dim}d")
    return " ".join(parts), dim


def probe_out_dim(obj: Any) -> Optional[int]:
    """Measure a complete module's embedding width with one dummy forward pass."""
    if not (isinstance(obj, nn.Module) or callable(obj)):
        return None
    try:
        if hasattr(obj, "eval"):
            obj.eval()
    except Exception:
        pass
    out = None
    for shape in ((1, 3, 224, 224), (1, 3, 256, 128), (1, 3, 128, 64)):
        try:
            with torch.no_grad():
                out = obj(torch.zeros(*shape))
            break
        except Exception:
            out = None
    if out is None:
        return None
    t = out
    if isinstance(t, dict):
        for k in ("x_norm_clstoken", "pooler_output", "last_hidden_state", "x", "features"):
            if k in t:
                t = t[k]
                break
        else:
            t = next((v for v in out.values() if hasattr(v, "shape")), None)
    if isinstance(t, (tuple, list)):
        t = next((v for v in t if hasattr(v, "shape")), None)
    if t is None or not hasattr(t, "shape"):
        return None
    if len(t.shape) == 1:
        return int(t.shape[0])
    if len(t.shape) >= 2:
        return int(t.shape[1])
    return None


def _head_spec(sd: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Detect a simple stacked Linear(+BN) projection head -> (in_dim, out_dim)."""
    linears = [(k, v) for k, v in sd.items() if k.endswith(".weight") and getattr(v, "ndim", 0) == 2]
    if not linears or len(linears) > 6:
        return None
    linears.sort(key=lambda kv: _idx(kv[0]))
    in_dim = int(linears[0][1].shape[1])
    out_dim = int(linears[-1][1].shape[0])
    if in_dim < 64 or out_dim >= in_dim:
        return None
    # Must look like a head: every other entry is a Linear, no conv-style 4-D weights
    if any(getattr(v, "ndim", 0) == 4 for v in sd.values()):
        return None
    return in_dim, out_dim


def _idx(key: str) -> int:
    try:
        return int(key.split(".")[0])
    except ValueError:
        return 0


# ── torchreid OSNet (rebuilt to match the checkpoint's parameter names) ───────

class ConvLayer(nn.Module):
    """conv + bn (+ relu in forward)."""

    def __init__(self, in_c: int, out_c: int, k: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class LightConv3x3(nn.Module):
    """1x1 -> depthwise 3x3 -> bn -> relu."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(self.bn(x), inplace=True)


class ChannelGate(nn.Module):
    """Squeeze-and-excitation gate used by OSBlock (fc1/fc2 as 1x1 convs, reduction 16)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=True)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)), inplace=True)))


class OSBlock(nn.Module):
    """Omni-scale block: four parallel depthwise streams fused through a channel gate."""

    def __init__(self, in_c: int, out_c: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid = max(1, out_c // bottleneck_reduction)
        self.conv1 = ConvLayer(in_c, mid, 1)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = ConvLayer(mid, out_c, 1)
        self.downsample = ConvLayer(in_c, out_c, 1) if in_c != out_c else None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        t = self.conv1(x)
        agg = (self.gate(self.conv2a(t)) + self.gate(self.conv2b(t))
               + self.gate(self.conv2c(t)) + self.gate(self.conv2d(t)))
        return F.relu(self.conv3(agg) + identity, inplace=True)


class OSNet(nn.Module):
    """torchreid OSNet backbone. Channel widths are read from the checkpoint."""

    def __init__(self, channels: List[int], feature_dim: int = 512, num_classes: int = 0):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.conv1 = ConvLayer(3, c0, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # stage2 / stage3 additionally halve the spatial size through a trailing 1x1 conv
        # (checkpoints store these as a nested Sequential -> "conv2.2.0.conv.weight")
        self.conv2 = nn.Sequential(OSBlock(c0, c1), OSBlock(c1, c1),
                                   nn.Sequential(ConvLayer(c1, c1, 1, stride=2)))
        self.conv3 = nn.Sequential(OSBlock(c1, c2), OSBlock(c2, c2),
                                   nn.Sequential(ConvLayer(c2, c2, 1, stride=2)))
        self.conv4 = nn.Sequential(OSBlock(c2, c3), OSBlock(c3, c3))
        self.conv5 = ConvLayer(c3, c3, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c3, feature_dim), nn.BatchNorm1d(feature_dim))
        self.classifier = nn.Linear(feature_dim, num_classes) if num_classes else None
        self.feature_dim = feature_dim

    def forward(self, x, return_featuremaps: bool = False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x).flatten(1)
        if self.fc is not None:
            v = F.relu(self.fc(v), inplace=True)
        return self.classifier(v) if self.classifier is not None else v


def _osnet_channels(sd: Dict[str, Any]) -> Optional[List[int]]:
    """Infer [c0, c1, c2, c3] widths from an OSNet state dict."""
    try:
        c0 = int(sd["conv1.bn.weight"].shape[0])
        c1 = int(sd["conv2.0.conv3.bn.weight"].shape[0])
        c2 = int(sd["conv3.0.conv3.bn.weight"].shape[0])
        c3 = int(sd["conv4.0.conv3.bn.weight"].shape[0])
        return [c0, c1, c2, c3]
    except KeyError:
        return None


def build_osnet(sd: Dict[str, Any]) -> Optional[OSNet]:
    """Build an OSNet matching the state dict, then load the weights (non-strict)."""
    channels = _osnet_channels(sd)
    if not channels:
        return None
    fc_w = sd.get("fc.0.weight")
    feat = int(fc_w.shape[0]) if fc_w is not None else 512
    cls_w = sd.get("classifier.weight")
    n_cls = int(cls_w.shape[0]) if cls_w is not None else 0
    net = OSNet(channels, feature_dim=feat, num_classes=n_cls)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # Only tolerate the classifier being absent; everything else must match.
    core_missing = [k for k in missing if not k.startswith("classifier")]
    if core_missing:
        raise RuntimeError(f"权重与 OSNet 架构不匹配，缺少参数：{', '.join(core_missing[:5])}")
    net.eval()
    return net


def _as_2d(out: Any) -> Optional[torch.Tensor]:
    """Coerce whatever a trunk returned into a (B, D) feature matrix."""
    t = out
    if isinstance(t, dict):
        for k in ("x_norm_clstoken", "pooler_output", "x", "features"):
            if k in t:
                t = t[k]
                break
        else:
            t = next((v for v in out.values() if hasattr(v, "shape")), None)
    if isinstance(t, (tuple, list)):
        t = next((v for v in t if hasattr(v, "shape")), None)
    if t is None or not hasattr(t, "shape"):
        return None
    if len(t.shape) == 4:                      # feature map -> global average pool
        return t.mean(dim=(2, 3))
    if len(t.shape) == 3:                      # token sequence -> CLS token
        return t[:, 0]
    if len(t.shape) == 1:
        return t.unsqueeze(0)
    return t


class BackboneAdapter(nn.Module):
    """Make an arbitrary trunk behave like a ReID encoder: (B, C, H, W) -> (B, D).

    Vision trunks disagree on what they return — a tensor, a dict (DINOv2/DINOv3
    hand back `x_norm_clstoken` / `x_norm_patchtokens`), or a tuple — so the
    return value is normalised here before the projection head sees it.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = _as_2d(self.module(x))
        if feat is None:
            raise RuntimeError("Backbone 没有返回可用的特征张量")
        return feat


class ReIDWrapper(nn.Module):
    """Batched crop tensor (BCHW, RGB, 0..1) -> L2-normalised embeddings (B, D).

    Ultralytics resizes crops to a square `imgsz` before calling the encoder; the wrapper
    re-resizes to the model's expected (h, w) and applies the standard ReID normalisation.
    An optional `head` (projection) is applied between the backbone and the normalisation.
    """

    def __init__(self, backbone: nn.Module, size: Tuple[int, int] = (256, 128),
                 head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.size = (int(size[0]), int(size[1]))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feat = self.backbone(x)
        if self.head is not None:
            feat = self.head(feat)
        return F.normalize(feat, p=2, dim=1)


# ── Generic projection head (rebuilt from its state dict) ────────────────────

def _group_sd(sd: Dict[str, Any]) -> Optional[Dict[int, Dict[str, Any]]]:
    """Group a flat Sequential-style state dict into {index: {suffix: tensor}}."""
    groups: Dict[int, Dict[str, Any]] = {}
    for k, v in sd.items():
        parts = k.split(".")
        if len(parts) < 2 or not parts[0].isdigit():
            return None
        groups.setdefault(int(parts[0]), {})[".".join(parts[1:])] = v
    return groups or None


def build_head_from_sd(sd: Dict[str, Any]) -> Optional[nn.Module]:
    """Rebuild a Linear/BatchNorm/LayerNorm stack from its state dict.

    Parameter names are preserved exactly (including the original indices), because a
    `Sequential`'s positional slots must line up for a strict `load_state_dict`. Positions
    with no parameters are filled with `ReLU` — activation+dropout slots are the only
    param-free ones that appear in a projection head, and `ReLU∘ReLU == ReLU`, so filling
    them is safe whether the original slot was an activation or a dropout.
    """
    groups = _group_sd(sd)
    if groups is None:
        return None

    specs: Dict[int, Tuple[str, int, int]] = {}   # idx -> (kind, in, out)
    for idx, g in groups.items():
        w = g.get("weight")
        if w is None or not hasattr(w, "ndim"):
            continue
        if w.ndim == 2:                                     # Linear
            specs[idx] = ("linear", int(w.shape[1]), int(w.shape[0]))
        elif w.ndim == 1 and "running_mean" in g and "running_var" in g:   # BatchNorm1d
            specs[idx] = ("bn", int(w.shape[0]), int(w.shape[0]))
        elif w.ndim == 1:                                   # LayerNorm / norm w/o stats
            specs[idx] = ("ln", int(w.shape[0]), int(w.shape[0]))
        else:
            return None                                     # convolutions -> not a head
    if not specs:
        return None

    order = sorted(specs)
    last = order[-1]
    layers: List[nn.Module] = [nn.Identity()] * (last + 1)
    for idx in order:
        kind, dim_in, dim_out = specs[idx]
        if kind == "linear":
            layers[idx] = nn.Linear(dim_in, dim_out)
        elif kind == "bn":
            layers[idx] = nn.BatchNorm1d(dim_out)
        else:
            layers[idx] = nn.LayerNorm(dim_out)

    # A gap between a Linear/norm and the next Linear is an activation slot -> ReLU.
    for idx in range(last + 1):
        if idx in specs:
            continue
        prev = max((i for i in order if i < idx), default=None)
        nxt = min((i for i in order if i > idx), default=None)
        layers[idx] = nn.ReLU(inplace=True) if (prev is not None and nxt is not None
                                                and specs[nxt][0] == "linear") else nn.Identity()

    net = nn.Sequential(*layers)
    try:
        net.load_state_dict(sd, strict=True)
    except RuntimeError:
        return None
    net.eval()
    return net


def head_dim(sd_or_module: Any) -> Optional[Tuple[int, int]]:
    """Return (in_dim, out_dim) of a projection head, or None."""
    sd = sd_or_module
    if isinstance(sd_or_module, nn.Module):
        sd = sd_or_module.state_dict()
    return _head_spec(sd) if isinstance(sd, dict) else None


def build_encoder_from_sd(sd: Dict[str, Any], size: Tuple[int, int] = (256, 128),
                          head_sd: Optional[Dict[str, Any]] = None) -> Optional[nn.Module]:
    """Build a wrapped embedding extractor from a recognised state dict."""
    if any(k.startswith("conv1.") for k in sd) and any(k.startswith("conv2.") for k in sd):
        net = build_osnet(sd)
        if net is not None:
            net.classifier = None  # embeddings only
            head = build_head_from_sd(head_sd) if head_sd else None
            if head_sd and head is None:
                raise RuntimeError("Projection Head 权重无法重建为可运行的模块")
            # The width check only makes sense when a head is actually attached — with
            # no head the trunk's own width is the output, so there is nothing to pair.
            if head is not None:
                want = head_dim(head_sd)
                got = net.feature_dim
                if want and want[0] != got:
                    raise RuntimeError(
                        f"Projection Head 需要输入 {want[0]} 维特征，"
                        f"当前特征聚合方式输出 {got} 维，两者不匹配，无法串接")
            return ReIDWrapper(net, size=size, head=head)
    return None


def export_builtin_encoder(spec_id: str, out_dir: str, size: Tuple[int, int] = (256, 128),
                           head_ckpt: Optional[str] = None,
                           feature_strategy: Optional[str] = None
                           ) -> Tuple[Optional[str], Dict[str, Any]]:
    """Build a built-in backbone (+ aggregation + optional head) into a runnable artifact.

    Transformer trunks take the eager route (recipe artifact, no trace); the others are
    exported as TorchScript as before. Returns (path, spec) — the spec carries the
    *measured* widths (trunk, aggregated, head) rather than the nominal catalogue value.
    """
    net, spec = build_builtin_backbone(spec_id)
    strat = (feature_strategy or DEFAULT_FEATURE_STRATEGY).strip() or DEFAULT_FEATURE_STRATEGY
    stem = spec["id"] + f"-{strat}" + (f"+{Path(head_ckpt).stem}" if head_ckpt else "")
    if spec.get("vit"):
        out, info = export_eager_encoder(net, stem, out_dir, size=size, head_ckpt=head_ckpt,
                                         feature_strategy=strat, label=spec.get("label", ""),
                                         backbone=BUILTIN_PREFIX + spec["id"])
    else:
        out, info = export_reid_module(net, stem, out_dir, size=size, head_ckpt=head_ckpt,
                                       feature_strategy=strat, label=spec.get("label", ""))
    # Keep the catalogue contract: callers read the *measured* width from the spec.
    spec = dict(spec, out_dim=info.get("backbone_dim") or spec.get("out_dim"))
    return out, {**spec, **info}


def ascii_safe_name(stem: str, suffix: str) -> str:
    """ASCII-only file name for TorchScript output.

    `torch.jit.save` / `torch.jit.load` go through narrow-character I/O on Windows, so a
    non-ASCII path silently produces a mojibake file that can then never be loaded back.
    Everything exported therefore gets an ASCII name; the original stays as `display_name`.
    """
    safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "._-")) else "_" for ch in stem)
    safe = safe.strip("._-") or "model"
    digest = hashlib.sha1(stem.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{safe}-{digest}{suffix}"


def export_encoder(ckpt: str, out_path: str, size: Tuple[int, int] = (256, 128),
                   head_ckpt: Optional[str] = None, tag: str = "") -> Optional[str]:
    """Convert a recognised checkpoint into a TorchScript embedding module.

    `head_ckpt` optionally folds a projection head into the exported graph, so the single
    artifact ultralytics loads already produces the final embedding.
    """
    obj, err = _safe_load(ckpt)
    if obj is None:
        raise RuntimeError(f"无法读取权重：{err}")
    module = obj if isinstance(obj, nn.Module) else None
    sd = obj.state_dict() if module is not None else obj
    if not isinstance(sd, dict):
        raise RuntimeError(f"无法识别的权重对象：{type(sd).__name__}")

    head_sd = None
    if head_ckpt:
        hobj, herr = _safe_load(head_ckpt)
        if hobj is None:
            raise RuntimeError(f"无法读取 Projection Head 权重：{herr}")
        head_sd = hobj.state_dict() if isinstance(hobj, nn.Module) else hobj
        if not isinstance(head_sd, dict):
            raise RuntimeError(f"无法识别的 Projection Head 权重：{type(head_sd).__name__}")

    enc = build_encoder_from_sd(sd, size=size, head_sd=head_sd)
    if enc is None and module is not None:
        # Not a state dict we can rebuild — but a complete module can be wrapped
        # as-is, which is how an arbitrary backbone (DINOv3, CLIP, timm…) is used.
        head = build_head_from_sd(head_sd) if head_sd else None
        if head is not None:
            want = head_dim(head_sd)
            got = probe_out_dim(BackboneAdapter(module))
            if want and got and want[0] != got:
                raise RuntimeError(
                    f"Projection Head 需要输入 {want[0]} 维特征，当前特征聚合方式输出 {got} 维，"
                    f"两者不匹配，无法串接")
        enc = ReIDWrapper(BackboneAdapter(module), size=size, head=head)
    if enc is None:
        return None
    enc.eval()
    out = Path(out_path)
    # Always write an ASCII name (see ascii_safe_name) — jit.save mangles non-ASCII paths.
    stem = Path(ckpt).stem + (f"+{Path(head_ckpt).stem}" if head_ckpt else "") + tag
    if out.suffix:
        out = out.parent / ascii_safe_name(stem, out.suffix)
    else:
        out = out / ascii_safe_name(stem, ".torchscript")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        scripted = torch.jit.script(enc)
    except Exception:
        scripted = torch.jit.trace(enc, torch.zeros(1, 3, *size), strict=False)
    scripted.save(str(out))
    return str(out)
