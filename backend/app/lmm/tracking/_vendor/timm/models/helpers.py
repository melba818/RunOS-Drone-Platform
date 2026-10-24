"""timm.models.helpers 的最小替身。

`build_model_with_cfg` / `adapt_input_conv` 在 OSTrack 里**只被 import，没有被调用**
（它的模型构造函数直接调 `_create_vision_transformer`），`named_apply` 只在
`init_weights` 里用一次。所以这里给的是"够用且诚实"的实现：
  * `named_apply`：与 timm 相同的遍历顺序；
  * `build_model_with_cfg`：直接构造模型（不做 HF hub 下载，我们本来就自己 load 权重）；
  * `adapt_input_conv`：只实现"灰度→RGB 或通道裁剪/复制"这一种真实用法。
"""
import torch
import torch.nn as nn

__all__ = ["build_model_with_cfg", "named_apply", "adapt_input_conv"]


def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def build_model_with_cfg(model_cls, variant, pretrained, default_cfg=None,
                         model_cfg=None, feature_cfg=None, pretrained_strict=True,
                         pretrained_filter_fn=None, pretrained_custom_load=False,
                         **kwargs):
    model = model_cls(**kwargs)
    if pretrained:
        raise RuntimeError(
            "timm 替身不支持联网下载预训练权重：R4A 一律自己 torch.load 本地权重")
    return model


def adapt_input_conv(in_chans, conv_weight):
    """把预训练的 patch_embed 卷积权重适配到当前输入通道数。"""
    conv_type = type(conv_weight)
    conv_weight = conv_weight.float()
    out_chans, _, kh, kw = conv_weight.shape
    if in_chans == 1:
        conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        repeat = int(in_chans) // 3
        if repeat == 0:
            conv_weight = conv_weight[:, :in_chans]
        else:
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans]
        conv_weight *= 3 / float(in_chans)
    return conv_weight.to(conv_type) if isinstance(conv_weight, torch.Tensor) else conv_weight
