"""timm.models.layers 的最小替身：to_2tuple / trunc_normal_ / lecun_normal_ /
DropPath / Mlp —— OSTrack 只用到这五个。

实现与 timm 一致（这些是公开且十年不变的初始化/正则化工具），
**初始化随机性必须一致**，否则同一份权重在不同环境下数值行为会变。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["to_2tuple", "trunc_normal_", "lecun_normal_", "DropPath", "Mlp"]


def _ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return tuple([x] * n)
    return parse


to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        # 与 timm 一样：先在 [2l-1, 2u-1] 均匀分布，再用 erfinv 变回截断正态
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _calculate_fan(tensor, mode="fan_in"):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in, fan_out = tensor.size(1), tensor.size(0)
    else:
        num_input_fmaps, num_output_fmaps = tensor.size(1), tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in if mode == "fan_in" else fan_out


def lecun_normal_(tensor, gain=1.0, mode="fan_in"):
    fan = _calculate_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0.0, std)


class DropPath(nn.Module):
    """Stochastic depth（推理时是恒等，训练时按 p 随机丢弃整条残差支路）。"""

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """timm 的 MLP：fc1 → act → drop → (norm) → fc2 → drop。"""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, norm_layer=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mode="bilinear"):
    """把位置编码从旧网格插值到新网格（与 OSTrack 自带实现一致的语义）。"""
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)
