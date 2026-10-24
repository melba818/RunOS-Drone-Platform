# -*- coding: utf-8 -*-
"""STEP 4：Tile Pass 的切片与坐标换算。

为什么必须有这个文件：小目标在 1920×1080 全图上只占二三十像素，VLX 在 Full Image
上容易漏。把图切成有重叠的小块再逐块问模型，等于给每个目标一次"放大镜"。

刻意**不**写死 3×3 / 4×4：tile 的数量必须由
    图片尺寸 / tile_size / overlap
三者动态算出来。写死网格在换分辨率时会出现"最后一行只切到半个目标"。

坐标约定（与 `lmm/tracking/frame_cache.py to_source()` 同一套范式）：
    Tile 保存的是**源图坐标** (x0, y0, x1, y1)；
    `to_tile()` 把源图框换成 tile 内坐标，`from_tile()` 换回来；
    所有出界都会被 `clamp_box()` 夹住 —— 宁可框被截断，也不让坐标越界写进标注。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]


@dataclass
class Tile:
    """一块切片。**存的是源图坐标**，tile 内的图由调用方自己按这个区域裁。"""

    index: int
    x0: int
    y0: int
    x1: int                      # 不含（Python 切片语义）
    y1: int
    row: int = 0
    col: int = 0
    #: 是否是被图像边界"挤回来"的边缘 tile（合并时要特殊对待，见 merge_ops）
    edge: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        return int(self.y1 - self.y0)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def clamp_box(box: Sequence[float], width: int, height: int) -> Box:
    """把框夹进 [0,width]×[0,height]。宽度为 0 的退化框直接返回原样（调用方判）。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    return (x1, y1, x2, y2)


def valid_box(box: Sequence[float], min_side: float = 1.0) -> bool:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x2 - x1) >= min_side and (y2 - y1) >= min_side


def generate_tiles(width: int, height: int, tile_size: int,
                   overlap: int = 0, min_tile: int = 64) -> List[Tile]:
    """按 尺寸 / tile_size / overlap **动态**生成切片网格。

    规则（逐条对应"必须处理"）：
      * `stride = tile_size - overlap`，且强制 `0 <= overlap < tile_size`；
      * 起点从 0 开始按 stride 步进；
      * **边界 tile 回退**：若 `start + tile_size > 尺寸`，把起点拉回到
        `尺寸 - tile_size`（保证每块都是完整 tile_size，不产生碎块）；
        仍然越界（图比 tile 还小）就整块图只切一块；
      * 拉回后与上一块起点相同 → 跳过，避免重复切片。
    """
    width = int(width)
    height = int(height)
    tile_size = int(tile_size)
    overlap = int(overlap)
    if width <= 0 or height <= 0 or tile_size <= 0:
        return []
    if overlap < 0:
        overlap = 0
    if overlap >= tile_size:
        overlap = tile_size - 1
    stride = max(1, tile_size - overlap)

    def axis_stops(size: int) -> List[int]:
        """该轴上的切片起点。**不许产生碎块**。

        第一版写成"先 append 再判断是否停"，结果在 1920/tile640/stride512 下
        起点序列变成 [0,512,1024,1536,1280] —— 1536 那块只剩 384 宽（碎块），
        而且顺序还乱了。正确做法是：**只有 `pos + tile_size < size` 才继续步进**，
        一旦放不下完整一块就交给 `last = size - tile_size` 贴边。
        """
        if size <= tile_size:
            return [0]
        stops: List[int] = []
        pos = 0
        while pos + tile_size < size:
            stops.append(pos)
            pos += stride
        #: 最后一块必须贴右/下边界且大小正好 tile_size
        last = size - tile_size
        if not stops or stops[-1] != last:
            stops.append(last)
        return stops

    xs = axis_stops(width)
    ys = axis_stops(height)
    tiles: List[Tile] = []
    idx = 0
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            x1 = min(width, x0 + tile_size)
            y1 = min(height, y0 + tile_size)
            edge = (x0 == 0 or y0 == 0 or x1 == width or y1 == height)
            if (x1 - x0) < min_tile or (y1 - y0) < min_tile:
                continue
            tiles.append(Tile(index=idx, x0=x0, y0=y0, x1=x1, y1=y1,
                              row=r, col=c, edge=edge))
            idx += 1
    return tiles


def to_tile(box: Sequence[float], tile: Tile) -> Box:
    """源图坐标 → tile 内坐标。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1 - tile.x0, y1 - tile.y0, x2 - tile.x0, y2 - tile.y0)


def from_tile(box: Sequence[float], tile: Tile,
              width: Optional[int] = None, height: Optional[int] = None) -> Box:
    """tile 内坐标 → 源图坐标（可选 clamp 回整图）。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    out = (x1 + tile.x0, y1 + tile.y0, x2 + tile.x0, y2 + tile.y0)
    if width is not None and height is not None:
        out = clamp_box(out, int(width), int(height))
    return out


def box_in_tile(box: Sequence[float], tile: Tile, min_visible: float = 0.35) -> bool:
    """这个框有多少落在 tile 里（用于判断该不该由这块 tile 负责）。

    `min_visible` 用**面积占比**而不是"中心点在内"：中心点判据会把"目标横跨
    两块 tile 各一半"算成两块都不管或都管，用面积占比能稳定地归属给占比大的。
    """
    x1, y1, x2, y2 = clamp_box(box, 1 << 30, 1 << 30)
    ix1 = max(x1, float(tile.x0))
    iy1 = max(y1, float(tile.y0))
    ix2 = min(x2, float(tile.x1))
    iy2 = min(y2, float(tile.y1))
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    return (ix2 - ix1) * (iy2 - iy1) / area >= float(min_visible)


def near_tile_edge(box: Sequence[float], tile: Tile, margin: float = 8.0) -> bool:
    """框是否贴着 tile 边界（贴边 = 目标被切了一半，这块 tile 的框不完整）。

    合并时把贴边框的可信度调低：它很可能是**目标的一部分**，而不是一个完整目标。
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1 - tile.x0 <= margin or y1 - tile.y0 <= margin
            or tile.x1 - x2 <= margin or tile.y1 - y2 <= margin)


def crop_with_context(box: Sequence[float], width: int, height: int,
                      context: float = 1.5, min_side: int = 64,
                      max_side: int = 768) -> Box:
    """Verify 用的裁剪：把框按 `context` 倍向外扩，给模型一点上下文。

    外扩后仍要 clamp 进图像；并保证最小边长 `min_side`（太小的 crop 模型看不清）。
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nw = max(float(min_side), min(float(max_side), w * float(context)))
    nh = max(float(min_side), min(float(max_side), h * float(context)))
    out = (cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5)
    return clamp_box(out, int(width), int(height))
