"""timm.models.vision_transformer 的最小替身：只有 resize_pos_embed。"""
from .layers import resize_pos_embed  # noqa: F401

__all__ = ["resize_pos_embed"]
