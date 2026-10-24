"""timm.models.registry 的最小替身：只有 register_model 装饰器。

OSTrack 只是 import 它（并未真正用到装饰器），这里给一个"原样返回"的实现，
保证 `@register_model` 若被使用也只是登记进一个本地字典，不会报错。
"""
_MODEL_REGISTRY = {}


def register_model(fn):
    _MODEL_REGISTRY[fn.__name__] = fn
    return fn


def list_models(filter="", exclude_filters="", pretrained=False):
    return sorted(_MODEL_REGISTRY.keys())


__all__ = ["register_model", "list_models"]
