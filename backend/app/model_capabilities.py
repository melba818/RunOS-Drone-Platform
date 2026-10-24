"""Detection threshold capabilities, independent of model size or display name."""

_UNSCORED_TYPES = frozenset({"vlm", "vlx", "vlx_qwen"})

def supports_confidence_threshold(model_type: str) -> bool:
    return str(model_type or "").lower() not in _UNSCORED_TYPES
