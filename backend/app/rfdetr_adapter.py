"""RF-DETR detector adapter for the existing frame/render/tracker pipeline."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


@lru_cache(maxsize=32)
def _metadata(path, stamp):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not str(checkpoint.get("model_name", "")).startswith("RFDETR"):
        raise ValueError("请选择包含 RFDETR model_name 的 RF-DETR checkpoint，不能使用 YOLO 或 ReID 权重。")
    args = checkpoint.get("args", {})
    names = args.get("class_names", []) if isinstance(args, dict) else []
    return {"architecture": checkpoint["model_name"], "names": names}


def metadata(path):
    p = Path(path).resolve()
    return _metadata(str(p), p.stat().st_mtime_ns)


class RFDetectorAdapter:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.device = next(model.model.model.parameters()).device
        names = model.model.class_names or metadata(path)["names"]
        self.names = dict(enumerate(names))
        self.predictor = SimpleNamespace(trackers=[])

    def predict(self, frame, conf=.25, imgsz=576, classes=None, **kwargs):
        from ultralytics.engine.results import Results
        multiple = self.model.model_config.patch_size * self.model.model_config.num_windows
        size = int(imgsz)
        if size <= 0 or size % multiple:
            raise ValueError(f"RF-DETR 图像尺寸必须是 {multiple} 的倍数。")
        # RF-DETR accepts RGB, while camera and file sources deliver BGR.
        detections = self.model.predict(np.ascontiguousarray(frame[..., ::-1]),
                                        threshold=conf, shape=(size, size), include_source_image=False)
        # RF-DETR places weights lazily on the device during the first predict call.
        self.device = next(self.model.model.model.parameters()).device
        boxes = np.column_stack((detections.xyxy, detections.confidence, detections.class_id)).astype(np.float32)
        if classes:
            boxes = boxes[np.isin(boxes[:, 5], classes)]
        return [Results(frame, path=self.path, names=self.names, boxes=torch.from_numpy(boxes).to(self.device))]

    def track(self, frame, tracker, persist=True, **kwargs):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
        result = self.predict(frame, **kwargs)[0]
        if not self.predictor.trackers:
            cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
            cfg.device = self.device
            if getattr(cfg, "with_reid", False) and getattr(cfg, "model", "auto") == "auto":
                raise ValueError("RF-DETR 跟踪使用 ReID 时，请选择 OSNet 或已配套的 Backbone + Head；不支持 YOLO 原生特征 auto。")
            self.predictor.trackers = [TRACKER_MAP[cfg.tracker_type](args=cfg)]
        tracks = self.predictor.trackers[0].update(result.boxes.cpu().numpy(), frame)
        # Tracker rows: xyxy, track_id, score, class_id, source_detection_index.
        data = np.asarray(tracks, dtype=np.float32)
        result.update(boxes=torch.from_numpy(data[:, :-1] if len(data) else np.empty((0, 7), np.float32)).to(self.device))
        return [result]
