from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, field_validator

from camera.camera_worker import CameraService
from routers.inference import FrameRequest, TRACKERS, DEFAULT_TRACKER, infer_numpy

router = APIRouter(prefix='/api/camera', tags=['camera'])
service = CameraService(infer_numpy)


class CameraOptions(FrameRequest):
    compute_node_id: str | None = None
    # The UI defaults to the GPU, so the API default matches it (a request that omits
    # `device` must not silently land on the CPU).
    device: Literal["cpu", "cuda:0"] = "cuda:0"
    frame_b64: str = ''
    task: Literal['detect','segment','pose','track'] = 'detect'
    confidence: float = Field(default=.25, ge=0, le=1)
    iou: float = Field(default=.45, ge=0, le=1)
    imgsz: int = Field(default=640, ge=32, le=4096)
    # Accept any tracker the backend actually ships (kept in sync with inference.TRACKERS).
    tracker: str = DEFAULT_TRACKER

    @field_validator('tracker')
    @classmethod
    def _known_tracker(cls, v: str) -> str:
        if v not in TRACKERS:
            raise ValueError(f"Unknown tracker '{v}'. Supported: {', '.join(TRACKERS)}")
        return v

    def inference_options(self):
        if not Path(self.model_path).is_file():
            raise HTTPException(404, '模型文件不存在，请重新选择模型。')
        # Built from the explicit contract rather than `model_dump()`: dumping the model
        # would pass the flat tracker tunables straight to `predict_frame`, which expects
        # them nested under `track_options`, and every start would raise
        # "got an unexpected keyword argument 'track_high_thresh'".
        options = self.predict_kwargs()
        options['device'] = self.device
        if self.compute_node_id: options['compute_node_id'] = self.compute_node_id
        return options


class CameraStart(CameraOptions):
    index: int = Field(default=0, ge=0, le=63)
    width: int = Field(default=1280, ge=160, le=4096)
    height: int = Field(default=720, ge=120, le=2160)


def session(session_id):
    with service.lock: found = service.sessions.get(session_id)
    if found is None: raise HTTPException(404,'摄像头会话不存在或已过期。')
    return found


@router.post('/scan')
def scan():
    try: return service.scan()
    except RuntimeError as exc: raise HTTPException(409,str(exc))


@router.get('/devices')
def devices():
    with service.lock: return dict(service.scan_state)


@router.post('/start')
def start(req: CameraStart, request: Request):
    if req.compute_node_id:
        from routers.compute_nodes import local_only
        local_only(request)
    options = req.inference_options()
    try: return service.start(req.index,req.width,req.height,options).snapshot()
    except RuntimeError as exc: raise HTTPException(409,str(exc))


@router.get('/{session_id}/frame')
def frame(session_id: str, after: int = 0):
    return session(session_id).snapshot(after)


@router.patch('/{session_id}/options')
def options(session_id: str, req: CameraOptions, request: Request):
    if req.compute_node_id:
        from routers.compute_nodes import local_only
        local_only(request)
    try: session(session_id).update(req.inference_options()); return {'ok':True}
    except RuntimeError as exc: raise HTTPException(409,str(exc))


@router.post('/{session_id}/pause')
def pause(session_id: str, paused: bool = True):
    try: session(session_id).pause(paused); return {'ok':True}
    except RuntimeError as exc: raise HTTPException(409,str(exc))


@router.post('/{session_id}/stop')
def stop(session_id: str):
    found = session(session_id)
    found.stop()
    return found.snapshot()
