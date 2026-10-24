"""输入源：本地视频文件 / RTSP 流 / 摄像头。

为什么不复用 `camera/camera_manager.py` 的 BaseVideoSource：那个类是给「摄像头会话 +
单槽丢帧队列」设计的，绑定了会话生命周期。LMM 任务需要的是「一次性打开 → 顺序读 → 结束
释放」，语义不同。这里保持独立，避免为了复用去改动既有模块（需求文档要求最小侵入）。

统一抽象与那边一致（open / read / close），将来要合并不难。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional

LOG = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """输入源不可用。"""


#: 常见视频扩展名（补全用）。
_VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v",
               ".ts", ".mpg", ".mpeg", ".m2ts", ".rmvb", ".3gp", ".mts")


def resolve_video_path(raw: str) -> Path:
    """把用户给的路径解析成磁盘上真实存在的视频文件。

    **为什么需要**：Windows 资源管理器默认隐藏已知文件类型的扩展名，用户从那里复制
    文件名粘进来时带的是 `D:\\AI\\video\\9月22日`，而磁盘上是 `9月22日.mp4`。
    旧实现直接 `is_file()` 判否 → 任务在建源阶段就失败，而前端只会一直显示"等待第一帧"，
    用户完全不知道发生了什么。这里替用户补回扩展名，并在真的找不到时给出同目录候选。

    依次尝试：原样 →（无扩展名时）补常见视频扩展名 → 当目录取其下第一个视频。
    """
    raw = str(raw or "").strip()
    # 从资源管理器「复制文件地址」粘进来会带外层引号
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        raw = raw[1:-1].strip()
    if not raw:
        raise SourceError("未指定视频文件路径")

    p = Path(raw)
    if p.is_file():
        return p

    if not p.suffix:
        for ext in _VIDEO_EXTS:
            for cand in (p.with_suffix(ext), p.with_suffix(ext.upper())):
                if cand.is_file():
                    LOG.info("[LMM] 源路径缺少扩展名，已自动补全：%s → %s", p, cand.name)
                    return cand

    if p.is_dir():
        vids = sorted(q for q in p.iterdir()
                      if q.is_file() and q.suffix.lower() in _VIDEO_EXTS)
        if vids:
            LOG.info("[LMM] 源路径是目录，改用其中第一个视频：%s", vids[0].name)
            return vids[0]

    # 失败：给出同目录候选，帮用户看出自己少写了什么
    near: list[str] = []
    if p.parent.is_dir():
        stem = p.stem.lower()
        try:
            for q in p.parent.iterdir():
                if q.is_file() and (q.stem.lower() == stem or stem in q.name.lower()):
                    near.append(q.name)
        except OSError:
            pass
    hint = f"。同目录下的相近文件：{'、'.join(sorted(near)[:5])}" if near else ""
    raise SourceError(f"视频文件不存在：{p}{hint}")


class BaseSource:
    """视频源契约。"""

    kind = "base"

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        return 0.0

    @property
    def total_frames(self) -> int:
        return 0

    @property
    def size(self) -> tuple:
        return (0, 0)

    @property
    def position(self) -> int:
        """刚读到的那一帧的**绝对帧号**（1-based；0 = 还没读）。

        R0 需要它：`frame_id` 必须能追溯"这一帧在视频里的位置"，否则
        `frame_age` / drift 都只能按"处理过的第几帧"估算 —— 而抽样间隔一变
        估算就错。实时模式更要靠它判断"最新帧 vs 已处理帧"。
        """
        return 0


class FileSource(BaseSource):
    """本地视频文件。按顺序读，`frame_stride>1` 时由调用方决定丢帧方式（这里用 grab 跳过）。"""

    kind = "file"

    def __init__(self, path: str, *, resolve: bool = True) -> None:
        # 在这里就把路径解析干净（补扩展名 / 目录取文件），后续 open() 只负责开流。
        #
        # `resolve=False` 是给「值不是磁盘路径」的子类留的（StreamSource 的 URL、
        # DeviceSource 的设备索引）：它们交给 resolve_video_path() 会在**构造阶段**
        # 就被判成"视频文件不存在"，永远到不了 open() 里的 cv2.VideoCapture。
        # 默认 True —— 本地视频行为与改动前逐字一致。
        raw = "" if path is None else str(path)
        self.path = str(resolve_video_path(raw)) if resolve else raw.strip()
        self._cap = None
        self._fps = 0.0
        self._total = 0
        self._size = (0, 0)

    def open(self) -> None:
        import cv2
        from pathlib import Path

        if not Path(self.path).is_file():
            raise SourceError(f"视频文件不存在：{self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise SourceError(f"无法打开视频：{self.path}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))

    def read(self, stride: int = 1) -> Optional[Any]:
        """读一帧；stride>1 时用 grab() 跳过中间帧（比 read 后丢弃快得多）。"""
        if self._cap is None:
            return None
        for _ in range(max(0, int(stride) - 1)):
            self._cap.grab()
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def position(self) -> int:
        if self._cap is None:
            return 0
        try:
            import cv2

            # OpenCV 的 POS_FRAMES 在 read() 之后指向**下一帧**的序号，
            # 减 1 才是刚读出来的这一帧（1-based 帧号）。
            return max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1)
        except Exception:  # noqa: BLE001 —— 拿不到帧号不该让任务挂掉
            return 0


class StreamSource(FileSource):
    """RTSP / HTTP 流。与文件同样的读取方式，但没有总帧数。

    OpenCV 的 RTSP 后端在断流时 read() 会一直阻塞，所以加一个软超时：连续 N 次读空
    就判定断流并结束任务，而不是把线程挂死。
    """

    kind = "stream"

    def __init__(self, url: str, max_empty: int = 60) -> None:
        # URL 必须原样保存到 open()：它不是磁盘路径，一旦走 resolve_video_path()
        # 构造阶段就会抛"视频文件不存在"（BLOCK-1），cv2.VideoCapture 永远拿不到它。
        super().__init__(url, resolve=False)
        self._max_empty = int(max_empty)
        self._empty = 0

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise SourceError(f"无法连接视频流：{self.path}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        self._total = 0
        self._size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))

    def read(self, stride: int = 1) -> Optional[Any]:
        frame = super().read(stride)
        if frame is None:
            self._empty += 1
            if self._empty >= self._max_empty:
                raise SourceError("视频流中断（连续读空帧）")
            time.sleep(0.05)
            return None
        self._empty = 0
        return frame


class DeviceSource(FileSource):
    """本地摄像头（设备索引）。"""

    kind = "device"

    def __init__(self, device: str = "0") -> None:
        # 设备索引同样不是磁盘路径：`"0"` 走 resolve_video_path() 会被判成
        # "视频文件不存在：0"（与 StreamSource 同一个继承问题）。
        super().__init__(device, resolve=False)

    def open(self) -> None:
        import cv2

        idx = int(self.path) if str(self.path).isdigit() else 0
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise SourceError(f"无法打开摄像头 {idx}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        self._total = 0
        self._size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))


def open_source(source_type: str, source: str) -> BaseSource:
    """按类型构造输入源。类型由前端显式传，避免"猜"路径是文件还是 URL。"""
    st = (source_type or "").strip().lower()
    if st in ("rtsp", "stream", "url"):
        return StreamSource(source)
    if st in ("camera", "device", "webcam"):
        return DeviceSource(source)
    if st in ("file", "video", ""):
        return FileSource(source)
    raise SourceError(f"未知输入源类型：{source_type}")
