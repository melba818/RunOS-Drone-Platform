# -*- coding: utf-8 -*-
"""R4A：远程 OSTrack 后端 —— 把跟踪计算迁到 A6000 GPU1 上的独立服务。

实现与 `ostrack.py OSTrackBackend` **完全相同的 `TrackerBackend` 接口**，
因此 `TargetManager` / `TrackerWorker` / `task_manager` 一行都不用改。

★STEP 2.1 传输优化★
────────────────────────────────────────────────────────────────────────────
优化前：RunOS → SSH → OSTrack 端到端 update ≈ 224 ms（≈4.5 FPS），不满足追帧。
先做了拆分实测（`.workbuddy/ostrack_size_sweep.txt`），结论**推翻了原假设**：

    原假设        "每次 update 重新建立 SSH channel，约 200 ms 通信开销"
    实测          channel_open 只有 ~115 ms，真正的大头是**上传体积**：
                    b64 745 KB（1080p q90）→ upload 2428 ms（吞吐仅 0.30 MB/s）
                    b64  29 KB（384²  q75）→ upload    1 ms
                  体积降下来后地板是 channel_open 115 ms + wait_first_byte 82 ms。

所以两个措施必须一起上，缺一个都到不了 10 FPS：
  1. **持久连接**（`tunnel_http.tunnel_http_request_persistent`）：复用同一条
     direct-tcpip channel + HTTP/1.1 keep-alive，摊掉 115 ms 建连；
  2. **只传目标周围的 crop**（本文件的 `_crop_around`）：按上一帧框 × SEARCH_FACTOR
     (4.0) 裁一块，745 KB → 约 30 KB。坐标换算靠请求里的 `offset` 字段，服务端
     负责把 tracker 状态与输出框在两个坐标系间搬运，**调用方看到的仍是全图坐标**。

约定（与服务端 server.py 对齐）：
  * 失败一律转 `TrackerResult(ok=False, reason=...)`，绝不抛 —— 与 NullTracker 同语义；
  * `health()` 探服务端，失败返回 `{ok: False, error: ...}`，不影响任务继续；
  * 连接断开自动重建、单次请求失败最多重试 1 次，都不让 RunOS 后端退出；
  * **Latest Frame Wins**：本后端只做"把当前这帧送上去"，从不缓存待处理帧队列 ——
    追哪一帧由 `TargetManager.step()` 决定（它本来就只取 `cache.newest()`）。

为什么 crop 后不会跟丢：OSTrack 的搜索区本来就是"上一帧框 × 4.0"，客户端提前裁
同一块区域，喂给它的像素完全等价（多出来的全图部分 sample_target 也不会用到）。
唯一风险是目标跑出 crop —— 由 `_crop_ok` 兜底：框贴边或分数掉下来就回退全图重来。
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..tunnel_http import tunnel_http_request_persistent
from .base import TrackerBackend
from .types import TrackerResult

LOG = logging.getLogger(__name__)

_DEFAULT_PORT = 8022
_DEFAULT_TIMEOUT = 15.0
#: ★参数由 `.workbuddy/ostrack_param_sweep.txt` 实测选出★（不是拍脑袋）：
#:   q75/cf4.0 → 119.9 ms (8.3 FPS)   q60/cf4.0 → 86.3 ms (11.6 FPS)
#:   q70/cf3.0 →  82.0 ms (12.2 FPS)  ← 更快但 crop 小于 SEARCH_FACTOR 4.0
#: cf3.0 会把 OSTrack 设计好的搜索区边缘裁掉，目标靠近 crop 边界时容易丢；
#: 为了 0.6 FPS 冒跟丢的风险不值，所以**保 cf4.0，把质量压到 60**——
#: crop 里只有目标那一块，JPEG 伪影对 ViT 影响很小（实测中心偏差 2.0 px，与
#: q75 的 1.7 px 同一量级）。
#: 注：真实航拍目标只有 20~40 px（本测试用的是 100 px 的"大目标"），crop 会被
#: min_crop 兜到 256，实际比这里的 86 ms 还要更快。
_DEFAULT_JPEG_QUALITY = 60
_DEFAULT_CROP_FACTOR = 4.0          # 与 OSTrack cfg.TEST.SEARCH_FACTOR 一致
_DEFAULT_MIN_CROP = 320
_DEFAULT_MAX_CROP = 960
#: initialize 只需要模板（TEMPLATE_FACTOR 2.0），给 2.5 留点余量
_DEFAULT_INIT_FACTOR = 2.5
_DEFAULT_MIN_INIT_CROP = 192
#: 分数低于此值视为"这一帧没跟上" → 下一帧回退全图，重新建立搜索区
_LOW_SCORE = 0.10
#: 延迟样本保留条数（p50/p95/max 用）
_LAT_SAMPLES = 256
#: health 真探的最小间隔（秒）
_HEALTH_TTL = 5.0


def _pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return float(s[k])


class RemoteOSTrackBackend(TrackerBackend):
    name = "ostrack-remote"

    def __init__(self, node_id: str = "", port: int = _DEFAULT_PORT,
                 token: str = "", timeout: float = _DEFAULT_TIMEOUT,
                 jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
                 crop_enabled: bool = True,
                 crop_factor: float = _DEFAULT_CROP_FACTOR,
                 min_crop: int = _DEFAULT_MIN_CROP,
                 max_crop: int = _DEFAULT_MAX_CROP,
                 init_factor: float = _DEFAULT_INIT_FACTOR,
                 min_init_crop: int = _DEFAULT_MIN_INIT_CROP,
                 parallel_channels: int = 1,
                 ref_mode: bool = False,
                 hub_client: Any = None) -> None:
        self.node_id = node_id or ""
        self.port = int(port or _DEFAULT_PORT)
        self.token = token or ""
        self.timeout = float(timeout)
        self.jpeg_quality = int(jpeg_quality)
        self.crop_enabled = bool(crop_enabled)
        self.crop_factor = float(crop_factor)
        self.min_crop = int(min_crop)
        self.max_crop = int(max_crop)
        self.init_factor = float(init_factor)
        self.min_init_crop = int(min_init_crop)
        #: ★并发通道数★：>1 时每个 track 按 `track_id % parallel_channels` 分到
        #: 不同的 SSH 隧道槽位，多个目标的 update 才能真正并行。
        #: 串行时 N 个目标共享 1/(单次延迟) 的预算，每目标只有 1/N —— 这是
        #: 多目标追帧频率上不去的**根本原因**（单条 SSH 隧道上的请求是排队的）。
        self.parallel_channels = max(1, int(parallel_channels or 1))
        #: ★服务器侧共享帧★：True 时**不再上传任何图片**，只发
        #: `{track_id, frame_id, media_time_ms}`，由服务端自己从 StreamHub 取帧。
        #: 旧 image 模式（crop / JPEG / base64）全部保留，可随时切回。
        self.ref_mode = bool(ref_mode)
        #: 可选的 StreamHub 客户端句柄（只用于 `health()` 里回报状态，不参与取帧）
        self.hub_client = hub_client

        self._lock = threading.RLock()
        #: track_id -> 上一帧框（全图坐标），决定下一次 crop 的位置
        self._last_box: Dict[int, Tuple[float, float, float, float]] = {}
        #: 统计
        self._stats = {
            "updates": 0, "inits": 0, "errors": 0, "reconnects": 0,
            "crop_updates": 0, "full_updates": 0, "fallback_full": 0,
            "bytes_sent": 0,
            #: ★共享帧链路的错误回显★：只记计数器（`ref_errors`）的话，一旦偶发
            #: 失败就只能看到"9 次错误"却不知道错在哪 —— 实测排查这类尾巴（Tracker
            #: Age 偶发飙到 1~2 s）全靠它。
            "ref_inits": 0, "ref_updates": 0, "ref_errors": 0,
            "last_ref_error": "",
        }
        self._lat: List[float] = []
        self._last_health: Dict[str, Any] = {"ok": False, "error": "未连接"}
        self._health_ts = 0.0
        self._probe_health()

    # ── 内部工具 ──────────────────────────────────────────────────────────
    def _probe_health(self) -> None:
        try:
            self._last_health = self._http("GET", "/health") or self._last_health
        except Exception as exc:  # noqa: BLE001
            self._last_health = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        self._health_ts = time.monotonic()

    @staticmethod
    def _encode_jpeg(image: Any, quality: int) -> str:
        if image is None:
            raise ValueError("frame is None")
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("frame must be 3-channel BGR ndarray")
        import cv2
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(arr),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise ValueError("jpeg encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _crop_around(self, frame: Any, box: Tuple[float, float, float, float],
                     factor: Optional[float] = None,
                     min_side: Optional[int] = None) -> Tuple[Any, int, int]:
        """按已知框裁一块正方形区域，返回 `(crop, ox, oy)`。

        crop 边长 = clamp(目标最大边 × factor, min_side, max_crop)，
        居中于框中心并 clamp 进图像边界。**返回真实原点** (ox, oy)，
        服务端靠它把框换算回全图坐标。

        `factor` 默认取 `crop_factor`（= OSTrack 的 SEARCH_FACTOR 4.0，update 用）；
        initialize 传 `init_factor`（≈2.5，对应 TEMPLATE_FACTOR 2.0 加一点余量），
        模板只需要目标本身那一圈，没必要按搜索区那么大的块上传。
        """
        arr = np.asarray(frame)
        h, w = arr.shape[0], arr.shape[1]
        bw = max(1.0, float(box[2] - box[0]))
        bh = max(1.0, float(box[3] - box[1]))
        side = int(max(int(min_side if min_side is not None else self.min_crop),
                       min(self.max_crop,
                           max(bw, bh) * float(factor if factor is not None
                                               else self.crop_factor))))
        if side >= min(h, w):
            #: crop 已经跟全图差不多大 —— 传全图，省掉一次切片与坐标换算
            return arr, 0, 0
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        x0 = int(max(0, min(w - side, cx - side * 0.5)))
        y0 = int(max(0, min(h - side, cy - side * 0.5)))
        return arr[y0:y0 + side, x0:x0 + side], x0, y0

    @staticmethod
    def _crop_ok(box: Tuple[float, float, float, float],
                 ox: int, oy: int, side: int) -> bool:
        """这一帧的框还好好地在 crop 里吗（贴边说明目标快跑出去了）。"""
        if side <= 0:
            return False
        m = side * 0.06
        return (box[0] - ox >= m and box[1] - oy >= m
                and (ox + side) - box[2] >= m and (oy + side) - box[3] >= m)

    def _http(self, method: str, path: str,
              payload: Optional[Dict[str, Any]] = None,
              timing: Optional[Dict[str, Any]] = None,
              slot_suffix: str = "") -> Dict[str, Any]:
        body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") \
            if payload is not None else None
        headers = {"Content-Type": "application/json",
                   "x-ostrack-token": self.token}
        if body is not None:
            headers["Content-Length"] = str(len(body))
            with self._lock:
                self._stats["bytes_sent"] += len(body)
        status, data = tunnel_http_request_persistent(
            self.node_id, self.port, method, path, body, headers,
            self.timeout, timing, slot_suffix=slot_suffix)
        if status != 200:
            raise ConnectionError("http %d: %s"
                                  % (status, (data or b"")[:200].decode("utf-8", "replace")))
        return json.loads(data.decode("utf-8"))

    def _result(self, track_id: int, d: Dict[str, Any]) -> TrackerResult:
        bbox = d.get("bbox_xyxy")
        extra = dict(d.get("extra") or {})
        #: ★服务端实际用的那一帧★：服务端把它放在响应**顶层**（`extra` 里没有），
        #: 而"VLX 帧号 == OSTrack 帧号"是验收项之一 —— 必须能从上层的
        #: `TrackerResult.extra["frame_id"]` 读到，否则只能靠日志比对。
        for k in ("frame_id", "media_time_ms"):
            if d.get(k) is not None and k not in extra:
                extra[k] = d.get(k)
        return TrackerResult(
            track_id=track_id,
            bbox_xyxy=tuple(float(v) for v in bbox) if bbox else None,
            score=float(d.get("score") or 0.0),
            ok=bool(d.get("ok")),
            latency_ms=float(d.get("latency_ms") or 0.0),
            reason=d.get("reason") or "",
            extra=extra,
        )

    def _record(self, latency_ms: float) -> None:
        with self._lock:
            self._lat.append(float(latency_ms))
            if len(self._lat) > _LAT_SAMPLES:
                del self._lat[:len(self._lat) - _LAT_SAMPLES]

    # ── TrackerBackend 契约 ───────────────────────────────────────────────
    def initialize(self, track_id: int, frame: Any, bbox_xyxy: Sequence[float]) -> TrackerResult:
        t0 = time.perf_counter()
        box = tuple(float(v) for v in bbox_xyxy)
        try:
            #: init 也走 crop：模板只需要目标周围 2.5× 那一圈，传 1080p 全图
            #: 实测要 2.1 s（745 KB@0.3MB/s），裁完只剩十几 KB。
            if self.crop_enabled:
                patch, ox, oy = self._crop_around(
                    frame, box, factor=self.init_factor, min_side=self.min_init_crop)
                box_in = (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)
            else:
                patch, ox, oy = frame, 0, 0
                box_in = box
            d = self._http("POST", "/track/init", {
                "track_id": int(track_id),
                "image": self._encode_jpeg(patch, self.jpeg_quality),
                "bbox": [float(v) for v in box_in],
                "offset": [int(ox), int(oy)],
            })
            r = self._result(track_id, d)
            r.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            with self._lock:
                self._stats["inits"] += 1
                if r.ok and r.bbox_xyxy is not None:
                    self._last_box[int(track_id)] = r.bbox_xyxy
                else:
                    self._last_box.pop(int(track_id), None)
            self._record(r.latency_ms)
            return r
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats["errors"] += 1
                self._last_box.pop(int(track_id), None)
            LOG.warning("[OSTrack-remote] initialize 失败 track=%s: %s", track_id, exc)
            return TrackerResult(track_id, ok=False,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 reason="remote:%s:%s" % (type(exc).__name__, exc))

    def update(self, track_id: int, frame: Any) -> TrackerResult:
        t0 = time.perf_counter()
        tid = int(track_id)
        timing: Dict[str, Any] = {}
        try:
            #: 有上一帧框就只传那一小块；没有（首帧 / 刚 reset / 上一帧失败）传全图
            box = self._last_box.get(tid)
            if self.crop_enabled and box is not None:
                patch, ox, oy = self._crop_around(frame, box)
                cropped = (ox, oy) != (0, 0)
                prev_side = int(patch.shape[1])
            else:
                patch, ox, oy = frame, 0, 0
                cropped = False
                prev_side = 0

            t_enc = time.perf_counter()
            b64 = self._encode_jpeg(patch, self.jpeg_quality)
            enc_ms = (time.perf_counter() - t_enc) * 1000.0

            d = self._http("POST", "/track/update", {
                "track_id": tid,
                "image": b64,
                "offset": [int(ox), int(oy)],
            }, timing,
                slot_suffix=(str(tid % self.parallel_channels)
                             if self.parallel_channels > 1 else ""))
            r = self._result(tid, d)
            r.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            st = d.get("server_timing") or {}
            r.extra.update({
                "transport": timing,
                "encode_ms": round(enc_ms, 2),
                "cropped": cropped,
                "offset": [int(ox), int(oy)],
                "patch": "%dx%d" % (patch.shape[1], patch.shape[0]),
                "server_timing": st,
            })

            with self._lock:
                self._stats["updates"] += 1
                self._stats["crop_updates" if cropped else "full_updates"] += 1
                if timing.get("rebuilt") or timing.get("persistent_retry"):
                    self._stats["reconnects"] += 1
            self._record(r.latency_ms)

            #: 下一次要不要继续 crop：没跟上 / 分数低 / 框贴边 → 回退全图重来
            keep_crop = False
            if r.ok and r.bbox_xyxy is not None:
                if (not cropped) or self._crop_ok(r.bbox_xyxy, ox, oy, prev_side):
                    keep_crop = r.score >= _LOW_SCORE
                else:
                    with self._lock:
                        self._stats["fallback_full"] += 1
                    LOG.debug("[OSTrack-remote] track=%s 框贴 crop 边 → 下一帧回退全图", tid)
            with self._lock:
                if keep_crop and r.bbox_xyxy is not None:
                    self._last_box[tid] = r.bbox_xyxy
                else:
                    self._last_box.pop(tid, None)
            return r
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats["errors"] += 1
                self._last_box.pop(tid, None)
            LOG.debug("[OSTrack-remote] update 失败 track=%s: %s", track_id, exc)
            return TrackerResult(tid, ok=False,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 reason="remote:%s:%s" % (type(exc).__name__, exc))

    # ── ★服务器侧共享帧（reference frame）★ ───────────────────────────────
    @property
    def supports_ref(self) -> bool:
        return bool(self.ref_mode)

    def init_by_ref(self, track_id: int, bbox_xyxy: Sequence[float],
                    frame_id: int, media_time_ms: Optional[float] = None,
                    label: str = "", pts_ms: Optional[float] = None) -> TrackerResult:
        """只传 `track_id + bbox + frame_id`，**服务端自己从 StreamHub 取帧**。

        ★入参白名单★（用户逐条拍定，越界即视为设计错误）：
            track_id / bbox（label 仅初始化需要）/ frame_id / media_time_ms
        禁止：image / jpeg / base64 / crop / offset —— 一个字节的像素都不许上行。

        ★必须与 VLX 检出框同帧★：实测隔 27 帧初始化会让模板直接错位，框一路
        扩到全屏（曾误判成"跟踪器漂移"）。`frame_id` 必须是 VLX 检出时那一帧。
        """
        t0 = time.perf_counter()
        tid = int(track_id)
        box = tuple(float(v) for v in bbox_xyxy)
        body: Dict[str, Any] = {"track_id": tid, "bbox": [float(v) for v in box]}
        if frame_id is not None:
            body["frame_id"] = int(frame_id)
        if media_time_ms is not None:
            body["media_time_ms"] = float(media_time_ms)
        #: ★真实 PTS★（STEP1）。服务端不认识这个字段时会自然忽略它 —— 对本链路
        #: 没有任何功能影响，只是带上真实时间便于端到端对齐诊断。
        if pts_ms is not None:
            body["pts_ms"] = float(pts_ms)
        if label:
            body["label"] = str(label)
        try:
            d = self._http("POST", "/track/init_by_ref", body)
            r = self._result(tid, d)
            r.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            with self._lock:
                self._stats["ref_inits"] = int(self._stats.get("ref_inits", 0)) + 1
                if not r.ok:
                    self._stats["ref_errors"] = int(self._stats.get("ref_errors", 0)) + 1
                    self._stats["last_ref_error"] = "init_by_ref: %s" % r.reason
            self._record(r.latency_ms)
            return r
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats["errors"] += 1
                self._stats["ref_errors"] = int(self._stats.get("ref_errors", 0)) + 1
                self._stats["last_ref_error"] = "init_by_ref: %s: %s" % (
                    type(exc).__name__, exc)
            LOG.warning("[OSTrack-remote] init_by_ref 失败 track=%s: %s", tid, exc)
            return TrackerResult(tid, ok=False,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 reason="remote-ref:%s:%s" % (type(exc).__name__, exc))

    def update_at(self, track_id: int, frame_id: int,
                  media_time_ms: Optional[float] = None,
                  pts_ms: Optional[float] = None) -> TrackerResult:
        """只传 `track_id + frame_id`，**服务端自己从 StreamHub 取帧**。

        实测（A6000 本地 mp4，5 目标并发 40 轮）：逐目标 **15.46 Hz**、p50 60 ms、
        每次请求 **37 B**；旧 JPEG 模式是 2.80 Hz / p50 340 ms / 约 24.8 KB。
        """
        t0 = time.perf_counter()
        tid = int(track_id)
        timing: Dict[str, Any] = {}
        body: Dict[str, Any] = {"track_id": tid}
        if frame_id is not None:
            body["frame_id"] = int(frame_id)
        if media_time_ms is not None:
            body["media_time_ms"] = float(media_time_ms)
        if pts_ms is not None:
            body["pts_ms"] = float(pts_ms)
        try:
            d = self._http("POST", "/track/update_at", body, timing,
                           slot_suffix=(str(tid % self.parallel_channels)
                                        if self.parallel_channels > 1 else ""))
            r = self._result(tid, d)
            r.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            st = d.get("server_timing") or {}
            r.extra.update({
                "transport": timing,
                "server_timing": st,
                "ref": True,
                "req_frame_id": int(frame_id) if frame_id is not None else None,
                "req_pts_ms": None if pts_ms is None else round(float(pts_ms), 3),
            })
            with self._lock:
                self._stats["ref_updates"] = int(self._stats.get("ref_updates", 0)) + 1
                if not r.ok:
                    self._stats["ref_errors"] = int(self._stats.get("ref_errors", 0)) + 1
                    self._stats["last_ref_error"] = "update_at: %s" % r.reason
            self._record(r.latency_ms)
            return r
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats["errors"] += 1
                self._stats["ref_errors"] = int(self._stats.get("ref_errors", 0)) + 1
                self._stats["last_ref_error"] = "update_at: %s: %s" % (
                    type(exc).__name__, exc)
            LOG.debug("[OSTrack-remote] update_at 失败 track=%s: %s", tid, exc)
            return TrackerResult(tid, ok=False,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 reason="remote-ref:%s:%s" % (type(exc).__name__, exc))

    def update_latest(self, track_id: int) -> TrackerResult:
        """★STEP3/P2★ 只发 `track_id`，由服务端在**收到请求的那一刻**自己取最新帧。

        为什么能压 Age：
            `update_at`   —— 帧号由**客户端**在发出时钉死，结果回来画面已走 3 帧，
                             Age = 一次**往返** RTT（实测 p50 200 ms）；
            `update_latest` —— 取帧时刻挪到服务端收到请求时，结果帧龄
                             = 往返 − 单程 = **单程**。

        ★响应里必须带回服务端实际用的那一帧★（`frame_id` / `media_time_ms` /
        `pts_ms`）：latest 模式下客户端**根本不知道**服务端取的是哪一帧，
        状态机若还拿客户端自己的 `cache.newest()` 去记 `last_frame_id`，
        就会把框记在错误的帧上（客户端时间线被 10 Hz 轮询量化过，通常落后
        于服务端）。所以这三个字段是本接口的**正确性**要求，不是诊断附加项。
        """
        t0 = time.perf_counter()
        tid = int(track_id)
        timing: Dict[str, Any] = {}
        try:
            d = self._http("POST", "/track/update_latest", {"track_id": tid},
                           timing,
                           slot_suffix=(str(tid % self.parallel_channels)
                                        if self.parallel_channels > 1 else ""))
            r = self._result(tid, d)
            r.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            st = d.get("server_timing") or {}
            _pts = d.get("pts_ms")
            r.extra.update({
                "transport": timing,
                "server_timing": st,
                "latest": True,
                "frame_id": (int(d["frame_id"])
                             if d.get("frame_id") is not None else None),
                "media_time_ms": float(d.get("media_time_ms") or 0.0),
                "pts_ms": (None if _pts is None else float(_pts)),
                "update_time": d.get("update_time"),
            })
            with self._lock:
                self._stats["latest_updates"] = int(
                    self._stats.get("latest_updates", 0)) + 1
                if not r.ok:
                    self._stats["ref_errors"] = int(
                        self._stats.get("ref_errors", 0)) + 1
                    self._stats["last_ref_error"] = "update_latest: %s" % r.reason
            self._record(r.latency_ms)
            return r
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._stats["errors"] += 1
                self._stats["ref_errors"] = int(
                    self._stats.get("ref_errors", 0)) + 1
                self._stats["last_ref_error"] = "update_latest: %s: %s" % (
                    type(exc).__name__, exc)
            LOG.debug("[OSTrack-remote] update_latest 失败 track=%s: %s", tid, exc)
            return TrackerResult(tid, ok=False,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 reason="remote-latest:%s:%s" % (
                                     type(exc).__name__, exc))

    def reset(self, track_id: int) -> None:
        with self._lock:
            self._last_box.pop(int(track_id), None)
        try:
            self._http("POST", "/track/reset", {"track_id": int(track_id)})
        except Exception as exc:  # noqa: BLE001
            LOG.debug("[OSTrack-remote] reset 失败 track=%s: %s", track_id, exc)

    def remove(self, track_id: int) -> None:
        with self._lock:
            self._last_box.pop(int(track_id), None)
        try:
            self._http("POST", "/track/remove", {"track_id": int(track_id)})
        except Exception as exc:  # noqa: BLE001
            LOG.debug("[OSTrack-remote] remove 失败 track=%s: %s", track_id, exc)

    def health(self) -> Dict[str, Any]:
        #: ★不能只返回构造时的快照★：那样服务端中途挂掉之后，`health()` 会一直
        #: 报 ok=True —— 前端和日志看到的是"一切正常，但没有框"。所以带 5 秒
        #: TTL 真探一次（GET /health 很轻），既能反映真实状态又不会打爆服务端。
        if (time.monotonic() - self._health_ts) > _HEALTH_TTL:
            self._probe_health()
        with self._lock:
            h = dict(self._last_health or {})
            st = dict(self._stats)
            lat = list(self._lat)
        h.setdefault("name", self.name)
        h["remote"] = True
        h["port"] = self.port
        h["crop_enabled"] = self.crop_enabled
        h["jpeg_quality"] = self.jpeg_quality
        #: ★服务器侧共享帧★：让前端/日志一眼看出当前跑的是哪条链路
        h["ref_mode"] = bool(self.ref_mode)
        h["frame_mode"] = "streamhub-ref" if self.ref_mode else "image-jpeg"
        if self.hub_client is not None:
            try:
                h["hub"] = self.hub_client.stats()
            except Exception:  # noqa: BLE001
                h["hub"] = {"ok": False}
        h["stats"] = st
        if lat:
            h["latency_ms"] = {
                "avg": round(sum(lat) / len(lat), 2),
                "p50": round(_pct(lat, 50), 2),
                "p95": round(_pct(lat, 95), 2),
                "max": round(max(lat), 2),
                "samples": len(lat),
            }
            h["update_fps"] = round(1000.0 / (sum(lat) / len(lat)), 2) if lat else 0.0
        if st.get("updates"):
            total = st["crop_updates"] + st["full_updates"]
            h["crop_ratio"] = round(st["crop_updates"] / max(1, total), 3)
            h["avg_kb_per_update"] = round(st["bytes_sent"] / max(1, total) / 1024.0, 1)
        return h

    def track_ids(self) -> List[int]:
        try:
            d = self._http("POST", "/track/ids", {})
            return [int(v) for v in (d.get("track_ids") or [])]
        except Exception:  # noqa: BLE001
            return []

    def close(self) -> None:
        # 服务端模型是常驻共享的，不因某个任务结束而关闭。
        with self._lock:
            self._last_box.clear()
