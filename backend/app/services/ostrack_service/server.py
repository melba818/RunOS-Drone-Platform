# -*- coding: utf-8 -*-
"""OSTrack 跟踪服务（A6000 / GPU1 / FP16，常驻）。

零额外依赖：标准库 `ThreadingHTTPServer` + numpy + PIL + torch（复用 vlxseek15 venv）。
OSTrack 的 timm 依赖由 `tracking/_vendor/timm` 最小替身解决，不装真 timm。

只绑 **127.0.0.1:8022** —— 公网不放行，RunOS 通过算力节点的 SSH `direct-tcpip`
隧道访问。鉴权用 `x-ostrack-token` 头（token 在 token.txt，与 VLX 同款约定）。

端点（与 `lmm/tracking/ostrack.py OSTrackBackend` 接口一一对应）：
    GET  /health         → {ok, name, device, fp16, tracks, load_seconds, vram_*, ...}
    POST /track/init     → {track_id, image(b64 jpeg), bbox:[x1,y1,x2,y2]} → TrackerResult
    POST /track/update   → {track_id, image(b64 jpeg), offset?:[ox,oy]} → TrackerResult
    POST /track/reset    → {track_id} → {ok}
    POST /track/remove   → {track_id} → {ok}
    POST /track/clear    → {} → {ok}   （清空本服务所有 track）
    POST /track/ids      → {} → {track_ids:[...]}

frame 约定：客户端发的是 **BGR ndarray 经 JPEG 编码后的 base64**；服务端用 PIL 解码
成 RGB 再转回 BGR，喂给 OSTrackBackend（它内部 `_to_rgb` 再转 RGB）——顺序两两抵消，
服务端不感知、不误转。

────────────────────────────────────────────────────────────────────────────
★STEP 2.1 传输优化★（`.workbuddy/ostrack_size_sweep.txt` 实测）

端到端 2.6 s 的真凶**不是** SSH 握手，而是**上传体积**：
    b64 745 KB（1080p q90） → upload 2428 ms，吞吐只有 0.30 MB/s
    b64  49 KB（384² q90）  → upload   49 ms
    b64  29 KB（384² q75）  → upload    1 ms
体积降下来后剩下的地板是 channel_open ~115 ms + wait_first_byte ~82 ms。

因此两件事一起做：
  1. `protocol_version = "HTTP/1.1"` + 持久 channel  → 摊掉 115 ms 的建连；
  2. `offset` 字段：客户端只传**目标周围的一小块 crop**，服务端把 tracker 状态
     与输出框在两个坐标系之间换算，调用方看到的仍是全图坐标。

`offset:[ox,oy]` 语义：本次上传的图是**全图**从 (ox,oy) 处裁下来的。
服务端为每条 track 记住"上次那张图的 offset"，本次 offset 变了就把 tracker 内部
状态（上一帧框）平移 `(ox-prev_ox, oy-prev_oy)` 到本次图的坐标系，推理完再把输出框
加回 (ox,oy) 变回全图坐标。这样**调用方完全不需要知道裁剪这件事**。
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

HOST = os.environ.get("OSTRACK_HOST", "127.0.0.1")
PORT = int(os.environ.get("OSTRACK_PORT", "8022"))
TOKEN_FILE = os.environ.get("OSTRACK_TOKEN_FILE", HERE + "/token.txt")
CKPT = os.environ.get(
    "OSTRACK_CKPT",
    "/srv/runhive/models/ostrack/vitb_256_mae_ce_32x4_ep300.pth.tar")
MAX_BODY = int(os.environ.get("OSTRACK_MAX_BODY_MB", "32")) * 1048576
#: keep-alive 连接最多保持多久（秒）。太长会白占线程，太短会退化成每请求建连。
KEEPALIVE_TIMEOUT = int(os.environ.get("OSTRACK_KEEPALIVE_S", "60"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tracking.ostrack import OSTrackBackend  # noqa: E402

_TOKEN: str = ""
_backend = None
_load_started = 0.0
#: track_id -> (ox, oy)：这条 track 的 tracker 状态当前处在哪个偏移的图上。
#: 只在服务端内部使用，调用方无感。
_OFFSETS: dict = {}
_OFF_LOCK = threading.Lock()
#: 统计
_STATS = {"requests": 0, "updates": 0, "inits": 0, "errors": 0,
          "bytes_in": 0, "crop_updates": 0, "full_updates": 0}


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_backend():
    global _backend, _load_started
    _load_started = time.time()
    log("loading OSTrack ckpt=%s" % CKPT)
    _backend = OSTrackBackend(checkpoint=CKPT, device="cuda", fp16=True)
    h = _backend.health()
    log("loaded in %.1fs: %s" % (time.time() - _load_started, json.dumps(h, ensure_ascii=False)))
    return h


def b64_to_bgr(b64: str):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must be 3-channel")
    return arr[:, :, ::-1]  # RGB -> BGR


def _offset_of(body) -> tuple:
    off = body.get("offset") if isinstance(body, dict) else None
    if isinstance(off, (list, tuple)) and len(off) == 2:
        try:
            return (float(off[0]), float(off[1]))
        except (TypeError, ValueError):
            return (0.0, 0.0)
    return (0.0, 0.0)


class Handler(BaseHTTPRequestHandler):
    #: ★HTTP/1.1★：默认 HTTP/1.0 会在每次响应后关连接，客户端想复用也复用不了
    #: （实测现象就是 IncompleteRead / ConnectionRefused 交替出现）。
    protocol_version = "HTTP/1.1"
    #: 有了 HTTP/1.1 keep-alive，一个连接就占一个线程，必须有超时兜底。
    timeout = KEEPALIVE_TIMEOUT

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:  # noqa: BLE001
            pass

    def _authed(self) -> bool:
        got = self.headers.get("x-ostrack-token", "")
        return bool(_TOKEN) and got == _TOKEN

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            raise ValueError("bad content-length %d" % n)
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            if _backend is None:
                return self._send(503, {"ok": False, "error": "model not loaded yet"})
            h = dict(_backend.health())
            h.setdefault("load_seconds",
                         round(time.time() - _load_started, 1) if _load_started else None)
            h["stats"] = dict(_STATS)
            h["keepalive"] = self.protocol_version
            return self._send(200, h)
        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        t_recv = time.perf_counter()
        try:
            body = self._read_json()
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": "bad json: %s" % exc})
        t_body = time.perf_counter()
        if _backend is None:
            return self._send(503, {"error": "model not loaded yet"})
        try:
            path = self.path.rstrip("/")
            _STATS["requests"] += 1

            if path == "/track/init":
                tid = int(body["track_id"])
                t0 = time.perf_counter()
                img = b64_to_bgr(body["image"])
                t_dec = time.perf_counter()
                ox, oy = _offset_of(body)
                with _OFF_LOCK:
                    _OFFSETS[tid] = (ox, oy)
                r = _backend.initialize(
                    tid, img, [float(v) for v in body["bbox"]])
                t_end = time.perf_counter()
                #: init 传的 bbox 是**本次图**的坐标系（客户端已减掉 offset），
                #: 输出同样要加回 offset 变回全图坐标，调用方才无感。
                if r.bbox_xyxy is not None and (ox or oy):
                    x1, y1, x2, y2 = r.bbox_xyxy
                    r.bbox_xyxy = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
                _STATS["inits"] += 1
                return self._send(200, self._decorate(
                    r, t_recv, t_body, t_dec, t_end))

            if path == "/track/update":
                tid = int(body["track_id"])
                t0 = time.perf_counter()
                img = b64_to_bgr(body["image"])
                t_dec = time.perf_counter()
                ox, oy = _offset_of(body)
                with _OFF_LOCK:
                    prev = _OFFSETS.get(tid)
                    if prev is None:
                        prev = (ox, oy)          # 首帧：以本次为准
                    _OFFSETS[tid] = (ox, oy)
                    if (ox, oy) != prev:
                        #: tracker 状态还停在上一张图的坐标系 → 平移到本次图上
                        try:
                            _backend.shift_state(tid, ox - prev[0], oy - prev[1])
                        except Exception as exc:  # noqa: BLE001
                            log("shift_state failed: %s" % exc)
                r = _backend.update(tid, img)
                t_end = time.perf_counter()
                #: 输出框从 crop 坐标 → 全图坐标（调用方永远只看到全图坐标）
                if r.bbox_xyxy is not None and (ox or oy):
                    x1, y1, x2, y2 = r.bbox_xyxy
                    r.bbox_xyxy = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
                _STATS["updates"] += 1
                if (ox or oy):
                    _STATS["crop_updates"] += 1
                else:
                    _STATS["full_updates"] += 1
                return self._send(200, self._decorate(r, t_recv, t_body, t_dec, t_end))

            if path == "/track/reset":
                tid = int(body["track_id"])
                _backend.reset(tid)
                with _OFF_LOCK:
                    _OFFSETS.pop(tid, None)
                return self._send(200, {"ok": True})
            if path == "/track/remove":
                tid = int(body["track_id"])
                _backend.remove(tid)
                with _OFF_LOCK:
                    _OFFSETS.pop(tid, None)
                return self._send(200, {"ok": True})
            if path == "/track/clear":
                for tid in list(_backend.track_ids()):
                    _backend.remove(tid)
                with _OFF_LOCK:
                    _OFFSETS.clear()
                return self._send(200, {"ok": True, "cleared": True})
            if path == "/track/ids":
                return self._send(200, {"track_ids": _backend.track_ids()})
            return self._send(404, {"error": "not found: %s" % path})
        except Exception as exc:  # noqa: BLE001
            _STATS["errors"] += 1
            log("request error: %s" % traceback.format_exc())
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    @staticmethod
    def _decorate(r, t_recv, t_body, t_dec, t_end):
        """给响应补上服务端细分耗时，让客户端能分清"慢在网络还是慢在推理"。"""
        d = r.to_dict()
        try:
            d["server_timing"] = {
                "body_read_ms": round((t_body - t_recv) * 1000.0, 2),
                "decode_ms": round((t_dec - t_body) * 1000.0, 2),
                "infer_ms": round(float(r.latency_ms or 0.0), 2),
                #: 从收到请求到推理结束；与纯推理的差 = 解码 + 调度排队
                "server_total_ms": round((t_end - t_recv) * 1000.0, 2),
            }
        except Exception:  # noqa: BLE001
            pass
        return d

    def log_message(self, *args):  # 静音默认访问日志，避免刷屏
        pass


def main() -> None:
    global _TOKEN
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as fh:
            _TOKEN = fh.read().strip()
    load_backend()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    #: keep-alive 下 daemon_threads 必须开，否则主线程退出时线程会挂着
    srv.daemon_threads = True
    log("listening on %s:%d (token=%s, http=%s, keepalive=%ss)"
        % (HOST, PORT, "set" if _TOKEN else "EMPTY",
           Handler.protocol_version, KEEPALIVE_TIMEOUT))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
