# -*- coding: utf-8 -*-
"""StreamHub 客户端 —— RunOS 侧**只问帧号，不搬像素**。

为什么要有它
────────────────────────────────────────────────────────────────────────────
旧链路里 RunOS 把每一帧 JPEG 化、base64、穿过 SSH 隧道发给 OSTrack。实测那条
路的物理天花板是 **2.2 Mbps / RTT 65 ms**，5 个目标只能跑到 2.8 Hz（p50 340 ms）。

StreamHub 把取帧搬到 A6000 本地之后，RunOS 与服务端之间只剩**几百字节的 JSON**：

    RunOS ──{frame_id}──►  VLX     /infer_at      （服务端自己取帧）
    RunOS ──{frame_id}──►  OSTrack /track/init_by_ref
                          OSTrack /track/update_at

本模块就是 RunOS 拿 `frame_id` 的地方：`GET /health` 返回 StreamHub 当前最新
帧号与流时间，RunOS 据此在**本机**维护一条"帧时间线"（只存 id 与时间，不存像素），
TargetManager 的 replay / 追帧逻辑因此完全不用改。

硬约束（与 hub.py 对齐）
  * **绝不接受图片**——本客户端没有、也不该有任何返回 ndarray / JPEG 的方法；
  * 失败一律抛 `StreamHubError`，由上层决定是否降级到旧 image 模式；
  * 走 `tunnel_http` 的持久连接槽位，与 VLX / OSTrack 各自隔离。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from ..tunnel_http import tunnel_http_request_persistent

LOG = logging.getLogger(__name__)

_DEFAULT_PORT = 8030
_DEFAULT_TIMEOUT = 10.0
_HEALTH_TTL = 2.0


class StreamHubError(RuntimeError):
    """StreamHub 不可达 / 返回异常。"""


class HubFrame:
    """一帧的**引用**（只有编号与时间，没有像素）。

    三个时间并存（与 hub.py 的文件头对齐）：
      media_time_ms   合成值 = frame_id × 1000/fps。保留只为兼容既有索引/外推。
      pts_ms          ★真实流内 PTS★。None = 未知（旧 hub / 该帧没抽到）。
                      这是 STEP2 算 `age_ms = video_pts - tracker_pts` 的依据。
      hub_wall_ms     hub 侧收到该帧的**墙钟**（A6000 时钟）。跨机比较要先估钟差，
                      不能直接与本机 `time.time()` 相减。
    """

    __slots__ = ("frame_id", "media_time_ms", "pts_ms", "hub_wall_ms")

    def __init__(self, frame_id: int, media_time_ms: float,
                 hub_wall_ms: float = 0.0,
                 pts_ms: Optional[float] = None) -> None:
        self.frame_id = int(frame_id)
        self.media_time_ms = float(media_time_ms)
        self.hub_wall_ms = float(hub_wall_ms)
        self.pts_ms = None if pts_ms is None else float(pts_ms)

    @property
    def has_pts(self) -> bool:
        return self.pts_ms is not None

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return "HubFrame(id=%d, t=%.1fms, pts=%s)" % (
            self.frame_id, self.media_time_ms,
            "na" if self.pts_ms is None else "%.1fms" % self.pts_ms)


class StreamHubClient:
    """`GET /health` 的薄封装 + 一次性自检。

    `/health` 足够轻（几百字节），可以十几 Hz 轮询；真正的帧传输全程不经过
    RunOS —— 这是整套架构的意义所在。
    """

    name = "streamhub"

    def __init__(self, node_id: str = "", port: int = _DEFAULT_PORT,
                 token: str = "", timeout: float = _DEFAULT_TIMEOUT,
                 slot: str = "hub", streamhub_url: str = "") -> None:
        self.node_id = node_id or ""
        self.port = int(port or _DEFAULT_PORT)
        self.token = token or ""
        self.timeout = float(timeout or _DEFAULT_TIMEOUT)
        self.slot = str(slot or "hub")
        #: 预留：将来 StreamHub 暴露到内网直连地址时不必改调用方
        self.streamhub_url = streamhub_url or ""
        self._stats: Dict[str, Any] = {
            "polls": 0, "errors": 0, "last_error": "",
            "last_frame_id": None, "last_media_time_ms": None,
            "fps": 0.0, "ring_used": 0, "frames_total": 0,
        }
        self._last_health: Dict[str, Any] = {}

    # ── 低层 ──────────────────────────────────────────────────────────────
    def _get(self, path: str, timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["x-streamhub-token"] = self.token
        status, data = tunnel_http_request_persistent(
            self.node_id, self.port, "GET", path, None, headers,
            self.timeout, timing, slot_suffix=self.slot)
        if status != 200:
            raise StreamHubError("http %d: %s"
                                 % (status, (data or b"")[:200].decode("utf-8", "replace")))
        try:
            return json.loads(data.decode("utf-8"))
        except ValueError as exc:
            raise StreamHubError("响应不是合法 JSON：%s" % exc) from exc

    # ── 公开 API ──────────────────────────────────────────────────────────
    def health(self, timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """探一次 `/health`。**会抛** `StreamHubError`。"""
        try:
            d = self._get("/health", timing)
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            self._stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
            raise
        self._stats["polls"] += 1
        self._last_health = d
        self._stats.update({
            "last_frame_id": d.get("last_frame_id"),
            "last_media_time_ms": d.get("last_media_time_ms"),
            "last_pts_ms": d.get("last_pts_ms"),
            "fps": float(d.get("fps") or 0.0),
            "ring_used": int(d.get("ring_used") or 0),
            "frames_total": int(d.get("frames_total") or 0),
        })
        self._stats["last_error"] = ""
        return d

    def latest(self, timing: Optional[Dict[str, Any]] = None) -> Optional[HubFrame]:
        """当前最新帧的引用；StreamHub 还没出帧时返回 None。

        ★`hub_wall_ms` / `pts_ms` 以前从不填（恒 0 / None）★ —— 字段一直在，
        但 `/health` 里那两个值没人读。STEP1 之后这里把它们填上，消费端才第一次
        拿到真实时间。旧 hub（没有这些字段）会自然拿到 None，不会炸。
        """
        d = self.health(timing)
        fid = d.get("last_frame_id")
        if fid is None:
            return None
        pts = d.get("last_pts_ms")
        return HubFrame(int(fid), float(d.get("last_media_time_ms") or 0.0),
                        hub_wall_ms=float(d.get("last_recv_wall_ms") or 0.0),
                        pts_ms=None if pts is None else float(pts))

    def frame_jpeg(self, frame_id: int, timing: Optional[Dict[str, Any]] = None,
                   quality: int = 95) -> Optional[bytes]:
        """★告警证据★：按帧号取一张 JPEG 原始字节，拿不到返回 None。

        ★为什么要带 `quality`★：StreamHub 的 `/frame.jpg` 默认按 q80 编码，
        而证据图是要**放大看远距离目标**的 —— 源文件先压成 q80、我们再解码画框
        重编码一次，两代有损叠加之后小目标就只剩马赛克了。所以这里显式要 q95
        （`?q=95`，服务端不支持时忽略该参数，行为与旧版一致，不会更差）。

        为什么破例：本客户端的硬约束是"绝不搬像素"——那条约束针对的是**逐帧
        追帧**（2.2 Mbps 的物理天花板就是被它撞上的）。告警证据是**低频、按需**
        的（一条告警三张图，一天也就几十张），与每秒十几次的追帧不是一个量级，
        撞不到那条天花板。

        ★失败一律返回 None，绝不抛★：取不到证据图只该让告警少三张缩略图，
        绝不能把告警本身弄丢，更不能让 VLX 循环挂掉。
        """
        try:
            headers = {"Accept": "image/jpeg"}
            if self.token:
                headers["x-streamhub-token"] = self.token
            status, data = tunnel_http_request_persistent(
                self.node_id, self.port, "GET",
                "/frame.jpg?id=%d&q=%d" % (int(frame_id), int(quality)),
                None, headers,
                self.timeout, timing, slot_suffix="evidence")
            if status != 200 or not data:
                LOG.warning("[StreamHub] 取证据帧失败 frame=%s status=%s",
                            frame_id, status)
                return None
            if not data[:2] == b"\xff\xd8":
                LOG.warning("[StreamHub] 证据帧不是 JPEG frame=%s", frame_id)
                return None
            return data
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[StreamHub] 取证据帧异常 frame=%s：%s", frame_id, exc)
            return None

    def try_health(self) -> Dict[str, Any]:
        """**不抛**版本：失败返回 `{ok: False, error: ...}`，给启动期自检用。"""
        try:
            d = self.health()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        d.setdefault("ok", True)
        return d

    # ── 观测 ──────────────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return float(self._stats.get("fps") or 0.0)

    def stats(self) -> Dict[str, Any]:
        out = dict(self._stats)
        out["name"] = self.name
        out["port"] = self.port
        out["slot"] = self.slot
        return out


class StreamHubFeed:
    """★把 StreamHub 的帧时间线搬到本机★（只搬 id 与时间，不搬像素）。

    TargetManager 的 `step()` 靠"出现了新帧号"来决定要不要发 `update_at`，
    `_reinit_and_replay()` 靠"VLX 那一帧之后有哪些帧"来做 replay —— 两者都只要
    帧号与时间，不需要像素。所以这里以固定频率轮询 `/health`，把新出现的
    frame_id **逐个补齐**（两个轮询之间 Hub 可能已经出了好几帧）写进
    `TrackerFrameCache`。

    补号而不是只记最新：只记最新会让 replay_plan 拿到一条"跳号"的时间线，
    `bbox_at()` 按最近邻找历史位置时会跨过真实运动过程。
    """

    def __init__(self, client: StreamHubClient, cache: Any,
                 poll_hz: float = 10.0, stop: Any = None) -> None:
        self.client = client
        self.cache = cache
        self.poll_hz = float(poll_hz or 10.0)
        self.stop = stop
        self._thread = None
        self._last_id = 0
        self._first_id = None
        #: ★外推基准★：最近一次轮询拿到的 (帧号, 流时间) + 本机墙钟。
        #: 轮询有 RTT（实测 ~130 ms），两次轮询之间 Hub 其实一直在出帧；
        #: 直接把"上次轮询到的时间"当 now 会把 Tracker Age 抬高一个轮询周期。
        #: Hub 的 `media_time_ms = frame_id / fps` 是**恒定速率**，所以按墙钟
        #: 线性外推是准确的（不需要再打一次请求）。
        self._ref_id = 0
        self._ref_media_ms = 0.0
        self._ref_wall_ms = 0.0
        #: ★真实 PTS 的外推基准★：Hub 的 pts 步长与 media_time 步长同为
        #: 1000/fps（fps 滤镜已把 PTS 均匀化，实测恒为 66.67 ms），所以同样可以
        #: 按帧数线性外推。旧 hub 没有 pts → 恒为 None，下游自动退化为只用帧号。
        self._ref_pts_ms: Optional[float] = None
        self._step_ms = 0.0          # 每帧间隔（= 1000 / fps），首次轮询后才有
        self.stats: Dict[str, Any] = {
            "polls": 0, "frames": 0, "errors": 0, "last_error": "",
            "last_frame_id": 0, "gap_max": 0,
            "last_pts_ms": None, "pts_seen": 0, "pts_missing": 0,
        }

    # ── "现在"是多少 ──────────────────────────────────────────────────────
    def now(self) -> Optional[HubFrame]:
        """Hub 当前帧的**外推值**（不额外发请求）。

        为什么要外推：轮询一次要一个 RTT（实测 ~130 ms），两次轮询之间 Hub 已经
        又出了好几帧。若直接拿"上次轮询到的值"当 now，Tracker Age 会被平白加上
        一个轮询周期（实测 p95 因此从 ~300 ms 涨到 533 ms）。

        Hub 的 `media_time_ms = frame_id / fps` 恒定速率 → 按墙钟线性外推即可，
        误差只有两侧时钟的漂移，远小于一个 RTT。
        """
        if self._ref_wall_ms <= 0:
            return None
        dt = max(0.0, time.time() * 1000.0 - self._ref_wall_ms)
        add_f = int(dt / self._step_ms) if self._step_ms > 0 else 0
        pts = (None if self._ref_pts_ms is None
               else self._ref_pts_ms + add_f * self._step_ms)
        return HubFrame(self._ref_id + add_f,
                        self._ref_media_ms + add_f * self._step_ms,
                        pts_ms=pts)

    # ── 线程 ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        import threading

        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="streamhub-feed",
                                        daemon=True)
        self._thread.start()

    def stop_feed(self, timeout: float = 3.0) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        period = 1.0 / max(0.5, self.poll_hz)
        while True:
            if self.stop is not None and self.stop.is_set():
                break
            t0 = time.monotonic()
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
                LOG.debug("[streamhub-feed] 轮询失败：%s", exc)
            dt = time.monotonic() - t0
            wait = period - dt
            if wait > 0 and self.stop is not None:
                self.stop.wait(wait)
            elif wait > 0:
                time.sleep(wait)

    def _poll_once(self) -> None:
        d = self.client.health()
        self.stats["polls"] += 1
        fid = d.get("last_frame_id")
        if fid is None:
            return
        fid = int(fid)
        mt = float(d.get("last_media_time_ms") or 0.0)
        fps = float(d.get("fps") or 0.0)
        step_ms = (1000.0 / fps) if fps > 0 else 0.0
        _pts = d.get("last_pts_ms")
        pts_top = None if _pts is None else float(_pts)
        if self._first_id is None:
            #: 首次对齐：只认最新这一帧，避免把 Hub 已经跑掉的几千帧灌进缓存
            self._first_id = fid
            self._last_id = fid - 1
        if fid <= self._last_id:
            return
        gap = fid - self._last_id
        if gap > self.stats["gap_max"]:
            self.stats["gap_max"] = int(gap)
        if step_ms > 0:
            self._step_ms = step_ms
        #: ★外推基准★ 在**读到最新帧的这一刻**取墙钟 —— 取晚一点（比如放在写
        #: 缓存之后）会把写缓存的时间也算进"Hub 又出了几帧"，外推就会偏大。
        self._ref_id = fid
        self._ref_media_ms = mt
        self._ref_pts_ms = pts_top
        self._ref_wall_ms = time.time() * 1000.0
        #: 每帧的流时间 = 最新帧时间往前推（Hub 的 media_time_ms = frame_id/fps；
        #: pts 同理，因为 fps 滤镜已把 PTS 均匀化，步长恒为 1000/fps）
        for k in range(self._last_id + 1, fid + 1):
            back = (fid - k) * step_ms if step_ms > 0 else 0.0
            t_ms = mt - back
            p_ms = None if pts_top is None else pts_top - back
            self.cache.put_ref(k, max(0.0, t_ms), pts_ms=p_ms)
        self._last_id = fid
        self.stats["frames"] += int(gap)
        self.stats["last_frame_id"] = fid
        self.stats["last_pts_ms"] = pts_top
        if pts_top is None:
            self.stats["pts_missing"] += int(gap)
        else:
            self.stats["pts_seen"] += int(gap)
