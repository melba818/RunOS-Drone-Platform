# -*- coding: utf-8 -*-
"""★P1：客户端侧 Tracker 异步化运行时★（STEP3 第一阶段）。

为什么要有它
────────────────────────────────────────────────────────────────────────────
改造前的 StreamHub 链路里，**三件事互相阻塞**：

  1. `on_vlx`（VLX 线程）里**同步**跑 `reset + init + ≤N 次 replay`
     —— 实测 ~983 ms，VLX 一来所有目标的逐帧 update 全部饿死；
  2. `TrackerWorker` 的 `_tick()` 发完 `update_at` 后**原地等返回**，
     一个 tick 的耗时 ≈ 一次 RTT，5 个目标挤在同一个 tick 里；
  3. tick 靠"轮询 + 跳过"驱动，实测 `step_skip_current` 2067 vs
     `step_calls` 2022 —— 几乎每次都在空转，更新频率被 feed 的 10 Hz
     轮询节拍卡住。

本文件把它们拆成**每 track 一条线程 + 一个状态机线程 + 一个节流发布线程**，
做到：

  ★ StreamHub feed 永不等待 tracker（它本来就只写帧时间线，本文件不碰它）
  ★ VLX 永不执行 init / replay（只投 `InitReq`，投完立刻返回）
  ★ 每个 track 独立 worker：一个 track 卡住不影响其它 track
  ★ 工作槽长度 = 1：Latest Frame Wins 是**结构上**的，不依赖任何丢帧判断

★本阶段**不降低 Age**★（P1 理论上限）
────────────────────────────────────────────────────────────────────────────
Age = "框所在帧" 与 "当前画面帧" 的差。P1 没有改服务端取帧语义，客户端仍然
指定 `frame_id=F`，RTT 200 ms 里画面照走 3 帧 → **Age p50 仍是 200 ms**。
把 Age 从往返压到单程要靠 P2（服务端取最新帧），那是**另一个阶段**、另一次改动。

P1 的收益是另外四项：吞吐、VLX 不再卡住跟踪、失败不阻塞整条链、Age 的
**长尾**（p95）显著下降。

线程模型
────────────────────────────────────────────────────────────────────────────
    shub-feed（外部，本文件不创建）  ──►  FrameTimeline(= TrackerFrameCache)
                                              │
    shub-vlx（外部）──► targets.on_vlx ──► InitReq（per track，长度 1）
                                              │
                            ┌─────────────────┴─────────────────┐
                            ▼                                   ▼
              rt-track-<tid>（每 track 一条）            rt-state（状态机）
              单槽流水线 in-flight = 1                   串行应用所有结果
                            │                                   │
                            └────────► ResultQueue ◄────────────┘
                                              │
                                      rt-ws-pub（脏触发 + 节流 + 心跳）
                                              │
                                      hub.publish（Queue 8，满丢最旧）

硬约束（与用户逐条拍定的一致）
  * 不改 OSTrack 模型 / 不改 VLX 模型 / 不改前端 Canvas / 不改 PTS 逻辑；
  * **前端 WS 协议一个字段都不许少**（`track_update` 沿用 `_make_track_publisher`）；
  * 旧 remote / image 模式**一行不动**（本文件只在 streamhub 模式下被装配）。
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

#: 发布节流。20 Hz 远超人眼可辨，也远超 tracker 的产出速度 —— 目的是**不要把
#: 发布频率绑死在 tracker 上**；没有它时，worker 跑多快就发多快，WS 队列会被
#: 冲垮（而队列满了丢最旧，反而把最新状态丢了）。
PUBLISH_HZ = 20.0

#: worker 空闲时的让出间隔。不能是 0（会把 feed 线程饿死），也不能太大
#: （会重新引入"节拍"）。2 ms 对应理论上限 500 Hz，实际受 RTT 限制远低于此。
IDLE_SLEEP_S = 0.002

#: 连续失败多少次后退避（OSTrack 服务挂掉时不要疯狂重试）
_ERR_BACKOFF = 3
_BACKOFF_MAX_S = 2.0


class InitSlot:
    """★per-track 的 init 请求槽，长度 = 1，Latest Wins★。

    为什么长度必须是 1：VLX 每 1.5 s 来一次，而 init 只要 ~200 ms，正常情况
    下不会积压。真正会积压的是"init 卡住"的时候 —— 那时排队的旧请求在被处理
    时早已过期（VLX 框属于好几秒前的帧），照着它初始化等于用旧位置建模板。
    所以新请求直接**覆盖**旧请求，旧的自然消失，不需要任何"丢"的判断。
    """

    __slots__ = ("_lock", "_req")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._req: Optional[Dict[str, Any]] = None

    def put(self, req: Dict[str, Any]) -> None:
        with self._lock:
            self._req = req

    def take(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            r = self._req
            self._req = None
            return r


class UpdateResult:
    """一条 update 的结果（worker → 状态机）。**不可变**，跨线程只读。"""

    __slots__ = ("track_id", "ok", "bbox_xyxy", "score", "frame_id",
                 "media_time_ms", "pts_ms", "reason", "latency_ms",
                 "newest", "req_wall_ms")

    def __init__(self, track_id: int, ok: bool,
                 bbox_xyxy: Optional[List[float]], score: float,
                 frame_id: Optional[int], media_time_ms: float,
                 pts_ms: Optional[float], reason: str, latency_ms: float,
                 newest: Any, req_wall_ms: float) -> None:
        self.track_id = int(track_id)
        self.ok = bool(ok)
        self.bbox_xyxy = bbox_xyxy
        self.score = float(score or 0.0)
        self.frame_id = frame_id
        self.media_time_ms = float(media_time_ms or 0.0)
        self.pts_ms = pts_ms
        self.reason = str(reason or "")
        self.latency_ms = float(latency_ms or 0.0)
        #: ★请求时用的那一帧（CachedFrame 引用）★：漂移守卫要用它的 frame_id
        #: 算 gap，状态机也要用它的 pts 更新 `last_pts_ms`。
        self.newest = newest
        self.req_wall_ms = float(req_wall_ms or 0.0)


class TrackWorker(threading.Thread):
    """★每 track 一条线程，单槽流水线（in-flight 恒为 1）★。

    为什么"每 track 一条"而不是"一个线程池 + 任务队列"：
      OSTrack 的状态是**链式依赖**的，`update(F+1)` 必须基于 `update(F)` 之后的
      内部状态。所以同一个 track 的请求必须严格串行 —— 线程内顺序执行天然满足，
      不需要任何调度；不同 track 之间本来就互相独立，天然并行。
      线程池方案反而要额外维护"这个 track 现在有没有在飞的请求"，容易出错。

    init 优先：每轮循环先处理属于自己的 `InitSlot`（init 没做完 track 是
    `pending_init`，根本不能 update），没有 init 才发 update。
    """

    def __init__(self, track_id: int, targets: Any, cache: Any,
                 init_slot: InitSlot, result_q: "queue.Queue",
                 stop: threading.Event, on_init_done: Any = None,
                 max_replay_frames: int = 8) -> None:
        super().__init__(daemon=True, name="rt-track-%d" % int(track_id))
        self.track_id = int(track_id)
        self.targets = targets
        self.cache = cache
        self.init_slot = init_slot
        self.result_q = result_q
        self.stop = stop
        self.on_init_done = on_init_done
        self.max_replay_frames = max(0, int(max_replay_frames))
        self._err_run = 0
        #: ★P2★ 自适应空闲间隔（见 `_pace`）
        self._idle = IDLE_SLEEP_S
        self._last_server_frame: Optional[int] = None
        self.stats: Dict[str, Any] = {
            "track_id": self.track_id,
            "inits": 0, "init_failed": 0, "updates": 0, "update_failed": 0,
            "replay_frames": 0, "errors": 0, "last_error": "",
            "last_latency_ms": 0.0, "alive": False,
        }

    # ── init ────────────────────────────────────────────────────────────
    def _do_init(self, req: Dict[str, Any]) -> None:
        """执行 init + replay。**必须在 VLX 线程之外**（P1 目标 2）。

        ★`frame_id` 与 `pts_ms` 严格来自 `InitReq`，一个字节都不重算★：
        init 隔帧会让模板错位、框一路扩到全屏（踩过）。
        """
        tid = int(req.get("track_id") or self.track_id)
        fid = int(req.get("frame_id") or 0)
        pts = req.get("pts_ms")
        box = tuple(float(v) for v in (req.get("bbox") or ())[:4])
        label = str(req.get("label") or "")
        mt = float(req.get("media_time_ms") or 0.0)
        kind = str(req.get("kind") or "init")
        #: ★STEP5★ init 端到端耗时（含 replay）。**必须量它** —— 超时阈值定得
        #: 合不合理全靠这个数：拍脑袋定的 2000 ms 实测把大量本可成功的 init
        #: 误杀成 timeout（日志里表现为"timeout 后紧接着同一请求 success"）。
        _t0 = time.time()
        #: ★STEP5★ 打"已开工"的点：超时阈值从此放宽，避免误杀本可成功的 init。
        try:
            fn = getattr(self.targets, "mark_init_started", None)
            if fn is not None:
                fn(tid)
        except Exception:  # noqa: BLE001
            pass
        ok = False
        #: ★STEP5★ 失败原因要一路带到 `mark_init_done`：日志里必须能分辨
        #: 「hub 404（帧被挤出）」「服务端返回失败」「客户端拿不到锚点帧」，
        #: 否则只会看到一句 `init_failed`，无从下手。
        why = ""
        try:
            backend = getattr(self.targets, "backend", None)
            if backend is None:
                why = "no_backend"
                return
            if kind == "reset":
                try:
                    backend.reset(tid)
                except Exception:  # noqa: BLE001
                    pass
            #: ref 模式：只发帧号（+ pts 作诊断），服务端自己取帧
            if getattr(self.targets, "ref_mode", False):
                res = backend.init_by_ref(tid, box, fid, mt, label, pts_ms=pts)
            elif len(box) == 4:
                anchor = self.cache.get(fid) if self.cache is not None else None
                frame = getattr(anchor, "frame", None) if anchor else None
                if frame is None:
                    self.stats["init_failed"] += 1
                    #: 目标帧已不在缓存里（环形缓冲挤出）→ 这次 init 没有意义
                    why = "expired"
                    return
                res = backend.initialize(tid, frame, box)
            else:
                why = "bad_bbox"
                return
            ok = bool(getattr(res, "ok", False))
            if not ok:
                why = str(getattr(res, "reason", "") or "server_failed")
            self.stats["inits"] += 1
            #: ★replay：回到 VLX 那一帧之后，沿缓存补到最新帧★
            if ok and self.cache is not None:
                for item in self.cache.replay_plan(fid, self.max_replay_frames):
                    if self.stop.is_set():
                        break
                    try:
                        if getattr(self.targets, "ref_mode", False):
                            r2 = backend.update_at(
                                tid, int(item.frame_id),
                                float(item.media_time_ms),
                                pts_ms=getattr(item, "pts_ms", None))
                        else:
                            r2 = backend.update(tid, item.frame)
                    except Exception as exc:  # noqa: BLE001
                        self.stats["errors"] += 1
                        self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
                        break
                    self.stats["replay_frames"] += 1
                    #: replay 的结果**不进 ResultQueue** —— replay 是"补位置"，
                    #: 中间帧的状态由状态机的既有逻辑（漂移守卫）统一处理，
                    #: 这里只把**最后一帧**作为一次正常 update 结果交出去。
                    if r2 is not None and getattr(r2, "ok", False):
                        newest = self.cache.newest()
                        self._push(tid, r2, newest or item, req.get("requested_at"))
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
            #: ★STEP5★ 远程后端把 hub 的 404/500 包成 `ConnectionError`，
            #: 异常消息里带着原始 HTTP 状态 —— 原样带出去才能被归类成 http_404。
            why = "%s: %s" % (type(exc).__name__, exc)
            LOG.warning("[rt-track-%d] init 异常 track=%s frame=%s",
                        self.track_id, tid, fid)
        finally:
            if not ok:
                self.stats["init_failed"] += 1
            if self.on_init_done is not None:
                try:
                    self.on_init_done(tid, ok, why, kind,
                                      (time.time() - _t0) * 1000.0)
                except Exception:  # noqa: BLE001
                    pass

    # ── update ──────────────────────────────────────────────────────────
    def _do_update(self) -> None:
        """发一次 `update_at`（ref 模式）/ `update`（旧模式）。

        ★每次都取"当下最新帧"，绝不排队★ —— 这就是 Latest Frame Wins。
        """
        newest = self.cache.newest() if self.cache is not None else None
        #: ★STEP3/P2★ latest 模式：帧由服务端自己取，客户端不需要先有最新帧
        latest = (str(getattr(self.targets, "update_mode", "") or "")
                  .strip().lower() == "latest")
        if newest is None and not latest:
            return
        tr = self.targets.get_track(self.track_id)
        if latest:
            #: ★故意不比较帧号★：服务端取的帧可能超前于客户端时间线，
            #: 用 `newest.frame_id` 比较会恒为 False → 跟踪永久停摆。
            if not self.targets.should_update_latest(tr):
                return
        elif not self.targets.should_update(tr, newest.frame_id):
            return
        backend = getattr(self.targets, "backend", None)
        if backend is None:
            return
        t0 = time.time()
        try:
            if getattr(self.targets, "ref_mode", False):
                if latest:
                    res = backend.update_latest(self.track_id)
                else:
                    res = backend.update_at(
                        self.track_id, int(newest.frame_id),
                        float(newest.media_time_ms),
                        pts_ms=getattr(newest, "pts_ms", None))
            else:
                res = backend.update(self.track_id, getattr(newest, "frame", None))
        except Exception as exc:  # noqa: BLE001
            self._err_run += 1
            self.stats["errors"] += 1
            self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
            return
        self._err_run = 0
        self.stats["updates"] += 1
        self.stats["last_latency_ms"] = round((time.time() - t0) * 1000.0, 2)
        if res is not None and not getattr(res, "ok", False):
            self.stats["update_failed"] += 1
        #: ★P2★ 服务端连续两次取到同一帧 = 画面没新帧，再快打也没意义
        self._pace(getattr(res, "extra", None) if res is not None else None)
        #: ★P2★ latest 模式下必须把**服务端回带的那一帧**交给状态机，
        #: 否则 `last_frame_id` 会被写成客户端时间线上一个不相干的帧号。
        real = (self.targets.real_frame(res, newest)
                if hasattr(self.targets, "real_frame") else newest)
        self._push(self.track_id, res, real, t0)

    def _pace(self, extra: Any) -> None:
        """按"服务端是否拿到了新帧"自适应调整空闲间隔。

        latest 模式下请求不再被客户端帧号节流，节奏完全由 RTT 决定；RTT 很小
        （服务端在本机时）时会变成对服务端的无意义轮询 —— 画面 15 fps，一秒
        打几十次拿到的都是同一帧。这里用**服务端回带的帧号**判断是否重复，
        重复就把间隔指数放大（上限 50 ms），拿到新帧立刻恢复。
        """
        fid = None
        if isinstance(extra, dict):
            try:
                fid = int(extra.get("frame_id"))
            except (TypeError, ValueError):
                fid = None
        if fid is None or fid != self._last_server_frame:
            self._last_server_frame = fid
            self._idle = IDLE_SLEEP_S
            return
        self._idle = min(0.05, max(IDLE_SLEEP_S, (self._idle or IDLE_SLEEP_S) * 2.0))

    def _push(self, tid: int, res: Any, newest: Any, req_wall: Any) -> None:
        try:
            self.result_q.put_nowait(UpdateResult(
                track_id=tid,
                ok=bool(getattr(res, "ok", False)),
                bbox_xyxy=(list(res.bbox_xyxy) if getattr(res, "bbox_xyxy", None)
                           is not None else None),
                score=float(getattr(res, "score", 0.0) or 0.0),
                frame_id=(int(newest.frame_id) if newest is not None else None),
                media_time_ms=(float(newest.media_time_ms)
                               if newest is not None else 0.0),
                pts_ms=(getattr(newest, "pts_ms", None)
                        if newest is not None else None),
                reason=str(getattr(res, "reason", "") or ""),
                latency_ms=float(getattr(res, "latency_ms", 0.0) or 0.0),
                newest=newest,
                req_wall_ms=(req_wall if isinstance(req_wall, (int, float))
                             else time.time()),
            ))
        except Exception:  # noqa: BLE001
            #: 队列满 → 丢最旧再塞（过期结果比没有结果更有害：会把框拉回旧位置）
            try:
                self.result_q.get_nowait()
            except Exception:
                pass
            try:
                self.result_q.put_nowait(UpdateResult(
                    track_id=tid, ok=bool(getattr(res, "ok", False)),
                    bbox_xyxy=None, score=0.0, frame_id=None,
                    media_time_ms=0.0, pts_ms=None, reason="dropped",
                    latency_ms=0.0, newest=newest,
                    req_wall_ms=time.time()))
            except Exception:
                pass

    # ── 主循环 ──────────────────────────────────────────────────────────
    def run(self) -> None:
        self.stats["alive"] = True
        try:
            while not self.stop.is_set():
                try:
                    req = self.init_slot.take()
                    if req is not None:
                        self._do_init(req)
                        continue
                    self._do_update()
                except Exception as exc:  # noqa: BLE001
                    #: ★异常绝不退出线程★ —— 线程死了就是"某个目标的跟踪静默
                    #: 消失"，比报错难查得多（与 TrackerWorker 同一原则）。
                    self.stats["errors"] += 1
                    self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
                    LOG.exception("[rt-track-%d] 循环异常，继续", self.track_id)
                if self._err_run >= _ERR_BACKOFF:
                    #: 连续失败 → 退避，别把挂掉的服务打成 DoS
                    self.stop.wait(min(_BACKOFF_MAX_S, 0.1 * (2 ** min(4, self._err_run))))
                else:
                    self.stop.wait(self._idle)
        finally:
            self.stats["alive"] = False


class StateLoop(threading.Thread):
    """★状态机线程：所有 track 状态的**唯一写入者**★。

    为什么要把状态写收敛到一条线程：改造前 `on_vlx`（VLX 线程）、`step()`
    （worker 线程）、HTTP health 三条线程都在改 track 状态，而且 `_reinit_and_replay`
    还在**锁外**改 `tr.state`（现存隐患）。收敛后锁只需保护字典本身。

    ★本线程绝不做网络请求★ —— 这是它能保持亚毫秒响应的前提，所有耗时操作
    都在 `TrackWorker` 里。
    """

    def __init__(self, targets: Any, result_q: "queue.Queue",
                 stop: threading.Event,
                 dirty: Optional[threading.Event] = None) -> None:
        super().__init__(daemon=True, name="rt-state")
        self.targets = targets
        self.result_q = result_q
        self.stop = stop
        #: ★"有东西变了"信号★：每应用一条结果就置位，发布线程据此立刻发一帧，
        #: 不需要靠轮询猜。没有它就得定时全量发 —— 那等于把发布频率又绑回
        #: 到固定节拍上，与"解耦"的初衷相悖。
        self.dirty = dirty or threading.Event()
        self.stats: Dict[str, Any] = {"applied": 0, "errors": 0,
                                      "last_error": "", "alive": False}

    def run(self) -> None:
        self.stats["alive"] = True
        try:
            while not self.stop.is_set():
                try:
                    ev = self.result_q.get(timeout=0.05)
                except Exception:  # noqa: BLE001
                    #: ★STEP5★ 队列空（每 50 ms 一次）正好是巡检 `pending_init`
                    #: 的节拍：卡死的 init **不会**产生任何结果事件，只靠
                    #: "有结果才巡检"永远轮不到它们。
                    self._sweep()
                    continue
                try:
                    self._apply(ev)
                except Exception as exc:  # noqa: BLE001
                    self.stats["errors"] += 1
                    self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
                    LOG.exception("[rt-state] 应用结果异常")
                self._sweep()
        finally:
            self.stats["alive"] = False

    def _sweep(self) -> None:
        """★STEP5★ 周期性巡检（本线程是状态唯一写入者，超时判 LOST 归它）。"""
        try:
            fn = getattr(self.targets, "sweep_pending_init", None)
            if fn is None:
                return
            if fn():
                self.dirty.set()
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)

    def _apply(self, ev: UpdateResult) -> None:
        tr = self.targets.get_track(ev.track_id)
        if tr is None:
            return
        #: 构造一个最小 `TrackerResult`，让状态机走**与同步路径完全相同**的
        #: `apply_update_result()` —— 判定逻辑始终只有一份。
        res = _MiniResult(ev)
        self.targets.apply_update_result(tr, res, ev.newest)
        self.stats["applied"] += 1
        self.dirty.set()


class _MiniResult:
    """把 `UpdateResult` 伪装成 `TrackerResult`，喂给既有判定逻辑。"""

    __slots__ = ("track_id", "ok", "bbox_xyxy", "score", "latency_ms",
                 "reason", "extra")

    def __init__(self, ev: UpdateResult) -> None:
        self.track_id = ev.track_id
        self.ok = ev.ok
        self.bbox_xyxy = tuple(ev.bbox_xyxy) if ev.bbox_xyxy else None
        self.score = ev.score
        self.latency_ms = ev.latency_ms
        self.reason = ev.reason
        self.extra: Dict[str, Any] = {"async": True}


class WsPublisher(threading.Thread):
    """★把"发布"从 tracker 的产出节奏里摘出来★。

    消息结构**完全沿用** `task_manager._make_track_publisher`：字段一个不少，
    前端协议零变化；连"什么时候该发"也沿用 `TrackerWorker` 的两条规则：

      * **有结果就发**（`dirty` 置位）—— 延迟只受一个节流下限约束；
      * **没有结果也要按心跳发**（默认 500 ms）—— 前端要靠它区分
        "跟踪在跑但没目标" 与 "跟踪线程已经死了"（第 44 节）。

    ★为什么不按固定 20 Hz 无脑发★：没有 track 更新时全量快照是**同一份数据**
    重复广播，WS 的 hub 队列只有 8 格、满了丢最旧 —— 高频空转会把真正
    有用的最新状态挤掉。所以改成"**脏了才发 + 心跳保底**"。

    ★为什么必须有节流下限★：tracker 理论上可以比 33 Hz 更快（RTT 只有
    200 ms 时不会，但服务端在本机时可能），那时每来一条结果就发一帧会把
    WS 队列冲垮。合并成一次再发，前端看到的是同一个最新位置。
    """

    def __init__(self, publish_fn: Any, targets: Any, stop: threading.Event,
                 dirty: threading.Event, hz: float = PUBLISH_HZ,
                 heartbeat_s: float = 0.5) -> None:
        super().__init__(daemon=True, name="rt-ws-pub")
        self.publish_fn = publish_fn
        self.targets = targets
        self.stop = stop
        self.dirty = dirty
        self.hz = float(hz or PUBLISH_HZ)
        self.heartbeat_s = max(0.05, float(heartbeat_s or 0.5))
        self._last_emit = 0.0
        self.stats: Dict[str, Any] = {"emits": 0, "errors": 0,
                                      "last_error": "", "alive": False}

    def run(self) -> None:
        self.stats["alive"] = True
        min_gap = 1.0 / max(1.0, self.hz)
        try:
            while not self.stop.is_set():
                #: 等 50 ms 的"脏"信号：没有更新时这条 wait 就是心跳的节拍器
                got = self.dirty.wait(timeout=0.05)
                if got:
                    self.dirty.clear()
                now = time.time()
                gap = now - self._last_emit
                if not got and gap < self.heartbeat_s:
                    continue
                if got and gap < min_gap:
                    #: ★节流★：这一帧太密，睡够下限再发（期间新结果继续合并）
                    self.stop.wait(min_gap - gap)
                self._emit()
        finally:
            self.stats["alive"] = False

    def _emit(self) -> None:
        now = time.time()
        try:
            snaps = self.targets.active_snapshots(now)
            items = [s.to_ws(now) for s in snaps]
            self.publish_fn(items)
            self.stats["emits"] += 1
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            self.stats["last_error"] = "%s: %s" % (type(exc).__name__, exc)
        self._last_emit = time.time()


class AsyncTrackerRuntime:
    """把上面三组线程装配起来，并提供**与 `TrackerWorker` 同名的生命周期方法**
    （`start` / `stop` / `stats`），这样 `task_manager` 侧的替换只需换一个类名。

    它**不创建也不触碰** `shub-feed` 与 `shub-vlx` —— 那两条是 P1 目标 1 与
    目标 2 的另一端，本来就独立。
    """

    name = "async-tracker"

    def __init__(self, targets: Any, cache: Any, publish_fn: Any,
                 stop: Optional[threading.Event] = None,
                 max_tracks: int = 5, max_replay_frames: int = 8,
                 hz: float = PUBLISH_HZ) -> None:
        self.targets = targets
        self.cache = cache
        self.publish_fn = publish_fn
        #: ★`stop` 只做**只读的外部信号**，绝不 set 它★：上层传进来的是
        #: **任务级**停止事件，`stop()` 一旦 set 它，等于"停止 tracker"顺手
        #: 把整个任务（采集 / VLX / WS）一起停了 —— 这是把两个生命周期
        #: 混成一个的典型事故。运行时用自己的事件，两者互不干涉。
        #:
        #: ★事件名不能叫 `self.stop`★：`stop` 是**方法名**（上层调
        #: `worker.stop(timeout=3.0)`），同名实例属性会把方法整个遮蔽掉，
        #: 调用时得到的是 `TypeError: 'Event' object is not callable`。
        self.external_stop = stop
        self.evt_stop = threading.Event()
        self.max_tracks = max(1, int(max_tracks or 5))
        self.max_replay_frames = int(max_replay_frames)
        self.hz = hz

        self._slots: Dict[int, InitSlot] = {}
        self._slots_lock = threading.Lock()
        self._workers: Dict[int, TrackWorker] = {}
        #: 结果队列容量 = max_tracks × 2。它是**控制消息**队列不是帧队列，
        #: 只用来在 worker 与状态机之间做解耦；满时丢最旧（过期结果有害）。
        self.result_q: "queue.Queue" = queue.Queue(maxsize=max(4, self.max_tracks * 2))

        #: ★VLX 投递 init 的入口★：`targets.on_init_request` 指向它
        self.on_init_request = self._enqueue_init
        self._dirty = threading.Event()
        self._state = StateLoop(targets, self.result_q, self.evt_stop, self._dirty)
        self._pub = WsPublisher(publish_fn, targets, self.evt_stop, self._dirty,
                                hz)
        self._started = False
        self._enqueued = 0
        self._dropped_no_worker = 0

    # ── InitReq 入口（由 VLX 线程调用，必须**瞬间返回**）──────────────────
    def _enqueue_init(self, req: Dict[str, Any]) -> bool:
        """返回 **True = 有 worker 会处理这条请求**，False = 请求将石沉大海。

        ★STEP5★ 为什么必须回这个 bool：改造前这里只做 `self._dropped_no_worker += 1`
        然后**照样返回 None**，`TargetManager._request_init` 无从得知请求被丢弃，
        于是 `pending_init` 留在 True → `should_update()` 永远否决 →
        track 永远停在 new、框一动不动（实测挂 862 帧 ≈ 57 s）。
        现在由调用方据此**立即判 init 失败**并置 LOST / 删除。
        """
        tid = int(req.get("track_id") or 0)
        slot = self._slot_for(tid)
        slot.put(req)            # ★长度 1，覆盖写★
        self._enqueued += 1
        self._ensure_worker(tid)
        if tid not in self._workers:
            self._dropped_no_worker += 1
            LOG.warning("[AsyncTracker] init 请求无 worker 接手 track=%s "
                        "(workers=%d/max=%d)", tid, len(self._workers),
                        self.max_tracks)
            return False
        return True

    def _reap_workers(self) -> None:
        """★STEP5★ 回收已经没用的 track worker。

        为什么必须有：`_workers` 改造前**只增不减**（只有 `stop()` 才 join），
        而 `max_tracks` 默认只有 5。于是第 6 条及以后创建的 track 永远拿不到
        worker（ `_ensure_worker` 在 `len(_workers) >= max_tracks` 时直接返回），
        它们的 init 请求全部石沉大海 —— 这就是"track 长时间 new"的直接原因
        （实测 8 条 track 里有 3 条如此）。

        回收条件（满足任一）：
          * 该 track 已从 `TargetManager` 里消失（被 drop / 判失败删除）；
          * 该 track 已 LOST —— 它不会再被 update，留着 worker 纯属占位。
        """
        with self._slots_lock:
            for tid in [t for t in list(self._workers.keys())]:
                tr = None
                try:
                    tr = self.targets.get_track(tid)
                except Exception:  # noqa: BLE001
                    tr = None
                gone = tr is None
                lost = (tr is not None
                        and str(getattr(tr, "state", "")) in ("lost", "removed"))
                if not (gone or lost):
                    continue
                self._workers.pop(tid, None)
                self._slots.pop(tid, None)

    def _slot_for(self, tid: int) -> InitSlot:
        with self._slots_lock:
            s = self._slots.get(tid)
            if s is None:
                s = InitSlot()
                self._slots[tid] = s
            return s

    def _ensure_worker(self, tid: int) -> None:
        if tid in self._workers:
            return
        #: ★STEP5★ 槽位满时**先回收**再判断：改造前这里直接 return，
        #: 而死掉的/已 LOST 的 track 的 worker 永远占着名额。
        if len(self._workers) >= self.max_tracks:
            self._reap_workers()
        if len(self._workers) >= self.max_tracks:
            return
        w = TrackWorker(tid, self.targets, self.cache, self._slot_for(tid),
                        self.result_q, self.evt_stop,
                        on_init_done=self._on_init_done,
                        max_replay_frames=self.max_replay_frames)
        self._workers[tid] = w
        if self._started:
            w.start()

    def _on_init_done(self, tid: int, ok: bool, reason: str = "",
                      kind: str = "", latency_ms: float = 0.0) -> None:
        try:
            self.targets.mark_init_done(tid, ok, reason=reason, kind=kind,
                                        latency_ms=latency_ms)
        except Exception:  # noqa: BLE001
            pass

    # ── 生命周期 ────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._started:
            return False
        self._started = True
        self._state.start()
        for w in list(self._workers.values()):
            if not w.is_alive():
                w.start()
        self._pub.start()
        LOG.info("[AsyncTracker] 启动：state + %d track worker + ws-pub@%.0fHz",
                 len(self._workers), self.hz)
        return True

    def stop(self, timeout: float = 3.0) -> None:
        self.evt_stop.set()
        #: ★唤醒发布线程★：它此刻可能正阻塞在 `dirty.wait(0.05)` 上，
        #: 不置位就要多等最多 50 ms（join 3 s 当然等得起，但停止应当干脆）。
        self._dirty.set()
        for w in list(self._workers.values()):
            if w.is_alive():
                w.join(timeout=max(0.1, timeout))
        if self._state.is_alive():
            self._state.join(timeout=max(0.1, timeout))
        if self._pub.is_alive():
            self._pub.join(timeout=max(0.1, timeout))
        self._started = False

    # ── 观测（沿用 `worker.stats()` 的字段名，前端协议不变）─────────────
    def stats(self) -> Dict[str, Any]:
        alive = sum(1 for w in self._workers.values() if w.is_alive())
        upd = sum(int(w.stats.get("updates", 0)) for w in self._workers.values())
        init = sum(int(w.stats.get("inits", 0)) for w in self._workers.values())
        err = sum(int(w.stats.get("errors", 0)) for w in self._workers.values())
        lat = [float(w.stats.get("last_latency_ms") or 0.0)
               for w in self._workers.values()
               if float(w.stats.get("last_latency_ms") or 0.0) > 0]
        return {
            "running": self._started,
            "ticks": self._state.stats.get("applied", 0),
            "emits": self._pub.stats.get("emits", 0),
            "errors": err + int(self._state.stats.get("errors", 0)),
            "last_error": (self._state.stats.get("last_error")
                           or self._pub.stats.get("last_error") or ""),
            "last_tick_ms": 0.0,
            "avg_tick_ms": 0.0,
            "uptime_s": 0.0,
            "track_fps": 0.0,
            #: ★P1 新增观测★（纯附加，前端不读也不影响）
            "async": {
                "workers": len(self._workers),
                "workers_alive": alive,
                "updates": upd,
                "inits": init,
                "replay_frames": sum(int(w.stats.get("replay_frames", 0))
                                     for w in self._workers.values()),
                "init_failed": sum(int(w.stats.get("init_failed", 0))
                                   for w in self._workers.values()),
                "update_failed": sum(int(w.stats.get("update_failed", 0))
                                     for w in self._workers.values()),
                "init_enqueued": self._enqueued,
                "init_no_worker": self._dropped_no_worker,
                "result_q": self.result_q.qsize(),
                "state_applied": int(self._state.stats.get("applied", 0)),
                "state_alive": bool(self._state.stats.get("alive", False)),
                "pub_alive": bool(self._pub.stats.get("alive", False)),
                "pub_hz": self.hz,
                "last_latency_ms": round(max(lat), 2) if lat else 0.0,
            },
        }
