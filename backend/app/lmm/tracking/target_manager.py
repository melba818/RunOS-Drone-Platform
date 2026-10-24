# -*- coding: utf-8 -*-
"""R4A：TargetManager —— VLX 的"语义"与 OSTrack 的"位置"之间的唯一裁决者。

它负责（第 12 节）：创建 / 关联 / 校正 / 丢失 / 重捕获。

三条不可违背的规则：

  1. **VLX 是语义权威**（第 16 节）：label 只能由 VLX 给，tracker 的 score 再低
     也不能把 swimmer 改成 boat。track_score 是"位置可信度"，不是识别置信度。
  2. **迟到结果必须 replay**（第 6/9 节）：VLX 给的是 frame=100 的框，画面已经在
     165 —— 必须在缓存的 frame 100 上初始化，再沿历史帧追到最新，禁止拿旧框
     直接用于当前帧。
  3. **历史帧已不在缓存就丢弃**（第 11 节）：宁可这一轮没有跟踪，也不要把过期
     的框当成"当前位置"（那是骗用户）。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .base import bbox_clip
from .types import (STATE_LOST, STATE_NEW, STATE_REMOVED, STATE_TRACKING,
                    STATE_UNCERTAIN, SOURCE_TRACKER, SOURCE_VLX, TrackerResult,
                    TrackSnapshot, center_dist, iou_xyxy, xywh_to_xyxy)

LOG = logging.getLogger(__name__)

#: ★STEP3/P2★ update 模式。`legacy` = 客户端指定帧号（`update_at`）；
#: `latest` = 服务端在收到请求时自己取最新帧（`update_latest`）。
UPDATE_MODE_LEGACY = "legacy"
UPDATE_MODE_LATEST = "latest"


@dataclass
class ServerFrame:
    """★STEP3/P2★ 服务端**实际用来推理的那一帧**（`update_latest` 回带）。

    为什么必须另造一个结构：`apply_update_result()` 拿 `newest` 去写
    `tr.last_frame_id / last_media_time_ms / last_pts_ms`。legacy 模式下
    `newest` 就是客户端缓存里的最新帧，帧号是客户端自己钉的，一致；
    latest 模式下服务端取的帧**可能比客户端时间线更新**（客户端 10 Hz 轮询
    量化过），继续用客户端的 `newest` 会把框记在一个**错误的帧号**上 ——
    表现为 Age 变成负数、帧号乱跳、漂移守卫的 gap 算错。

    三个字段与 `CachedFrame` 的语义完全对齐，因此状态机与漂移守卫一行都不用改。
    """

    frame_id: int
    media_time_ms: float = 0.0
    pts_ms: Optional[float] = None


@dataclass
class TrackConfig:
    """第 19/21 节：这些参数**必须可调**，不许硬编码散落。"""

    #: 0.35 → 0.15：MX250 实测（`R4A-OSTrack-本机GPU实测报告.md` 第 4 节）
    #: track_score 落在 0.167~0.715、均值 0.35。阈值 0.35 会把**一半的帧**
    #: 判成低分 → 立刻 UNCERTAIN → LOST，等于"接上了却看不见"。
    score_threshold: float = 0.15
    uncertain_after: int = 3           # 连续几帧低分 → UNCERTAIN
    lost_after: int = 10               # 再持续低分 → LOST
    max_tracks: int = 5                # 第 14 节：OSTrack 是单目标，不许无限建
    max_without_vlx_ms: float = 7000.0  # 第 21 节：VLX 太久没确认 → 不再信任
    #: 15 → 8：实测 replay 15 帧要 **2615 ms**，而这期间 tracker 一个框都不出，
    #: 前端就是 2.6 s 空白。8 帧只需 797 ms，代价是初始位置略旧 ——
    #: 连续但略旧，远好过精确但空白。
    max_replay_frames: int = 8
    match_iou: float = 0.25            # 第 13 节：同 label + IoU 优先
    #: 0.6 → 1.0：IoU 与中心距是**"或"的关系**（见 `_match`）。实测
    #: （`_r4a_pipeline_run.log`）目标折返时跟踪器会漂移 35 px，IoU 掉到 0.221
    #: 差一点不过 0.25，于是同一个目标被建成第二条 track —— 画面上出现两个框，
    #: 比"关联错"难看得多。1.0 = 中心距在一个目标对角线内，对航拍小目标
    #: （彼此间距远大于自身尺寸）足够区分不同个体。
    match_center_ratio: float = 1.0
    # ── 漂移守卫（真实 VLX 视频 E2E 实测后新增，见 `_accept`）─────────────
    #: 真实任务实测（`_r4a_e2e_stdout.log`）：OSTrack 在 2 s 无校正窗口里会把
    #: 32×23 的小目标**膨胀到 938×513**（面积涨 650 倍，几乎铺满画面），
    #: 而同一时刻 VLX 给出的真值只是 23×16。膨胀后的框中心离真值 366 px，
    #: 于是下一次 VLX 检测必然关联失败 → 同一个目标被拆成 8 条 track。
    #: 面积守卫锚在**VLX 给的 anchor 尺寸**上，与步数无关，膨胀一次就被拦下。
    #:
    #: ★4.0 → 12.0（实测修正，`.workbuddy/realtime_probe.txt`）★：
    #: 5 分钟真实链路里 87 次漂移拒绝有 **85 次是这道门**，而它拦下的其实是
    #: OSTrack 的**正常输出** —— OSTrack 的 TEMPLATE_FACTOR 是 2.0，初始化后
    #: 它给出的框在**每个维度上就是模板的 2 倍**，面积正好 ×4，刚好顶到 4.0
    #: 的门槛。于是 track 连续 3 次被判漂移 → LOST → `step()` 直接跳过 →
    #: 整个跟踪通道事实上是死的（total_update_fps 只有 1.02）。
    #: 真正的失控膨胀（实测 32×23 → 938×513，面积 ×650）在 12.0 下照样被拦。
    drift_area_grow: float = 50.0      # 面积相对 anchor 的上限倍数
    drift_area_shrink: float = 50.0    # 面积相对 anchor 的下限倒数
    #: ★单步面积变化上限（相对**上一步的框**）★：真正能区分"漂移"和"正常"的
    #: 是**每一步的变化率**，不是相对 2 秒前那个 VLX anchor 的绝对倍数 ——
    #: 航拍里目标远近距离一变，面积本来就会整体涨落；而失控膨胀是**一步步**
    #: 涨上去的（实测 32×23 → 938×513 是逐帧累积的）。
    #: 2.5 = 单步面积最多涨 1.5 倍，真漂移几步就撞上，正常抖动撞不上。
    drift_area_step: float = 2.5
    #: 单步中心位移上限 = `drift_move_ratio × 目标对角线 × sqrt(跨过的帧数)`。
    #: 用 sqrt 而不是线性缩放：缓存是稀疏采样的，一次 step 可能跨几十帧，
    #: 线性放开会大到等于没设门。实测目标 45 帧只移动 40 px（<1 px/帧）。
    drift_move_ratio: float = 1.5
    #: 3 → 5：连续 3 次太容易被"VLX 框本身抖动"带偏（VLX 每 2~3 s 才来一帧，
    #: 一次误判要等两秒才可能被纠正）。5 次仍能在一秒内拦住真漂移。
    drift_after: int = 5               # 连续几次判漂移 → LOST
    #: ★并行 update 数★：1 = 串行（旧行为）。远程 tracker 一次 update 约
    #: 110~190 ms，N 个目标串行就只有 1/N 的每目标频率；>1 时把不同 track 的
    #: 请求并发发出（配合 `RemoteOSTrackBackend.parallel_channels` 分槽）。
    parallel_updates: int = 1
    #: 运动一致性关联（第三道门，见 `_match`）：VLX 两次检测之间目标能走多远。
    #: 单位是"目标对角线 / 秒"。实测移动目标约 25 px/s（对角线 25 px）≈ 1.0，
    #: 留 3 倍余量；而误检/换目标那种几百 px 的跳变会被正确拒绝。
    match_move_per_sec: float = 3.0
    match_move_fps: float = 30.0       # 把帧间隔换算成秒用的标称帧率
    #: ★时间上限★：没有它，一条**很久没被 VLX 确认**的 track 的门槛会随时间
    #: 线性放宽到形同虚设 —— 实测 anchor 停在 357 帧（11.9 s）前的 track，
    #: 门槛被放到 1213 px，于是新出现的目标被误关联到那条早已 LOST 的旧 track
    #: （`_r4a_e2e_stdout2.log`：`anchor@102 move=601.6` 却判 MATCH）。
    #: 3 s 足够覆盖正常的目标运动，超出的时间不再给额外的位移额度。
    match_move_max_sec: float = 3.0
    #: ★周期校正门槛★：VLX 每 2~3 s 来一次，**绝大多数只是"确认目标还在"**，
    #: 此时 tracker 的模板是好的，重置一次纯属浪费。只有下面几种情况才真的
    #: 需要 reset + initialize（= 重换模板）：
    #:   * track 处于 LOST / UNCERTAIN / NEW（本来就该重建）；
    #:   * VLX 框与该帧的 tracker 历史位置 IoU < `correction_iou`（tracker 已偏）；
    #:   * tracker 自己给的 score 低于 `score_threshold`（自己都说没跟上）。
    #: 实测（5 分钟真实链路）75 次周期校正里几乎全是"确认"，每次付
    #: init + 8 帧 replay ≈ 983 ms 并**独占远程后端锁**，把其它目标的逐帧
    #: update 全部饿死 → 所有 track 掉 LOST → 画面变回"每 2~3 s 瞬移一次"。
    correction_iou: float = 0.5
    #: ★STEP5★ `pending_init` 超时（毫秒）。
    #: 为什么必须有它：init 请求投出去后，**没有任何东西保证它一定会被处理** ——
    #: worker 槽位满了（`_ensure_worker` 在 `len(_workers) >= max_tracks` 时
    #: 直接返回）、或者 `on_init_request` 抛异常，请求就永远躺在 `InitSlot` 里，
    #: `pending_init` 恒为 True → `should_update()` 永远 False → 这条 track
    #: **永远停在 new、框一动不动**，而 `active_snapshots()` 又只排除 LOST/REMOVED
    #: （不排除 new），于是前端一直收到同一个框 = 用户看到的「静止大框」。
    #: 实测一条 track 能在 new 上挂 862 帧（≈57 s）。
    #: 2000 ms = init 正常只要 ~200 ms，10 倍余量；超时即判失败 → LOST。
    pending_init_timeout_ms: float = 2000.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TrackConfig":
        #: ★这里的默认值必须**与字段声明逐字一致**★。曾经是 0.35/15/0.6，
        #: 与字段的 0.15/8/1.0 打架 —— 谁路过调一下 `from_dict` 就会静默回到
        #: 旧阈值（一半帧被判低分 → 立刻 LOST，等于"接上了却看不见"）。
        d = dict(d or {})
        return cls(
            score_threshold=float(d.get("score_threshold", 0.15)),
            uncertain_after=int(d.get("uncertain_after", 3)),
            lost_after=int(d.get("lost_after", 10)),
            max_tracks=int(d.get("max_tracks", 5)),
            max_without_vlx_ms=float(d.get("max_without_vlx_ms", 7000.0)),
            max_replay_frames=int(d.get("max_replay_frames", 8)),
            match_iou=float(d.get("match_iou", 0.25)),
            match_center_ratio=float(d.get("match_center_ratio", 1.0)),
            drift_area_grow=float(d.get("drift_area_grow", 50.0)),
            drift_area_shrink=float(d.get("drift_area_shrink", 50.0)),
            drift_move_ratio=float(d.get("drift_move_ratio", 1.5)),
            drift_after=int(d.get("drift_after", 5)),
            match_move_per_sec=float(d.get("match_move_per_sec", 3.0)),
            match_move_fps=float(d.get("match_move_fps", 30.0)),
            match_move_max_sec=float(d.get("match_move_max_sec", 3.0)),
            correction_iou=float(d.get("correction_iou", 0.5)),
            parallel_updates=int(d.get("parallel_updates", 1)),
            #: ★STEP5★ 与字段声明逐字一致（这里的默认值曾经与字段打架，
            #: 谁路过调一下 `from_dict` 就会静默回到旧阈值）。
            pending_init_timeout_ms=float(d.get("pending_init_timeout_ms", 2000.0)),
        )


@dataclass
class Track:
    """一条目标的状态。**不含模型权重**，只有位置与判定用的计数器。"""

    track_id: int
    label: str
    state: str = STATE_NEW
    bbox_xyxy: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    score: float = 0.0
    last_frame_id: int = 0
    last_media_time_ms: float = 0.0
    #: ★真实流内 PTS★（STEP1），与 `last_media_time_ms`（合成值）并存
    last_pts_ms: Optional[float] = None
    #: ★P1 异步初始化★：True = 已投递 init 请求但还没拿到服务端结果。
    #: 此时**绝不能**对它发 `update_at` —— 服务端还没有这个 track 的模板，
    #: 会直接回 "track 未初始化"（实测踩过，表现为 ref_errors 一路涨）。
    pending_init: bool = False
    #: ★STEP5★ `pending_init` 置位的时间戳（time.time()）。没有它就**无从判断**
    #: 这一位是不是卡死了 —— 实测出现过挂 862 帧（≈57 s）的情况。
    pending_init_at: float = 0.0
    #: ★STEP5★ worker **已经开始处理**这条 init 请求。
    #: 为什么必须分开两个时间点：`pending_init_at` 是"入队时刻"，但 worker
    #: 此刻可能正在给这条 track 做 update（`update_at` 一次 110~190 ms，
    #: 5 条 track 共享一条 SSH 隧道时排队更长），要过好几秒才轮到它 take 这个
    #: 请求。若从"入队"起算 2000 ms 就判超时，会把**大量本可成功的 init**
    #: 全部误杀（实测 67 次 init 里 25 次被误杀，日志里表现为
    #: "timeout 之后紧接着同一请求 success"）。
    #: 判据：未开工 + 超过阈值 = 真的没人接手（worker 槽位满 / 请求丢失）；
    #: 已开工 = worker 的 `finally` 一定会给回执，不再用超时催它。
    pending_init_started: bool = False
    #: ★STEP5★ 这次 init 请求**要用**的那一帧（= 触发它的 VLX 检出帧）。
    #: 打日志时 `gap = current_frame - init_req_frame` 就是"VLX 帧已经过去多久"，
    #: 这是判断 init 该不该放弃（`expired`）的唯一依据。
    init_req_frame: Optional[int] = None
    #: ★STEP5★ `init`（新目标）/ `reset`（周期校正）。失败处理不同：
    #: 新目标一次都没成功过 → 直接删除；周期校正是给本来跟得好好的 track
    #: 换模板，失败只置 LOST，等下一次 VLX 再校正。
    init_kind: str = ""
    last_vlx_frame_id: Optional[int] = None
    last_vlx_confirmed_at: Optional[float] = None
    low_score_frames: int = 0
    created_at: float = field(default_factory=time.time)
    event: str = ""
    #: 第 44 节：事件统计必须分开，tracker 的更新不能算成"AI 识别"
    reacquire_count: int = 0
    #: 漂移守卫用的连续漂移计数（见 `TargetManager._accept`）
    drift_frames: int = 0
    #: ★VLX anchor 的尺寸与帧号★：漂移守卫的**唯一基准**。tracker 自己的历史
    #: 会漂，拿它当基准等于拿误差量误差；VLX 给的是真值，只能拿它做尺子。
    anchor_area: float = 0.0
    #: ★单步面积基准（EMA）★：漂移主门比的是"这一步相对**基准框**涨了多少"。
    #: 用 EMA 而不是上一步的框：OSTrack 输出的框口径与 VLX 框本来就差一个
    #: TEMPLATE_FACTOR（线性 2×，面积 4×），拿"上一步"或"VLX anchor"当基准
    #: 都会把这点**恒定差异**误判成漂移（实测 87 次拒绝里 85 次是它）。
    #: EMA 会在几步之内把这个恒定差异吸收掉，而逐帧累积的失控膨胀（单步
    #: 涨 1.5 倍）追不上 0.3 的跟踪速度，仍会被拦下。
    drift_ref_area: float = 0.0
    vlx_anchor_frame: Optional[int] = None
    vlx_anchor_box: Optional[Tuple[float, float, float, float]] = None
    #: ★关键★ 近期位置历史 `(frame_id, bbox)`，只留最近若干条。
    history: Deque[Tuple[int, Tuple[float, float, float, float]]] = field(
        default_factory=lambda: deque(maxlen=64))

    def remember(self, frame_id: int,
                 bbox: Tuple[float, float, float, float]) -> None:
        self.history.append((int(frame_id), tuple(float(v) for v in bbox)))

    def bbox_at(self, frame_id: int) -> Optional[Tuple[float, float, float, float]]:
        """取这条 track 在**指定帧**上的位置（取帧号最接近的一条）。

        为什么必须留历史：VLX 给的是 2.3 s **之前**那一帧的框，而 track 的
        `bbox_xyxy` 是**现在**的位置。拿"过去的框"跟"现在的位置"比 IoU，
        目标只要动起来就必然失配 → 同一个目标被反复建成新 track（实测：
        8 秒里单个运动目标建了 2 条）。留历史才能**拿同一时刻的位置比**。
        """
        if not self.history:
            return None
        best = None
        best_gap = None
        for fid, box in self.history:
            gap = abs(int(fid) - int(frame_id))
            if best_gap is None or gap < best_gap:
                best_gap, best = gap, box
        return best

    def snapshot(self, now: Optional[float] = None) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=self.track_id, label=self.label, bbox_xyxy=self.bbox_xyxy,
            state=self.state, source=SOURCE_TRACKER, track_score=self.score,
            last_vlx_frame_id=self.last_vlx_frame_id,
            last_vlx_confirmed_at=self.last_vlx_confirmed_at,
            event=self.event, vlx_confidence=None,
            frame_id=self.last_frame_id, media_time_ms=self.last_media_time_ms,
            pts_ms=self.last_pts_ms,
        )


class TargetManager:
    """VLX detection → track 的关联/校正，以及 tracker 的逐帧推进。"""

    def __init__(self, backend: Any, cache: Any,
                 cfg: Optional[TrackConfig] = None,
                 ref_mode: Optional[bool] = None,
                 on_init_request: Optional[Any] = None,
                 update_mode: str = UPDATE_MODE_LEGACY) -> None:
        self.backend = backend
        self.cache = cache
        self.cfg = cfg or TrackConfig()
        #: ★STEP3/P2★ `legacy`（默认）/ `latest`。**只影响 update**，init 与
        #: replay 一律不变 —— init 必须落在 VLX 那一帧上，隔帧初始化会让模板
        #: 错位、框扩全屏，所以 P2 明确禁止 latest init。
        self.update_mode = (str(update_mode or UPDATE_MODE_LEGACY)
                            .strip().lower())
        if self.update_mode not in (UPDATE_MODE_LEGACY, UPDATE_MODE_LATEST):
            self.update_mode = UPDATE_MODE_LEGACY
        #: ★P1 异步初始化★：`None` = 旧行为（`on_vlx` 里**同步**跑
        #: reset + init + replay，实测 ~983 ms，VLX 一来整条跟踪链就卡住）。
        #: 给了回调 → `on_vlx` 只投递 `InitReq` 就立刻返回，真正的 init 与
        #: replay 由**该 track 自己的 worker 线程**执行。
        #: 旧模式（remote / 本地 image）不设置它 → 行为与改造前逐字一致。
        self.on_init_request = on_init_request
        #: ★服务器侧共享帧★：`None` = 自动探测（后端说自己支持就用）。
        #: 打开后**只改帧的传递方式**：`update()` → `update_at()`、
        #: `initialize()` → `init_by_ref()`。状态机 / 漂移守卫 / 周期校正 /
        #: replay 的判定逻辑一行不动 —— 它们本来就只看框与帧号，不看像素。
        self.ref_mode = (bool(getattr(backend, "supports_ref", False))
                         if ref_mode is None else bool(ref_mode))
        self._lock = threading.RLock()
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        #: 统计（第 44 节：与 VLX 检测计数完全分开）
        self.stats = {
            "vlx_detections": 0, "expired_detections": 0,
            "tracks_created": 0, "tracks_removed": 0,
            "tracker_updates": 0, "reacquire_count": 0, "lost_count": 0,
            "replay_total": 0, "replay_frames": 0, "replay_ms": 0.0,
            "drift_rejected": 0,
            "drift_reject_step": 0, "drift_reject_abs": 0,
            #: ★追帧闭环埋点★（第 5/6/10 节）：用来回答"生产只有 4 FPS
            #: 到底卡在哪"以及"104 次 init 分别是什么"。
            "step_calls": 0,          #: TargetManager.step 被调用次数
            "step_update_calls": 0,   #: 真正下发 backend.update 的次数
            "step_update_ok": 0,      #: 返回 ok 的次数
            "step_skip_state": 0,     #: 因 LOST/REMOVED 跳过的次数
            "step_skip_current": 0,   #: 因"已追到最新帧"跳过的次数
            "init_new": 0,            #: 新目标初始化
            "init_correction": 0,     #: ★已有 track 的周期校正（reset+init）★
            #: ★"VLX 只是确认、没有真的重建模板"的次数★。没有这个计数时，
            #: 75 次周期校正和 6 次新目标在日志里长得一模一样，根本看不出
            #: 校正是不是被滥用（用户第 10 节要的就是这个分类）。
            "init_skipped": 0,
            "init_failed": 0,
            #: ★LOST 原因分类★：只知道"lost_count=99"没用 —— 必须知道是
            #: drift（几何守卫判漂移）/ low_score（tracker 自己说没跟上）/
            #: vlx_ttl（VLX 太久没确认）哪一种，否则无从下手。
            "lost_drift": 0,
            "lost_low_score": 0,
            "lost_vlx_ttl": 0,
            "drift_reject_move": 0,
            "drift_reject_area": 0,
            "update_not_ok": 0,
            "per_track_updates": {},  #: track_id -> 成功 update 次数（第 6 节：
                                      #: 多目标串行共享预算，必须分开统计）
        }
        self._started_at = time.time()

    # ── VLX 结果入口 ──────────────────────────────────────────────────────
    def on_vlx(self, frame_id: int, media_time_ms: float,
               detections: Sequence[Dict[str, Any]],
               now: Optional[float] = None,
               pts_ms: Optional[float] = None) -> Dict[str, Any]:
        """一条 VLX detection（可能含多个框）进来后的全部决策。

        返回值是**这一次的诊断**，给日志与 WS timing 用：
        `expired` 为 True 表示整条 detection 因历史帧已失效被丢弃（第 11 节）。

        ★`pts_ms`★：VLX 检出那一帧的真实 PTS。异步初始化时**必须原样带进
        `InitReq`** —— init 只能在这一帧上做（隔帧会让模板错位），PTS 也要
        与该帧严格对应，不能事后用"当前 pts"补。
        """
        now = float(now if now is not None else time.time())
        #: ★STEP5★ 每次 VLX 来都先清一遍：卡在 `pending_init` 的 track 会一直
        #: 占着 `_tracks` 名额，让后面的新目标被 `max_tracks` 淘汰掉。
        self.sweep_pending_init(now)
        info: Dict[str, Any] = {"frame_id": int(frame_id), "expired": False,
                                "created": 0, "updated": 0, "replay_ms": 0.0,
                                "replay_frames": 0, "debug": []}
        dets = [d for d in (detections or []) if self._bbox_of(d) is not None]
        if not dets:
            return info
        with self._lock:
            self.stats["vlx_detections"] += len(dets)

        anchor = self.cache.get(int(frame_id)) if self.cache is not None else None
        if anchor is None:
            #: 第 11 节：宁可丢弃，也不要拿旧框初始化当前帧
            with self._lock:
                self.stats["expired_detections"] += len(dets)
            info["expired"] = True
            info["reason"] = "detection frame no longer in cache"
            return info

        t0 = time.time()
        claimed: set = set()
        replay_steps = 0
        for det in dets:
            box = self._bbox_of(det)
            label = str(det.get("label") or det.get("label_zh") or "").strip() or "目标"
            matched = self._match(label, box, int(frame_id), exclude=claimed)
            if matched is not None:
                claimed.add(matched.track_id)
            #: 诊断：关联失败是 R4A 最容易"静默退化"的地方 —— 一旦每次检测
            #: 都新建 track，画面上同一个目标会出现两个框，而日志里什么都不报。
            #:
            #: ★必须记**真正匹配上的那条**的比对数据★：记"第一个候选"会把
            #: "有候选但没匹配上"误报成 MATCH，看日志的人会以为关联是好的。
            row: Dict[str, Any] = {}
            if matched is not None:
                row = self._row_for(matched, box, int(frame_id))
                row["tid"] = matched.track_id
                #: 与 `_explain` 的候选行区分开：日志里一旦出现"数据对不上"，
                #: 第一步就是确认这行到底是 matched 还是某个候选。
                row["matched_tid"] = matched.track_id
                #: 区分"靠重叠/中心距命中"与"只有运动门命中"：后者意味着
                #: tracker 已经漂出去了，是漂移守卫该被调紧的信号，不能混在
                #: 普通 MATCH 里让人误以为跟踪很健康。
                near = ((row.get("iou") or 0.0) >= self.cfg.match_iou
                        or (row.get("dist") or 1e9)
                        <= self.cfg.match_center_ratio * (row.get("diag") or 1.0))
                row["verdict"] = "MATCH" if near else "MATCH(仅运动门)"
            else:
                cands = (self._explain(label, box, int(frame_id)).get("rows") or [])
                cands = [c for c in cands if c.get("tid") not in claimed] or cands
                if cands:
                    row = dict(cands[0])
                    row["verdict"] = "NEW(候选都没过门槛)"
                else:
                    row = {"verdict": "NEW(无同 label 候选)"}
            info["debug"].append({
                "frame_id": int(frame_id), "label": label,
                "matched": None if matched is None else matched.track_id,
                "row": row,
                "why": self._explain(label, box, int(frame_id)),
            })
            if matched is not None:
                #: ★第 10 节★：这是**周期校正**，不是新目标 —— track_id 不变。
                #: 但"校正"不等于"每次都要 reset + initialize"：
                #: `_reinit_and_replay` 一次要付 init + 最多 max_replay_frames
                #: 次 update（实测 983 ms）并**独占远程后端锁**，把所有目标的逐帧
                #: update 饿死。tracker 本来就跟得好好的时候，VLX 这一帧只该当作
                #: "确认目标还在"，不动模板、不占锁。
                if self._correction_needed(matched, box, int(frame_id)):
                    with self._lock:
                        self.stats["init_correction"] += 1
                    if self.on_init_request is not None:
                        replay_steps += self._request_init(
                            matched, frame_id, media_time_ms, pts_ms, box, now,
                            label, kind="reset")
                    else:
                        replay_steps += self._reinit_and_replay(
                            matched, frame_id, media_time_ms, box, now, label)
                    info["updated"] += 1
                else:
                    with self._lock:
                        self.stats["init_skipped"] += 1
                        #: 只刷新"VLX 认过它"这件事：位置仍由 tracker 逐帧给，
                        #: 但 TTL（`max_without_vlx_ms`）要跟着走，否则会被判
                        #: `vlx_ttl_expired` 掉 LOST。
                        matched.last_vlx_frame_id = int(frame_id)
                        matched.last_vlx_confirmed_at = now
                        matched.vlx_anchor_frame = int(frame_id)
                        matched.vlx_anchor_box = tuple(float(v) for v in box)
                        matched.drift_frames = 0
                        matched.low_score_frames = 0
                        if matched.state in (STATE_LOST, STATE_UNCERTAIN, STATE_NEW):
                            matched.state = STATE_TRACKING
                            matched.event = "confirmed"
                        else:
                            matched.event = "vlx_confirmed"
                    info["updated"] += 1
                    continue
            else:
                tr = self._create(label, box, frame_id, media_time_ms, now)
                if tr is None:
                    continue
                with self._lock:
                    self.stats["init_new"] += 1
                if self.on_init_request is not None:
                    replay_steps += self._request_init(
                        tr, frame_id, media_time_ms, pts_ms, box, now, label,
                        kind="init")
                else:
                    replay_steps += self._reinit_and_replay(
                        tr, frame_id, media_time_ms, box, now, label)
                info["created"] += 1
        #: `current_gap` 是"当前帧离 VLX 帧有多远"，`hist_gap` 是"实际用来比对的
        #: 历史帧离 VLX 帧有多远"。前者几十帧、后者个位数，才能证明关联是对的。
        try:
            newest_now = self.cache.newest() if self.cache is not None else None
            if newest_now is not None:
                info["current_gap_frames"] = abs(int(newest_now.frame_id) - int(frame_id))
        except Exception:  # noqa: BLE001
            pass
        info["replay_ms"] = round((time.time() - t0) * 1000.0, 2)
        #: ★当次，不是累计★：`self.stats["replay_frames"]` 是任务级累加值，
        #: 打进日志会显示成 8→16→24… 让人以为是"这一帧 replay 了 24 帧"。
        info["replay_frames"] = int(replay_steps)
        return info

    def _update_one(self, tr: Any, newest: Any) -> Any:
        """把一条 track 推进到 `newest` 这一帧。

        ★唯一被"换帧传递方式"改动的地方★：
          * ref_mode（服务器侧共享帧）→ 只发 `{track_id, frame_id}`，
            服务端自己从 StreamHub 取帧（实测 15.46 Hz / p50 60 ms）；
          * 否则 → 老办法，把像素喂给后端（JPEG + crop，2.8 Hz / p50 340 ms）。

        两种路径返回**同一个 `TrackerResult`**，因此后面的状态机 / 漂移守卫 /
        校正判定完全不需要知道自己走的是哪条路。

        ★STEP3/P2★ 第三条路径 `update_latest`：连帧号都不发（发什么帧由服务端
        在收到请求时自己决定）。**只有 update 走这条路** —— init 与 replay 仍
        必须指定帧号。
        """
        if self.ref_mode:
            if self.update_mode == UPDATE_MODE_LATEST:
                return self.backend.update_latest(tr.track_id)
            return self.backend.update_at(tr.track_id, int(newest.frame_id),
                                          float(newest.media_time_ms),
                                          pts_ms=getattr(newest, "pts_ms", None))
        return self.backend.update(tr.track_id, newest.frame)

    def real_frame(self, res: Any, newest: Any) -> Any:
        """★STEP3/P2★ 这条结果**真正属于**的那一帧。

        * legacy —— 就是客户端指定的 `newest`（帧号是客户端钉的，必然一致）；
        * latest —— 取服务端回带的 `frame_id/media_time_ms/pts_ms`
          （`remote_backend.update_latest` 已把它们放进 `extra`）。
          拿不到（旧服务端 / 异常）时退回 `newest`，绝不因此丢弃结果。
        """
        if self.update_mode != UPDATE_MODE_LATEST:
            return newest
        ex = getattr(res, "extra", None) or {}
        fid = ex.get("frame_id") if isinstance(ex, dict) else None
        if fid is None:
            return newest
        try:
            return ServerFrame(int(fid),
                               float(ex.get("media_time_ms") or 0.0),
                               ex.get("pts_ms"))
        except (TypeError, ValueError):
            return newest

    # ── 逐帧推进（TrackerWorker 调）────────────────────────────────────────
    def step(self) -> List[TrackSnapshot]:
        """把每条 track 推进到缓存里的最新帧。返回本次被更新的快照。"""
        if self.cache is None or self.backend is None:
            return []
        #: ★STEP5★ 每轮先清掉卡死的 `pending_init`：它们既不会自己恢复，
        #: 又会一直占着画面（state 仍是 new）。
        self.sweep_pending_init()
        newest = self.cache.newest()
        if newest is None:
            return []
        out: List[TrackSnapshot] = []
        now = time.time()
        with self._lock:
            tracks = list(self._tracks.values())
            self.stats["step_calls"] += 1
        #: 先挑出"这一轮真的要 update 的 track"，再决定串行还是并行下发 ——
        #: 串行时 N 个目标共享 1/(单次延迟) 的预算，每目标只有 1/N。
        pend: List[Track] = []
        #: ★P2★ latest 模式下**绝不能**用 `tr.last_frame_id >= newest.frame_id`
        #: 做判据：服务端取的帧会超前于客户端时间线，这条判据恒为 False →
        #: 同步路径下跟踪**永久停摆**（一次都 update 不了）。
        _latest = (self.update_mode == UPDATE_MODE_LATEST)
        for tr in tracks:
            if _latest:
                if not self.should_update_latest(tr):
                    with self._lock:
                        self.stats["step_skip_state"] += 1
                    continue
                pend.append(tr)
                continue
            if tr.state in (STATE_LOST, STATE_REMOVED):
                with self._lock:
                    self.stats["step_skip_state"] += 1
                continue
            if tr.last_frame_id >= newest.frame_id:
                with self._lock:
                    self.stats["step_skip_current"] += 1
                continue
            pend.append(tr)
        if not pend:
            return out
        with self._lock:
            self.stats["step_update_calls"] += len(pend)
        par = int(getattr(self.cfg, "parallel_updates", 1) or 1)
        if par > 1 and len(pend) > 1:
            #: ★并行下发★：不同 track 走不同的 SSH 隧道槽位（见
            #: `RemoteOSTrackBackend.parallel_channels`），否则它们仍会排在同一
            #: 条隧道上，并行只是把等待从"客户端"挪到"服务端"。
            with ThreadPoolExecutor(max_workers=min(par, len(pend))) as pool:
                results = list(pool.map(
                    lambda t: self._update_one(t, newest), pend))
        else:
            results = [self._update_one(t, newest) for t in pend]

        for tr, res in zip(pend, results):
            #: ★P2★ latest 模式下用**服务端回带**的帧，不是客户端的 `newest`
            snap = self.apply_update_result(
                tr, res, self.real_frame(res, newest), now)
            if snap is not None:
                out.append(snap)
        return out

    def apply_update_result(self, tr: Any, res: Any, newest: Any,
                            now: Optional[float] = None) -> Optional[TrackSnapshot]:
        """★单条 track 的 update 结果应用★ —— `step()` 与 P1 异步运行时共用同一段。

        与改造前 `step()` 里的判定**逐行一致**，只是从"一次处理一批"变成"一次
        处理一条"：异步运行时里每个 track 有自己的 worker，结果回来一条就应用
        一条，不必等其它 track 的请求都返回。

        两条路径共用这一个方法，是为了保证**判定逻辑永远只有一份** —— 否则
        异步路径迟早会和同步路径算出不同的结果，而这种差异极难排查。
        """
        now = float(now if now is not None else time.time())
        with self._lock:
            if res.ok and res.bbox_xyxy is not None:
                self.stats["tracker_updates"] += 1
                self.stats["step_update_ok"] += 1
                #: ★第 6 节★：多目标**串行**共享同一个 tracker 预算，
                #: 总 update/s 不等于单目标 FPS —— 必须逐 track 分开记。
                key = str(int(tr.track_id))
                pt = self.stats["per_track_updates"]
                pt[key] = int(pt.get(key, 0)) + 1
                prev = tr.bbox_xyxy
                gap = max(1, int(newest.frame_id) - int(tr.last_frame_id))
                ok, why = self._accept(tr, prev, res.bbox_xyxy, gap)
                if ok:
                    tr.drift_frames = 0
                    tr.bbox_xyxy = res.bbox_xyxy
                    tr.score = float(res.score or 0.0)
                    #: 单步面积基准跟着被接受的框走（EMA 0.7/0.3）：
                    #: 恒定的口径差异会被吸收，逐帧膨胀追不上。
                    na = max(1e-6, (res.bbox_xyxy[2] - res.bbox_xyxy[0])
                             * (res.bbox_xyxy[3] - res.bbox_xyxy[1]))
                    tr.drift_ref_area = (float(tr.drift_ref_area or 0.0) * 0.7
                                         + na * 0.3) or na
                    if tr.score < self.cfg.score_threshold:
                        tr.low_score_frames += 1
                    else:
                        tr.low_score_frames = 0
                else:
                    #: 漂移：**冻结**在上一帧位置（不丢弃！），历史里写冻结值，
                    #: 这样下一次 VLX 来时 `bbox_at()` 还能取到正确的位置。
                    tr.drift_frames += 1
                    tr.event = "drift:%s" % why
                    tr.last_drift = why  # type: ignore[attr-defined]
                    if why.startswith("area_step"):
                        self.stats["drift_reject_step"] += 1
                    elif why.startswith("area_abs"):
                        self.stats["drift_reject_abs"] += 1
                    else:
                        self.stats["drift_reject_move"] += 1
                    self.stats["drift_rejected"] += 1
                tr.last_frame_id = newest.frame_id
                tr.last_media_time_ms = newest.media_time_ms
                #: ★真实 PTS★：frame 是 CachedFrame（StreamHub 模式下由
                #: `put_ref` 写入，带 pts_ms）；旧链路/旧 hub 时为 None。
                tr.last_pts_ms = getattr(newest, "pts_ms", None)
                tr.remember(newest.frame_id, tr.bbox_xyxy)
                if tr.drift_frames >= self.cfg.drift_after:
                    if tr.state != STATE_LOST:
                        self.stats["lost_count"] += 1
                        self.stats["lost_drift"] += 1
                    tr.state = STATE_LOST
                    tr.event = "drift"
                else:
                    self._apply_state(tr, now)
            else:
                tr.low_score_frames += 1
                self.stats["update_not_ok"] += 1
                self._apply_state(tr, now)
            return tr.snapshot(now)

    # ── ★P1 异步运行时接口★ ──────────────────────────────────────────────
    #: 这三个方法是给 `async_runtime` 用的**只读视图**，不改任何既有判定。
    def iter_tracks(self) -> List[Track]:
        with self._lock:
            return list(self._tracks.values())

    def get_track(self, track_id: Any) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(int(track_id))

    def should_update(self, tr: Any, newest_frame_id: Any) -> bool:
        """这条 track 现在该不该发 update？

        三条否决，与 `step()` 里的 `step_skip_*` 口径一一对应，再加 P1 新增的
        `pending_init`：
          * LOST / REMOVED —— 没有意义（服务端模板已经废了），原来是
            `step_skip_state`；
          * `pending_init` —— ★P1 新增★：init 请求还在路上，此时发 update
            服务端只回 "track 未初始化"（实测 ref_errors 一路涨就是它）；
          * 已经追到最新帧 —— 原来是 `step_skip_current`。
        """
        if tr is None:
            return False
        if tr.state in (STATE_LOST, STATE_REMOVED):
            return False
        if getattr(tr, "pending_init", False):
            return False
        try:
            return int(tr.last_frame_id) < int(newest_frame_id)
        except Exception:  # noqa: BLE001
            return True

    def should_update_latest(self, tr: Any) -> bool:
        """★STEP3/P2★ latest 模式的"该不该发" —— **故意不看帧号**。

        为什么不能沿用 `should_update(tr, newest.frame_id)`：latest 模式下服务端
        取的是**它自己眼中**的最新帧，可能**超前**于客户端时间线（客户端只按
        10 Hz 轮询、且被量化过）。于是 `tr.last_frame_id` 会大于
        `cache.newest().frame_id`，那条"已追到最新帧"的判据恒为 False →
        **跟踪永久停摆**（一次都 update 不了，比 Age 大还糟）。

        所以这里只留两条"服务端视角下无意义"的否决：
          * LOST / REMOVED —— 模板已经废了，发过去只会得到 not ok；
          * `pending_init` —— init 还在飞，服务端此刻没有模板。

        节奏交给 worker 自己：**单槽串行**（一次只飞一个请求）+ 相同帧号时
        的自适应退避，不再依赖任何客户端帧号比较。
        """
        if tr is None:
            return False
        if tr.state in (STATE_LOST, STATE_REMOVED):
            return False
        if getattr(tr, "pending_init", False):
            return False
        return True

    def active_snapshots(self, now: Optional[float] = None) -> List[TrackSnapshot]:
        now = float(now if now is not None else time.time())
        with self._lock:
            return [t.snapshot(now) for t in self._tracks.values()
                    if t.state not in (STATE_LOST, STATE_REMOVED)]

    # ── 内部：关联 ────────────────────────────────────────────────────────
    def _bbox_of(self, det: Any) -> Optional[Tuple[float, float, float, float]]:
        """从一条 detection 里取框。**两种字段格式都要认**：

        * `bbox_xyxy` / `bbox`：本模块自己的 `TrackerResult` 与部分内部结构的写法；
        * `x1/y1/x2/y2`：**真实链路里 `RuleEngine.decorate()` 输出的格式**。
          （注：前端 2026-09-26 起已不画框、`LmmBox` 一并删除，但这里的字段约定
          是后端内部契约，不受影响。）

        只认前一种时，真实任务里每条 detection 都会被当成"没有框"直接丢弃
        —— `on_vlx` 看起来没报错、`track_update` 也照发，但 tracks 永远是空的。
        这个坑只有在真实 VLX 任务上跑才会暴露，离线单测喂的是 `bbox_xyxy`。
        """
        if isinstance(det, dict):
            b = det.get("bbox_xyxy") or det.get("bbox")
            if b and len(b) == 4:
                try:
                    x1, y1, x2, y2 = (float(v) for v in b)
                except (TypeError, ValueError):
                    return None
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)
            if all(det.get(k) is not None for k in ("x1", "y1", "x2", "y2")):
                try:
                    x1, y1, x2, y2 = (float(det[k]) for k in ("x1", "y1", "x2", "y2"))
                except (TypeError, ValueError):
                    return None
                if x2 > x1 and y2 > y1:
                    return (x1, y1, x2, y2)
        return None

    def _match(self, label: str, box: Tuple[float, float, float, float],
               frame_id: Optional[int] = None,
               exclude: Optional[set] = None) -> Optional[Track]:
        """同 label + IoU + 中心距 + 运动一致性（第 13 节的规则）。

        **必须在同一时刻比较**：`frame_id` 是 VLX 检测帧，用 track 在那
        一帧附近的历史位置比，而不是它当前的位置 —— 否则目标一动就失配。

        `exclude` 是本轮 VLX 里**已经被前面的框占用**的 track_id：一条 VLX
        detection 常带多个框，不排他的话两个框会抢同一条 track，画面上就只剩
        一个框（实测 frame=94 一次给了 2 个框）。
        """
        best: Optional[Track] = None
        best_score = 0.0
        diag = ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5 or 1.0
        skip = set(exclude or ())
        with self._lock:
            cands = [t for t in self._tracks.values()
                     if t.state not in (STATE_REMOVED,) and t.label == label
                     and t.track_id not in skip]
        for t in cands:
            ref = t.bbox_at(frame_id) if frame_id is not None else None
            if ref is None:
                ref = t.bbox_xyxy
            if not ref or max(ref) == 0:
                continue
            ov = iou_xyxy(ref, box)
            d = center_dist(ref, box)
            #: **"或"而不是"且"**：两个门都设成硬门槛时，跟踪器只要漂移一点
            #: 就两个门同时卡住 → 同一个目标被反复新建 track。重叠充分
            #: **或** 中心够近，二者有其一就认为是同一个目标。
            ok_iou = ov >= self.cfg.match_iou
            ok_dist = d <= self.cfg.match_center_ratio * diag
            #: 第三道门：**VLX↔VLX 的运动一致性**。前两道比的是"tracker 历史
            #: 位置 vs 新 VLX 框"，tracker 一旦漂了就全废 —— 实测 8 次检测里
            #: 7 次因为这道门废掉而新建 track，同一个移动目标被拆成 8 条。
            #: 改用**上一轮 VLX 的真值 anchor** 做基准：真值对真值，tracker 漂
            #: 不漂不影响这道门，只要目标在两次 VLX 之间的位移符合运动学即可。
            ok_move, move_gate = self._motion_ok(t, box, frame_id)
            if not (ok_iou or ok_dist or ok_move):
                continue
            if ok_iou or ok_dist:
                score = ov + max(0.0, 1.0 - d / diag) * 0.1
            else:
                #: 只有运动门命中 → 低优先级，且**按距离加权**：多个候选都靠
                #: 这道门时（目标挨得近）要选更近的那个，固定 0.05 会选到第一个。
                score = 0.05 * max(0.0, 1.0 - center_dist(t.vlx_anchor_box, box)
                                   / max(1e-6, move_gate))
            if score > best_score:
                best_score, best = score, t
        return best

    def _motion_ok(self, t: Track, box: Tuple[float, float, float, float],
                   frame_id: Optional[int]) -> Tuple[bool, float]:
        """`(是否通过, 允许位移)`：两次 VLX 之间的位移是否符合运动学。

        只认 `vlx_anchor_box`（VLX 真值），不认 tracker 的历史位置 —— 后者
        正是要被这道门兜底的错误来源。
        """
        if frame_id is None or t.vlx_anchor_box is None or t.vlx_anchor_frame is None:
            return False, 0.0
        gap = abs(int(frame_id) - int(t.vlx_anchor_frame))
        diag = ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5 or 1.0
        secs = gap / max(1e-6, self.cfg.match_move_fps)
        #: 时间只放宽到 `match_move_max_sec`：再久也不给更多额度，否则"很久没被
        #: 确认的旧 track"的门槛会大到把任何新目标都吸过去。
        secs = min(secs, self.cfg.match_move_max_sec)
        allowed = self.cfg.match_move_per_sec * diag * max(0.5, secs)
        return center_dist(t.vlx_anchor_box, box) <= allowed, allowed

    def _correction_needed(self, tr: Track, box: Tuple[float, float, float, float],
                           frame_id: Optional[int] = None) -> bool:
        """这条 VLX 检测**值不值得**把 tracker 的模板换掉（reset + initialize）。

        为什么必须问这一句（实测 5 分钟真实链路，`.workbuddy/realtime_probe.txt`）：
        VLX 每 ~3.5 s 来一次，每次都对已匹配目标做 reset + init + 8 帧 replay
        ≈ 983 ms，且远程后端是**一把锁串行**的 —— 5 分钟里 75 次这样的校正
        吃掉了约 25% 的时间，并把逐帧 update 挤到 0.84 次/秒，最后所有 track
        全部掉成 LOST，画面退化成"每 2~3 s 瞬移一次"。

        判定：tracker 本来就跟得住（状态正常 + 该帧历史位置与 VLX 框 IoU 够
        + 自己给的分数够）→ 这一帧只是**确认**，不重建模板。
        """
        if tr.state in (STATE_LOST, STATE_UNCERTAIN, STATE_NEW):
            return True
        ref = tr.bbox_at(frame_id) if frame_id is not None else None
        if not ref:
            ref = tr.bbox_xyxy
        if not ref or max(ref) == 0:
            return True
        try:
            ov = iou_xyxy(ref, box)
        except Exception:  # noqa: BLE001
            return True
        if ov < float(self.cfg.correction_iou):
            return True
        if float(tr.score or 0.0) < float(self.cfg.score_threshold):
            return True
        return False

    def _accept(self, tr: Track, prev: Tuple[float, float, float, float],
                new: Tuple[float, float, float, float],
                gap_frames: int) -> Tuple[bool, str]:
        """这一步 tracker 的输出**能不能信**（漂移守卫）。

        为什么必须有这一层：OSTrack 的 `score` 是分类头的峰值，**不代表框没漂**。
        实测里框从 32×23 膨胀到 938×513 的过程中，score 反而从 0.168 涨到
        0.671 —— 完全靠 score 判漂移会 100% 漏判，只能靠几何约束。

        两道门：
          1. 面积相对 VLX anchor 的倍率（与步数无关，膨胀一次就拦下）；
          2. 单步中心位移 / 目标对角线，按跨过的帧数开方缩放。

        被拦下时**位置冻结而不是丢弃**：框停在上一帧的位置，这样下一次 VLX
        来的时候历史里还是正确的地方，关联才有机会命中（丢弃了就永远对不上）。
        """
        cfg = self.cfg
        if not prev or max(prev) == 0 or not new or max(new) == 0:
            return True, ""
        pw = max(1e-6, float(prev[2] - prev[0]))
        ph = max(1e-6, float(prev[3] - prev[1]))
        nw = max(1e-6, float(new[2] - new[0]))
        nh = max(1e-6, float(new[3] - new[1]))
        area_new = nw * nh
        #: 门 1：**单步**面积变化（主门）—— 相对"近期基准框"的 EMA。
        #: 失控膨胀是逐帧累积的，单步涨 1.5 倍就会被这里连续拦下。
        ref = float(getattr(tr, "drift_ref_area", 0.0) or 0.0)
        #: ★门槛必须按"跨了多少帧"开方缩放★（与下面的位移门同一个道理）：
        #: 多目标串行时单条 track 两次 update 之间能隔上百帧（实测 3~6 s），
        #: 航拍里目标远近一变，面积本来就会整体涨落。用固定 2.5 倍去卡，
        #: 等于把**正常的尺度变化**全判成漂移（实测 118/120 次拒绝是它）。
        _scale = max(1.0, float(max(1, gap_frames)) ** 0.5)
        _up = 1.0 + (float(cfg.drift_area_step) - 1.0) * _scale
        if ref <= 0:
            #: 还没有基准（replay 0 帧等情形）：这一帧直接采信并把基准立起来。
            #: 不这么做会掉进一个死循环 —— 基准为空 → 永远拿 VLX 框当基准 →
            #: 恒定口径差被反复判漂移 → 连续 5 次 → LOST。
            tr.drift_ref_area = max(1e-6, area_new)
            return True, ""
        step_ratio = area_new / max(1e-6, ref)
        if step_ratio > _up:
            return False, "area_step_x%.1f>%.1f" % (step_ratio, _up)
        if step_ratio < 1.0 / max(1e-6, _up):
            return False, "area_step_x%.2f<%.2f" % (step_ratio, 1.0 / _up)
        #: 门 2：相对 VLX anchor 的**绝对上限**（兜底）—— 只拦真正的失控，
        #: 不拦"OSTrack 输出的框本来就和 VLX 框不是一个口径"这种正常差异。
        base = float(tr.anchor_area or 0.0) or (pw * ph)
        if base > 0:
            ratio = area_new / base
            if ratio > cfg.drift_area_grow:
                return False, "area_abs_x%.1f" % ratio
            if ratio < 1.0 / max(1e-6, cfg.drift_area_shrink):
                return False, "area_abs_x%.2f" % ratio
        diag = (nw * nw + nh * nh) ** 0.5 or 1.0
        move = center_dist(prev, new)
        allowed = cfg.drift_move_ratio * diag * max(1.0, float(gap_frames) ** 0.5)
        if move > allowed:
            return False, "move_%.0f>%.0f" % (move, allowed)
        return True, ""

    def _row_for(self, t: Track, box: Tuple[float, float, float, float],
                 frame_id: Optional[int] = None) -> Dict[str, Any]:
        """单条 track 与某个框的比对数据（诊断用，不参与判定）。"""
        diag = ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5 or 1.0
        ref = t.bbox_at(frame_id) if frame_id is not None else None
        if ref is None:
            return {"state": t.state, "no_history": True}
        gap = None
        if t.history:
            gap = min(abs(f - int(frame_id or 0)) for f, _ in t.history)
        ok_move, allowed = self._motion_ok(t, box, frame_id)
        out = {
            "state": t.state,
            "iou": round(iou_xyxy(ref, box), 3),
            "dist": round(center_dist(ref, box), 1),
            "diag": round(diag, 1),
            "hist_gap_frames": gap,
            "hist_len": len(t.history),
            "ref_box": [round(float(v), 1) for v in ref],
        }
        if t.vlx_anchor_box is not None:
            out["anchor_move"] = round(center_dist(t.vlx_anchor_box, box), 1)
            out["anchor_gate"] = round(allowed, 1)
            out["anchor_ok"] = ok_move
            #: ★必须把 anchor 帧号也打出来★：只打 move/gate 时，一旦两者对不上
            #: （真实 E2E 出现过 move=536 / gate=443 却判 MATCH），看日志的人
            #: 没有任何依据判断是"门放太宽"还是"拿错了 track 的 anchor"。
            out["anchor_frame"] = t.vlx_anchor_frame
        return out

    def _explain(self, label: str, box: Tuple[float, float, float, float],
                 frame_id: Optional[int] = None) -> Dict[str, Any]:
        """为什么没匹配上（只用于日志/诊断，不参与判定）。"""
        with self._lock:
            cands = [t for t in self._tracks.values()
                     if t.state not in (STATE_REMOVED,) and t.label == label]
        out: Dict[str, Any] = {"cands": len(cands)}
        out["rows"] = [dict({"tid": t.track_id}, **self._row_for(t, box, frame_id))
                       for t in cands]
        return out

    def _create(self, label: str, box: Tuple[float, float, float, float],
                frame_id: int, media_time_ms: float, now: float) -> Optional[Track]:
        with self._lock:
            alive = [t for t in self._tracks.values() if t.state != STATE_REMOVED]
            if len(alive) >= self.cfg.max_tracks:
                #: 超上限：淘汰"最久没被 VLX 确认"的那条（第 14 节）
                victim = min(alive, key=lambda t: (t.last_vlx_confirmed_at or 0.0,
                                                   t.created_at))
                self._drop(victim.track_id)
            tid = self._next_id
            self._next_id += 1
            tr = Track(track_id=tid, label=label, state=STATE_NEW, bbox_xyxy=box,
                       last_frame_id=int(frame_id),
                       last_media_time_ms=float(media_time_ms),
                       last_vlx_frame_id=int(frame_id), last_vlx_confirmed_at=now,
                       anchor_area=max(1e-6, (box[2] - box[0]) * (box[3] - box[1])),
                       vlx_anchor_frame=int(frame_id),
                       vlx_anchor_box=tuple(float(v) for v in box))
            tr.remember(int(frame_id), box)
            self._tracks[tid] = tr
            self.stats["tracks_created"] += 1
            return tr

    # ── ★P1 异步初始化★ ──────────────────────────────────────────────────
    def _request_init(self, tr: Track, frame_id: int, media_time_ms: float,
                      pts_ms: Optional[float],
                      box: Tuple[float, float, float, float], now: float,
                      label: str, kind: str = "init") -> int:
        """★把 init 交给该 track 自己的 worker，VLX 线程立刻返回★（P1 目标 2）。

        为什么必须异步：同步路径 `_reinit_and_replay` 要付 `reset + init +
        最多 N 次 update`，实测 **~983 ms**，而且是在 **VLX 线程里**跑的 ——
        VLX 一来，所有目标的逐帧 update 全部饿死（STEP3 阻塞点 B3）。

        本方法只做两件事：
          1. 同步落下"VLX 是真值"这一层的状态（anchor / 漂移基准 / TTL）——
             这些是纯内存操作，微秒级，且**必须立刻生效**，否则下一次 VLX
             到来时关联判定会用到过期的 anchor；
          2. 投递 `InitReq`。**帧号与 pts 原样带过去**，不做任何换算。

        ★`frame_id` 与 `pts_ms` 必须严格对应同一帧★：init 隔帧会让模板错位、
        框一路扩到全屏（踩过，曾被误判成跟踪器漂移）。
        """
        planned = 0
        try:
            if self.cache is not None:
                planned = len(self.cache.replay_plan(
                    int(frame_id), self.cfg.max_replay_frames))
        except Exception:  # noqa: BLE001
            planned = 0

        req: Dict[str, Any] = {
            "track_id": int(tr.track_id),
            "frame_id": int(frame_id),
            "media_time_ms": float(media_time_ms),
            "pts_ms": None if pts_ms is None else float(pts_ms),
            "bbox": [float(v) for v in box],
            "label": str(label or ""),
            "kind": str(kind),
            "requested_at": float(now),
        }
        with self._lock:
            #: 先落下 VLX 这一层的真值（与 `_reinit_and_replay` 成功后做的等价）
            tr.anchor_area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
            tr.drift_ref_area = 0.0
            tr.vlx_anchor_frame = int(frame_id)
            tr.vlx_anchor_box = tuple(float(v) for v in box)
            tr.drift_frames = 0
            tr.low_score_frames = 0
            tr.last_vlx_frame_id = int(frame_id)
            tr.last_vlx_confirmed_at = float(now)
            #: ★已知 PTS 绝不能被覆盖成 None★：拿不到新 PTS 时保持旧值，
            #: 否则前端那行"真实时间轴"诊断会因为一次 init 整行消失
            #: （pts 缺失 → 诊断降级），看起来像时间轴断了。
            if pts_ms is not None:
                tr.last_pts_ms = float(pts_ms)
            #: ★pending 标记★：worker 见到 True 就**不发 update**（服务端此刻
            #: 还没有模板，发过去只会得到 "track 未初始化"）。
            tr.pending_init = True
            tr.pending_init_at = float(now)
            tr.pending_init_started = False
            tr.init_req_frame = int(frame_id)
            tr.init_kind = str(kind)
            self.stats["init_async_queued"] = int(
                self.stats.get("init_async_queued", 0)) + 1

        #: ★STEP5★ `on_init_request` 现在**必须返回 bool**（True = 有人接手）。
        #: 返回 False = 根本没有 worker 会处理这条请求（worker 槽位满 / 请求被
        #: 丢弃）—— 这时如果还把 `pending_init` 留在 True，这条 track 就**永远**
        #: 停在 new：`should_update()` 恒为 False → 一次 update 都不发 →
        #: `last_frame_id` 不动 → 前端一直收到同一个框（实测挂过 862 帧）。
        accepted = True
        try:
            ret = self.on_init_request(req)
            if ret is False:
                accepted = False
        except Exception:  # noqa: BLE001
            LOG.exception("[TargetManager] 投递 InitReq 失败 track=%s", tr.track_id)
            accepted = False
        if not accepted:
            self._fail_init(tr.track_id, "no_worker")
            return 0
        return planned

    # ── ★STEP5★ init 成败的**唯一出口** ────────────────────────────────────
    def _init_log(self, tr: "Track", ok: bool, reason: str,
                  level: str = "info") -> None:
        """★STEP5 第 4 步★：每次 init 一条结构化日志。

        格式（用户指定）：
            track=3
            vlx_frame=12345
            init_frame=12345
            gap=0
            success

        失败时最后一行换成失败原因（`expired` / `no_worker` / `timeout` /
        `http_404` / 服务端给的 `reason`）。
        """
        try:
            cur = self.cache.newest() if self.cache is not None else None
            cur_fid = int(getattr(cur, "frame_id", 0) or 0)
        except Exception:  # noqa: BLE001
            cur_fid = 0
        vf = int(tr.vlx_anchor_frame or 0)
        rf = int(tr.init_req_frame if tr.init_req_frame is not None else vf)
        line = ("init track=%s kind=%s vlx_frame=%s init_frame=%s "
                "latest=%s gap=%s %s"
                % (tr.track_id, tr.init_kind or "init", vf, rf, cur_fid,
                   cur_fid - rf, ("success" if ok else str(reason or "failed"))))
        if level == "warning" or not ok:
            LOG.warning("[TargetManager] %s", line)
        else:
            LOG.info("[TargetManager] %s", line)
        #: 同时留一份到 stats，便于 WS/报告里直接读最近几次
        with self._lock:
            log = self.stats.setdefault("init_log", [])
            if isinstance(log, list):
                log.append({"t": round(time.time(), 3), "track": tr.track_id,
                            "kind": tr.init_kind or "init", "vlx_frame": vf,
                            "init_frame": rf, "latest": cur_fid,
                            "gap": cur_fid - rf,
                            "result": ("success" if ok
                                       else str(reason or "failed"))})
                #: ★只留最近 50 条★：这是环形诊断缓冲，不是审计日志，
                #: 让它无限长会把 WS 消息撑爆。
                del log[:-50]

    def _fail_init(self, track_id: int, reason: str,
                   drop: Optional[bool] = None) -> None:
        """init 失败的统一处理：**绝不让它继续留在 new**。

        三件事，一件都不能少：
          1. `pending_init = False` —— 否则 `should_update()` 永远否决，
             这条 track 再也不会被 update（画面上就是静止框）；
          2. 状态离开 `new` —— `active_snapshots()` 只排除 LOST/REMOVED，
             不排除 new，所以置 LOST 是"从画面上摘掉这个框"的唯一办法；
          3. 新目标（`kind == "init"`）**直接删除** —— 它一次都没成功过，
             留着只会占 `_tracks` 名额（`_create` 的 max_tracks 淘汰会因此
             误杀正常的 track）。周期校正失败只置 LOST：那条 track 本来
             跟得好好的，下一次 VLX 还能校正回来。
        """
        with self._lock:
            tr = self._tracks.get(int(track_id))
            if tr is None:
                return
            tr.pending_init = False
            tr.pending_init_at = 0.0
            tr.pending_init_started = False
            kind = str(tr.init_kind or "init")
            if drop is None:
                #: ★`timeout` 不删除、不动服务端★：它只是"太久没回"，
                #: 那条请求**大概率还在 worker 里跑着**。此刻删除等于把马上就要
                #: 建好的模板拆掉，后续 `update_at` 全报 "track 未初始化"
                #: （实测 ref_errors 37 → 158 就是这么来的）。
                #: 置 LOST 已经满足"不占画面 / 不占 worker / 不占 update 队列"。
                drop = (kind == "init" and reason != "timeout")
            tr.state = STATE_LOST
            tr.event = "init_failed:%s" % reason
            self.stats["init_failed"] = int(self.stats.get("init_failed", 0)) + 1
            self.stats["init_fail_%s" % reason] = int(
                self.stats.get("init_fail_%s" % reason, 0)) + 1
        self._init_log(tr, False, reason)
        if drop:
            #: 删掉 = 彻底不占 bbox 显示 / worker / update 队列，
            #: 下一次 VLX 重新发现时会新建一条干净的 track。
            self._drop(int(track_id))
        elif reason != "timeout" and self.backend is not None:
            #: ★只有"确认失败"才清服务端模板★。
            #: `timeout` **绝对不能**清 —— 那条 init 请求大概率还在 worker 里
            #: 跑着，此刻 remove 会把**它马上就要建好的模板**删掉，于是后续
            #: `update_at` 全报 "track 未初始化"（实测 ref_errors 从 37 涨到
            #: 158 就是这么来的）。超时只清本地状态就够了。
            try:
                self.backend.remove(int(track_id))
            except Exception:  # noqa: BLE001
                pass

    def sweep_pending_init(self, now: Optional[float] = None) -> int:
        """★STEP5 第 3 步★：`pending_init` 超时巡检。

        没有这一层时，`pending_init` 只能靠 worker 的回执清除。一旦请求没被
        处理（worker 槽位满 / 线程异常 / 服务端不回），它就**永远**是 True。
        这里给它一个硬性上限，超时一律判失败。

        由 `StateLoop`（状态唯一写入者）与 `step()` 周期性调用 —— 不能放在
        `should_update()` 里：那是 worker 每 2 ms 调一次的只读视图。
        """
        now = float(now if now is not None else time.time())
        limit_ms = float(getattr(self.cfg, "pending_init_timeout_ms", 2000.0))
        if limit_ms <= 0:
            return 0
        #: ★已开工的请求给一个宽松得多的硬上限（纯兜底）★：worker 的 `finally`
        #: 一定会回执，正常情况下根本用不到；留着只是为了防"worker 线程真的
        #: 死了"这种极端情况，避免又变回永久 new。
        limit_started_ms = max(5.0 * limit_ms, 10000.0)
        due = []
        with self._lock:
            for t in self._tracks.values():
                if not getattr(t, "pending_init", False):
                    continue
                at = float(getattr(t, "pending_init_at", 0.0) or 0.0)
                if at <= 0:
                    continue
                age_ms = (now - at) * 1000.0
                cap = (limit_started_ms if getattr(t, "pending_init_started",
                                                   False) else limit_ms)
                if age_ms > cap:
                    due.append(t.track_id)
        n = 0
        for tid in due:
            self._fail_init(tid, "timeout")
            n += 1
        if n:
            self.stats["init_timeout"] = int(self.stats.get("init_timeout", 0)) + n
        return n

    def mark_init_started(self, track_id: int) -> None:
        """★STEP5★ worker 真正开始处理这条 init 请求时打个点。

        打了这个点之后，超时阈值从 `pending_init_timeout_ms`（默认 2000 ms）
        放宽到 `max(5×, 10000 ms)` —— 从"入队"起算 2000 ms 会把大量本可成功
        的 init 误杀（实测 67 次里 25 次）。
        """
        with self._lock:
            tr = self._tracks.get(int(track_id))
            if tr is None:
                return
            tr.pending_init_started = True

    def mark_init_done(self, track_id: int, ok: bool = True,
                       reason: str = "", kind: str = "",
                       latency_ms: float = 0.0) -> None:
        """worker 完成 init 后的回执（P1）。

        ★STEP5★ 失败路径统一走 `_fail_init`：置 LOST（从画面上摘掉框），
        新目标直接删除。改造前只置了 LOST 但**没有清 `pending_init` 之外
        的资源**，而且"请求根本没被处理"这种情况压根不会走到这里 —— 那才是
        track 卡在 new 的真凶（见 `_request_init` 与 `sweep_pending_init`）。
        """
        with self._lock:
            tr = self._tracks.get(int(track_id))
            if tr is None:
                return
            tr.pending_init = False
            tr.pending_init_at = 0.0
            tr.pending_init_started = False
            if kind:
                tr.init_kind = str(kind)
            #: ★STEP5★ init 端到端耗时分布。超时阈值是否合理全靠它 ——
            #: 拍脑袋的 2000 ms 实测把大量本可成功的 init 误杀成 timeout。
            if latency_ms > 0:
                ms = float(latency_ms)
                self.stats["init_ms_total"] = float(
                    self.stats.get("init_ms_total", 0.0)) + ms
                self.stats["init_ms_count"] = int(
                    self.stats.get("init_ms_count", 0)) + 1
                if ms > float(self.stats.get("init_ms_max", 0.0)):
                    self.stats["init_ms_max"] = round(ms, 1)
                hist = self.stats.setdefault("init_ms_recent", [])
                if isinstance(hist, list):
                    hist.append(round(ms, 1))
                    del hist[:-50]
        if not ok:
            #: ★失败原因归一化★：服务端回的 `reason` 里最常见的就是 hub 404
            #: （目标帧已被环形缓冲挤出），日志里要能一眼认出来。
            why = str(reason or "").strip()
            low = why.lower()
            if "404" in low or "not found" in low:
                why = "http_404"
            elif not why:
                why = "failed"
            self._fail_init(track_id, why)
            return
        with self._lock:
            tr = self._tracks.get(int(track_id))
            if tr is None:
                return
            self.stats["init_async_done"] = int(
                self.stats.get("init_async_done", 0)) + 1
        self._init_log(tr, True, "success")

    def _reinit_and_replay(self, tr: Track, frame_id: int, media_time_ms: float,
                           box: Tuple[float, float, float, float], now: float,
                           label: str) -> int:
        """第 9 节：在 detection 对应的历史帧上初始化 → 沿缓存 replay 到最新帧。

        这一步是 R4A 的核心：**VLX 给的是 2.3 秒前的位置**，直接拿来画是错的，
        但丢掉又浪费，所以"回到那一帧 + 追到现在"。

        返回本次 replay 的帧数 —— **当次，不是累计**（调用方要记当次）。
        """
        anchor = self.cache.get(int(frame_id)) if self.cache is not None else None
        if anchor is None:
            return 0
        #: 校正时先把旧状态清掉（第 17 节：新 VLX = 新的 ground-truth anchor）
        self.backend.reset(tr.track_id)
        if self.ref_mode:
            #: ★服务器侧共享帧★：只给 `frame_id`，服务端自己取那一帧。
            #: 必须传**VLX 检出时那一帧**（`frame_id`），否则模板错位 → 框一路
            #: 扩到全屏（实测隔 27 帧初始化的后果，曾被误判成跟踪器漂移）。
            res = self.backend.init_by_ref(tr.track_id, box, int(frame_id),
                                           float(media_time_ms), label,
                                           pts_ms=getattr(anchor, "pts_ms", None))
        else:
            res = self.backend.initialize(tr.track_id, anchor.frame, box)
        if not res.ok:
            tr.state = STATE_LOST
            tr.event = "init_failed"
            with self._lock:
                self.stats["init_failed"] += 1
            return 0
        #: 新 VLX = 新的真值 → 漂移基准重置（第 17 节）
        tr.anchor_area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        #: 单步面积基准从"tracker 自己的输出"起步（不是 VLX 框）——
        #: 否则第一帧就会把 OSTrack 与 VLX 的口径差当成漂移。
        tr.drift_ref_area = 0.0
        tr.vlx_anchor_frame = int(frame_id)
        tr.vlx_anchor_box = tuple(float(v) for v in box)
        tr.drift_frames = 0
        t0 = time.time()
        plan = self.cache.replay_plan(int(frame_id), self.cfg.max_replay_frames)
        steps = 0
        rejected = 0
        cur: Tuple[float, float, float, float] = tuple(float(v) for v in box)
        last_score = 0.0
        for item in plan:
            if self.ref_mode:
                r = self.backend.update_at(tr.track_id, int(item.frame_id),
                                           float(item.media_time_ms),
                                           pts_ms=getattr(item, "pts_ms", None))
            else:
                r = self.backend.update(tr.track_id, item.frame)
            steps += 1
            if r.ok and r.bbox_xyxy is not None:
                #: replay 是**连续帧**，这里漂移比 step() 更能说明 tracker 已经
                #: 跟不住了 —— 但只统计不中断：replay 中途停掉会让位置停在旧帧。
                ok, why = self._accept(tr, cur, r.bbox_xyxy,
                                       max(1, int(item.frame_id) - int(frame_id)))
                if not ok:
                    rejected += 1
                    tr.remember(item.frame_id, cur)
                    continue
                cur = r.bbox_xyxy
                last_score = float(r.score or 0.0)
                tr.last_frame_id = item.frame_id
                tr.last_media_time_ms = item.media_time_ms
                tr.last_pts_ms = getattr(item, "pts_ms", None)
                tr.remember(item.frame_id, r.bbox_xyxy)
        replay_ms = (time.time() - t0) * 1000.0
        with self._lock:
            self.stats["replay_total"] += 1
            self.stats["replay_frames"] += steps
            self.stats["replay_ms"] += replay_ms
        if rejected:
            tr.drift_frames += 1
            tr.event = "drift:replay_rejected_%d" % rejected
        if rejected < steps:
            #: 至少有一步被接受才更新位置/分数；全被拒时**保留 VLX 的 anchor**，
            #: 不能把 score 归零（归零会立刻触发 low_score 计数 → 提前 LOST）。
            tr.bbox_xyxy = cur
            tr.score = float(last_score or tr.score)
        else:
            tr.bbox_xyxy = box
        #: 锚点帧也要进历史：下一次 VLX 可能还是拿这一帧附近的框来比
        tr.remember(int(frame_id), box)
        #: replay 结束时 `cur` 已经是 tracker 自己的输出 → 以此作为单步面积基准
        tr.drift_ref_area = max(1e-6, (cur[2] - cur[0]) * (cur[3] - cur[1]))
        tr.label = label
        tr.last_vlx_frame_id = int(frame_id)
        tr.last_vlx_confirmed_at = now
        tr.low_score_frames = 0
        if tr.state in (STATE_LOST, STATE_UNCERTAIN, STATE_NEW):
            tr.event = "reacquired" if tr.state == STATE_LOST else "confirmed"
            if tr.state == STATE_LOST:
                self.stats["reacquire_count"] += 1
                tr.reacquire_count += 1
        else:
            tr.event = "updated"
        tr.state = STATE_TRACKING
        tr.last_replay = {"frames": steps, "ms": round(replay_ms, 2)}  # type: ignore[attr-defined]
        return steps

    def _apply_state(self, tr: Track, now: float) -> None:
        """UNCERTAIN / LOST 判定 + VLX 确认年龄 TTL（第 19/20/21 节）。"""
        age_ms = None
        if tr.last_vlx_confirmed_at:
            age_ms = (now - tr.last_vlx_confirmed_at) * 1000.0
        stale = age_ms is not None and age_ms > self.cfg.max_without_vlx_ms
        if stale:
            #: 第 21 节：VLX 太久没确认，即使 tracker 看着还跟得住也不再信任
            if tr.state != STATE_LOST:
                self.stats["lost_count"] += 1
                self.stats["lost_vlx_ttl"] += 1
            tr.state = STATE_LOST
            tr.event = "vlx_ttl_expired"
            return
        if tr.low_score_frames >= self.cfg.lost_after:
            if tr.state != STATE_LOST:
                self.stats["lost_count"] += 1
                self.stats["lost_low_score"] += 1
            tr.state = STATE_LOST
            tr.event = "lost"
        elif tr.low_score_frames >= self.cfg.uncertain_after:
            tr.state = STATE_UNCERTAIN
            tr.event = "uncertain"
        else:
            tr.state = STATE_TRACKING
            tr.event = tr.event if tr.event in ("reacquired", "confirmed") else "updated"

    def _drop(self, track_id: int) -> None:
        #: ★STEP5★ 自己持锁：现在有两处调用方 —— `_create`（已在锁内，RLock
        #: 可重入）与 `_fail_init`（在锁外，否则会在锁里做网络调用）。
        with self._lock:
            tr = self._tracks.pop(int(track_id), None)
            if tr is None:
                return
            self.stats["tracks_removed"] += 1
            tr.state = STATE_REMOVED
        try:
            self.backend.remove(int(track_id))
        except Exception:  # noqa: BLE001
            pass

    # ── 观测 / 清理 ───────────────────────────────────────────────────────
    def snapshot_stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._tracks)
            states: Dict[str, int] = {}
            for t in self._tracks.values():
                states[t.state] = states.get(t.state, 0) + 1
            st = dict(self.stats)
        st["tracks"] = n
        st["states"] = states
        if st.get("replay_total"):
            st["avg_replay_ms"] = round(st["replay_ms"] / st["replay_total"], 2)
            st["avg_replay_frames"] = round(st["replay_frames"] / st["replay_total"], 1)
        #: ★第 5/6 节★：帧率一律按"成功 update / 运行时长"现算。
        #: `total_update_fps` 是**所有目标的总和**；`per_track_fps` 才是单目标
        #: 真正拿到的频率 —— 多目标串行时前者会是后者的 N 倍，绝不能混用。
        up = max(0.001, time.time() - self._started_at)
        st["uptime_s"] = round(up, 1)
        st["total_update_fps"] = round(st["step_update_ok"] / up, 2)
        st["step_fps"] = round(st["step_calls"] / up, 2)
        st["per_track_fps"] = {
            k: round(float(v) / up, 2)
            for k, v in (st.get("per_track_updates") or {}).items()}
        return st

    def track_ids(self) -> List[int]:
        with self._lock:
            return list(self._tracks.keys())

    def clear(self) -> None:
        with self._lock:
            ids = list(self._tracks.keys())
            self._tracks.clear()
        for tid in ids:
            try:
                self.backend.remove(tid)
            except Exception:  # noqa: BLE001
                pass
