"""LMM 模块的可调参数。

要求（来自需求文档）：参数不硬编码在业务代码里，通过配置文件覆盖。

优先级：**环境变量 > 配置文件 > 内置默认值**
配置文件：`backend/workspace/lmm_config.json`（与 `vlm_providers.json` 同目录，沿用项目习惯）

注意：RunOS 后端本身**不使用 python-dotenv**（全仓无 dotenv 依赖）。

**不再有 `RUNOS_LLM_*` 相关配置。** 那些变量原本是给 PromptAnalyzer 调外部大模型做
中文 → 英文关键词拆解用的；该链路已于 2026-09-23 删除（`lmm/prompt_analyzer.py`）——
用户输入不会经过关键词拆解；若配置了附加提示词，请求时会在原文后追加该说明。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

from config import settings

#: 配置文件位置（放在 models 目录下，与项目其它运行时配置保持一致）
CONFIG_FILE: Path = settings.PLATFORM_MODELS_DIR / "lmm_config.json"

_LOCK = threading.RLock()

#: 检测与任务默认参数。键名与前端"高级配置"面板一一对应。
DEFAULTS: Dict[str, Any] = {
    # ── 在线视觉检测 ──────────────────────────────────────────────────────
    "engine": "vlm",                     # vlm | vlx | yoloe（兼容旧 qwen-vl）
    "vision_provider_id": "",            # 已保存的在线视觉大模型配置 ID
    "qwen_provider_id": "",              # 旧版配置兼容
    "qwen_interval_seconds": 2.0,        # 最新帧优先，不排队处理过期视频帧
    # ── 模型 ────────────────────────────────────────────────────────────────
    #: 首选权重（相对名由 ultralytics 解析，绝对路径亦可）
    "model_name": "yoloe-26s-seg.pt",
    "model_candidates": ["yoloe-26s-seg.pt", "yoloe-11s-seg.pt"],
    #: 文本编码器（MobileCLIP2-B TorchScript）。ultralytics 会去 WEIGHTS_DIR 找同名文件。
    "text_encoder": "mobileclip2_b.ts",
    #: 输入尺寸。**航拍小目标实测**：640 下 YOLOE 对水面人员（框只占画面 3%）完全漏检，
    #: 1280 才能稳定召回（MX250 上约 6 FPS，3060 上无压力）。所以默认取 1280。
    "imgsz": 1280,
    #: 开放词汇检测的置信度普遍低于封闭集专用模型（实测同场景 0.10~0.23），
    #: 沿用 0.25 会把真目标滤掉，这里放低到 0.20。
    "confidence": 0.20,
    "iou": 0.45,
    #: 半精度：**本机 MX250 是 Pascal(CC 6.1)，FP16 吞吐仅为 FP32 的 1/64，开启只会更慢**。
    #: 目标环境 RTX 3060 12GB 上建议开启。默认关闭。
    "half": False,
    "device": "auto",  # auto | cpu | cuda:0
    # ── 跟踪 ────────────────────────────────────────────────────────────────
    "tracker": "botsort",  # botsort | bytetrack
    # ── 任务 ────────────────────────────────────────────────────────────────
    "frame_stride": 1,
    "max_frame_queue": 1,      # 单槽丢帧：推理跟不上时丢弃旧帧而不是堆积
    # ── 告警规则 ────────────────────────────────────────────────────────────
    "hit_frames": 3,           # 同一 track 连续命中 N 帧才告警
    "cooldown_seconds": 30,    # 同一 track 的告警冷却
    "min_targets": 1,          # 至少检出多少个目标才触发
    #: 无轨迹候选的复核兜底。ultralytics 只给"已激活轨迹"回填 boxes.id，而远距离小目标
    #: （航拍里的人/游泳者）根本激活不了轨迹 —— 实测 stride=1 下 person 框 0/350 拿到 id，
    #: 同一素材的 boat 却有稳定轨迹。若不兜底，这类目标的 VLM 调用数恒为 0，二阶段形同虚设。
    #: 打开后：按画面网格把无轨迹候选归到"空间格"当伪轨迹，继续送 VLM 复核（缓存键同网格）。
    "trackless_verify": True,
    #: 空间格边长（画面切成 grid×grid）。越小归属越精细，但跨格移动会把伪轨迹切断。
    "trackless_grid": 6,
    # ── 二阶段：视觉大模型复核 ──────────────────────────────────────────────
    # 配置本身**不在这里**：复用「大模型自动标注」已有的 workspace/models/vlm_providers.json，
    # 这里只存"用哪一条"（config_id）以及调度参数。
    "vlm_enabled": True,
    "vlm_config_id": "",              # 空 = 未选择；前端下拉读现有配置
    "vlm_strategy": "smart",          # smart=智能触发 | interval=固定间隔
    "vlm_interval_seconds": 5.0,      # strategy=interval 时每隔多久复核一次
    "semantic_cache_ttl": 2.0,        # 同一 track 的语义结论有效期（秒）
    "vlm_bbox_area_delta": 0.35,      # 框面积变化超过该比例 → 重新复核
    "vlm_conf_drop_ratio": 0.6,       # 检测置信度跌到上次的该比例以下 → 重新复核
    "vlm_gap_frames": 15,             # 目标消失超过该帧数又出现 → 重新复核
    "vlm_context_scale": 1.5,         # ROI 扩框倍数（判定"在水里"需要环境像素）
    "vlm_max_side": 768,              # 送给 VLM 的裁剪图长边上限
    #: 航拍小目标专用：倍数对小框是失效的（10px 的框扩 1.5 倍仍是 15px，看不到环境），
    #: 所以再加绝对下限 —— 窗口长边不足 256px 就向外补足，送图长边不足 384px 就放大。
    "vlm_min_window": 256,
    "vlm_min_side": 384,
    "vlm_concurrency": 1,             # VLM worker 数
    "vlm_timeout": 60,                # 单次调用超时（秒）
    "vlm_max_repair": 1,              # JSON 解析失败的修复次数上限
    "vlm_max_tokens": 1024,
    "vlm_queue_size": 16,             # VLM 队列上限，满了丢最旧
    "vlm_job_ttl_seconds": 3.0,       # 排队超过该时长仍未处理 → 丢弃（画面已经跑过去了）
    "vlm_on_error": "allow",          # allow=降级告警并标未复核 | block=严格不告警
    "vlm_temporal_frames": 0,         # >1 时启用连续多帧行为理解
    "vlm_temporal_span": 8,           # 多帧采样间隔（帧）
    "visual_prompt_path": "",         # Visual Prompt 参考图（磁盘路径）
    # ── 三阶段：VLX-Seek（自然语言目标检测，替代 YOLOE + Qwen-VL 二阶段）──────
    #: 开关。关掉则回退到旧的 YOLOE 链路。
    "vlx_seek_enabled": False,
    #: 传输方式：ssh=复用算力节点凭据走 direct-tcpip 隧道（服务端只绑 127.0.0.1，
    #: 不暴露公网）；direct=直连 base_url。
    "vlx_seek_transport": "ssh",
    #: 算力节点 id。留空时自动挑名为 3060 的节点。
    "vlx_seek_node_id": "",
    "vlx_seek_port": 8021,
    "vlx_seek_base_url": "",          # 仅 transport=direct 时用，不写死 IP
    "vlx_seek_token": "",             # 与云端服务 .env 的 VLX_TOKEN 一致
    "vlx_seek_timeout": 300.0,        # 单次请求超时（秒）；offload 模式单图可达数十秒
    "vlx_seek_task": "detection",     # detection | reasoning_detection | grounding_single | counting
    "vlx_seek_lang": "zh",
    "vlx_seek_max_new_tokens": 256,   # 重复块自动停机，实际通常只用到十几到几十个 token
    "vlx_seek_frame_interval": 1.0,   # 视频/摄像头按秒抽样间隔
    "vlx_seek_extra_instruction": "", # 所有 VLX 入口共用的附加提示词；空值保持原有行为
    #: ★R2B★ Fast Transport 总开关（0=每请求新建 channel+HTTP 连接，1=复用）。
    #: R2B-1 已实测证伪两个猜想：SSH channel window（默认已是 2 MB，反向对照
    #: 256 KB 毫无变化）与 socket 缓冲区（写入快了但端到端没变，只是把耗时挪到
    #: wait_first_byte）。本开关只启用**唯一被证实有效**的那一项：复用
    #: direct-tcpip channel + HTTP/1.1 keep-alive，消掉约 148 ms/次的建连。
    "vlx_seek_fast_transport": 0,

    # ── R1：Realtime Latest-Frame Fast Path ─────────────────────────────────
    #: 总开关。0 = legacy_sync（读一帧→推理→推画面，行为与 R1 之前完全一致）；
    #: 1 = realtime_latest（采集与推理解耦、最新帧优先）。
    #: 默认 0：新链路出问题可以**一键回退**，旧实现一行没删。
    "lmm_realtime_fast_path": 0,
    #: 预览帧率（只影响 UI 画面，不影响送给 VLX 的原图）。
    #: VLX 只有 0.2~1 FPS，画面必须靠这条独立通道维持 5~10 FPS。
    "lmm_realtime_preview_fps": 8.0,
    "lmm_realtime_preview_quality": 75,
    "lmm_realtime_preview_max_side": 960,
    #: 本地文件是否按媒体时间匀速采集。**必须 True**：否则后端一路读到文件尾，
    #: 浏览器还停在第 10 秒，detection 与画面完全对不上。
    "lmm_realtime_pacing": 1,
    #: ★WebRTC 高清预览★ MediaMTX 的**对外基础地址**（形如 `http://<media-host>:8889`）。
    #: 空串 = 关闭 WebRTC，浏览器继续走现有的 JPEG preview 通道（fallback，本轮不删）。
    #: ★绝不写死 `127.0.0.1:8889`★：RunOS 部署到服务器之后，浏览器里的
    #: 127.0.0.1 是**用户自己的电脑**，不是 RunOS 服务器。所以这个地址必须由后端
    #: 下发（前端从 `GET /api/lmm/config` 读），后续也可换成按任务下发 `webrtc_url`。
    "lmm_webrtc_base_url": "",

    # ── ★技术路线切换（2026-09-25）：VLX 单模型 + 三帧确认告警 ──────────
    #: 唯一检测链路 = StreamHub → VLX → DetectionManager → AlertEvent。
    #: **不再有任何跟踪器参与**：没有 init / 模板 / replay / 状态机 / 漂移守卫。
    #: 下面 `lmm_alert_*` 一组是告警机制的阈值；`lmm_tracking_*` / `lmm_ostrack_*`
    #: 那两组**保留但不再被运行时读取**（文件与配置都在，便于将来重新接入）。
    #:
    #: ★1-Hit 即时告警（2026-09-26 产品定案）★：`min_hits` / `max_frame_gap` /
    #: `max_time_ms` / `expire_ms` 四项**已废弃、读了也不再生效** —— 不存在
    #: "连续命中确认"和"候选存活"这两个概念了。留着键名是为了不让旧配置文件报错，
    #: 也是为了将来要回退时不用改配置加载白名单。
    "lmm_alert_min_hits": 3,
    #: 相邻两次命中判定为"同一目标"的 IoU 下限。0.5 是用户指定值。
    "lmm_alert_iou": 0.5,
    #: ★IoU 之外的第二道门★：中心位移 ≤ 目标对角线的这个倍数也算同一目标。
    #: 0 = 只认 IoU。为什么需要它：VLX 实测 0.23 Hz（4.3 s/次），无人机还在摇摄，
    #: 同一目标相邻两次采样能位移好几个身位，IoU 直接掉到 0 → **永远不出告警**。
    #: 1.5 ≈ "位移不超过一个半身位"，仍是"同一目标"的语义，不是跟踪。
    "lmm_alert_center_ratio": 1.5,
    #: 【已废弃】原"连续命中的帧号差上限"。1-Hit 之后不再参与判定。
    "lmm_alert_max_frame_gap": 90,
    #: 【已废弃】原"连续命中的媒体时间差上限"。1-Hit 之后不再参与判定。
    "lmm_alert_max_time_ms": 8000.0,
    #: 【已废弃】原"候选存活时长"。1-Hit 之后没有候选池，不再参与判定。
    "lmm_alert_expire_ms": 8000.0,
    #: ★同一目标两次告警之间的冷却（毫秒）—— 1-Hit 之后这是**唯一**的防刷屏闸门。
    #: 按「label + 空间位置」判定：位置明显不同的人/船各自告警，只有"同一处
    #: 又出现"才被压住。持续可见的目标不该每 1~2 秒刷一条。
    "lmm_alert_cooldown_ms": 12000.0,
    #: 确认状态保持时长（毫秒）：`Confirmed Target` 指标的显示窗口。
    "lmm_alert_confirm_hold_ms": 20000.0,
    #: 1 = 每条告警抓**一张**证据图存盘（1-Hit：检出帧 = 告警帧 = 证据帧）。
    "lmm_alert_evidence": 1,
    #: 前端红色闪烁光晕时长（毫秒）。2~3 s，只作为默认值下发给前端。
    "lmm_alert_pulse_ms": 2600.0,

    # ── R4A：VLX（低频语义）+ OSTrack（高频位置）双通道 ──────────────────
    #: 总开关。0 = 关闭（行为与 R4A 之前完全一致，只有 VLX）；1 = 启用跟踪通道。
    #: **默认 0**：MX250 上 tracker 实测只有 ~7 FPS（`R4A-OSTrack-本机GPU实测报告.md`），
    #: 属于"低帧率补充通道"而不是逐帧跟踪，必须先小流量验过再开。
    "lmm_tracking_enabled": 0,
    #: 任务级模式：off | ostrack。走 opts 合并（同 lmm_realtime_fast_path）。
    "lmm_tracking_mode": "ostrack",
    #: track_score 低于它算"这一帧没跟上"。**0.15 是 MX250 实测值**：
    #: 响应分数落在 0.167~0.715，设 0.35 会把一半的帧误判成低分。
    "lmm_track_score_threshold": 0.15,
    "lmm_track_uncertain_frames": 3,
    "lmm_track_lost_frames": 10,
    #: 第 5 个目标时整轮已到 1.5 FPS（5 × ~130 ms），再多条就盖过收益了。
    "lmm_track_max_tracks": 5,
    #: VLX 多久没确认就不再信任这条 track。VLX 约 2.3 s 一帧，7 s ≈ 3 轮没确认。
    "lmm_track_max_without_vlx_ms": 7000.0,
    #: 历史帧缓存时长。**必须 > VLX 端到端延迟**（实测 2.3 s），否则 VLX 回来时
    #: 它检测的那帧已经被顶掉，第 11 节就只能丢弃。
    "lmm_track_cache_seconds": 5.0,
    #: 缓存帧的长边。1080p 原始帧 5 s 就是 ~933 MB，必须缩。
    "lmm_track_frame_longest_side": 960,
    #: replay 上限。**8 是实测值**：15 帧要 2615 ms 且期间不出任何框，
    #: 8 帧只要 797 ms。宁可初始位置略旧，也不要 2.6 s 空白。
    "lmm_track_max_replay_frames": 8,
    #: 同 label 关联用：IoU 下限 / 中心距与目标对角线的比值上限。
    #: **两者是"或"的关系**：都设成硬门槛时跟踪器一漂移就两边都卡住，
    #: 同一个目标会被建成第二条 track（画面上出现两个框）。
    "lmm_track_match_iou": 0.25,
    "lmm_track_match_center_ratio": 1.0,
    #: ★周期校正门槛★：tracker 状态正常、且它在该帧的历史位置与 VLX 框的
    #: IoU ≥ 这个值时，VLX 这一帧只当作"确认"，**不** reset + init + replay。
    #: 实测 5 分钟真实链路里 75/81 次"校正"都是这种确认，每次白付 ~983 ms
    #: 并独占远程后端锁，把逐帧 update 挤到 0.84 次/秒（所有 track 掉 LOST）。
    "lmm_track_correction_iou": 0.5,
    #: ★STEP5★ init 请求投出去后最多等多久（毫秒）。超时即判失败 → LOST，
    #: 绝不让 track 卡在 new（`pending_init` 卡死 = 前端静止大框，实测挂过 862 帧）。
    #: 正常 init 只要 ~200 ms，2000 ms 是 10 倍余量。
    "lmm_track_pending_init_timeout_ms": 2000.0,
    #: 漂移守卫（面积相对 VLX anchor 的倍数 / 连续几次判漂移才 LOST）。
    #: **12.0 不是随手放宽**：OSTrack 的 TEMPLATE_FACTOR=2.0 让它正常输出的框
    #: 面积就是 anchor 的 4 倍，门槛 4.0 会把正常输出全判成漂移 —— 实测 5 分钟
    #: 里 87 次拒绝有 85 次来自这道门，结果是所有 track 掉 LOST、逐帧 update
    #: 停摆。真正的失控膨胀（面积 ×650）在 12.0 下依然被拦住。
    "lmm_track_drift_area_grow": 12.0,
    "lmm_track_drift_area_shrink": 12.0,
    "lmm_track_drift_after": 5,
    #: ★STEP3/P1 异步跟踪★（只在 `streamhub` 模式下生效）：
    #: 1 = 每 track 一条 worker 线程（`async_runtime.AsyncTrackerRuntime`）；
    #: 0 = 退回改造前的同步 `TrackerWorker`（A/B 与一键回退用）。
    #: ★必须写进 DEFAULTS★：`_ALLOWED = set(DEFAULTS.keys())` 是配置白名单，
    #: 没在这里登记的键会被 `load()` **静默丢弃** —— 表现为"明明传了
    #: `lmm_track_async=0`，跑起来还是异步"，排查时极难发现是键被丢了。
    "lmm_track_async": 1,
    #: ★STEP3/P2★ 逐帧 update 取哪一帧：
    #:   "legacy" = 客户端指定帧号（`/track/update_at`）—— **默认**，P1 现状；
    #:   "latest" = 只发 track_id，服务端收到时自己取最新帧
    #:              （`/track/update_latest`）→ Age 从**往返**降到**单程**。
    #: ★只影响 update★：init 与 replay 永远走指定帧号（`init_by_ref` /
    #: `update_at`）—— init 必须落在 VLX 那一帧，隔帧初始化会让模板错位。
    #: 旧端点 `/track/update_at` 服务端保留不动，切回 legacy 即完全回到改造前。
    "lmm_track_update_mode": "legacy",
    #: ★远程跟踪（服务隔离）★：`lmm_tracking_mode == "remote"` 时，跟踪计算迁到
    #: A6000 GPU1 上的独立 OSTrack 服务，本机只发 JPEG 收 bbox（经 SSH 隧道）。
    #: node_id 留空 = 自动挑名为 A6000 的节点（与 VLX 同款逻辑）。
    "lmm_ostrack_node_id": "",
    "lmm_ostrack_port": 8022,
    "lmm_ostrack_token": "",
    #: ★服务器侧共享帧（StreamHub）★：`lmm_tracking_mode == "streamhub"` 时，
    #: 帧**不再从本机上传** —— A6000 上的 StreamHub 是唯一取帧者，RunOS 只发
    #: `{track_id, frame_id, media_time_ms}`，由服务端自己从 StreamHub 取帧。
    #: 这条链路的上行是几百字节而不是几十 KB，实测 5 目标 15.46 Hz（旧链路 2.8 Hz）。
    "lmm_streamhub_node_id": "",
    "lmm_streamhub_port": 8030,
    "lmm_streamhub_token": "",
    #: RunOS 轮询 `GET /health` 拿最新 frame_id 的频率。它同时是 tracker 的
    #: 更新频率上限（step 只在出现新帧时才下发 update_at）。
    #: **不要设太高**：本机→A6000 的 SSH 隧道 RTT ≈65 ms，15 Hz 会把链路打满。
    "lmm_streamhub_poll_hz": 10.0,
    #: StreamHub 帧时间线在 RunOS 侧的缓存时长（replay 用，必须 > VLX 端到端延迟）。
    "lmm_streamhub_cache_seconds": 5.0,
    #: VLX `/infer_at` 的间隔（秒）。VLX 单次 ~600 ms，1.5 s ≈ 40% 占用。
    "lmm_streamhub_vlx_interval": 1.5,
    #: StreamHub 不可用时退化到哪个模式（"remote" = 旧 JPEG/crop 模式，"off" = 不跟踪）。
    "lmm_streamhub_fallback_mode": "remote",
    #: 兜底画面尺寸：首次 `/infer_at` 返回前 detection 消息里的 width/height。
    "lmm_streamhub_width": 1280,
    "lmm_streamhub_height": 720,
    #: ★STEP3/4：Qwen2.5-VL-3B-Instruct 服务★（A6000 GPU1，端口 8023）。
    #: 与 OSTrack **互斥占用 GPU1**：实时模式 OSTrack ON / Qwen OFF，
    #: 自动标注模式 Qwen ON / OSTrack OFF（GPU0 的 VLX 始终 ON）。
    #: node_id 留空 = Qwen 服务尚未部署，`vlx-qwen` 模型也不会出现。
    "lmm_qwen_node_id": "",
    "lmm_qwen_port": 8023,
    "lmm_qwen_token": "",
}

#: 允许通过配置文件覆盖的键（白名单，防止写坏结构）
_ALLOWED = set(DEFAULTS.keys())


def _coerce(key: str, value: Any) -> Any:
    """按默认值的类型做一次宽松转换，避免前端传来字符串把数值参数写坏。"""
    ref = DEFAULTS.get(key)
    if ref is None or isinstance(value, type(ref)):
        return value
    try:
        if isinstance(ref, bool):
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(ref, int):
            return int(float(value))
        if isinstance(ref, float):
            return float(value)
        if isinstance(ref, list) and isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
    except (TypeError, ValueError):
        return ref
    return value


def load() -> Dict[str, Any]:
    """读取配置（默认值 + 文件覆盖）。文件损坏时静默回退默认值并保留原文件。"""
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_FILE.is_file():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k, v in (data or {}).items():
                if k in _ALLOWED:
                    cfg[k] = _coerce(k, v)
    except Exception:
        # 配置坏了不能拖垮整个模块，用默认值继续
        pass
    return cfg


def save(patch: Dict[str, Any]) -> Dict[str, Any]:
    """合并写入（原子替换，与 training_manager._persist_jobs 同样的做法）。"""
    with _LOCK:
        cfg = load()
        for k, v in (patch or {}).items():
            if k in _ALLOWED:
                cfg[k] = _coerce(k, v)
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, CONFIG_FILE)
        return cfg


# ── 外部 LLM 配置（供 PromptAnalyzer 做关键词拆解）已于 2026-09-23 连同解析器一起删除 ──
# 这里刻意**不保留 `llm_settings()`**：留着它等于留一个"还能偷偷接别的模型"的口子，
# 而现在的链路要求用户输入逐字透传到 VLX-Seek，中间不允许再过任何模型。
