"""跟踪器配置解析。

复用项目已有的派生 YAML 逻辑（`routers/inference._resolve_tracker_yaml`），
不另起一套：那份实现已经处理了「workspace/temp 会被启动清理器删除 → 每次调用都要
mkdir」和「内容哈希缓存」两个坑。只读地借用，不修改原模块。

**为什么必须按检测置信度派生阈值（关键）**

ultralytics 内置 `botsort.yaml` 的 `track_high_thresh` 是 0.25、`bytetrack.yaml` 是 0.5。
而 LMM 的开环词表检测为追求高召回，`confidence` 默认 0.2、实测常压到 0.15。于是
**送进跟踪器的框绝大多数落在 `track_low_thresh`~`track_high_thresh` 之间，被当作
"低分框"**：BYTETracker/BoT-SORT 的低分框只参与对 `lost`/`unconfirmed` 轨迹的二次
关联，**不会激活新轨迹**；而未激活的轨迹不会被回填到 `boxes.id`。

实测后果（本次验收）：
- 同一段视频里，conf≈0.73 的船拿到了 `track_id=3` 并成功走完 VLM 复核；
- conf≈0.15~0.30 的人/小船全部 `track_id=None` → 既无法按 track 缓存，也无法归属
  VLM 结论 → VLM 调用数为 0、告警只能靠"无 id 兜底"产出（已在下游修掉）。

因此这里把 `track_high_thresh` / `new_track_thresh` 派生到略低于 `confidence`，
让**跟踪器的工作点与检测器保持一致**；`track_low_thresh` 再按比例下压，保持两级关联
的层次。不传 `confidence` 时行为与改造前完全一致（用内置 YAML）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

#: 允许使用的跟踪器。ultralytics 内置，无 strongsort（项目既有结论）。
TRACKERS: Dict[str, str] = {
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "deepocsort": "Deep OC-SORT",
    "fasttrack": "FastTrack",
    "tracktrack": "TrackTrack",
}

DEFAULT_TRACKER = "botsort"


def normalize(name: Any) -> str:
    s = str(name or "").strip().lower()
    if s.endswith(".yaml"):
        s = s[: -len(".yaml")]
    return s if s in TRACKERS else DEFAULT_TRACKER


#: 允许的跟踪器阈值下限：太低会让跟踪器把噪声当轨迹，反而增加 ID 数量（=更多 VLM 调用）
_MIN_THRESH = 0.02


def tracker_tunables(confidence: Any) -> Optional[Dict[str, Any]]:
    """按检测置信度推导跟踪器阈值。无法解析时返回 None（用内置 YAML）。

    - `track_high_thresh` / `new_track_thresh` 取 `confidence` 的 0.95（略低，避开
      `scores > thresh` 的严格比较边界），保证检测器给出的候选都能进入一级关联。
    - `track_low_thresh` 取 high 的 0.6，保留"低分框做二次关联"的原有层次。
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if not (0.0 < conf <= 1.0):
        return None
    high = max(_MIN_THRESH, round(conf * 0.95, 4))
    low = max(_MIN_THRESH, round(high * 0.6, 4))
    if low >= high:
        low = round(max(_MIN_THRESH, high * 0.5), 4)
    return {
        "track_high_thresh": high,
        "track_low_thresh": low,
        "new_track_thresh": high,
    }


def resolve_tracker_yaml(tracker: Any, confidence: Any = None) -> str:
    """返回可交给 `model.track(tracker=...)` 的 YAML 名或路径。

    这里不需要 ReID：外观匹配交给二阶段的 VLM Verifier，跟踪器只做运动关联。
    但阈值必须与检测置信度对齐，否则拿不到 track_id（见模块 docstring）。
    """
    name = normalize(tracker)
    tunables = tracker_tunables(confidence)
    if tunables:
        LOG.debug("[TRACKER] 按 confidence=%s 派生阈值：%s", confidence, tunables)
    try:
        from routers.inference import _resolve_tracker_yaml as _impl
        return _impl(name, False, None, tunables)
    except Exception:
        # 借用失败不能让检测跑不起来：退回 ultralytics 内置 YAML 名
        LOG.debug("[TRACKER] 派生 YAML 不可用，回退内置 %s.yaml", name, exc_info=True)
        return f"{name}.yaml"
