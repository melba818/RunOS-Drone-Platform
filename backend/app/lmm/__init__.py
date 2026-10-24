"""LMM — 多模态目标检测（自然语言 → 开放词汇检测 → 跟踪 → 规则 → 告警）。

本包是对 RunOS 现有推理链路的**增量扩展**，不改动 `routers/inference.py`、
`model_integration.py`、`camera/*` 与训练链路。

管道（2026-09-23 起）：
    用户原文（自然语言，逐字透传）
             → VLX-Seek（云端 10B，WeDetect proposals + reasoning_detection）
             → RuleEngine（semantic_final：有框即定案）→ DetectionEvent → 前端
    本机 YOLOE 仅作为 `vlx_seek_enabled=false` 时的回退，同样吃整句 Prompt。

**不再有 Prompt 解析环节**：`prompt_analyzer.py`（本地关键词词表 + 外部 LLM）已删除。
把「寻找水面上未穿救生衣的人员」拆成 `person / swimmer / water` 会丢掉这些词之间的
语义关系，而这正是 VLX-Seek 能直接理解的部分。

日志前缀约定（沿用项目既有风格）：
    [LMM]     任务级事件
    [VLX]     VLX-Seek 调用（task / lang / raw query 每次任务开跑时打印一次）
    [YOLOE]   模型加载 / 推理（仅回退路径）
    [TRACKER] 跟踪（仅回退路径）
    [RULE]    规则与告警
"""

__all__ = ["LOG_PREFIX"]

#: 该模块统一日志前缀，避免和其它模块的输出混在一起
LOG_PREFIX = "[LMM]"
