"""
Global configuration: path constants, singleton service instances, and shared mutable state.

All mutable dicts/lists are defined here so every router accesses the same objects.
Routers import this module as:   from config import settings
then reference globals as:       settings.active_datasets, settings.DATASETS_DIR, etc.
This ensures that reassignments (e.g. settings.DATASETS_DIR = new_path) are visible
across all modules without needing to re-import.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
# Anchored to this file so paths work regardless of the process working directory.
_DEFAULT_WORKSPACE_DIR = Path(__file__).parent.parent / "workspace"
WORKSPACE_DIR = Path(os.environ.get("RUNOS_DATA_ROOT", str(_DEFAULT_WORKSPACE_DIR))).expanduser().resolve()
# Shared platform configuration lives with the owner workspace; datasets and exports remain per-account.
PLATFORM_CONFIG_ADMIN_ID = os.environ.get("RUNOS_PLATFORM_CONFIG_ADMIN_ID", "1").strip() or "1"
_SHARED_CONFIG_ROOT = os.environ.get("RUNOS_SHARED_CONFIG_ROOT")
if _SHARED_CONFIG_ROOT:
    PLATFORM_CONFIG_DIR = Path(_SHARED_CONFIG_ROOT).expanduser().resolve()
elif os.environ.get("RUNOS_DATA_ROOT") and WORKSPACE_DIR.name.startswith("admin-"):
    PLATFORM_CONFIG_DIR = (WORKSPACE_DIR.parent / f"admin-{PLATFORM_CONFIG_ADMIN_ID}").resolve()
else:
    PLATFORM_CONFIG_DIR = WORKSPACE_DIR
PLATFORM_MODELS_DIR = PLATFORM_CONFIG_DIR / "models"
PLATFORM_COMPUTE_NODES_DIR = PLATFORM_CONFIG_DIR / "compute_nodes"
DATASETS_DIR  = WORKSPACE_DIR / "datasets"
MODELS_DIR    = PLATFORM_MODELS_DIR
EXPORTS_DIR   = WORKSPACE_DIR / "exports"
SNAPSHOTS_DIR = WORKSPACE_DIR / "snapshots"
TEMP_DIR      = WORKSPACE_DIR / "temp"
VIDEOS_DIR    = WORKSPACE_DIR / "videos"
JOBS_FILE     = WORKSPACE_DIR / "batch_jobs.json"
#: ★告警证据图★：每条告警的连续三帧截图。刻意**不放 TEMP_DIR** —— 启动清理器
#: 会清空 temp，告警证据是"事后可追溯"的产物，必须长期留存。
ALERTS_DIR    = WORKSPACE_DIR / "alerts"

METADATA_FILENAME = "dataset_metadata.json"

# Create directories on import
for _d in [WORKSPACE_DIR, PLATFORM_CONFIG_DIR, PLATFORM_MODELS_DIR, PLATFORM_COMPUTE_NODES_DIR, DATASETS_DIR, MODELS_DIR, EXPORTS_DIR, SNAPSHOTS_DIR, TEMP_DIR, VIDEOS_DIR, ALERTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Service singletons ────────────────────────────────────────────────────────
from dataset_parsers import DatasetParser
from format_converter import FormatConverter
from annotation_tools import AnnotationManager
from model_integration import ModelManager
from training import TrainingManager
from dataset_merger import DatasetMerger
from video_utils import VideoFrameExtractor, DuplicateDetector, CLIPEmbeddingManager
from augmentation import DatasetAugmenter

dataset_parser    = DatasetParser()
format_converter  = FormatConverter()
annotation_manager = AnnotationManager()
model_manager     = ModelManager(MODELS_DIR)
training_manager  = TrainingManager(jobs_file=WORKSPACE_DIR / "training_jobs.json")
dataset_merger    = DatasetMerger()
video_extractor   = VideoFrameExtractor()
duplicate_detector = DuplicateDetector()
clip_manager      = CLIPEmbeddingManager()
augmenter         = DatasetAugmenter()

# ── Shared mutable state ──────────────────────────────────────────────────────
active_datasets: Dict[str, Any] = {}
sorting_sessions: Dict[str, Dict] = {}
annotation_history: Dict[str, List[Dict]] = {}
_images_cache: Dict[str, List] = {}
_download_status: Dict[str, Dict] = {}
_uploaded_videos: Dict[str, Dict[str, Any]] = {}

_gpu_status: dict = {
    "state": "unknown",
    "message": "",
    "gpu_name": "",
}

_text_annotate_jobs: Dict[str, Dict] = {}
_job_controls: Dict[str, Dict] = {}
_extraction_jobs: Dict[str, Dict] = {}
_augmentation_jobs: Dict[str, Dict] = {}

_app_settings: dict = {
    "models_path":   str(MODELS_DIR),
    "datasets_path": str(DATASETS_DIR),
    "output_path":   str(EXPORTS_DIR),
    "use_gpu":       True,
    "gpu_device":    "0",
}

# ── model_runtime（P0/P1 新增：仅契约层，未接入任何推理链路）─────────────────
# 新架构总开关。为 0 时 Model/Deployment/Resolver
# 一律不得进入 _run_inference_unlocked、VLX 视频、批量标注、Qwen 调用。
# P2B 受控在线验证通过（新链路成功 / legacy 未强制切换 / 配置与 data.yaml 未污染），正式置 1。
# 生效范围极窄：仅 model_id=vlx-seek 且 target.label 非空时走 model_runtime_v1，其余全部保持旧链路。
FEATURE_MODEL_RUNTIME_V1 = 1

# 旧 VLX 链路是"目标描述即类别"。False = 用 DetectionTarget.label 写 data.yaml（新行为）；
# 回放只带 text_prompt 的历史 job 时由调用方按 True 处理。
LEGACY_LABEL_FROM_DESCRIPTION = False

# Deployment **静态配置**文件。注意：P0/P1 阶段该文件不应存在，
# DeploymentStore.load() 在文件缺失时必须返回空集合，不得自动创建。
DEPLOYMENTS_FILE = PLATFORM_MODELS_DIR / "deployments.json"

# 只读自查/调试接口开关（与上面的功能开关分离，避免 FEATURE 必须为 0 时被牵连）。
# 默认 0：不注册任何 debug endpoint。
MODEL_RUNTIME_DEBUG = 0

# ── P2A：DetectionTarget 接入标注请求 ────────────────────────────────────────
# 0（当前默认）：请求侧已完成 label/description/extra_prompt → DetectionTarget 标准化，
#   但**执行侧仍用 text_prompt**，写盘语义与 P2A 之前逐字节一致（不产生半迁移状态）。
# 1（P2B 起）：执行侧改用 DetectionTarget.prompt_text()，类别改用 DetectionTarget.label。
TARGET_STRUCTURE_EXECUTION = 0
