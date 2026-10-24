from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Dict, Any, Literal


class DatasetInfo(BaseModel):
    id: str
    name: str
    path: str
    format: str
    task_type: str
    num_images: int
    num_annotations: int
    classes: List[str]
    created_at: str
    splits: Optional[Dict[str, int]] = None


class AnnotationUpdate(BaseModel):
    image_id: str
    annotations: List[Dict[str, Any]]


class ConversionRequest(BaseModel):
    dataset_id: str
    target_format: str
    output_name: Optional[str] = None


class MergeRequest(BaseModel):
    dataset_ids: List[str]
    output_name: str
    output_format: str
    class_mapping: Optional[Dict[str, str]] = None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    compute_node_id: Optional[str] = None
    dataset_id: str = Field(min_length=1,max_length=240)
    name: str = Field(default="",max_length=160)
    model_type: str
    model_arch: str = "yolov8n"
    initial_weights: Literal["builtin", "custom"] = "builtin"
    custom_weights: Optional[str] = None
    epochs: int = Field(default=100,ge=1,le=10000)
    batch_size: int = Field(default=16,ge=1,le=1024)
    image_size: int = Field(default=640,ge=128,le=4096)
    pretrained: bool = True
    device: str = "auto"
    save_period: int = Field(default=0,ge=0,le=10000)  # save checkpoint every N epochs (0 = disabled)
    lr0: float = Field(default=.01,gt=0,le=1)
    lrf: float = Field(default=.01,ge=0,le=1)
    optimizer: Literal["SGD","Adam","AdamW","NAdam","RAdam","RMSProp","auto"] = "SGD"
    patience: int = Field(default=50,ge=0,le=10000)
    cos_lr: bool = False
    warmup_epochs: float = Field(default=3,ge=0,le=10000)
    weight_decay: float = Field(default=.0005,ge=0,le=1)
    mosaic: float = Field(default=1.0,ge=0,le=1)
    hsv_h: float = Field(default=0.015,ge=0,le=1)
    hsv_s: float = Field(default=0.7,ge=0,le=1)
    hsv_v: float = Field(default=0.4,ge=0,le=1)
    flipud: float = Field(default=0.0,ge=0,le=1)
    fliplr: float = Field(default=0.5,ge=0,le=1)
    amp: bool = True
    dropout: float = Field(default=0.0,ge=0,le=1)


class SettingsConfig(BaseModel):
    models_path: Optional[str] = None
    datasets_path: Optional[str] = None
    output_path: Optional[str] = None
    use_gpu: bool = True
    gpu_device: str = "0"


class ClassAddRequest(BaseModel):
    dataset_id: str
    new_classes: List[str]
    use_model: bool = False
    model_id: Optional[str] = None


class ClassExtractRequest(BaseModel):
    dataset_id: str
    classes_to_extract: List[str]
    output_name: str
    output_format: Optional[str] = None


class ClassDeleteRequest(BaseModel):
    dataset_id: str
    classes_to_delete: List[str]


class ClassMergeRequest(BaseModel):
    dataset_id: str
    source_classes: List[str]
    target_class: str


class ClassRenameRequest(BaseModel):
    dataset_id: str
    old_name: str
    new_name: str


class SplitRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    dataset_id: Optional[str] = None
    train_ratio: float = Field(default=0.7,ge=0,le=1)
    val_ratio: float = Field(default=0.2,ge=0,le=1)
    test_ratio: float = Field(default=0.1,ge=0,le=1)
    output_name: Optional[str] = Field(default=None,min_length=1,max_length=160,pattern=r'^[^/\\\x00-\x1f]+$')
    shuffle: bool = True
    seed: Optional[int] = Field(default=None,ge=0,le=2147483647)
    stratify: bool = False

    @model_validator(mode='after')
    def valid_split(self):
        if abs(self.train_ratio+self.val_ratio+self.test_ratio-1)>1e-6:
            raise ValueError('划分比例合计必须为 100%。')
        if self.stratify:
            raise ValueError('当前划分接口不支持类别分层，请使用随机划分。')
        return self


class AugmentationConfig(BaseModel):
    dataset_id: str
    output_name: str
    target_size: int
    augmentations: Dict[str, Dict[str, Any]]


class SortingAction(BaseModel):
    image_id: str
    action: str


class AnnotationHistoryEntry(BaseModel):
    timestamp: str
    action: str
    annotation_data: Dict[str, Any]


class LocalFolderRequest(BaseModel):
    folder_path: str
    dataset_name: Optional[str] = None
    format_hint: Optional[str] = None


class LocalPathRequest(BaseModel):
    path: str
    dataset_name: Optional[str] = None
    format_hint: Optional[str] = None


class VideoExtractRequest(BaseModel):
    video_id: Optional[str] = None
    video_path: Optional[str] = None
    output_name: str
    mode: str = "interval"
    nth_frame: int = 30
    frame_interval: Optional[int] = None
    uniform_count: Optional[int] = 100
    manual_frames: Optional[List[int]] = None
    max_frames: Optional[int] = None
    start_time: float = 0
    end_time: Optional[float] = None
    existing_dataset_id: Optional[str] = None


class DuplicateDetectionRequest(BaseModel):
    method: str = "perceptual"
    threshold: int = 10
    include_near_duplicates: bool = True


class ClipRegroupRequest(BaseModel):
    threshold: int = 90


class RemoveDuplicatesRequest(BaseModel):
    groups: List[List[Dict[str, Any]]]
    keep_strategy: str = "first"


class SimplePreviewRequest(BaseModel):
    dataset_id: str
    config: Dict[str, Any]
    num_previews: int = 6


class SimpleAugmentRequest(BaseModel):
    dataset_id: str
    config: Dict[str, Any]
    augment_factor: Optional[float] = None
    target_size: Optional[int] = None
    output_name: Optional[str] = None
    class_targets: Optional[Dict[str, int]] = None


class EnhancedAugmentationRequest(BaseModel):
    output_name: str
    target_size: int
    target_multiplier: Optional[float] = None
    augmentations: Dict[str, Dict[str, Any]]
    preserve_originals: bool = True


class EnhancedSplitRequest(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    output_name: Optional[str] = None
    shuffle: bool = True
    seed: Optional[int] = None
    stratified: bool = False
    by_folder: bool = False


class YamlWizardConfig(BaseModel):
    class_names: List[str]
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    image_ids: List[str]


class BatchSplitRequest(BaseModel):
    image_ids: List[str]
    split: str


class SnapshotRequest(BaseModel):
    name: str
    description: str = ""


class DatasetRenameRequest(BaseModel):
    new_name: str
