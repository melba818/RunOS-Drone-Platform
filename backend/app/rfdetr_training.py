"""RF-DETR 1.9 training bridge using its native Lightning lifecycle."""
from pathlib import Path
import time
import math


def augmentation_config(config):
    """Map shared UI controls to RF-DETR's supported Albumentations interface."""
    def bounded(name, default, maximum=1):
        value = float(config.get(name, default))
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f'{name} 必须在 0 到 {maximum} 之间。')
        return value
    return {
        'HorizontalFlip': {'p': bounded('fliplr', .5)},
        'VerticalFlip': {'p': bounded('flipud', 0)},
        'ColorJitter': {
            'hue': bounded('hsv_h', .015, .5),
            'saturation': bounded('hsv_s', .5),
            'brightness': bounded('hsv_v', .3),
            'contrast': 0, 'p': 1.0,
        },
    }


def train_job(manager, job):
    from rfdetr import RFDETR, RFDETRBase, RFDETRMedium, RFDETRLarge
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
    from pytorch_lightning import Callback
    from training import _resolve_device

    config = job['config']; root = Path(job['dataset_path'])
    output = root / 'runs' / f"rfdetr_{job['id']}"
    output.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_device(config, job)
    device = f'cuda:{resolved}' if isinstance(resolved, int) or str(resolved).isdigit() else str(resolved)
    architecture = config.get('base_model', 'rfdetr_medium')
    if config.get('initial_weights') == 'custom':
        model = RFDETR.from_checkpoint(architecture, device=device, amp=config.get('amp', True))
    else:
        model_class = {'rfdetr_base': RFDETRBase, 'rfdetr_medium': RFDETRMedium, 'rfdetr_large': RFDETRLarge}[architecture]
        kwargs = {} if config.get('pretrained', True) else {'pretrain_weights': None}
        model = model_class(device=device, amp=config.get('amp', True), **kwargs)
    mc = model.model_config
    size = int(config.get('image_size', mc.resolution))
    stride = mc.patch_size * mc.num_windows
    if size <= 0 or size % stride:
        raise ValueError(f'RF-DETR 输入尺寸必须是 {stride} 的倍数，当前为 {size}。')
    mc.resolution = size
    dataset_file = 'yolo' if (root / 'data.yaml').is_file() else 'roboflow'
    tc = model.get_train_config(
        aug_config=augmentation_config(config), augmentation_backend='albumentations',
        dataset_dir=str(root), dataset_file=dataset_file, output_dir=str(output),
        epochs=config.get('epochs', 80), batch_size=config.get('batch_size', 1),
        grad_accum_steps=max(1,16//max(1,config.get('batch_size',1))),
        lr=config.get('lr0',.0001), lr_encoder=config.get('lr0',.0001),
        optimizer=(config.get('optimizer','AdamW').lower() if config.get('optimizer') != 'auto' else 'adamw'),
        lr_scheduler='cosine' if config.get('cos_lr') else 'step', lr_min_factor=config.get('lrf',.01),
        warmup_epochs=config.get('warmup_epochs',3), weight_decay=config.get('weight_decay',.0005),
        early_stopping=config.get('patience',0)>0, early_stopping_patience=max(1,config.get('patience',20)),
        checkpoint_interval=config.get('save_period') or config.get('epochs',80),
        num_workers=0, multi_scale=False, expanded_scales=False, tensorboard=False,
        wandb=False, run_test=False, progress_bar=None)
    # RF-DETR's coarse format detector misses VisOS images/train layouts.
    # Its actual YOLO loader supports them; read their declared classes directly.
    if dataset_file == 'yolo':
        import yaml
        dataset_yaml = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8-sig'))
        names = dataset_yaml.get('names')
        if isinstance(names, dict):
            if sorted(int(k) for k in names) != list(range(len(names))):
                raise ValueError('RF-DETR 数据集类别 ID 必须从 0 开始连续编号。')
        if not isinstance(names, (list, dict)) or not names:
            raise ValueError('RF-DETR 数据集 data.yaml 缺少有效 names。')
        mc.num_classes = len(names)
    else:
        model._align_num_classes_from_dataset(str(root))
    module = RFDETRModelModule(mc, tc)
    data = RFDETRDataModule(mc, tc)
    job['total_epochs'] = tc.epochs
    job['logs'].extend([f'RF-DETR: {type(model).__name__}; initial weights: {architecture}',
                        f'Dataset: {root}; format: {dataset_file}; device: {device}; imgsz: {size}',
                        f'RF-DETR augmentation: {tc.aug_config}; Mosaic/Dropout 不适用。'])

    class Monitor(Callback):
        last_update = 0
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if job.get('_stop_requested'): trainer.should_stop = True
            if time.monotonic()-self.last_update > 3:
                self.last_update=time.monotonic()
                job['logs'].append(f"RF-DETR epoch {trainer.current_epoch+1}/{tc.epochs}, batch {batch_idx+1}/{trainer.num_training_batches}")
        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:return
            metrics=trainer.callback_metrics
            def value(key):
                v=metrics.get(key)
                return float(v.detach().cpu()) if v is not None else None
            row={'epoch':trainer.current_epoch+1,'train_box_loss':value('train/loss_bbox'),
                 'train_cls_loss':value('train/loss_ce'),'train_dfl_loss':None,
                 'val_box_loss':value('val/loss_bbox'),'val_cls_loss':value('val/loss_ce'),'val_dfl_loss':None,
                 'mAP50':value('val/mAP_50'),'mAP50_95':value('val/mAP_50_95'),
                 'train_loss':value('train/loss'),'val_loss':value('val/loss')}
            job['epoch_history'].append(row);job['metrics']=row
            job['current_epoch']=row['epoch'];job['progress']=100*row['epoch']/tc.epochs
            job['logs'].append(f"RF-DETR epoch {row['epoch']}: loss={row['train_loss']}, val={row['val_loss']}, mAP50={row['mAP50']}")
            manager._persist_jobs()

        def on_train_epoch_end(self, trainer, module):
            if not job['epoch_history']: return
            row = job['epoch_history'][-1]
            for field, key in [('train_box_loss', 'train/loss_bbox'), ('train_cls_loss', 'train/loss_ce'), ('train_loss', 'train/loss')]:
                value = trainer.callback_metrics.get(key)
                if value is not None: row[field] = float(value.detach().cpu())
            job['metrics'] = row
            manager._persist_jobs()

    accelerator = 'gpu' if device.startswith('cuda') else device
    kwargs = {'accelerator':accelerator,'devices':[int(device.split(':')[-1])] if accelerator=='gpu' else 1}
    trainer = build_trainer(tc, mc, **kwargs)
    trainer.callbacks.append(Monitor())
    job['_trainer']=trainer;job['status']='running'
    try:
        trainer.fit(module, data)
        best = output/'checkpoint_best_total.pth'
        if not best.exists():
            best=next(iter(sorted(output.rglob('*.pth'),key=lambda p:p.stat().st_mtime,reverse=True)),None)
        if best is None:raise RuntimeError('RF-DETR 未生成模型文件，请检查训练日志。')
        job['model_path']=str(best)
        last=output/'checkpoint.pth'
        job['last_checkpoint']=str(last) if last.exists() else str(best)
        job['status']='stopped' if job.get('_stop_requested') else 'completed'
        if job['status']=='completed':job['progress']=100
    finally:
        job['_trainer']=None
