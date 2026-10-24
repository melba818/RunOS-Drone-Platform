"""Shared training architecture catalogue and bounded parameters."""
import json
from pathlib import Path
from fastapi import HTTPException

ARCHITECTURES=json.loads(Path(__file__).with_name('training-architectures.json').read_text(encoding='utf-8'))

def validation_error(field, message, code='INVALID_VALUE'):
    raise HTTPException(422, {'code':'VALIDATION_FAILED','message':'参数校验未通过。','errors':[{'field':field,'code':code,'message':message}]})

def model_type(arch):
    if arch.startswith('rtdetrv2'):return 'rtdetrv2'
    if arch.startswith('rtdetr'):return 'rtdetr'
    if arch.startswith('rfdetr'):return 'rfdetr'
    if '-seg' in arch:return 'segmentation'
    if '-cls' in arch:return 'classification'
    return 'yolo'

def validate_architecture(config):
    selected=next((i for i in ARCHITECTURES if i['value']==config.model_arch),None)
    if selected is None:validation_error('model_arch','请选择平台支持的训练架构。')
    if model_type(config.model_arch)!=config.model_type:validation_error('model_type','模型类型与训练架构不匹配。')
    if config.initial_weights=='custom':
        # Read metadata without unpickling model code. Unknown files remain selectable in the
        # native form but require explicit task metadata before an assistant can recommend them.
        from config import settings
        name=config.custom_weights or ''
        info=settings.model_manager.loaded_models.get(Path(name).stem,{})
        task=info.get('task') or info.get('task_type')
        if task and task!=selected['taskType']:validation_error('custom_weights','权重任务类型与训练任务不匹配。')
        if '-seg' in name and selected['taskType']!='segmentation':validation_error('custom_weights','分割权重不能用于当前任务。')
        if '-cls' in name and selected['taskType']!='classification':validation_error('custom_weights','分类权重不能用于当前任务。')
