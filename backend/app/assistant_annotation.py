"""Load catalog models and append accepted boxes without replacing formal labels."""
import asyncio,base64
from pathlib import Path
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from config import settings

_load_lock=asyncio.Lock()

async def ensure_models(data):
    async with _load_lock:
        for ident in [data['model_id'],data.get('verify_model_id')]:
            if not ident:continue
            info=settings.model_manager.loaded_models.get(ident)
            from vlm_annotator import load_providers
            provider=load_providers().get(ident)
            if info and provider and info.get('model')!=provider:
                await run_in_threadpool(settings.model_manager._try_autoload,ident)
                info=settings.model_manager.loaded_models.get(ident)
            if not info or not info.get('model') or info.get('error'):
                await run_in_threadpool(settings.model_manager._try_autoload,ident)
                info=settings.model_manager.loaded_models.get(ident)
            if not info or not info.get('model') or info.get('error'):
                raise HTTPException(422,{'code':'MODEL_UNAVAILABLE','message':'所选标注模型未能加载。','errors':[{'field':'model_id' if ident==data['model_id'] else 'verify_model_id','code':'MODEL_UNAVAILABLE','message':'请检查模型配置与服务连接。'}]})

def save_accepted(dataset_path,dataset,image,annotations,label):
    import human_review
    from core.annotation_lock import annotation_lock
    root=Path(dataset_path).resolve();image=Path(image).resolve()
    if not image.is_relative_to(root):raise ValueError('标注图片路径无效。')
    saved=0;before={}
    with annotation_lock(root):
        try:
            for ann in annotations:
                if ann.get('class_name')!=label or not ann.get('bbox') or ann.get('decision','ACCEPT')!='ACCEPT':continue
                changes,duplicate=human_review._changes(root,dataset,image,ann,'assistant-accepted')
                if duplicate:continue
                for change in changes:
                    target=human_review._target(root,change['path'])
                    if target not in before:before[target]=(target.exists(),target.read_bytes() if target.exists() else b'')
                    human_review.atomic_bytes(target,base64.b64decode(change['after']))
                saved+=1
        except Exception:
            for path,(exists,content) in before.items():
                if exists:human_review.atomic_bytes(path,content)
                elif path.exists():path.unlink()
            raise
    return saved
