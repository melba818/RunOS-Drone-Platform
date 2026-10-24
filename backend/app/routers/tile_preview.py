"""Cancellable, read-only single-image Tile Pass jobs (per-account API process)."""
import copy
import threading
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from config import settings
from tile_pass import TilePassCancelled, validate_options

router = APIRouter()
_jobs = {}
_controls = {}
_lock = threading.Lock()

def start_preview(image_file, classes, *, dataset_id, image_id, model_id,
                  verify_model_id='', text_prompt='', label='', confidence=.5,
                  tile_size=640, tile_overlap=128, tile_batch_size=64):
    validate_options(tile_size, tile_overlap, tile_batch_size)
    if verify_model_id == model_id:
        raise ValueError('标注模型和校验模型必须不同')
    with _lock:
        now = time.time()
        for key, job in list(_jobs.items()):
            if job['status'] not in ('running','paused') and now-job['created'] > 7200:
                _jobs.pop(key, None)
                _controls.pop(key, None)
        if sum(j['status'] in ('running','paused') for j in _jobs.values()) >= 4:
            raise HTTPException(status_code=429, detail='已有四个预览任务，请等待或取消后再试')
        job_id = uuid.uuid4().hex
        pause = threading.Event(); pause.set()
        control = dict(pause=pause, stop=threading.Event())
        _controls[job_id] = control
        _jobs[job_id] = dict(job_id=job_id, dataset_id=dataset_id, image_id=image_id,
                            status='running', created=now, phase='full', progress={})
    def update(state):
        with _lock:
            _jobs[job_id]['phase'] = state.get('phase', 'tile')
            _jobs[job_id]['progress'] = state
    def run():
        from annotation_chain import run_chain_dual
        try:
            annotations, debug = run_chain_dual(Path(image_file), text_prompt, classes,
                model_a=model_id, model_b=verify_model_id, label=label,
                confidence_threshold=confidence, tile_pass=True,
                tile_size=tile_size, tile_overlap=tile_overlap, tile_batch_size=tile_batch_size,
                control=control, progress=update)
            from routers.annotations import _dual_summary
            from human_review import register_preview
            with _lock:
                if control['stop'].is_set():
                    _jobs[job_id]['status'] = 'cancelled'
                else:
                    review = register_preview(job_id, dataset_id, image_file, annotations, model_id=model_id, verify_model_id=verify_model_id)
                    _jobs[job_id].update(status='done', result=dict(success=True, review_job_id=job_id, review_summary=review["summary"],
                        annotations=[a for a in annotations if a['decision']=='ACCEPT'],
                        review_annotations=[a for a in annotations if a['decision']=='REVIEW'],
                        dual=_dual_summary(debug), tile_pass=debug.get('tile_pass', {})))
        except TilePassCancelled:
            with _lock: _jobs[job_id]['status'] = 'cancelled'
        except Exception as exc:
            with _lock: _jobs[job_id].update(status='error', error=str(exc))
    threading.Thread(target=run, name='tile-preview-'+job_id[:8], daemon=True).start()
    return dict(job_id=job_id, status='running', preview=True)

@router.get('/api/auto-annotate/tile-preview/{job_id}/status')
def status(job_id: str):
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(404, '预览任务不存在或已过期')
        return copy.deepcopy(_jobs[job_id])

@router.post('/api/auto-annotate/tile-preview/{job_id}/{action}')
def control_job(job_id: str, action: str):
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(404, '预览任务不存在或已过期')
        if action not in ('pause','resume','cancel'):
            raise HTTPException(422, '不支持的任务操作')
        job, control = _jobs[job_id], _controls[job_id]
        if job['status'] not in ('running','paused'):
            return dict(status=job['status'])
        if action == 'pause':
            control['pause'].clear(); job['status']='paused'
        elif action == 'resume':
            control['pause'].set(); job['status']='running'
        else:
            control['stop'].set(); control['pause'].set(); job['status']='cancelled'
        return dict(status=job['status'])
