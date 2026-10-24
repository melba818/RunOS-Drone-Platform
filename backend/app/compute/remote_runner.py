"""Uploaded to one exclusive RunOS job directory; reuses existing training engines."""
import json
import os
from pathlib import Path
import sys
import threading
import time
import zipfile


def main():
    root=Path(sys.argv[1]).resolve()
    # Atomic directory creation prevents double launch after a lost SSH response.
    try: (root/'runner.lock').mkdir()
    except FileExistsError: return
    (root/'runner.pid').write_text(str(os.getpid()),encoding='ascii')
    from training import TrainingManager
    manager=TrainingManager()
    job=json.loads((root/'job.json').read_text())
    job.update(status='starting',logs=['远程训练进程已启动'],epoch_history=[],metrics={},current_epoch=0,progress=0)
    manager.training_jobs[job['id']]=job
    finished=threading.Event()
    def publish():
        status=manager.get_status(job['id'])
        status['logs']=list(job.get('logs',[]))
        status.update(heartbeat=time.time(),pid=os.getpid(),last_checkpoint=job.get('last_checkpoint'))
        temp=root/'state.json.tmp'
        temp.write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8')
        os.replace(temp,root/'state.json')
    def watch():
        while not finished.wait(2):
            if (root/'STOP').exists():
                job['_stop_requested']=True
                trainer=job.get('_trainer')
                if trainer is not None: trainer.stop=True
            try: publish()
            except Exception: pass
    publish()
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:
        import torch
        # Server environments are inspected, never automatically installed.
        if job['model_type']=='rfdetr':
            from rfdetr.training import RFDETRDataModule,RFDETRModelModule,build_trainer
            import albumentations
        else:
            import ultralytics
        device=job['config']['device']
        if device!='cpu':
            if not torch.cuda.is_available(): raise RuntimeError('远程 PyTorch CUDA 不可用，未回退 CPU')
            index=int(device)
            if index>=torch.cuda.device_count(): raise RuntimeError('所选 GPU 不可用')
            job['device_info']=torch.cuda.get_device_name(index)
        dataset=root/'dataset';dataset.mkdir(exist_ok=True)
        with zipfile.ZipFile(root/'dataset.zip') as archive:
            for member in archive.infolist():
                target=(dataset/member.filename).resolve()
                if not target.is_relative_to(dataset): raise ValueError('数据快照包含非法路径')
            archive.extractall(dataset)
        if (root/'STOP').exists(): job['status']='stopped'
        else:
            manager._run_training(job['id'])
            if job.get('_stop_requested') and job.get('status')=='completed': job['status']='stopped'
    except Exception as exc:
        job.update(status='failed',error=str(exc))
        job['logs'].append('远程训练失败：'+str(exc))
    finally:
        finished.set();watcher.join(timeout=5);publish()


if __name__=='__main__': main()
