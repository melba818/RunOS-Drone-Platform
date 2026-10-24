"""SSH training transport; model training stays in the existing training engines."""
import json
import os
from pathlib import Path
import posixpath
import shlex
import tempfile
import threading
import time
import uuid
import paramiko
from .training_bundle import bundle_dataset


def environment_command(node, python_args):
    q=shlex.quote
    if node['environment_type']=='conda':
        return 'bash -lc '+q(q(node.get('conda_executable') or 'conda')+' run --no-capture-output -n '+q(node['conda_env'])+' python '+python_args)
    python=node['venv_path'].rstrip('/')+'/bin/python' if node['environment_type']=='venv' else 'python3'
    return q(python)+' '+python_args


def read_json(ftp,path):
    with ftp.open(path,'r') as f:
        data=f.read(8*1024*1024+1)
        if len(data)>8*1024*1024: raise ValueError('远程状态文件过大')
        return json.loads(data)


def retry_operation(operation, job=None):
    """Each attempt owns a new client; never reuse a failed transport."""
    for attempt in range(2):
        try: return operation()
        except (paramiko.AuthenticationException,paramiko.BadHostKeyException): raise
        except (EOFError,paramiko.SSHException,ConnectionResetError,OSError):
            if attempt: raise
            if job is not None: job['logs'].append('SSH 连接中断，建立新连接重试一次')
            time.sleep(1)


AMP_PROBE_NAME = 'yolo26n.pt'


def _amp_probe_weights():
    """Locally cached probe weights for ultralytics' AMP self-check.

    ultralytics/utils/checks.py:1010 resolves them as WEIGHTS_DIR/'yolo26n.pt' and only
    downloads when that file is missing. WEIGHTS_DIR comes from SETTINGS['weights_dir'],
    whose default is `root/'weights'` with `root = GIT.root or Path()` — so a pip-installed
    ultralytics (no git checkout) gets the *relative* path "weights", resolved against the
    process CWD. The remote runner cds into the job directory, hence <job>/weights/.

    Seeding it costs one 5 MB upload and saves a GitHub download that, from the compute
    node, ran at a few KB/s and stalled every job for ~10 minutes before epoch 1.
    """
    from config import settings
    backend = Path(__file__).resolve().parent.parent
    for path in (settings.WORKSPACE_DIR / 'ultralytics_cache' / AMP_PROBE_NAME,
                 backend.parent / 'weights' / AMP_PROBE_NAME):
        if path.is_file():
            return path
    return None


def _human_size(num):
    for unit in ('B','KB','MB','GB'):
        if num<1024 or unit=='GB': return f'{num:.0f} {unit}' if unit=='B' else f'{num:.1f} {unit}'
        num/=1024
    return f'{num:.1f} GB'


class UploadReporter:
    """Percent-step upload progress.

    paramiko calls the callback once per SFTP block, so without throttling a few
    hundred MB would append thousands of log lines and the persistence thread
    would spend all its time writing job state to disk.

    We emit one line per completed STEP % of the current file, plus a "start"
    line on the first callback and a summary in done() — no time-based lines, so
    the log length depends only on file size, never on link speed.
    """

    STEP = 5  # percent granularity

    def __init__(self, job, label):
        self.job=job
        self.label=label
        self.total=0
        self.sent=0
        self.started=time.monotonic()
        self.last_time=self.started
        self.last_sent=0
        self.phase=0
        self.opened=False
        # Average speed is measured over the whole file. Keeping our own anchor
        # matters: __call__ rewrites last_time/last_sent on every emission, so
        # reusing them here would report the speed of the final 5 % band.
        self.avg_started=self.started
        self.avg_sent=0

    def set_stage(self, label, total=0):
        """Called for every uploaded file. Resets the step counter as well as the
        clock, otherwise a small engine file would be treated as a continuation of
        the multi-hundred-MB dataset and never emit a line."""
        self.label=label
        self.total=total
        self.sent=0
        self.phase=0
        self.opened=False
        self.started=time.monotonic()
        self.last_time=self.started
        self.last_sent=0
        self.avg_started=self.started
        self.avg_sent=0

    def __call__(self, sent, total):
        if self.job.get('remote_stop_requested'): raise ValueError('上传已由用户停止')
        self.sent=sent; self.total=total
        self.avg_sent=sent
        # Index of the 5 % band we have just completed. One line per band, full stop.
        step=int(sent/max(total,1)*(100/self.STEP))
        if step<=self.phase: return
        now=time.monotonic()
        elapsed=now-self.last_time or 1e-9
        speed=(sent-self.last_sent)/elapsed
        self.last_time=now; self.last_sent=sent
        self.phase=step
        pct=min(step*self.STEP,100)
        if not self.opened:
            self.opened=True
            self.job['logs'].append(f'{self.label} 开始上传（{_human_size(total)}）')
        self.job['logs'].append(
            f'{self.label} {pct}% — {_human_size(sent)}/{_human_size(total)}，{_human_size(speed)}/s')

    def done(self, label=None):
        now=time.monotonic()
        elapsed=now-self.avg_started
        avg=(self.avg_sent/elapsed) if elapsed>0 else 0
        text=f'{label or self.label} 完成 — {_human_size(self.avg_sent)}，用时 {elapsed:.0f} 秒'
        if avg: text+=f'，平均 {_human_size(avg)}/s'
        self.job['logs'].append(text)


def command(service,node,secret,text,job=None):
    def execute():
        with service.connect(node,secret) as client:
            return service.execute(client,text,timeout=20)
    return retry_operation(execute,job)


def read_state(service,node,secret,remote,job):
    def read():
        with service.sftp(node,secret) as ftp:
            if job.get('remote_stop_requested'):
                with ftp.open(remote+'/STOP','w') as f:f.write('stop requested by RunOS')
            try:return read_json(ftp,remote+'/state.json')
            except FileNotFoundError:return None
    return retry_operation(read,job)


def process_exists(service,node,secret,pid,job):
    if not str(pid).isdigit() or int(pid)<=1:return False
    code,_,_=command(service,node,secret,'kill -0 '+str(int(pid))+' 2>/dev/null',job)
    return code==0


def confirm_pid(manager,job,node,secret,service):
    remote=job['remote_dir']
    def read():
        with service.sftp(node,secret) as ftp:
            try:
                with ftp.open(remote+'/runner.pid') as f:return int(f.read(32))
            except FileNotFoundError:
                try:return int(read_json(ftp,remote+'/state.json').get('pid',0))
                except FileNotFoundError:return 0
    pid=retry_operation(read,job)
    if pid and process_exists(service,node,secret,pid,job):
        job['remote_pid']=pid;job['remote_workdir']=remote;job['log_path']=remote+'/console.log'
        job['remote_connection']='connected'
        manager._persist_jobs()
        return True
    return False


def run_remote(manager,job):
    from config import settings
    from routers.compute_nodes import store, service, required
    node=store.get(job['config']['compute_node_id'])
    secret=store.bundle(node)
    required(node, secret, '启动远程任务')
    q=shlex.quote
    if job.get('remote_stop_requested') and not job.get('remote_dir'):
        job['status']='stopped';return
    if not job.get('remote_dir'):
        probe=service.probe(node,secret)
        if not probe['connected']: raise ValueError('远程连接失败：'+'；'.join(probe['errors']))
        if not probe.get('workdir_exists'): raise ValueError('远程工作目录不存在或不可访问')
        if not probe.get('environment',{}).get('gpu_available'): raise ValueError('远程 CUDA 不可用，不能启动 GPU 训练')
        if job['model_type']=='rfdetr' and not probe['environment'].get('rfdetr'): raise ValueError('远程 RF-DETR 环境未就绪')
        gpu=node['gpu_id'] if node['gpu_mode']=='manual' else next((g['index'] for g in probe['gpus']),None)
        if gpu is None: raise ValueError('未检测到可选 GPU')
        if gpu not in [g['index'] for g in probe['gpus']]: raise ValueError('所选 GPU 不存在')
        remote=posixpath.join(node['workdir'].rstrip('/'),'runos_jobs',job['id']+'_'+uuid.uuid4().hex[:10])
        job['logs'].append(f"上传到算力节点：{node['name']} / GPU {gpu}")
        remote_config={**job['config'],'compute_node_id':None,'device':str(gpu),'models_dir':remote}
        source=Path(remote_config['base_model'])
        if source.is_file(): remote_config['base_model']=remote+'/initial'+source.suffix
        elif source.is_absolute(): raise ValueError('本地初始权重文件不存在')
        remote_job={k:job.get(k) for k in ('id','name','format','model_type','started_at','total_epochs')}
        remote_job.update(dataset_path=remote+'/dataset',config=remote_config)
        with tempfile.TemporaryDirectory(prefix='runos-upload-') as tmp:
            archive=Path(tmp)/'dataset.zip'
            job['logs'].append('正在打包数据集快照（只读，不改动本地文件）')
            bundle_started=time.monotonic()
            counts=bundle_dataset(Path(job['dataset_path']),archive,remote+'/dataset')
            job['logs'].append('数据快照：'+json.dumps(counts,ensure_ascii=False))
            archive_bytes=archive.stat().st_size
            job['logs'].append(f'快照打包完成 — {_human_size(archive_bytes)}，用时 {time.monotonic()-bundle_started:.0f} 秒（ZIP 不压缩，图片本就无法再压）')
            # weights/ is where ultralytics resolves WEIGHTS_DIR from (see _amp_probe_weights).
            code,_,_=command(service,node,secret,'mkdir -p -- '+q(remote)+' '+q(remote+'/weights'),job)
            if code: raise ValueError('无法创建本任务远程目录')
            job['logs'].append('远程任务目录已创建：'+remote)
            job.update(remote_dir=remote,remote_workdir=remote,log_path=remote+'/console.log',remote_phase='uploading',status='starting')
            job['device_info']=f"{node['name']} / GPU {gpu}"
            manager._persist_jobs()
            def upload():
                with service.sftp(node,secret) as ftp:
                    ftp.get_channel().settimeout(60)
                    reporter=UploadReporter(job,'数据集快照')
                    ftp.put(str(archive),remote+'/dataset.zip',callback=reporter)
                    reporter.done('数据集快照')
                    backend=Path(__file__).resolve().parent.parent
                    upload_files=[('训练引擎','training.py',backend/'training.py'),
                                  ('RF-DETR 引擎','rfdetr_training.py',backend/'rfdetr_training.py'),
                                  ('远程运行器','runner.py',Path(__file__).with_name('remote_runner.py'))]
                    for label,name,path in upload_files:
                        size=path.stat().st_size
                        reporter.set_stage(f'{label}（{name}）',size)
                        ftp.put(str(path),remote+'/'+name,callback=reporter)
                    # One line for all three engines — they are ~100 KB each and a
                    # "完成" line per file would just be noise.
                    job['logs'].append('训练引擎已上传（training.py / rfdetr_training.py / runner.py）')
                    if source.is_file():
                        size=source.stat().st_size
                        reporter.set_stage(f'初始权重（{source.name}）',size)
                        ftp.put(str(source),remote_config['base_model'],callback=reporter)
                        reporter.done(f'初始权重（{source.name}）')
                    probe=_amp_probe_weights()
                    if probe is not None:
                        ftp.put(str(probe),remote+'/weights/'+AMP_PROBE_NAME)
                        job['logs'].append(f'AMP 自检探针权重已预置（{probe.name}，{_human_size(probe.stat().st_size)}），跳过服务器 GitHub 下载')
                    with ftp.open(remote+'/job.json','w') as f:f.write(json.dumps(remote_job))
                    job['logs'].append('任务描述文件已上传，准备启动远程训练进程')
                    manager._persist_jobs()
            retry_operation(upload,job)  # closes SFTP and SSH before launching
            job['remote_phase']='uploaded';manager._persist_jobs()
    if job.get('remote_phase')=='uploading':
        raise ValueError('上传未完成；已保留任务目录，请重新提交，未启动远程训练')
    if job.get('remote_phase')=='uploaded':
        remote=job['remote_dir']
        cmd=environment_command(node,q(remote+'/runner.py')+' '+q(remote))
        # Repeating after a lost response only reuses the PID/lock, never training twice.
        launch=('cd '+q(remote)+' || exit 1; '
                'if [ -f launch.pid ] && kill -0 "$(cat launch.pid)" 2>/dev/null; then cat launch.pid; '
                'elif [ -d runner.lock ]; then :; '
                'else nohup '+cmd+' >> '+q(remote+'/console.log')+' 2>&1 < /dev/null & p=$!; echo "$p" > launch.pid; echo "$p"; fi')
        try:
            code,_,_=command(service,node,secret,'sh -c '+q(launch),job)
            if code:raise ValueError('远程训练启动命令失败，请查看 console.log')
        except (EOFError,paramiko.SSHException,ConnectionResetError,OSError):
            job['logs'].append('启动回执仍不可达，将通过 PID/状态文件确认；不再重复启动')
        job['remote_phase']='submitted';job['logs'].append('上传完成，已使用新 SSH 连接启动，正在确认远程 PID')
        manager._persist_jobs()
    monitor(manager,job,node,secret,service)


def monitor(manager,job,node,secret,service):
    from config import settings
    remote=job['remote_dir']
    job['status']='starting'
    job.setdefault('remote_workdir',remote);job.setdefault('log_path',remote+'/console.log')
    absent_since=None;last_notice=0;confirmed=False
    pid_confirmed=False;first_state=False;connected_notice=False
    while True:
        try:
            state=read_state(service,node,secret,remote,job)
            if job.get('remote_connection')!='connected':
                job['logs'].append('已连接算力节点，开始读取训练状态')
            job['remote_connection']='connected'
            terminal=state and state.get('status') in ('completed','failed','stopped')
            if not terminal and not confirmed:
                confirmed=confirm_pid(manager,job,node,secret,service)
                if confirmed and not pid_confirmed:
                    pid_confirmed=True
                    job['logs'].append(f"远程训练进程已确认运行（PID {job.get('remote_pid')}），工作目录：{remote}")
            if not state:
                absent_since=absent_since or time.monotonic()
                if time.monotonic()-absent_since>120:
                    raise ValueError('远程进程未生成状态；请检查 '+job['log_path'])
                manager._persist_jobs();time.sleep(3);continue
            if not first_state:
                first_state=True
                job['logs'].append('已收到远程状态文件，训练进程正在初始化运行环境')
            if state.get('pid'):job['remote_pid']=state['pid']
            for k in ('progress','current_epoch','total_epochs','metrics','epoch_history','gpu_mem_gb','batch_stats','device_info'):
                if k in state and state[k] is not None: job[k]=state[k]
            job['logs']=['算力节点：'+node['name']+'；远程目录：'+remote]+state.get('logs',[])
            if job.get('remote_stop_requested'):job['logs'].append('已请求停止，等待远程当前批次/轮次结束')
            if terminal:
                model=state.get('model_path')
                if model:
                    model=posixpath.normpath(model)
                    if not model.startswith(remote+'/') or Path(model).suffix not in ('.pt','.pth'):raise ValueError('远程返回了非法权重路径')
                    dest=settings.MODELS_DIR/f"remote_{job['model_type']}_{job['id']}{Path(model).suffix}"
                    partial=dest.with_suffix(dest.suffix+'.partial')
                    def download():
                        with service.sftp(node,secret) as ftp:
                            try:ftp.stat(model)
                            except FileNotFoundError:raise ValueError('远程任务结束但权重不存在，请检查 '+job['log_path'])
                            ftp.get(model,str(partial))
                    retry_operation(download,job);os.replace(partial,dest)
                    job['model_path']=str(dest);job['logs'].append('权重已取回模型库：'+dest.name)
                job['status']=state['status'];job['remote_phase']='finished'
                if state.get('error'):job['error']=state['error']
                manager._persist_jobs();return
            if not confirmed or time.time()-state.get('heartbeat',0)>180:
                if not confirm_pid(manager,job,node,secret,service):raise ValueError('远程 PID 已退出，请查看 '+job['log_path'])
                if time.time()-state.get('heartbeat',0)>180:raise ValueError('远程进程仍存在，但训练心跳已停止；未重新启动任务，请查看 '+job['log_path'])
                confirmed=True
            job['status']='running' if state.get('status')=='running' and confirmed else 'starting'
            job['remote_phase']='monitoring'
            manager._persist_jobs();time.sleep(3)
        except (paramiko.AuthenticationException,paramiko.BadHostKeyException):
            raise ValueError('远程认证或服务器身份已变化，已停止自动重连；远程任务不会被强制终止')
        except (OSError,EOFError,paramiko.SSHException):
            job['remote_connection']='reconnecting';confirmed=False
            if time.monotonic()-last_notice>30:
                job['logs'].append('远程连接中断，正在重连；服务器任务不会重复启动')
                last_notice=time.monotonic();manager._persist_jobs()
            time.sleep(5)


def restore_remote_jobs(manager):
    for id,job in list(manager.training_jobs.items()):
        if job.get('config',{}).get('compute_node_id') and job.get('remote_dir') and job.get('status') in ('interrupted','running','starting'):
            thread=threading.Thread(target=manager._run_training,args=(id,),daemon=True)
            manager.training_threads[id]=thread
            thread.start()
