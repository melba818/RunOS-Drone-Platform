"""SSH-only HTTP bridge to the unchanged RunOS inference engine."""
import base64
from contextlib import ExitStack
import hashlib
import http.client
import io
import json
from pathlib import Path
import posixpath
import secrets
import shlex
import threading
import time
from urllib.parse import urlencode
import uuid
import paramiko

from fastapi import HTTPException
from fastapi.responses import Response,StreamingResponse
from starlette.background import BackgroundTask
from compute.store import CONNECTION_KEYS
from .remote_training import environment_command,read_json


class InferenceNodes:
    def __init__(self):
        self.records={}
        self.job_records={}
        self.transfers={}
        self.lock=threading.RLock()

    def ensure(self,id):
        from routers.compute_nodes import store,service,required
        node=store.get(id)
        signature=json.dumps({k:node.get(k) for k in (*CONNECTION_KEYS,'workdir','environment_type','conda_env','conda_executable','venv_path','gpu_mode','gpu_id','updated_at')},sort_keys=True)
        with self.lock:
            old=self.records.get(id)
            if old and old['signature']==signature:
                if time.monotonic()-old['used']<840:
                    old['used']=time.monotonic();return old
                try:
                    response,stack=self.exchange(old,'GET','/health')
                    try:ready=response.status==200;response.read()
                    finally:stack.close()
                    if ready:
                        old['used']=time.monotonic();return old
                except Exception:pass
            credential=store.bundle(node)
            required(node, credential, '启动远程推理')
            report=service.probe(node,credential)
            if not report['connected']:raise HTTPException(422,'SSH 不可用：'+'；'.join(report['errors']))
            if not report.get('workdir_exists'):raise HTTPException(422,'远程工作目录不存在或不可访问')
            if not report['environment'].get('gpu_available'):raise HTTPException(422,'远程 PyTorch CUDA 不可用')
            gpu=node['gpu_id'] if node['gpu_mode']=='manual' else next((g['index'] for g in report['gpus']),None)
            if gpu not in [g['index'] for g in report['gpus']]:raise HTTPException(422,'未检测到所选 GPU')
            root=posixpath.join(node['workdir'].rstrip('/'),'runos_inference',uuid.uuid4().hex)
            token=secrets.token_urlsafe(32)
            with service.connect(node,credential) as client:
                code,_,_=service.execute(client,'umask 077; mkdir -p -- '+shlex.quote(posixpath.dirname(root))+' && mkdir -- '+shlex.quote(root))
                if code:raise HTTPException(422,'无法创建远程推理工作目录')
                backend=Path(__file__).resolve().parent.parent
                with client.open_sftp() as ftp:
                    ftp.get_channel().settimeout(60)
                    for name,path in [('worker.py',Path(__file__).with_name('inference_server.py')),('inference_engine.py',backend/'routers/inference.py'),('model_integration.py',backend/'model_integration.py'),('rfdetr_adapter.py',backend/'rfdetr_adapter.py'),('reid_encoders.py',backend/'reid_encoders.py')]:
                        ftp.put(str(path),root+'/'+name)
                    with ftp.open(root+'/worker.json','w') as f:f.write(json.dumps({'gpu':gpu,'token':token,'require_gpu':True}))
                    ftp.chmod(root+'/worker.json',0o600)
                cmd=environment_command(node,shlex.quote(root+'/worker.py')+' '+shlex.quote(root))
                launch='cd '+shlex.quote(root)+' && umask 077; nohup '+cmd+' > '+shlex.quote(root+'/console.log')+' 2>&1 < /dev/null &'
                code,_,_=service.execute(client,'sh -c '+shlex.quote(launch),timeout=15)
                if code:raise HTTPException(502,'远程推理进程启动失败')
                deadline=time.monotonic()+75
                with client.open_sftp() as ftp:
                    ftp.get_channel().settimeout(10)
                    while True:
                        try: ready=read_json(ftp,root+'/ready.json');break
                        except FileNotFoundError:
                            if time.monotonic()>deadline:raise HTTPException(502,'远程推理未启动。请检查该目录 console.log；服务器需安装 torch、ultralytics/RF-DETR、FastAPI、uvicorn、python-multipart 等依赖：'+root)
                            time.sleep(1)
            record=dict(node=node,credential=credential,service=service,root=root,token=token,port=int(ready['port']),signature=signature,used=time.monotonic(),models={},upload_lock=threading.Lock())
            self.records[id]=record
            return record

    def video_record(self,id,job_id):
        """Resolve existing jobs without starting a new worker, including after restart."""
        key=(id,job_id)
        if key in self.job_records:return self.job_records[key]
        from routers.compute_nodes import store,service
        node=store.get(id);credential=store.bundle(node)
        required(node, credential, '读取远端推理结果')
        base=posixpath.join(node['workdir'].rstrip('/'),'runos_inference')
        with service.sftp(node,credential) as ftp:
            try:entries=sorted(ftp.listdir_attr(base),key=lambda e:e.st_mtime or 0,reverse=True)
            except FileNotFoundError:entries=[]
            for entry in entries:
                if len(entry.filename)!=32 or any(c not in '0123456789abcdef' for c in entry.filename):continue
                root=base+'/'+entry.filename
                try:
                    ready=read_json(ftp,root+'/ready.json');cfg=read_json(ftp,root+'/worker.json')
                    record=dict(node=node,credential=credential,service=service,root=root,token=cfg['token'],port=int(ready['port']),used=time.monotonic(),models={},upload_lock=threading.Lock())
                    response,stack=self.exchange(record,'GET','/api/infer/video/'+job_id+'/status')
                    try:found=response.status==200;response.read()
                    finally:stack.close()
                    if found:
                        self.job_records[key]=record
                        return record
                except (OSError,ValueError,EOFError,paramiko.SSHException):continue
        raise HTTPException(404,'未找到原服务器视频任务；任务进程可能已退出，请勿重复提交以免重复推理')

    def model(self,record,path):
        source=Path(path).resolve()
        if not source.is_file() or source.suffix.lower() not in ('.pt','.pth','.yaml','.yml'):raise HTTPException(422,'请选择本地有效模型或跟踪器配置')
        stamp=f'{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}'
        with record['upload_lock']:
            if stamp in record['models']:return record['models'][stamp]
            target=record['root']+'/models/'+hashlib.sha256(stamp.encode()).hexdigest()+source.suffix.lower()
            with record['service'].sftp(record['node'],record['credential']) as ftp:ftp.put(str(source),target)
            record['models'][stamp]=target
            return target

    def fields(self,record,values):
        values=dict(values)
        if str(values.get('with_reid','')).lower() in ('true','1'):raise HTTPException(422,'远程推理暂不支持自定义 ReID，请选择本地或关闭 ReID')
        if values.get('model_path'):values['model_path']=self.model(record,values['model_path'])
        tracker=values.get('tracker')
        if tracker and Path(str(tracker)).is_file():values['tracker']=self.model(record,tracker)
        values.pop('compute_node_id',None)
        return values

    @staticmethod
    def exchange(record,method,path,body=None,headers=None):
        stack=ExitStack()
        try:
            client=stack.enter_context(record['service'].connect(record['node'],record['credential']))
            channel=client.get_transport().open_channel('direct-tcpip',('127.0.0.1',record['port']),('127.0.0.1',0),timeout=10)
            channel.settimeout(180)
            connection=http.client.HTTPConnection('127.0.0.1',record['port'],timeout=180)
            connection.sock=channel
            stack.callback(connection.close)
            connection.request(method,path,body=body,headers={**(headers or {}),'X-RunOS-Token':record['token']})
            response=connection.getresponse()
            return response,stack
        except Exception:
            stack.close();raise

    def forward(self,id,method,path,query=None,values=None,uploads=None):
        query=dict(query or {})
        transfer_id=query.pop('upload_id',None)
        transfer=None
        if transfer_id and method=='POST' and path=='video':
            # Bounded transient progress records; never contain credentials.
            if len(self.transfers)>100:
                self.transfers={k:v for k,v in self.transfers.items() if time.monotonic()-v['updated']<3600}
                while len(self.transfers)>100:
                    self.transfers.pop(next(iter(self.transfers)))
            transfer={'stage':'准备服务器及模型','sent':0,'total':0,'updated':time.monotonic()}
            self.transfers[(id,transfer_id)]=transfer
        record=self.video_record(id,path.split('/')[1]) if method=='GET' and path.startswith('video/') else self.ensure(id)
        values=self.fields(record,values or {})
        query=self.fields(record,query or {})
        body=None;headers={}
        if uploads is not None:
            boundary='RunOS'+uuid.uuid4().hex
            chunks=[];length=0
            for name,value in values.items():
                blob=(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()
                chunks.append(blob);length+=len(blob)
            for name,upload in uploads:
                # Browser paths are not used as remote paths; only extension is needed.
                ext=Path(upload.filename or '').suffix
                if len(ext)>12 or not all(c.isalnum() or c=='.' for c in ext):ext='.bin'
                blob=(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="input{ext}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
                upload.file.seek(0,2);size=upload.file.tell();upload.file.seek(0)
                if size>4*1024**3:raise HTTPException(413,'单文件超过 4 GiB')
                chunks.extend([blob,upload.file,b'\r\n']);length+=len(blob)+size+2
            end=f'--{boundary}--\r\n'.encode();chunks.append(end);length+=len(end)
            if transfer is not None:transfer.update(stage='上传至服务器',total=length)
            def stream():
                for chunk in chunks:
                    if isinstance(chunk,bytes):
                        yield chunk
                        if transfer is not None:transfer.update(sent=transfer['sent']+len(chunk),updated=time.monotonic())
                    else:
                        while True:
                            data=chunk.read(1024*1024)
                            if not data:break
                            yield data
                            if transfer is not None:transfer.update(sent=transfer['sent']+len(data),updated=time.monotonic())
                if transfer is not None:transfer.update(stage='服务器接收完成，创建推理任务')
            body=stream();headers={'Content-Type':f'multipart/form-data; boundary={boundary}','Content-Length':str(length)}
        elif method in ('POST','PATCH'):
            body=json.dumps(values).encode();headers={'Content-Type':'application/json'}
        try:
            response,stack=self.exchange(record,method,'/api/infer/'+path+('?' + urlencode(query) if query else ''),body,headers)
        except Exception:
            # Recheck the worker on the next request; never replay a submitted video.
            record['used']=0
            raise
        content_type=response.getheader('Content-Type','application/octet-stream')
        if 'json' in content_type:
            try:
                payload=response.read(32*1024*1024+1)
                if len(payload)>32*1024*1024:raise HTTPException(502,'远程响应过大')
                if method=='POST' and path=='video' and response.status==200:
                    job_id=json.loads(payload).get('job_id')
                    if job_id:
                        self.job_records[(id,job_id)]=record
                        from .inference_history import save
                        save(id,job_id,{'id':job_id,'status':'starting','progress':0,'message':'等待服务器进度'})
                if method=='GET' and path.startswith('video/') and path.endswith('/status') and response.status==200:
                    from .inference_history import save
                    save(id,path.split('/')[1],json.loads(payload))
                return Response(payload,status_code=response.status,media_type='application/json')
            finally:stack.close()
        def output():
            try:
                while True:
                    data=response.read(65536)
                    if not data:break
                    yield data
            finally:stack.close()
        out_headers={}
        for key in ('Content-Disposition','Content-Length'):
            if response.getheader(key):out_headers[key]=response.getheader(key)
        return StreamingResponse(output(),status_code=response.status,media_type=content_type,headers=out_headers,background=BackgroundTask(stack.close))

    def frame(self,id,frame,model_path,timings,options):
        import cv2
        _,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        values={**options,'model_path':model_path,'frame_b64':base64.b64encode(data).decode()}
        values.update(values.pop('track_options',{}) or {})
        values.pop('device',None)
        started=time.perf_counter()
        response=self.forward(id,'POST','frame',values=values)
        payload=json.loads(response.body)
        if response.status_code>=400:raise RuntimeError(payload.get('detail','远程推理失败'))
        if timings is not None:timings.update(inference_ms=(time.perf_counter()-started)*1000,device='SSH GPU（含传输）')
        return base64.b64decode(payload['annotated_b64']),payload['detections']

    def shutdown(self):
        for record in list(self.records.values()):
            try:
                response,stack=self.exchange(record,'POST','/shutdown',b'{}',{'Content-Type':'application/json'})
                try:response.read()
                finally:stack.close()
            except Exception:pass
        self.records.clear()


nodes=InferenceNodes()
