import collections, json, os, queue, subprocess, sys, threading, time
from pathlib import Path
from abc import ABC, abstractmethod

HOST = Path(__file__).with_name('capture_host.py')
FLAGS = getattr(subprocess,'CREATE_NO_WINDOW',0)

def terminate_host(process):
    """Also stop Windows venv redirector children, not just the launcher executable."""
    import psutil
    try:
        owner=psutil.Process(process.pid)
        tree=owner.children(recursive=True)+[owner]
        for child in reversed(tree):
            try: child.kill()
            except psutil.NoSuchProcess: pass
        psutil.wait_procs(tree,timeout=2)
    except psutil.NoSuchProcess: pass
    process.wait(timeout=2)


def latest_put(q, item):
    try: q.put_nowait(item)
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: pass
        try: q.put_nowait(item)
        except queue.Full: pass

def call_host(request, timeout):
    process = subprocess.Popen([sys.executable,str(HOST),json.dumps({**request,'parent':os.getpid()})],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=FLAGS)
    try:
        output,error=process.communicate(timeout=timeout)
        if process.returncode: raise RuntimeError(error.decode(errors='replace')[-500:])
        return json.loads(output.decode('utf-8'))
    except subprocess.TimeoutExpired:
        terminate_host(process)
        process.communicate(timeout=2)
        raise
    finally:
        process.stdout.close();process.stderr.close()

class BaseVideoSource(ABC):
    @abstractmethod
    def open(self): pass
    @abstractmethod
    def read(self, timeout=1): pass
    @abstractmethod
    def close(self): pass

class CameraSource(BaseVideoSource):
    def __init__(self, index, width, height, backends=None):
        self.index,self.width,self.height=index,width,height
        self.backends=backends or [(700,index),(1400,index)]
        self.queue=queue.Queue(maxsize=1)
        self.process=None; self.thread=None; self.error=None
        self.stop_event=threading.Event(); self.ready=threading.Event()
        self.actual=None; self.fps=0.; self.last_frame=0.; self.backend=None
        self._times=collections.deque(maxlen=30)
        self._close_lock=threading.Lock()
        self.cancelled=threading.Event()

    def open(self):
        errors=[]
        for backend,index in self.backends:
            with self._close_lock:
                if self.cancelled.is_set(): raise RuntimeError('摄像头启动已取消。')
                self.stop_event.clear(); self.ready.clear(); self.error=None
                self.backend=backend
                request=dict(action='stream',index=index,backend=backend,width=self.width,height=self.height,parent=os.getpid())
                self.process=subprocess.Popen([sys.executable,str(HOST),json.dumps(request)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=FLAGS)
                self.thread=threading.Thread(target=self._capture,name='CameraCapture',daemon=True)
                self.thread.start()
            if self.ready.wait(6) and self.actual and not self.error: return
            errors.append(self.error or '设备打开/首帧超时')
            self.close(retrying=True); self.actual=None
        raise RuntimeError(' / '.join(errors))

    def _capture(self):
        import cv2, numpy as np
        try:
            stream=self.process.stdout
            while not self.stop_event.is_set():
                line=stream.readline(4096)
                if not line: raise RuntimeError('摄像头连接已结束。')
                metadata=json.loads(line)
                if metadata.get('error'): raise RuntimeError(metadata['error'])
                length=metadata['bytes']
                if not 0 < length <= 32*1024*1024: raise RuntimeError('Invalid camera frame')
                data=bytearray()
                while len(data)<length:
                    chunk=stream.read(length-len(data))
                    if not chunk: raise RuntimeError('摄像头画面传输中断。')
                    data.extend(chunk)
                frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if frame is None: continue
                self.actual=[metadata['width'],metadata['height']]
                self.last_frame=time.monotonic();self._times.append(self.last_frame)
                if len(self._times)>1: self.fps=(len(self._times)-1)/max(self._times[-1]-self._times[0],1e-6)
                latest_put(self.queue,(metadata,frame));self.ready.set()
        except Exception as exc:
            if not self.stop_event.is_set(): self.error=str(exc)
            self.ready.set()

    def read(self, timeout=1):
        return self.queue.get(timeout=timeout)

    def close(self, retrying=False):
        with self._close_lock:
            if not retrying:self.cancelled.set()
            self.stop_event.set()
            self.ready.set()
            p=self.process
            if p is not None:
                if p.poll() is None:
                    try:
                        p.stdin.write(b'stop\n');p.stdin.flush();p.wait(timeout=1)
                    except (OSError,subprocess.TimeoutExpired):
                        terminate_host(p)
                if self.thread and self.thread is not threading.current_thread():self.thread.join(timeout=2)
                if p.stdout:p.stdout.close()
                if p.stdin:p.stdin.close()
            self.process=None
            while True:
                try:self.queue.get_nowait()
                except queue.Empty:break

class CameraManager:
    def __init__(self):
        self.lock=threading.Lock();self.devices=[]

    def scan(self):
        if not self.lock.acquire(blocking=False): raise RuntimeError('设备扫描已在进行中。')
        try:
            try: entries=call_host({'action':'enumerate'},8)
            except (ImportError,RuntimeError,subprocess.TimeoutExpired):
                entries=[dict(index=i,name=f'Camera {i}',backend=700) for i in range(4)]
            direct=[c for c in entries if c['backend']==700]
            if not direct: direct=[c for c in entries if c['backend']==1400]
            devices=[]
            for c in direct[:16]:
                backends=[(c['backend'],c['index'])]
                alternatives=[x for x in entries if x['name']==c['name'] and x['backend']!=c['backend']]
                if len(alternatives)>1:
                    # Identically named USB devices must not silently switch to a different camera.
                    identity=lambda d:(d.get('path') or '').lower().removeprefix('@device:pnp:')
                    alternatives=[x for x in alternatives if identity(c) and identity(c)==identity(x)]
                backends += [(x['backend'],x['index']) for x in alternatives]
                item={**c,'backends':backends,'openable':False,'resolutions':[],'error':None}
                for backend,index in backends:
                    try:
                        probe=call_host(dict(action='probe',backend=backend,index=index),6)
                        item.update(probe)
                        if probe.get('openable'):item['error']=None;break
                    except Exception as exc:item['error']=str(exc) or '设备探测超时'
                devices.append(item)
            self.devices=devices
            return devices
        finally:self.lock.release()

    def source(self,index,width,height):
        item=next((d for d in self.devices if d['index']==index),None)
        return CameraSource(index,width,height,item['backends'] if item else None)
