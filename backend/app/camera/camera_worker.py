"""Bounded live sessions; capture and shared inference never run on the API loop."""
import base64
import collections
import queue
import threading
import time
import uuid

from camera.camera_manager import CameraManager


class CameraSession:
    def __init__(self, source, options, infer):
        self.id = uuid.uuid4().hex
        self.source, self.options, self.infer = source, options, infer
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.status, self.error = 'starting', None
        self.last_client = time.monotonic()
        self.sequence = 0
        self.result = None  # One latest result slot, never a result backlog.
        self.version = 0
        self.times = collections.deque(maxlen=30)
        self.thread = threading.Thread(target=self._run, name='CameraInference', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            self.source.open()
            with self.lock:
                if self.stopped.is_set(): return
                self.status = 'running'
            while not self.stopped.is_set():
                with self.lock:
                    paused = self.status == 'paused'
                    options, version = dict(self.options), self.version
                if self.source.error: raise RuntimeError(self.source.error)
                if time.monotonic() - self.source.last_frame > 6:
                    raise RuntimeError('摄像头连续 6 秒没有新画面，请检查连接或设备占用。')
                if paused:
                    self.stopped.wait(.05)
                    continue
                try: metadata, frame = self.source.read(timeout=.25)
                except queue.Empty: continue
                started = time.monotonic()
                timings = {}
                jpeg, detections = self.infer(frame, session_id=f'camera:{self.id}', timings=timings, **options)
                finished = time.monotonic()
                with self.lock:
                    if self.stopped.is_set() or self.status == 'paused' or version != self.version:
                        continue
                    self.times.append(finished)
                    fps = ((len(self.times)-1)/(self.times[-1]-self.times[0])) if len(self.times)>1 else 0.
                    self.sequence += 1
                    self.result = dict(sequence=self.sequence, frame_sequence=metadata['sequence'],
                        annotated_b64=base64.b64encode(jpeg).decode('ascii'), detections=detections,
                        inference_fps=round(fps,2), latency_ms=round(timings.get('inference_ms',(finished-started)*1000),1),
                        processing_ms=round((finished-started)*1000,1),
                        device=timings.get('device','unknown'),
                        frame_age_ms=round((finished-metadata['captured'])*1000,1),
                        input_age_ms=round((started-metadata['captured'])*1000,1),
                        object_count=len(detections))
        except Exception as exc:
            with self.lock:
                if not self.stopped.is_set():
                    self.status = 'error'
                    message = str(getattr(exc,'detail',exc))
                    self.error = ('GPU 显存不足。请停止其他 GPU 任务，或调整左侧推理图像尺寸后重新启动。 ' + message
                                  if 'out of memory' in message.lower() else message)
        finally:
            self.source.close()
            with self.lock:
                if self.status != 'error': self.status = 'stopped'
                self.result = None

    def snapshot(self, after=0):
        with self.lock:
            self.last_client = time.monotonic()
            output = dict(id=self.id,status=self.status,error=self.error,index=self.source.index,
                resolution=self.source.actual,camera_fps=round(self.source.fps,2),
                model_path=self.options['model_path'],options=dict(self.options),
                queue_size=self.source.queue.qsize(),queue_capacity=1)
            if self.result:
                output.update({k:v for k,v in self.result.items() if k!='annotated_b64'})
                if self.sequence > after: output['annotated_b64'] = self.result['annotated_b64']
            return output

    def update(self, options):
        with self.lock:
            if self.status not in ('starting','running','paused'): raise RuntimeError('摄像头已停止。')
            self.options = dict(options)
            self.version += 1
            self.last_client = time.monotonic()

    def pause(self, paused):
        with self.lock:
            if self.status not in ('running','paused'): raise RuntimeError('摄像头尚未就绪。')
            self.status = 'paused' if paused else 'running'
            self.times.clear()
            self.version += 1
            self.last_client = time.monotonic()

    def stop(self):
        with self.lock:
            self.stopped.set()
            if self.thread.is_alive() and self.status != 'error': self.status = 'stopping'
            self.result = None
        self.source.close()
        if self.thread is not threading.current_thread(): self.thread.join(timeout=2)
        with self.lock:
            if not self.thread.is_alive() and self.status != 'error': self.status = 'stopped'


class CameraService:
    def __init__(self, infer, manager=None, lease_seconds=30):
        self.manager = manager or CameraManager()
        self.infer = infer
        self.lock = threading.RLock()
        self.sessions = {}
        self.scan_state = {'status':'idle','devices':[],'error':None}
        self.lease_seconds = lease_seconds
        self.closed = threading.Event()
        self.guard = None

    def _active(self):
        return any(s.thread.is_alive() for s in self.sessions.values())

    def scan(self):
        with self.lock:
            if self._active(): raise RuntimeError('请先停止摄像头再扫描设备。')
            if self.scan_state['status'] == 'scanning': return dict(self.scan_state)
            self.scan_state = {'status':'scanning','devices':[],'error':None}
            def work():
                try: state = dict(status='done',devices=self.manager.scan(),error=None)
                except Exception as exc: state = dict(status='error',devices=[],error=str(exc))
                with self.lock: self.scan_state = state
            threading.Thread(target=work,name='CameraScan',daemon=True).start()
            return dict(self.scan_state)

    def start(self,index,width,height,options):
        with self.lock:
            if self.closed.is_set(): raise RuntimeError('服务正在关闭。')
            if self._active() or self.scan_state['status']=='scanning':
                raise RuntimeError('已有摄像头任务或扫描正在运行，请等待停止完成。')
            while len(self.sessions)>=4:
                self.sessions.pop(next(iter(self.sessions)))
            session = CameraSession(self.manager.source(index,width,height),options,self.infer)
            self.sessions[session.id] = session
            session.start()
            if self.guard is None:
                self.guard = threading.Thread(target=self._guard,name='CameraLease',daemon=True)
                self.guard.start()
            return session

    def _guard(self):
        while not self.closed.wait(1):
            with self.lock: sessions = list(self.sessions.values())
            for s in sessions:
                if s.thread.is_alive() and time.monotonic()-s.last_client > self.lease_seconds:
                    s.stop()

    def shutdown(self):
        self.closed.set()
        with self.lock: sessions = list(self.sessions.values())
        for s in sessions: s.stop()
        if self.guard: self.guard.join(timeout=2)
