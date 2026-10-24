"""Isolated OpenCV handle owner. Only this child process opens VideoCapture."""
import json, os, sys, threading, time

def main():
    import cv2
    request = json.loads(sys.argv[1])
    if request['action'] == 'enumerate':
        from cv2_enumerate_cameras import enumerate_cameras
        cameras = []
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            for c in enumerate_cameras(backend):
                cameras.append(dict(index=c.index, name=c.name, backend=c.backend, path=c.path))
        print(json.dumps(cameras)); return
    parent = request['parent']
    def parent_guard():
        import psutil
        while True:
            time.sleep(0.5)
            if not psutil.pid_exists(parent):
                os._exit(0)
    threading.Thread(target=parent_guard, daemon=True).start()
    stopped = threading.Event()
    if request['action'] == 'stream':
        def stop_reader():
            sys.stdin.buffer.readline()
            stopped.set()
        threading.Thread(target=stop_reader, daemon=True).start()
    cap = None
    try:
        cap = cv2.VideoCapture(request['index'], request['backend'])
        if not cap.isOpened():
            raise RuntimeError('无法打开摄像头：设备可能被占用、已断开或未允许桌面应用访问。')
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if request['action'] == 'probe':
            supported = []
            for width, height in [(640,480),(1280,720),(1920,1080)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
                ok, frame = cap.read()
                if ok:
                    actual = [int(frame.shape[1]),int(frame.shape[0])]
                    if actual not in supported: supported.append(actual)
            if not supported: raise RuntimeError('摄像头已打开，但读取画面失败。')
            print(json.dumps(dict(openable=True,resolutions=supported))); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, request['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, request['height'])
        sequence = 0
        while not stopped.is_set():
            ok, frame = cap.read()
            if not ok: raise RuntimeError('摄像头读取失败或设备已断开。')
            sequence += 1
            captured = time.monotonic()
            ok, encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok: continue
            header = dict(bytes=len(encoded),sequence=sequence,captured=captured,
                          width=int(frame.shape[1]),height=int(frame.shape[0]))
            sys.stdout.buffer.write((json.dumps(header)+'\n').encode())
            sys.stdout.buffer.write(encoded.tobytes()); sys.stdout.buffer.flush()
    except Exception as exc:
        print(json.dumps({'error':str(exc)}),flush=True)
    finally:
        if cap is not None: cap.release()

if __name__ == '__main__': main()
