"""Private loopback inference worker, uploaded by RunOS and reached only over SSH."""
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import time
import types


def main():
    root=Path(sys.argv[1]).resolve()
    cfg=json.loads((root/'worker.json').read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg['gpu'])
    import torch
    if cfg.get('require_gpu',True) and not torch.cuda.is_available():
        raise RuntimeError('远程 PyTorch CUDA 不可用，未回退 CPU')
    from fastapi import FastAPI,Request
    from fastapi.responses import JSONResponse
    import uvicorn
    from model_integration import ModelManager
    models=root/'models';temp=root/'temp'
    models.mkdir(exist_ok=True);temp.mkdir(exist_ok=True)
    settings=types.SimpleNamespace(MODELS_DIR=models,TEMP_DIR=temp,model_manager=ModelManager(models),_download_status={})
    config=types.ModuleType('config');config.settings=settings;sys.modules['config']=config
    import inference_engine as engine
    original=engine.predict_frame
    def predict(*args,**kwargs):
        kwargs['device']='cuda:0' if cfg.get('require_gpu',True) else 'cpu'
        return original(*args,**kwargs)
    engine.predict_frame=predict
    app=FastAPI()
    last=[time.monotonic()]
    draining=[False]
    @app.middleware('http')
    async def auth(request:Request,call_next):
        import hmac
        if not hmac.compare_digest(request.headers.get('x-runos-token',''),cfg['token']):
            return JSONResponse({'detail':'Unauthorized'},status_code=403)
        last[0]=time.monotonic()
        return await call_next(request)
    app.include_router(engine.router)
    @app.get('/health')
    def health():return {'ready':True,'device':'cuda:0' if cfg.get('require_gpu',True) else 'cpu'}
    @app.post('/shutdown')
    def shutdown():
        draining[0]=True
        return {'draining':True}
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=sock.getsockname()[1],access_log=False,log_level='warning'))
    async def idle():
        while not server.should_exit:
            await asyncio.sleep(10)
            active=any(j.get('status') in ('pending','running') for j in engine._video_jobs.values())
            if time.monotonic()-last[0]>900 and not active and engine._video_jobs:
                with settings.model_manager.runtime_lock:
                    engine._sessions.clear()
                    settings.model_manager.loaded_models.clear()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                last[0]=time.monotonic()
            if (draining[0] and not active) or (time.monotonic()-last[0]>900 and not engine._video_jobs):
                server.should_exit=True
    async def serve():
        watcher=asyncio.create_task(idle())
        try:await server.serve(sockets=[sock])
        finally:watcher.cancel()
    (root/'ready.json').write_text(json.dumps({'port':sock.getsockname()[1],'pid':os.getpid()}))
    asyncio.run(serve())


if __name__=='__main__': main()
