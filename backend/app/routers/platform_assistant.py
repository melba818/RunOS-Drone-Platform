"""Account-local, versioned preparation of annotation and training jobs.

Drafts have no job side effects. Executions atomically claim the exact reviewed
revision before invoking the same handlers used by the native application.
"""
import hashlib,json,os,re,secrets,sqlite3,time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal,Optional
from fastapi import APIRouter,BackgroundTasks,HTTPException,Request
from pydantic import BaseModel,ConfigDict,Field,ValidationError
from config import settings
from schemas.common import TrainingConfig
from training_contract import ARCHITECTURES,model_type,validation_error
from routers import training,annotations

router=APIRouter()

class AnnotationConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    dataset_id:str=Field(min_length=1,max_length=240)
    model_id:str=Field(min_length=1,max_length=160)
    text_prompt:Optional[str]=Field(default=None,max_length=3000)
    label:str=Field(min_length=1,max_length=160)
    description:Optional[str]=Field(default=None,max_length=2000)
    extra_prompt:Optional[str]=Field(default=None,max_length=3000)
    confidence_threshold:float=Field(default=.5,ge=0,le=1,allow_inf_nan=False)
    verify_model_id:Optional[str]=Field(default=None,max_length=160)
    on_verify_fail:Literal['review']='review'
    tile_pass:bool=False
    tile_size:int=Field(default=640,ge=128,le=2048)
    tile_overlap:int=Field(default=128,ge=0,le=2047)
    tile_batch_size:int=Field(default=64,ge=1,le=512)

class DraftBody(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['annotation','training']
    data:dict
    revision:Optional[int]=None

class ExecuteBody(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1)

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def owner(request):
    value=getattr(request.state,'admin_id',None)
    if not isinstance(value,int) or value<1:raise HTTPException(401,'登录状态已失效。')
    return value

@contextmanager
def db():
    path=settings.WORKSPACE_DIR/'assistant-drafts.sqlite3'
    c=sqlite3.connect(path,timeout=10);c.row_factory=sqlite3.Row
    try:
        c.execute('CREATE TABLE IF NOT EXISTS drafts(id TEXT PRIMARY KEY,owner INTEGER NOT NULL,kind TEXT NOT NULL,revision INTEGER NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,binding TEXT NOT NULL,result TEXT,followup TEXT NOT NULL,created INTEGER NOT NULL,updated INTEGER NOT NULL)')
        c.execute('CREATE TABLE IF NOT EXISTS assistant_flows(id TEXT PRIMARY KEY,owner INTEGER NOT NULL,revision INTEGER NOT NULL,state TEXT NOT NULL,updated INTEGER NOT NULL)')
        os.chmod(path,0o600)
        with c:yield c
    finally:c.close()

def dataset_binding(dataset_id):
    if dataset_id not in settings.active_datasets:raise HTTPException(404,'数据集不存在。')
    root=settings.DATASETS_DIR.resolve();path=(root/dataset_id).resolve()
    if path.parent!=root or not path.is_dir():validation_error('dataset_id','数据集目录无效。')
    manifest=[]
    for f in sorted(path.rglob('*')):
        if not f.is_file():continue
        if f.is_symlink() or not f.resolve().is_relative_to(path):validation_error('dataset_id','数据集包含无效的文件引用。')
        stat=f.stat();manifest.append([f.relative_to(path).as_posix(),stat.st_size,stat.st_mtime_ns])
    return digest([settings.active_datasets[dataset_id],manifest])

def safe_models():
    allowed={'id','name','type','loaded','downloaded','remote','supports_confidence_threshold'}
    # Classification and embedding services cannot produce bounding-box labels.
    return [{k:v for k,v in m.items() if k in allowed} for m in settings.model_manager.list_models()
            if (m.get('loaded') or m.get('downloaded')) and not m.get('error')
            and m.get('type') in {'vlm','vlx','vlx_qwen','yolo','yoloworld','rfdetr','rtdetr','rcnn','grounding_dino','groundingdino','owlvit'}]

def model_binding(ident,model):
    stable={k:v for k,v in model.items() if k in {'id','type'}}
    from vlm_annotator import load_providers
    providers=load_providers()
    provider=providers.get(ident)
    if provider:
        from dataclasses import asdict,is_dataclass
        stable['connection_revision']=digest(asdict(provider) if is_dataclass(provider) else vars(provider))
    if model.get('type') in {'vlx','vlx_qwen'}:
        from lmm import config as lmm_config
        stable['connection_revision']=digest(lmm_config.load())
    info=next((m for m in settings.model_manager.list_models() if m['id']==ident),{})
    info={**info,**settings.model_manager.loaded_models.get(ident,{})}
    path=Path(str(info.get('path','')))
    if path.is_file():
        st=path.stat();stable['weights_revision']=[st.st_size,st.st_mtime_ns]
    return digest(stable)

def normalize(kind,data,request):
    try:cfg=AnnotationConfig.model_validate(data) if kind=='annotation' else TrainingConfig.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422,{'code':'VALIDATION_FAILED','message':'参数校验未通过。','errors':[{'field':'.'.join(map(str,i['loc'])),'code':i['type'],'message':'请检查此字段的类型和范围。'} for i in e.errors()]}) from None
    binding={'dataset':dataset_binding(cfg.dataset_id)}
    if kind=='training':
        training.validate_training_config(cfg,request)
        if cfg.initial_weights=='custom':
            f=(settings.MODELS_DIR/cfg.custom_weights).resolve();st=f.stat()
            binding['weights']=digest([cfg.custom_weights,st.st_size,st.st_mtime_ns])
        else:
            f=settings.MODELS_DIR/Path(training._resolve_base_model(cfg.model_arch,cfg.model_type)).name
            binding['weights']=digest([f.name,f.stat().st_size,f.stat().st_mtime_ns]) if f.is_file() else 'builtin-not-cached'
        binding['architecture']=digest(next(a for a in ARCHITECTURES if a['value']==cfg.model_arch))
    else:
        ds=settings.active_datasets[cfg.dataset_id]
        if ds.get('task_type','detection') in {'segmentation','classification'}:
            validation_error('dataset_id','当前自动标注流程支持目标检测数据集，请选择检测数据集。')
        models={m['id']:m for m in safe_models()}
        for field in ['model_id','verify_model_id']:
            ident=getattr(cfg,field)
            if ident and ident not in models:validation_error(field,'所选标注模型不可用，请刷新模型列表。')
            if ident:binding[field]=model_binding(ident,models[ident])
        if cfg.verify_model_id==cfg.model_id:validation_error('verify_model_id','标注模型与复核模型须分别选择。')
        ds=settings.active_datasets[cfg.dataset_id]
        classes=settings.model_manager._get_dataset_classes(settings.DATASETS_DIR/cfg.dataset_id,ds.get('format','yolo')) or ds.get('classes') or []
        if cfg.label not in [str(c) for c in classes]:validation_error('label','请选择数据集中的实际类别。')
        if cfg.tile_pass:
            from tile_pass import validate_options
            try:validate_options(cfg.tile_size,cfg.tile_overlap,cfg.tile_batch_size)
            except ValueError:validation_error('tile_size','切片尺寸、重叠和批次不匹配。')
        if not (cfg.text_prompt or '').strip():
            cfg.text_prompt='\n'.join(p for p in [cfg.label,cfg.description,cfg.extra_prompt] if p)
    return cfg.model_dump(mode='json'),binding

def read(owner_id,did):
    if not re.fullmatch('[a-f0-9]{32}',did):raise HTTPException(404,'参数草稿不存在。')
    with db() as c:r=c.execute('SELECT * FROM drafts WHERE id=? AND owner=?',(did,owner_id)).fetchone()
    if r is None:raise HTTPException(404,'参数草稿不存在。')
    return dict(r)

def public(r):
    return {k:r[k] for k in ['id','kind','revision','status','followup','created','updated']}|{
        'data':json.loads(r['payload']),'binding':json.loads(r['binding']),
        'result':json.loads(r['result']) if r['result'] else None,
        'target':{'view':'training' if r['kind']=='training' else 'annotate','draft_id':r['id']}}

@router.get('/api/assistant/catalog')
async def catalog(request:Request):
    owner(request)
    datasets=[]
    for ident,ds in settings.active_datasets.items():
        datasets.append({k:v for k,v in dict(ds,id=ident).items() if k in {'id','name','format','task_type','num_images','num_annotations','classes','splits'}})
    weights=await training.list_initial_weights()
    try:
        from routers.health import hardware_info
        hardware=await hardware_info()
        compute={k:v for k,v in hardware.items() if k in {'cpu_cores','ram_total_gb','ram_available_gb','cuda_available','gpu_count','gpus'}}
    except Exception:compute={'checked':False,'cpu_available':True}
    return {'datasets':datasets,'annotation_models':safe_models(),'training_architectures':ARCHITECTURES,
            'compute':compute,
            'initial_weights':weights['models'],'training_schema':TrainingConfig.model_json_schema(),
            'annotation_schema':AnnotationConfig.model_json_schema(),
            'capabilities':{'preserve_existing_labels':True,'review_candidates_are_training_labels':False,'full_checkpoint_resume':False,
            'annotation_jobs':'text-batch','draft_execution':'revision-bound-idempotent'}}

@router.post('/api/assistant/drafts',status_code=201)
async def prepare(body:DraftBody,request:Request):
    oid=owner(request);data,binding=normalize(body.kind,body.data,request);did=secrets.token_hex(16);now=int(time.time())
    with db() as c:c.execute("INSERT INTO drafts VALUES(?,?,?,1,'prepared',?,?,NULL,'pending',?,?)",(did,oid,body.kind,json.dumps(data,ensure_ascii=False),json.dumps(binding),now,now))
    return public(read(oid,did))

@router.get('/api/assistant/drafts')
async def list_drafts(request:Request):
    oid=owner(request)
    with db() as c:rows=c.execute('SELECT * FROM drafts WHERE owner=? ORDER BY updated DESC LIMIT 60',(oid,)).fetchall()
    return {'drafts':[public(dict(r)) for r in rows]}

@router.get('/api/assistant/drafts/{did}')
async def detail(did:str,request:Request):
    r=read(owner(request),did);v=public(r)
    if r['status']=='submitted' and v['result']:
        jid=v['result'].get('job_id') or v['result'].get('training_id')
        try:
            result=await annotations.get_text_batch_status(jid) if r['kind']=='annotation' else await training.get_training_status(jid)
            # Do not expose private filesystem paths or full raw model logs through assistant tools.
            v['job']={k:value for k,value in result.items() if k in {'status','progress','total','processed','annotated','failed','total_annotations','dual','pending_review','review_summary','current_epoch','total_epochs','metrics','error','dataset_id','model_id'}}
            if v['job'].get('error'):v['job']['error']='任务未完成，请在对应业务页面查看日志。'
            v['offer_training']=r['kind']=='annotation' and result.get('status')=='done' and r['followup']=='pending' and result.get('total',0)>result.get('failed',0)
        except HTTPException:v['job']={'status':'unavailable'}
    if r['kind']=='annotation':
        from assistant_workflow import current
        v['workflow']=await current(owner(request),did)
        v['offer_review']=v['workflow']['stage']=='review_required'
        v['offer_training']=v['workflow']['stage'] in {'split_done','training_ready'}
    return v

@router.put('/api/assistant/drafts/{did}')
async def update(did:str,body:DraftBody,request:Request):
    oid=owner(request);r=read(oid,did)
    if r['kind']!=body.kind or r['status']!='prepared' or r['revision']!=body.revision:raise HTTPException(409,'草稿已更新或已提交，请重新读取。')
    data,binding=normalize(body.kind,body.data,request)
    if body.kind=='annotation':
        with db() as c:flow=c.execute('SELECT state FROM assistant_flows WHERE id=? AND owner=?',(did,oid)).fetchone()
        if flow and json.loads(flow['state'])['source_dataset_id']!=data['dataset_id']:
            validation_error('dataset_id','当前流程已绑定数据集，更换数据集须重新准备标注草稿。')
    with db() as c:
        n=c.execute("UPDATE drafts SET payload=?,binding=?,revision=revision+1,updated=? WHERE id=? AND owner=? AND revision=? AND status='prepared'",(json.dumps(data,ensure_ascii=False),json.dumps(binding),int(time.time()),did,oid,body.revision)).rowcount
        if n!=1:raise HTTPException(409,'草稿版本已更新。')
    return public(read(oid,did))

@router.post('/api/assistant/drafts/{did}/execute')
async def execute(did:str,body:ExecuteBody,request:Request,background_tasks:BackgroundTasks):
    oid=owner(request);r=read(oid,did)
    if r['revision']!=body.revision:raise HTTPException(409,'参数版本已更新，请重新核对。')
    if r['status']=='submitted':return await detail(did,request)
    if r['status']!='prepared':raise HTTPException(409,'任务已提交或执行状态待核查，请勿重复提交。')
    data,binding=normalize(r['kind'],json.loads(r['payload']),request)
    if binding!=json.loads(r['binding']):raise HTTPException(409,'数据集、模型或权重已更新，请重新保存草稿并核对。')
    if r['kind']=='training':
        from assistant_workflow import validate_training_execution
        await validate_training_execution(oid,did,data)
    with db() as c:
        n=c.execute("UPDATE drafts SET status='submitting',updated=? WHERE id=? AND owner=? AND revision=? AND status='prepared'",(int(time.time()),did,oid,body.revision)).rowcount
        if n!=1:raise HTTPException(409,'任务提交中，请查询执行状态。')
    try:
        if r['kind']=='annotation':
            from assistant_annotation import ensure_models
            await ensure_models(data)
            _,latest_binding=normalize(r['kind'],data,request)
            if latest_binding!=binding:raise HTTPException(409,'模型或数据已更新，请重新核对。')
            result=await annotations.auto_annotate_text_batch(**data,preserve_existing=True,background_tasks=background_tasks)
        else:result=await training.start_training(TrainingConfig.model_validate(data),background_tasks,request)
    except HTTPException:
        with db() as c:c.execute("UPDATE drafts SET status='prepared' WHERE id=? AND owner=?",(did,oid))
        raise
    except Exception:
        with db() as c:c.execute("UPDATE drafts SET status='uncertain' WHERE id=? AND owner=?",(did,oid))
        raise HTTPException(503,'任务提交状态待核查，请查看任务列表。') from None
    with db() as c:c.execute("UPDATE drafts SET status='submitted',result=?,updated=? WHERE id=? AND owner=?",(json.dumps(result),int(time.time()),did,oid))
    return await detail(did,request)

@router.post('/api/assistant/drafts/{did}/control/{action}')
async def control(did:str,action:Literal['pause','resume','cancel','skip_training'],request:Request):
    oid=owner(request);r=read(oid,did);v=public(r)
    if action=='skip_training':
        with db() as c:c.execute("UPDATE drafts SET followup='skipped',updated=? WHERE id=? AND owner=?",(int(time.time()),did,oid))
        return await detail(did,request)
    if r['status']!='submitted':raise HTTPException(409,'尚未创建任务。')
    jid=(v['result'] or {}).get('job_id') or (v['result'] or {}).get('training_id')
    if r['kind']=='annotation':
        fn={'pause':annotations.pause_text_batch,'resume':annotations.resume_text_batch,'cancel':annotations.cancel_text_batch}[action]
        await fn(jid)
    else:
        if action=='pause':await training.pause_training(jid)
        elif action=='resume':await training.resume_training(jid)
        else:await training.stop_training(jid,request)
    return await detail(did,request)

from assistant_workflow import router as workflow_router
router.include_router(workflow_router)
