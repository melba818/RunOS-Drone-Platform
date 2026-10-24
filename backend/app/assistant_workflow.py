"""Persistent checkpoints around the application's native annotation/split/train UI."""
import json,re,time
from fastapi import APIRouter,HTTPException,Request
from pydantic import BaseModel,ConfigDict,Field,model_validator
from config import settings
from split_contract import split_counts

router=APIRouter()
def helpers():
    from routers import platform_assistant as a
    return a
def invalid(field,message):raise HTTPException(422,{'code':'WORKFLOW_VALIDATION','message':message,'errors':[{'field':field,'message':message}]})
def storage():return helpers().db()
def read(oid,fid):
    if not re.fullmatch('[a-f0-9]{32}',fid):raise HTTPException(404,'流程不存在。')
    with storage() as c:r=c.execute('SELECT * FROM assistant_flows WHERE id=? AND owner=?',(fid,oid)).fetchone()
    if not r:
        draft=helpers().read(oid,fid)
        if draft['kind']!='annotation':raise HTTPException(404,'标注流程不存在。')
        value={'id':fid,'stage':'annotation_ready','source_dataset_id':json.loads(draft['payload'])['dataset_id'],'annotation_draft':fid,'training_draft':None,'target_dataset_id':None,'split':None,'review':None}
        with storage() as c:c.execute('INSERT OR IGNORE INTO assistant_flows VALUES(?,?,1,?,?)',(fid,oid,json.dumps(value),int(time.time())))
        return read(oid,fid)
    return dict(r)
def public(row):return dict(json.loads(row['state']),revision=row['revision'])
def clean(v):return {k:x for k,x in v.items() if k not in {'revision','annotation_job','source_dataset','target_dataset','training_status','training_id','training_job'}}
def save(oid,fid,revision,value):
    with storage() as c:n=c.execute('UPDATE assistant_flows SET state=?,revision=revision+1,updated=? WHERE id=? AND owner=? AND revision=?',(json.dumps(value,ensure_ascii=False),int(time.time()),fid,oid,revision)).rowcount
    if n!=1:raise HTTPException(409,'流程已更新，请刷新后继续。')
    return public(read(oid,fid))
async def current(oid,fid):
    a=helpers();row=read(oid,fid);v=public(row);draft=a.public(a.read(oid,v['annotation_draft']))
    job={}
    if draft.get('result'):
        try:
            raw=await a.annotations.get_text_batch_status(draft['result']['job_id'])
            job={k:raw[k] for k in ['status','total','processed','annotated','failed','total_annotations','review_summary','dataset_id'] if k in raw}
        except HTTPException:job={'status':'unavailable'}
    if v['stage'] in {'annotation_ready','annotation_running','annotation_problem'}:
        stage='review_required' if job.get('status')=='done' else 'annotation_running' if job.get('status') in {'running','paused'} or draft['status']=='submitting' else 'annotation_problem' if job else 'annotation_ready'
        if stage!=v['stage']:
            v['stage']=stage
            try:v=save(oid,fid,row['revision'],clean(v))
            except HTTPException as e:
                if e.status_code!=409:raise
                v=public(read(oid,fid))
    v['annotation_job']=job;v['source_dataset']=safe_dataset(v['source_dataset_id'])
    if v.get('target_dataset_id'):v['target_dataset']=safe_dataset(v['target_dataset_id'])
    if v.get('training_draft'):
        d=a.public(a.read(oid,v['training_draft']));v['training_status']=d['status']
        if d.get('result'):
            v['training_id']=d['result'].get('training_id')
            try:
                raw=await a.training.get_training_status(v['training_id'])
                v['training_job']={k:raw[k] for k in ['status','progress','current_epoch','total_epochs','metrics'] if k in raw}
            except HTTPException:v['training_job']={'status':'unavailable'}
        status=(v.get('training_job') or {}).get('status')
        stage='training_done' if status=='completed' else 'training_running' if status in {'pending','queued','starting','running','pausing','paused','stopping'} or d['status']=='submitting' else 'training_problem' if status or d['status']=='uncertain' else 'training_ready'
        if stage!=v['stage']:
            v['stage']=stage
            try:v=save(oid,fid,v['revision'],clean(v))|{k:x for k,x in v.items() if k in {'training_status','training_id','training_job','annotation_job','source_dataset','target_dataset'}}
            except HTTPException as e:
                if e.status_code!=409:raise
                v=public(read(oid,fid))
    return v
def safe_dataset(ident):
    ds=settings.active_datasets.get(ident)
    return {k:v for k,v in dict(ds or {},id=ident).items() if k in {'id','name','format','task_type','num_images','num_annotations','classes','splits'}}
def images(ident):
    if ident not in settings.active_datasets:raise HTTPException(404,'流程数据集已不可用。')
    ds=settings.active_datasets[ident]
    return settings.dataset_parser.get_images_with_annotations(settings.DATASETS_DIR/ident,ds['format'],page=1,limit=999999)
class Review(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1)
    confirmed:bool
    exclude_pending:bool=False
    accept_partial:bool=False
class SplitPlan(BaseModel):
    model_config=ConfigDict(extra='forbid')
    train_ratio:float=Field(default=.7,ge=0,le=1,allow_inf_nan=False)
    val_ratio:float=Field(default=.2,ge=0,le=1,allow_inf_nan=False)
    test_ratio:float=Field(default=.1,ge=0,le=1,allow_inf_nan=False)
    output_name:str=Field(min_length=1,max_length=160,pattern=r'^[^/\\\x00-\x1f]+$')
    seed:int=Field(default=42,ge=0,le=2147483647)
    shuffle:bool=True
    stratify:bool=False
    @model_validator(mode='after')
    def coherent(self):
        if abs(self.train_ratio+self.val_ratio+self.test_ratio-1)>1e-6:raise ValueError('划分比例合计必须为 100%。')
        if self.stratify:raise ValueError('当前流程使用随机划分，不支持类别分层。')
        return self
class SplitBody(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1)
    data:SplitPlan
class BindTraining(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1)
    draft_id:str=Field(pattern='^[a-f0-9]{32}$')
class ReopenReview(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1)
@router.get('/api/assistant/flows/{fid}')
async def get_flow(fid:str,request:Request):return await current(helpers().owner(request),fid)
@router.post('/api/assistant/flows/{fid}/review')
async def review(fid:str,body:Review,request:Request):
    a=helpers();oid=a.owner(request);v=await current(oid,fid)
    if v['revision']!=body.revision:raise HTTPException(409,'流程已更新，请重新核对。')
    if v['stage'] not in {'review_required','split_ready'} or not body.confirmed:invalid('confirmed','请先在标注页检查效果并确认。')
    job=v['annotation_job'];pending=(job.get('review_summary') or {}).get('pending',0)
    if pending and not body.exclude_pending:invalid('exclude_pending',f'仍有 {pending} 条候选待复核。请处理后继续，或明确仅使用正式标签。')
    if job.get('failed',0) and not body.accept_partial:invalid('accept_partial','部分图片标注失败，请处理后继续，或确认使用当前结果。')
    rows=images(v['source_dataset_id']);formal=sum(len(i.get('annotations') or []) for i in rows)
    if formal==0:invalid('annotations','尚无正式标注，请调整标注参数或手动标注后再继续。')
    v['stage']='split_ready';v['review']={'confirmed_at':int(time.time()),'formal_annotations':formal,'pending_excluded':pending,'partial_accepted':body.accept_partial,'binding':a.dataset_binding(v['source_dataset_id'])}
    return save(oid,fid,body.revision,clean(v))
def split_preview(v,plan):
    ident=v['source_dataset_id'];ds=settings.active_datasets.get(ident,{})
    if not (v.get('review') and v['stage'] in {'split_ready','split_done','training_ready'}):invalid('review','请先确认标注效果。')
    if ds.get('format') not in {'yolo','yolov5','yolov8','yolov9','yolov10','yolov11','yolov12'}:invalid('dataset_id','此流程目前支持 YOLO 数据集划分，请先在格式转换页面处理。')
    try:settings.dataset_parser.validate_split_source(settings.DATASETS_DIR/ident)
    except ValueError:invalid('dataset_id','数据集存在重名或未纳入列表的图片，请检查文件名与 YAML 配置。')
    if v['review']['binding']!=helpers().dataset_binding(ident):raise HTTPException(409,'标注或数据已变化，请返回标注页重新确认效果。')
    count=len(images(ident));counts=split_counts(count,plan.train_ratio,plan.val_ratio,plan.test_ratio)
    if counts['train']<1 or counts['val']<1:invalid('ratios','训练集和验证集至少各需要一张图片，请调整比例。')
    return {'total':count,'counts':counts,'data':plan.model_dump(),'preserve_source':True}
@router.post('/api/assistant/flows/{fid}/split-preview')
async def preview(fid:str,body:SplitPlan,request:Request):return split_preview(await current(helpers().owner(request),fid),body)
@router.post('/api/assistant/flows/{fid}/split')
async def create_split(fid:str,body:SplitBody,request:Request):
    a=helpers();oid=a.owner(request);v=await current(oid,fid)
    if v['stage'] in {'split_done','training_ready'} and v.get('split',{}).get('data')==body.data.model_dump():return v
    if v['revision']!=body.revision:raise HTTPException(409,'流程已更新，请重新核对。')
    if v['stage']!='split_ready':invalid('stage','请先确认标注效果；如需重新划分，请从标注阶段继续。')
    plan=split_preview(v,body.data);state=clean(v);state['stage']='split_pending';state=save(oid,fid,body.revision,state)
    revision=state.pop('revision')
    try:
        from routers.datasets import split_dataset
        from schemas.common import SplitRequest
        result=await split_dataset(v['source_dataset_id'],SplitRequest(**body.data.model_dump()))
        target=result['new_dataset'];assert result['splits']==plan['counts']
        if a.dataset_binding(v['source_dataset_id'])!=v['review']['binding']:
            raise RuntimeError('Source changed while splitting')
        state.update(stage='split_done',target_dataset_id=target['id'],split={'data':body.data.model_dump(),'counts':result['splits'],'target_binding':a.dataset_binding(target['id']),'created_at':int(time.time())})
        save(oid,fid,revision,state)
    except HTTPException:
        state['stage']='split_ready';save(oid,fid,revision,state);raise
    except Exception:
        state['stage']='split_uncertain';save(oid,fid,revision,state)
        raise HTTPException(503,'划分状态待核查，请查看数据集列表，勿重复提交。') from None
    return await current(oid,fid)
@router.post('/api/assistant/flows/{fid}/bind-training')
async def bind_training(fid:str,body:BindTraining,request:Request):
    a=helpers();oid=a.owner(request);v=await current(oid,fid)
    if v['stage']=='training_ready' and v['training_draft']==body.draft_id:return v
    if body.revision!=v['revision']:raise HTTPException(409,'流程已更新，请重新读取。')
    if v['stage'] not in {'split_done','training_ready'}:invalid('stage','请先完成并确认数据划分。')
    draft=a.public(a.read(oid,body.draft_id))
    if draft['kind']!='training' or draft['data']['dataset_id']!=v['target_dataset_id']:invalid('dataset_id','训练须使用本流程划分后的数据集。')
    if draft['status']!='prepared':invalid('draft_id','请选择尚未提交的训练参数草稿。')
    if (v.get('split') or {}).get('target_binding')!=a.dataset_binding(v['target_dataset_id']):raise HTTPException(409,'划分后的数据已变化，请重新检查和划分。')
    v['training_draft']=body.draft_id;v['stage']='training_ready'
    return save(oid,fid,v['revision'],clean(v))

@router.post('/api/assistant/flows/{fid}/reopen-review')
async def reopen_review(fid:str,body:ReopenReview,request:Request):
    a=helpers();oid=a.owner(request);v=await current(oid,fid)
    if v['revision']!=body.revision:raise HTTPException(409,'流程已更新，请重新读取。')
    if v['stage'] not in {'split_ready','split_done','training_ready','training_problem','training_done'}:invalid('stage','请等待当前任务结束后再重新检查。')
    if v.get('training_draft') and a.read(oid,v['training_draft'])['status'] in {'submitting','uncertain'}:invalid('stage','训练提交状态待核查，请先在任务列表检查。')
    previous=v.get('previous_outputs',[])
    if v.get('target_dataset_id'):previous=previous+[{'dataset_id':v['target_dataset_id'],'training_draft':v.get('training_draft')}]
    v.update(stage='review_required',target_dataset_id=None,training_draft=None,split=None,review=None,previous_outputs=previous)
    return save(oid,fid,v['revision'],clean(v))

async def validate_training_execution(oid,did,data):
    """A training draft bound to a flow must retain that flow's reviewed output."""
    with storage() as c:rows=c.execute('SELECT * FROM assistant_flows WHERE owner=?',(oid,)).fetchall()
    for row in rows:
        value=public(row)
        if value.get('training_draft')!=did:continue
        v=await current(oid,value['id'])
        if v['stage']!='training_ready':invalid('stage','此流程尚未就绪或已提交，请检查当前任务。')
        if data['dataset_id']!=v['target_dataset_id']:invalid('dataset_id','请使用本流程划分后的数据集；更换数据集须重新准备流程。')
        if (v.get('split') or {}).get('target_binding')!=helpers().dataset_binding(v['target_dataset_id']):raise HTTPException(409,'划分后的图片或标注已变化，请重新检查和划分。')
    # Discarded prepared drafts cannot be executed after their flow is reopened.
    for row in rows:
        if any(output.get('training_draft')==did for output in public(row).get('previous_outputs',[])):
            invalid('draft_id','该草稿对应之前的划分，请使用当前流程的训练参数。')
