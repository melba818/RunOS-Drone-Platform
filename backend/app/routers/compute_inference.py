import re
from fastapi import APIRouter,Depends,Request,HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from routers.compute_nodes import local_only
from compute.remote_inference import nodes

router=APIRouter(prefix='/api/compute-inference',tags=['Remote Inference'],dependencies=[Depends(local_only)])

@router.get('/history')
def history():
    from compute.inference_history import listing
    return {'jobs':listing()}

@router.post('/history')
async def remember(request:Request):
    from compute.inference_history import save
    data=await request.json()
    node=str(data.get('node_id','local'));job=str(data.get('job_id',''))
    if not re.fullmatch('[a-zA-Z0-9_-]{1,64}',node) or not re.fullmatch('[a-zA-Z0-9_-]{1,64}',job):
        raise HTTPException(422,'任务标识无效')
    raw=data.get('snapshot') or {}
    snapshot={k:raw[k] for k in ('id','status','progress','message','frames','processed','total_frames','has_tracks','proc_fps','eta','logs') if k in raw}
    if len(str(snapshot))>100000:raise HTTPException(413,'任务记录过大')
    return save(node,job,snapshot)


@router.get('/{node_id}/uploads/{upload_id}')
def upload_status(node_id:str,upload_id:str):
    state=nodes.transfers.get((node_id,upload_id))
    return dict(state) if state else {'stage':'等待本机接收完成','sent':0,'total':0}


@router.api_route('/{node_id}/api/infer/{path:path}',methods=['GET','POST','DELETE'])
async def proxy(node_id:str,path:str,request:Request):
    allowed=(request.method=='GET' and (path=='classes' or path=='smart/catalog' or path=='smart/models' or re.fullmatch(r'smart/load-status/[a-zA-Z0-9_.-]+',path) or re.fullmatch(r'video/[a-zA-Z0-9_-]+/(status|result|tracks/(mot|json))',path))) or (request.method=='POST' and path in ('image','frame','video','smart/image','smart/frame','smart/load')) or (request.method=='DELETE' and re.fullmatch(r'session/[a-zA-Z0-9:_-]+',path))
    if not allowed:raise HTTPException(404,'远程推理接口不可用')
    try:
        if 'multipart/form-data' in request.headers.get('content-type',''):
            async with request.form() as form:
                values={};uploads=[]
                for k,v in form.multi_items():
                    if not re.fullmatch('[a-z_]+',k):raise HTTPException(422,'表单字段无效')
                    if isinstance(v,UploadFile):uploads.append((k,v))
                    else:values[k]=v
                return await run_in_threadpool(nodes.forward,node_id,request.method,path,dict(request.query_params),values,uploads)
        values=await request.json() if request.method=='POST' else {}
        return await run_in_threadpool(nodes.forward,node_id,request.method,path,dict(request.query_params),values)
    except HTTPException:raise
    except KeyError:raise HTTPException(422,'算力节点不存在')
    except Exception:
        raise HTTPException(502,'远程推理连接失败，请检查 SSH 凭据、端口转发权限和服务器环境；未回退本机')
