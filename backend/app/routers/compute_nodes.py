"""Loopback-only node management; SSH/SFTP never runs in the browser."""
import ipaddress
import re
from urllib.parse import urlparse
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from typing import Literal
from config import settings
from compute.store import CONNECTION_KEYS, NodeStore
from compute.remote import RemoteExecutionService


class SafeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def safe(request):
            try: return await handler(request)
            except RequestValidationError:
                # FastAPI's default error body may echo submitted secrets.
                return JSONResponse({'detail':'节点参数无效，请检查必填项、端口和环境路径'},status_code=422)
            except KeyError: return JSONResponse({'detail':'节点不存在'},status_code=404)
            except (ValueError,RuntimeError):
                return JSONResponse({'detail':'操作未完成，请检查凭据、节点配置或重新检测指纹'},status_code=400)
        return safe


def local_only(request: Request):
    try: local = ipaddress.ip_address(request.client.host).is_loopback
    except (ValueError,AttributeError): local=False
    origin=request.headers.get('origin')
    if origin:
        parsed=urlparse(origin)
        local=local and parsed.scheme in ('http','https') and parsed.hostname in ('localhost','127.0.0.1','::1') and parsed.port in (3000,None)
    if not local: raise HTTPException(403,'算力节点凭据操作仅允许本机 RunOS 访问')


router=APIRouter(prefix='/api/compute-nodes',tags=['Compute Nodes'],route_class=SafeRoute,dependencies=[Depends(local_only)])
store=NodeStore(settings.PLATFORM_COMPUTE_NODES_DIR)
service=RemoteExecutionService(store)


class Credential(BaseModel):
    password: SecretStr | None = None
    private_key: SecretStr | None = None
    passphrase: SecretStr | None = None
    jump_password: SecretStr | None = None
    jump_private_key: SecretStr | None = None
    jump_passphrase: SecretStr | None = None
    save: bool = True

    def secret(self, auth, prefix=''):
        field = lambda key: getattr(self, prefix + key)
        result={k:field(k).get_secret_value() for k in ('password','private_key','passphrase') if field(k) is not None}
        if not result.get('password' if auth=='password' else 'private_key'): raise ValueError('凭据不能为空')
        if any(len(v)>65536 for v in result.values()): raise ValueError('凭据过大')
        return result


def bundle(body):
    """打包成 {'target':…, 'jump':…}；未启用跳板时 jump 为 None。"""
    if not body.credential: return None
    return {'target': body.credential.secret(body.auth_type),
            'jump': body.credential.secret(body.jump_auth_type, 'jump_') if body.jump_enabled else None}


class NodeConfig(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    connection_type: Literal['ssh','api'] = 'ssh'
    host: str = Field(min_length=1,max_length=253)
    port: int = Field(default=22,ge=1,le=65535)
    username: str = Field(min_length=1,max_length=128)
    auth_type: Literal['password','key'] = 'password'
    workdir: str = Field(default='/tmp',min_length=1,max_length=512)
    environment_type: Literal['system','conda','venv'] = 'system'
    conda_env: str = Field(default='',max_length=100)
    conda_executable: str = Field(default='',max_length=512)
    venv_path: str = Field(default='',max_length=512)
    gpu_mode: Literal['auto','manual'] = 'auto'
    gpu_id: int | None = Field(default=None,ge=0,le=255)
    # 一级跳板（堡垒机）：本机 → jump → 目标机。目标机地址由跳板机解析，可填内网 IP/主机名。
    jump_enabled: bool = False
    jump_host: str = Field(default='',max_length=253)
    jump_port: int = Field(default=22,ge=1,le=65535)
    jump_username: str = Field(default='',max_length=128)
    jump_auth_type: Literal['password','key'] = 'password'

    @field_validator('host','jump_host')
    @classmethod
    def host_safe(cls,v,info):
        if info.field_name=='jump_host' and v=='': return v
        if not re.fullmatch(r'[A-Za-z0-9_.:\-]+',v) or v.startswith('-'): raise ValueError('地址无效')
        return v

    @model_validator(mode='after')
    def complete(self):
        for value in (self.name,self.username,self.workdir,self.conda_env,self.venv_path,self.conda_executable,self.jump_username):
            if any(ord(c)<32 for c in value): raise ValueError('字段包含控制字符')
        if self.jump_enabled:
            if not self.jump_host or not self.jump_username: raise ValueError('启用了跳板机时，跳板机地址与用户名必填')
            if self.jump_host==self.host and self.jump_port==self.port: raise ValueError('跳板机不能指向目标服务器本身')
        if not self.workdir.startswith('/'): raise ValueError('工作目录须为绝对路径')
        if self.environment_type=='conda' and not re.fullmatch(r'[A-Za-z0-9_.-]+',self.conda_env): raise ValueError('Conda 环境名称无效')
        if self.environment_type=='venv' and not self.venv_path.startswith('/'): raise ValueError('venv 路径无效')
        if self.conda_executable and not self.conda_executable.startswith('/'): raise ValueError('Conda 可执行文件须为绝对路径')
        if self.gpu_mode=='manual' and self.gpu_id is None: raise ValueError('请选择 GPU')
        return self


class NodeWrite(NodeConfig):
    credential: Credential | None = None
    node_id: str | None = None

    def config(self): return self.model_dump(exclude={'credential','node_id'})


class Trust(BaseModel):
    target: str = Field(max_length=280)
    fingerprint: str = Field(max_length=100)


def save(body,id=None):
    if id: ensure_idle(id)
    return store.save(body.config(),id,bundle(body),body.credential.save if body.credential else True)


def required(node, parts, action):
    """跑 SSH 前确保两跳凭据齐全，避免把'没录凭据'报成'认证失败'。"""
    missing=[]
    if not parts.get('target'): missing.append('目标服务器')
    if node.get('jump_enabled') and not parts.get('jump'): missing.append('跳板机')
    if missing: raise HTTPException(422,f'缺少{"、".join(missing)}凭据，请重新输入后再{action}')


def ensure_idle(id):
    if any(j.get('config',{}).get('compute_node_id')==id and j.get('status') in ('running','starting','interrupted') for j in settings.training_manager.training_jobs.values()):
        raise HTTPException(409,'此节点仍有关联远程训练任务，请先处理任务再修改节点或凭据')


@router.get('')
def list_nodes(): return {'nodes':[store.public(n) for n in store.list()]}


@router.post('')
def create_node(body:NodeWrite): return save(body)


@router.post('/test-draft')
def test_draft(body:NodeWrite):
    node=body.config()
    if body.credential: secret=bundle(body)
    elif body.node_id:
        saved=store.get(body.node_id)
        if any(node[k]!=saved[k] for k in CONNECTION_KEYS): raise HTTPException(422,'连接信息（含跳板机）已变更，请重新输入凭据')
        secret=store.bundle(saved)
        node['id']=body.node_id
        required(saved, secret, '测试连接')
    else: raise HTTPException(422,'请先输入 SSH 凭据')
    result=service.probe(node,secret)
    if body.node_id:
        saved=store.get(body.node_id)
        if all(node[k]==saved[k] for k in NodeConfig.model_fields):
            store.report(body.node_id,result)
    return result


@router.post('/trust')
def trust_host(body:Trust):
    service.trust(body.target,body.fingerprint)
    return {'success':True}


@router.put('/{id}')
def update_node(id:str,body:NodeWrite): return save(body,id)


@router.delete('/{id}')
def delete_node(id:str):
    ensure_idle(id)
    store.delete(id)
    return {'success':True}


@router.post('/{id}/credential')
@router.put('/{id}/credential')
def replace_credential(id:str,body:Credential):
    """换凭据：跳板机与目标机捆绑在同一份 blob，更新时必须整套给，否则会留下半残凭据。"""
    ensure_idle(id)
    node=store.get(id)
    config={k:node[k] for k in NodeConfig.model_fields}
    parts={'target':body.secret(node['auth_type']),
           'jump':body.secret(node['jump_auth_type'],'jump_') if node.get('jump_enabled') else None}
    if node.get('jump_enabled') and not parts['jump']: raise HTTPException(422,'此节点通过跳板机连接，请同时填写跳板机凭据')
    return store.save(config,id,parts,body.save)


@router.post('/{id}/test')
def test_node(id:str):
    node=store.get(id)
    try: parts=store.bundle(node)
    except RuntimeError: raise HTTPException(422,'凭据不可用，请更换凭据；加密凭据须由原 Windows 账户读取')
    required(node, parts, '测试连接')
    result=service.probe(node,parts)
    store.report(id,result)
    return result


@router.get('/{id}/gpu')
def node_gpu(id:str):
    node=store.get(id)
    return {'checked_at':node['last_check_time'],'gpus':(node.get('report') or {}).get('gpus',[])}


@router.get('/{id}/environment')
def node_environment(id:str):
    node=store.get(id)
    return {'checked_at':node['last_check_time'],'environment':(node.get('report') or {}).get('environment',{})}
