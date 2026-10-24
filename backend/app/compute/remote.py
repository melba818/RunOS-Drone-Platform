import base64
import csv
import hashlib
import io
import json
import posixpath
import shlex
import socket
import threading
import time
from contextlib import contextmanager

import paramiko
from .store import now


class HostTrustRequired(Exception):
    def __init__(self, info):
        self.info = info


class HopFailure(ValueError):
    """某一跳连不上。继承 ValueError —— 现有 400/错误收集逻辑照旧生效，不会漏成 500。"""

    def __init__(self, hop, message):
        super().__init__(message)
        self.hop = hop


HOP_LABEL = {'jump': '跳板机', 'target': '内网服务器'}


ENV_SCRIPT = '''import json,sys,importlib
r={"python":sys.version.split()[0]}
try:
 import torch
 r.update(torch=torch.__version__,cuda=torch.version.cuda,gpu_available=torch.cuda.is_available())
except ImportError: r["torch_error"]="PyTorch 未安装"
except Exception: r["torch_error"]="PyTorch 导入失败"
try:
 import rfdetr
 r["rfdetr"]="已安装"
except ImportError: r["rfdetr_error"]="RF-DETR 未安装或依赖缺失"
except Exception: r["rfdetr_error"]="RF-DETR 导入失败"
try:
 import transformers
 r["transformers"]=transformers.__version__
except Exception: pass
print("RUNOS_ENV="+json.dumps(r))
'''

#: 除了节点自己配的那个解释器，还顺带扫一遍工作目录里的 venv 布局。
#: 一台多人共用的 GPU 机器上，RF-DETR 与 VLX-Seek **必须各成一个独立环境**
#: （torch 版本互不兼容：前者 2.6.0+cu124，后者 2.10.0+cu126）。只报「主环境」
#: 会让卡片把两个都显示成未安装，看着像没部署。
#: **只读**：这里只发现 + 逐个 import 探测，不创建任何东西。
def _venv_discover_script(workdir: str) -> str:
    wd = workdir.rstrip('/') or '/tmp'
    # 结尾必须 `exit 0`：循环里最后一条 `[ -x ]` 在最后一个候选目录不存在时返回 1，
    # 会把「已经打印出来的有效路径」连带判成整条命令失败（实测踩过：两个 venv 都
    # 打印了却因为 .venv 不存在而 exit=1）。
    inner = ('for d in {wd}/envs/* {wd}/venv {wd}/.venv; do '
             '[ -x "$d/bin/python" ] && echo "$d/bin/python"; done; exit 0').format(wd=wd)
    return 'sh -c ' + shlex.quote(inner)


def _venv_name(path: str) -> str:
    """/x/y/envs/rfdetr/bin/python -> rfdetr"""
    return posixpath.basename(posixpath.dirname(posixpath.dirname(path))) or path


MAX_DISCOVERED_VENVS = 4


class RemoteExecutionService:
    def __init__(self, store):
        self.store = store
        self.candidates = {}
        self.lock = threading.RLock()
        self.busy = set()
        self.slots = threading.BoundedSemaphore(4)

    @staticmethod
    def endpoint(host, port): return f'[{host}]:{port}'

    @staticmethod
    def target(node): return f"[{node['host']}]:{node['port']}"

    @staticmethod
    def jump_target(node): return f"[{node['jump_host']}]:{node['jump_port']}"

    def trust(self, target, fingerprint):
        with self.lock:
            candidate = self.candidates.get(target)
            if not candidate or candidate['fingerprint'] != fingerprint or time.monotonic()-candidate['time'] > 600:
                raise ValueError('指纹确认已过期，请重新测试连接')
            self.store.trust(target, candidate['algorithm'], candidate['key'], fingerprint)
            del self.candidates[target]

    @staticmethod
    def _auth(auth_type, secret, hop):
        label = HOP_LABEL.get(hop, '')
        if auth_type == 'password':
            if not secret or not secret.get('password'): raise HopFailure(hop, f'{label}凭据缺失，请重新输入密码')
            return {'password': secret['password']}
        for cls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
            try:
                return {'pkey': cls.from_private_key(io.StringIO((secret or {}).get('private_key') or ''), password=(secret or {}).get('passphrase') or None)}
            except (paramiko.SSHException, ValueError): pass
        raise HopFailure(hop, f'{label} SSH Key 格式不支持或密钥口令错误')

    @staticmethod
    def failure(hop, exc):
        """把 paramiko 异常翻成能定位到具体哪一跳的中文提示。"""
        label = HOP_LABEL.get(hop, '服务器')
        if isinstance(exc, HopFailure): return exc
        if isinstance(exc, paramiko.AuthenticationException): return HopFailure(hop, f'{label} SSH 认证失败：用户名、密码或 SSH Key 错误')
        if isinstance(exc, paramiko.BadHostKeyException): return HopFailure(hop, f'{label} SSH Host Key 已变化，已拒绝连接，请联系管理员核实服务器身份')
        if isinstance(exc, (socket.timeout, TimeoutError)): return HopFailure(hop, f'{label}连接超时：请检查网络与服务器负载')
        if isinstance(exc, (socket.gaierror, ConnectionError, OSError)): return HopFailure(hop, f'{label}连接失败：请检查地址、端口、防火墙或服务器是否在线')
        if isinstance(exc, EOFError): return HopFailure(hop, f'{label} SSH 会话未建立')
        return HopFailure(hop, f'{label} SSH 连接失败：{exc}')

    def hop(self, endpoint, host, port, username, auth_type, secret, hop, sock=None):
        """建立单跳 SSH。sock 有值时走已有通道转发（第二跳）。"""
        client = paramiko.SSHClient()
        saved = self.store.host_key(endpoint)
        if saved:
            key = paramiko.PKey.from_type_string(saved[0], base64.b64decode(saved[1]))
            host_id = host if port == 22 else endpoint
            client.get_host_keys().add(host_id, saved[0], key)
        service = self
        class Verify(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key):
                if saved:
                    raise paramiko.BadHostKeyException(hostname, key, paramiko.PKey.from_type_string(saved[0], base64.b64decode(saved[1])))
                fp = 'SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
                candidate = dict(target=endpoint, fingerprint=fp, algorithm=key.get_name(), key=key.get_base64(), time=time.monotonic(), hop=hop)
                with service.lock: service.candidates[endpoint] = candidate
                raise HostTrustRequired({k: candidate[k] for k in ('target','fingerprint','algorithm','hop')})
        client.set_missing_host_key_policy(Verify())
        try:
            auth = self._auth(auth_type, secret, hop)
            client.connect(host, port=port, username=username,
                           timeout=10, banner_timeout=10, auth_timeout=15, allow_agent=False, look_for_keys=False,
                           **({'sock': sock} if sock else {}), **auth)
            transport = client.get_transport()
            if transport is None or not transport.is_active(): raise EOFError('SSH transport inactive')
            transport.set_keepalive(30)
            return client
        except BaseException as exc:
            client.close()
            if isinstance(exc, HostTrustRequired): raise
            raise self.failure(hop, exc) from exc

    @contextmanager
    def connect(self, node, secret):
        if node['connection_type'] != 'ssh': raise ValueError('API 连接尚未实现')
        parts = secret if isinstance(secret, dict) and ('target' in secret or 'jump' in secret) else {'target': secret, 'jump': None}
        if not node.get('jump_enabled'):
            with self.hop(self.target(node), node['host'], node['port'], node['username'], node['auth_type'], parts.get('target'), 'target') as client:
                yield client
            return
        jump_host, jump_port = node['jump_host'], node['jump_port']
        with self.hop(self.jump_target(node), jump_host, jump_port, node['jump_username'], node['jump_auth_type'], parts.get('jump'), 'jump') as jump:
            # 第二跳复用跳板机的 transport 开一条 direct-tcpip 通道，等价于 ssh -W / ProxyJump。
            transport = jump.get_transport()
            try:
                channel = transport.open_channel('direct-tcpip', (node['host'], node['port']), ('127.0.0.1', 0), timeout=10)
            except Exception as exc:
                raise HopFailure('jump', f'跳板机无法连通 {self.target(node)}：请确认跳板机能访问该内网地址与端口，并检查其 sshd 是否允许端口转发') from exc
            try:
                with self.hop(self.target(node), node['host'], node['port'], node['username'], node['auth_type'], parts.get('target'), 'target', sock=channel) as client:
                    yield client
            finally:
                try: channel.close()
                except Exception: pass

    @staticmethod
    def execute(client, command, timeout=20):
        """Bounded command execution; no user-supplied shell snippets exposed by HTTP."""
        transport = client.get_transport()
        if transport is None or not transport.is_active(): raise EOFError('SSH transport inactive')
        channel = transport.open_session(timeout=8)
        channel.settimeout(8)
        output, error = bytearray(), bytearray()
        try:
            channel.exec_command(command)
            deadline = time.monotonic()+timeout
            while True:
                if channel.recv_ready(): output.extend(channel.recv(32768))
                if channel.recv_stderr_ready(): error.extend(channel.recv_stderr(32768))
                if len(output)+len(error)>1024*1024: raise ValueError('检测输出过大，已停止读取')
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready(): break
                if time.monotonic()>deadline: raise TimeoutError('远程检测命令超时')
                time.sleep(.02)
            return channel.recv_exit_status(), output.decode('utf-8','replace'), error.decode('utf-8','replace')
        finally: channel.close()

    @contextmanager
    def sftp(self, node, secret):
        with self.connect(node, secret) as client:
            transport=client.get_transport()
            if transport is None or not transport.is_active(): raise EOFError('SSH transport inactive')
            session = client.open_sftp()
            session.get_channel().settimeout(30)
            try: yield session
            finally: session.close()

    def upload(self, node, secret, local_path, remote_path):
        with self.sftp(node, secret) as ftp: ftp.put(str(local_path), remote_path)

    def download(self, node, secret, remote_path, local_path):
        with self.sftp(node, secret) as ftp: ftp.get(remote_path, str(local_path))

    @staticmethod
    def python_command(node):
        q = shlex.quote
        script = ' -c '+q(ENV_SCRIPT)
        if node['environment_type']=='venv': return q(node['venv_path'].rstrip('/')+'/bin/python')+script
        if node['environment_type']=='conda':
            conda = node.get('conda_executable') or 'conda'
            cmd = q(conda)+' run --no-capture-output -n '+q(node['conda_env'])+' python'+script
            # Login shell discovers normal Conda installations; explicit executable handles others.
            return 'bash -lc '+q(cmd)
        return "sh -c "+q('(command -v python3 >/dev/null && exec python3'+script+') || exec python'+script)

    def _probe_venvs(self, client, node, result):
        """把 workdir 下的独立 venv 逐个探一遍，填进 result['venvs']。

        每台机器只跑 get_config 的十几秒(discover) + 每个环境一次 import；
        任何一环超时/失败都只记 error，**不往 result['errors'] 里塞** ——
        那是"节点不可用"的告警位， Aux 环境没装 torch 不代表这台机器不能用。
        """
        venvs=[]
        paths=[]
        try:
            code,out,_=self.execute(client,_venv_discover_script(node['workdir']),timeout=20)
            # 有输出就以输出为准（部分 glob 不匹配会让退出码非 0，那是正常的）
            paths=[l.strip() for l in out.splitlines() if l.strip()]
        except Exception as exc:  # noqa: BLE001
            result.setdefault('venvs_note',f'venv 发现失败：{exc}')
        seen=set()
        for path in paths[:MAX_DISCOVERED_VENVS]:
            root=posixpath.dirname(posixpath.dirname(path))
            if root in seen: continue
            seen.add(root)
            item={'name':_venv_name(path),'path':path,'active':path.rstrip('/').endswith(
                (node.get('venv_path') or '#').rstrip('/')+'/bin/python')}
            try:
                code,out,_=self.execute(client,shlex.quote(path)+' -c '+shlex.quote(ENV_SCRIPT),timeout=90)
                lines=[l[len('RUNOS_ENV='):] for l in out.splitlines() if l.startswith('RUNOS_ENV=')]
                item.update(json.loads(lines[-1]) if code==0 and lines else {'python_error':'Python 启动失败'})
            except Exception as exc:  # noqa: BLE001
                item['python_error']=str(exc)
            venvs.append(item)
        result['venvs']=venvs
        if venvs:
            ready=[v['name'] for v in venvs if v.get('torch')]
            # 主环境没 torch、但某个 venv 里有 —— 明确提示，别让人以为机器白装了
            if ready and not result['environment'].get('torch'):
                result['environment']['torch_note']='主解释器无 PyTorch；已就绪的虚拟环境：'+', '.join(ready)

    @staticmethod
    def _vlx_service_state(client, node):
        """/home/.../runos/services/vlxseek15/server.py 的进程状态（只读）。"""
        wd=node['workdir'].rstrip('/') or '/tmp'
        inner=('if pgrep -f "services/vlxseek15/server.py" >/dev/null 2>&1; then echo running; '
               'elif [ -f %s/services/vlxseek15/server.py ]; then echo deployed; else echo absent; fi') % wd
        try:
            code,out,_=RemoteExecutionService.execute(client,'sh -c '+shlex.quote(inner),timeout=20)
            state=(out.strip() if code==0 else '') or 'unknown'
        except Exception:  # noqa: BLE001
            state='unknown'
        return state if state in ('running','deployed','absent') else 'unknown'

    def probe(self, node, secret):
        key = node.get('id') or self.target(node)
        with self.lock:
            if key in self.busy: raise ValueError('此节点正在检测，请稍候')
            if not self.slots.acquire(blocking=False): raise ValueError('检测任务繁忙，请稍后重试')
            self.busy.add(key)
        jump = node.get('jump_enabled')
        result = dict(status='offline', checked_at=now(), connected=False, gpus=[], environment={}, errors=[], jump=None)
        if jump:
            result['jump'] = dict(target=self.jump_target(node), endpoint=f"{node['jump_username']}@{node['jump_host']}:{node['jump_port']}",
                                  connected=False)
        try:
            with self.connect(node, secret) as client:
                result.update(status='online',connected=True)
                if jump: result['jump'].update(connected=True)
                code,out,_ = self.execute(client,'uname -a')
                result['system'] = out.strip() if code==0 else '未检测'
                code,_,_ = self.execute(client, 'test -d '+shlex.quote(node['workdir']))
                result['workdir_exists'] = code==0
                if code: result['errors'].append('工作目录不存在或不可访问（未自动创建）')
                code,out,_ = self.execute(client,'nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits')
                if code==0:
                    for row in csv.reader(io.StringIO(out)):
                        if len(row)!=7: continue
                        def num(v):
                            try: return float(v.strip())
                            except ValueError: return None
                        result['gpus'].append(dict(index=int(row[0]),uuid=row[1].strip(),name=row[2].strip(),memory_total=num(row[3]),memory_used=num(row[4]),utilization=num(row[5]),temperature=num(row[6]),process_count=None))
                    code,out,_ = self.execute(client,'nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits')
                    if code==0:
                        rows=list(csv.reader(io.StringIO(out)))
                        for gpu in result['gpus']: gpu['process_count']=len({r[1].strip() for r in rows if len(r)==2 and r[0].strip()==gpu['uuid']})
                else: result['errors'].append('nvidia-smi 不可用，GPU 未检测')
                if node['gpu_mode']=='manual' and node['gpu_id'] not in [g['index'] for g in result['gpus']]:
                    result['errors'].append('所选 GPU 编号未检测到')
                code,out,_ = self.execute(client,self.python_command(node),timeout=60)
                lines=[line[len('RUNOS_ENV='):] for line in out.splitlines() if line.startswith('RUNOS_ENV=')]
                if code==0 and lines:
                    result['environment']=json.loads(lines[-1])
                    env=result['environment']
                    result['errors'].extend(env[k] for k in ('torch_error','rfdetr_error') if k in env)
                    if env.get('gpu_available') is False: result['errors'].append('GPU 不可用：torch.cuda.is_available() = False')
                else:
                    kind=node['environment_type']
                    result['errors'].append('Conda 不可用、环境不存在或 Python 启动失败：'+node['conda_env'] if kind=='conda' else 'Python 环境不存在或无法执行')
                self._probe_venvs(client,node,result)
                result['vlx_seek_service']=self._vlx_service_state(client,node)
        except HostTrustRequired as exc:
            result.update(status='untrusted',trust_required=exc.info)
            # 两跳共用一份信任论证：先过了哪一跳就标哪一跳已连通
            if result['jump']: result['jump']['connected'] = exc.info.get('hop') == 'target'
            label = HOP_LABEL.get(exc.info.get('hop'), '服务器')
            result['errors'].append(f'首次连接{label}：请向管理员核对 SSH 指纹后信任')
        except HopFailure as exc:
            # 已定位到跳：失败的那一跳之前都算连上了。
            if result['jump']: result['jump']['connected'] = exc.hop == 'target'
            result['errors'].append(str(exc))
        except paramiko.BadHostKeyException:
            result['errors'].append('SSH Host Key 已变化，已拒绝连接。请联系管理员核实服务器身份。')
        except paramiko.AuthenticationException:
            result['errors'].append('SSH 认证失败：用户名、密码或 SSH Key 错误')
        except (socket.timeout,TimeoutError): result['errors'].append('服务器或检测命令超时')
        except (socket.gaierror,ConnectionError,OSError): result['errors'].append('SSH 连接失败：检查地址、端口、网络或服务器是否在线')
        except ValueError as exc: result['errors'].append(str(exc))
        except Exception: result['errors'].append('SSH 协议或环境检测失败，请检查服务配置')
        finally:
            with self.lock: self.busy.discard(key)
            self.slots.release()
        return result
