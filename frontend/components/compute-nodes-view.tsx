'use client'

import { useEffect, useState } from 'react'
import { Server, Plus, RefreshCw, ShieldCheck, Loader2, ArrowRight, Route, KeyRound } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

type Config = {
  name: string; connection_type: 'ssh' | 'api'; host: string; port: number; username: string
  auth_type: 'password' | 'key'; workdir: string; environment_type: 'system' | 'conda' | 'venv'
  conda_env: string; conda_executable: string; venv_path: string; gpu_mode: 'auto' | 'manual'; gpu_id: number | null
  // 一级跳板（堡垒机）：本机 → 跳板机 → 内网服务器
  jump_enabled: boolean; jump_host: string; jump_port: number; jump_username: string; jump_auth_type: 'password' | 'key'
}
type GPU = { index: number; name: string; memory_total: number | null; memory_used: number | null; utilization: number | null; temperature: number | null; process_count: number | null }
type Hop = { target: string; endpoint: string; connected: boolean }
type Report = { status: string; connected: boolean; checked_at: string; system?: string; workdir_exists?: boolean; errors: string[]; gpus: GPU[]
  environment: { python?: string; torch?: string; cuda?: string; gpu_available?: boolean; rfdetr?: string; transformers?: string; torch_error?: string; rfdetr_error?: string; torch_note?: string }
  venvs?: Array<{ name: string; path: string; active?: boolean; python?: string; torch?: string; cuda?: string; gpu_available?: boolean; rfdetr?: string; transformers?: string; torch_error?: string; rfdetr_error?: string; python_error?: string }>
  trust_required?: { target: string; fingerprint: string; algorithm: string; hop?: 'jump' | 'target' }
  jump?: Hop | null
}
type Node = Config & { id: string; status: string; credential_saved: boolean; credential_available: boolean
  jump_credential_saved: boolean; jump_credential_available: boolean; last_check_time: string | null; report: Report | null }
const empty: Config = { name: '', connection_type: 'ssh', host: '', port: 22, username: '', auth_type: 'password', workdir: '/data/runos', environment_type: 'system', conda_env: '', conda_executable: '', venv_path: '', gpu_mode: 'auto', gpu_id: null,
  jump_enabled: false, jump_host: '', jump_port: 22, jump_username: '', jump_auth_type: 'password' }
const selectClass = 'h-9 w-full rounded-md border bg-background px-3 text-sm disabled:opacity-50'
const stateName: Record<string,string> = { online: '在线', offline: '离线', unknown: '未检测', untrusted: '待确认指纹' }
// 改了这些就必须重录凭据 —— 与后端 CONNECTION_KEYS 一致
const CONNECTION_KEYS = ['host','port','username','auth_type','jump_enabled','jump_host','jump_port','jump_username','jump_auth_type']

function Section({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="space-y-2 pt-1">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[10px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
        {hint && <p className="text-[10px] text-muted-foreground">{hint}</p>}
      </div>
      <div className="h-px bg-border" />
    </div>
  )
}

function HopRow({ label, endpoint, ok, last }: { label: string; endpoint: string; ok: boolean; last?: boolean }) {
  return (
    <div className="flex items-start gap-2">
      <div className="flex flex-col items-center pt-0.5">
        <span className={`size-2 rounded-full ${ok ? 'bg-primary' : 'bg-muted-foreground/40'}`} />
        {!last && <span className="my-0.5 w-px flex-1 bg-border" />}
      </div>
      <div className="min-w-0 flex-1 pb-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs">{label}</span>
          <span className={`text-[10px] ${ok ? 'text-primary' : 'text-muted-foreground'}`}>{ok ? '已连通' : '未连通'}</span>
        </div>
        <p className="truncate font-mono text-[10px] text-muted-foreground">{endpoint}</p>
      </div>
    </div>
  )
}

function CredentialFields({ title, auth, password, privateKey, passphrase, onPassword, onKey, onPassphrase }: {
  title: string; auth: 'password' | 'key'; password: string; privateKey: string; passphrase: string
  onPassword: (v: string) => void; onKey: (v: string) => void; onPassphrase: (v: string) => void
}) {
  return (
    <div className="space-y-2 rounded-md border border-dashed p-2.5">
      <p className="flex items-center gap-1.5 text-xs font-medium"><KeyRound className="size-3" />{title}</p>
      {auth === 'password'
        ? <Input type="password" autoComplete="new-password" value={password} onChange={e => onPassword(e.target.value)} placeholder="密码" />
        : <>
            <textarea autoComplete="off" spellCheck={false} className="w-full h-24 rounded-md border bg-background p-2 font-mono text-xs" value={privateKey} onChange={e => onKey(e.target.value)} placeholder="粘贴 PEM / OpenSSH 私钥" />
            <Input type="password" autoComplete="new-password" value={passphrase} onChange={e => onPassphrase(e.target.value)} placeholder="密钥口令（可选）" />
          </>}
    </div>
  )
}

export function ComputeNodesView({ apiUrl = 'http://localhost:8000' }: { apiUrl?: string }) {
  const [nodes, setNodes] = useState<Node[]>([])
  const [selected, setSelected] = useState<Node | null>(null)
  const [form, setForm] = useState<Config>({ ...empty })
  const [report, setReport] = useState<Report | null>(null)
  const [password, setPassword] = useState('')
  const [privateKey, setPrivateKey] = useState('')
  const [passphrase, setPassphrase] = useState('')
  const [jumpPassword, setJumpPassword] = useState('')
  const [jumpKey, setJumpKey] = useState('')
  const [jumpPassphrase, setJumpPassphrase] = useState('')
  const [saveCredential, setSaveCredential] = useState(true)
  const [replace, setReplace] = useState(true)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [trustChecked, setTrustChecked] = useState(false)

  async function api(path = '', method = 'GET', body?: unknown) {
    const response = await fetch(`${apiUrl}/api/compute-nodes${path}`, {
      method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(150000),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请检查配置')
    return data
  }
  async function reload() { const data = await api(); setNodes(data.nodes); return data.nodes as Node[] }
  useEffect(() => { let active = true; api().then(data => { if (active) setNodes(data.nodes) }).catch(e => { if(active) setError(e.message) }); return () => { active = false } }, [apiUrl])
  function clearCredentials() {
    setPassword(''); setPrivateKey(''); setPassphrase(''); setJumpPassword(''); setJumpKey(''); setJumpPassphrase('')
  }
  function choose(node: Node | null) {
    setSelected(node); setForm(node ? { ...node } : { ...empty }); setReport(node?.report ?? null)
    clearCredentials()
    // 任一一跳缺凭据就必须重录整套
    setReplace(!node?.credential_available || Boolean(node?.jump_enabled && !node?.jump_credential_available))
    setSaveCredential(true); setMessage(''); setError(''); setTrustChecked(false)
  }
  function edit<K extends keyof Config>(key: K, value: Config[K]) {
    setForm(previous => ({ ...previous, [key]: value })); setReport(null); setTrustChecked(false)
    if (CONNECTION_KEYS.includes(key)) setReplace(true)
  }
  function jumpCredentialMissing() {
    if (!form.jump_enabled) return false
    return form.jump_auth_type === 'password' ? !jumpPassword : !jumpKey
  }
  function payload() {
    const config = Object.fromEntries(Object.keys(empty).map(k => [k, form[k as keyof Config]]))
    if (form.connection_type !== 'ssh') throw new Error('API 连接将在后续版本提供')
    if (!form.name.trim() || !form.host.trim() || !form.username.trim()) throw new Error('请填写节点名称、服务器地址和用户名')
    if (form.jump_enabled && (!form.jump_host.trim() || !form.jump_username.trim())) throw new Error('请填写跳板机地址与用户名')
    if (form.jump_enabled && form.jump_host === form.host && form.jump_port === form.port) throw new Error('跳板机不能与内网服务器是同一台')
    if (replace) {
      if (!(form.auth_type === 'password' ? password : privateKey)) throw new Error('请输入内网服务器的 SSH 凭据')
      if (jumpCredentialMissing()) throw new Error('请输入跳板机的 SSH 凭据')
    }
    return { ...config, node_id: selected?.id,
      ...(replace ? { credential: { ...(form.auth_type === 'password' ? { password } : { private_key: privateKey, passphrase }),
        ...(form.jump_enabled ? (form.jump_auth_type === 'password' ? { jump_password: jumpPassword } : { jump_private_key: jumpKey, jump_passphrase: jumpPassphrase }) : {}),
        save: saveCredential } } : {}) }
  }
  async function action(fn: () => Promise<void>) {
    if (busy) return
    setBusy(true); setError(''); setMessage('')
    try { await fn() } catch(e) { setError(e instanceof Error ? e.message : '请求失败') } finally { setBusy(false) }
  }
  async function test() {
    await action(async () => {
      const result = await api('/test-draft','POST',payload()); setReport(result); setTrustChecked(false)
      if (selected) await reload()
    })
  }
  async function save() {
    await action(async () => {
      const node = await api(selected ? `/${selected.id}` : '',selected ? 'PUT' : 'POST',payload()) as Node
      setSelected(node); clearCredentials()
      setReplace(!node.credential_available || Boolean(node.jump_enabled && !node.jump_credential_available))
      setMessage('节点已保存');
      if (report?.connected) {
        const result = await api(`/${node.id}/test`,'POST'); setReport(result)
      }
      await reload()
    })
  }
  async function recheck() {
    if (!selected) return test()
    await action(async () => {
      const result = await api(`/${selected.id}/test`,'POST'); setReport(result); setTrustChecked(false); await reload()
    })
  }
  const env = report?.environment ?? {}
  const venvs = report?.venvs ?? []
  const gpuValue = (n: number | null, suffix='') => n == null ? '未检测' : `${n}${suffix}`
  const targetEndpoint = `${selected?.username || form.username}@${selected?.host || form.host}:${selected?.port || form.port}`
  const trustHop = report?.trust_required?.hop === 'jump' ? '跳板机' : '内网服务器'
  return <div className="min-w-0 space-y-6 p-4 md:p-6">
    <div className="flex items-center justify-between"><div><h1 className="text-2xl font-bold">算力节点</h1><p className="text-sm text-muted-foreground mt-1">管理远程 GPU 服务器。服务器在内网时，可经由公网跳板机转发连接。</p></div><Button disabled={busy} onClick={() => choose(null)}><Plus className="mr-2 size-4"/>添加节点</Button></div>
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {message && <p role="status" className="text-sm text-primary">{message}</p>}
    <div className="grid gap-5 xl:grid-cols-[220px_minmax(340px,1fr)_minmax(280px,0.85fr)]">
      <Card><CardHeader><CardTitle className="text-base">节点列表</CardTitle></CardHeader><CardContent className="space-y-2">
        {!nodes.length && <p className="text-sm text-muted-foreground">暂无节点，点击“添加节点”。</p>}
        {nodes.map(node => <button key={node.id} disabled={busy} onClick={() => choose(node)} className={`w-full rounded-lg border p-3 text-left ${selected?.id === node.id ? 'border-primary bg-primary/10' : 'hover:bg-muted'}`}><div className="flex items-center gap-2 font-medium text-sm"><Server className="size-4 shrink-0"/>{node.name}</div><div className="mt-2 text-xs text-muted-foreground flex flex-wrap items-center gap-1"><span className={node.status === 'online' ? 'text-primary' : ''}>● {stateName[node.status] ?? '未检测'}</span>{node.jump_enabled && <span className="inline-flex items-center gap-0.5 rounded border px-1 text-[10px]"><Route className="size-2.5"/>跳板</span>}</div><p className="truncate text-[11px] text-muted-foreground mt-1">{node.jump_enabled ? `${node.jump_host} → ${node.host}` : `${node.username}@${node.host}`}</p><p className="text-[11px] text-muted-foreground mt-1">{node.last_check_time ? `最近检测 ${new Date(node.last_check_time).toLocaleString()}` : '尚未检测'}</p></button>)}
      </CardContent></Card>
      <Card><CardHeader><CardTitle className="text-base">{selected ? '编辑节点' : '添加节点'}</CardTitle></CardHeader><CardContent>
        <fieldset disabled={busy} className="space-y-4 disabled:opacity-70">
          <label className="block text-sm">节点名称<Input value={form.name} onChange={e => edit('name',e.target.value)} placeholder="请输入节点名称" className="mt-1"/></label>
          <label className="block text-sm">连接方式<select className={`${selectClass} mt-1`} value={form.connection_type} onChange={e => edit('connection_type',e.target.value as Config['connection_type'])}><option value="ssh">SSH</option><option value="api" disabled>API（即将支持）</option></select></label>

          <Section label="连接路径" hint={form.jump_enabled ? '经跳板机转发' : '直连'} />
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={form.jump_enabled} onChange={e => edit('jump_enabled',e.target.checked)}/>经由公网跳板机（堡垒机）连接内网服务器</label>

          {form.jump_enabled && <div className="space-y-3 rounded-lg border bg-muted/30 p-3">
            <p className="flex items-center gap-1.5 text-xs font-medium"><Route className="size-3.5"/>跳板机（公网可达）</p>
            <div className="grid grid-cols-[1fr_84px] gap-2"><label className="text-xs">地址<Input value={form.jump_host} onChange={e => edit('jump_host',e.target.value)} placeholder="公网 IP 或域名" className="mt-1"/></label><label className="text-xs">端口<Input type="number" min={1} max={65535} value={form.jump_port} onChange={e => edit('jump_port',+e.target.value)} className="mt-1"/></label></div>
            <div className="grid grid-cols-[1fr_auto] gap-2"><label className="text-xs">用户名<Input autoComplete="off" value={form.jump_username} onChange={e => edit('jump_username',e.target.value)} className="mt-1"/></label><label className="text-xs">认证方式<select className={`${selectClass} mt-1 w-24`} value={form.jump_auth_type} onChange={e => edit('jump_auth_type',e.target.value as Config['auth_type'])}><option value="password">密码</option><option value="key">SSH Key</option></select></label></div>
          </div>}

          <div className={form.jump_enabled ? 'space-y-3 rounded-lg border p-3' : 'contents'}>
            {form.jump_enabled && <p className="text-xs font-medium">内网服务器（GPU 机器）</p>}
            <div className="grid grid-cols-[1fr_84px] gap-2"><label className="text-sm">服务器地址<Input value={form.host} onChange={e => edit('host',e.target.value)} placeholder="IP 或域名" className="mt-1"/></label><label className="text-sm">SSH 端口<Input type="number" min={1} max={65535} value={form.port} onChange={e => edit('port',+e.target.value)} className="mt-1"/></label></div>
            <div className="grid grid-cols-2 gap-2"><label className="text-sm">用户名<Input autoComplete="off" value={form.username} onChange={e => edit('username',e.target.value)} className="mt-1"/></label><label className="text-sm">认证方式<select className={`${selectClass} mt-1`} value={form.auth_type} onChange={e => edit('auth_type',e.target.value as Config['auth_type'])}><option value="password">密码</option><option value="key">SSH Key</option></select></label></div>
          </div>

          {form.jump_enabled && <div className="rounded-md border border-dashed p-2.5 text-[11px] leading-relaxed text-muted-foreground">
            <div className="flex flex-wrap items-center gap-1 font-mono"><span>本机</span><ArrowRight className="size-3"/><span className="text-foreground">跳板机</span><span>{form.jump_host || '?'}:{form.jump_port}</span><ArrowRight className="size-3"/><span className="text-foreground">内网服务器</span><span>{form.host || '?'}:{form.port}</span></div>
            <p className="mt-1">内网地址由跳板机解析，本机只需能连到跳板机；两跳各自的 SSH 指纹都需要确认一次。</p>
          </div>}

          <Section label="SSH 凭据" hint={form.jump_enabled ? '两跳各自一套' : undefined} />
          {!replace ? <div className="flex items-center justify-between gap-2 text-sm"><span className="flex gap-2 text-primary"><ShieldCheck className="size-4"/>{selected?.credential_saved ? '凭据已加密保存' : '凭据仅本次有效'}{selected?.jump_enabled && (selected?.jump_credential_saved ? '（含跳板机）' : '')}</span><Button variant="outline" size="sm" onClick={() => setReplace(true)}>更换凭据</Button></div> : <div className="space-y-2">
            <CredentialFields title={form.jump_enabled ? '内网服务器凭据' : 'SSH 凭据'} auth={form.auth_type} password={password} privateKey={privateKey} passphrase={passphrase}
              onPassword={setPassword} onKey={setPrivateKey} onPassphrase={setPassphrase} />
            {form.jump_enabled && <CredentialFields title="跳板机凭据" auth={form.jump_auth_type} password={jumpPassword} privateKey={jumpKey} passphrase={jumpPassphrase}
              onPassword={setJumpPassword} onKey={setJumpKey} onPassphrase={setJumpPassphrase} />}
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={saveCredential} onChange={e => setSaveCredential(e.target.checked)}/>保存凭据</label><p className="text-xs text-muted-foreground">由本机 Windows 账户加密保存，不向页面回传。</p>
          </div>}

          <Section label="运行环境" />
          <label className="block text-sm">远程工作目录<Input value={form.workdir} onChange={e => edit('workdir',e.target.value)} placeholder="/data/runos/huang" className="mt-1"/></label>
          <label className="block text-sm">Python 环境<select className={`${selectClass} mt-1`} value={form.environment_type} onChange={e => edit('environment_type',e.target.value as Config['environment_type'])}><option value="system">System Python</option><option value="conda">Conda</option><option value="venv">venv</option></select></label>
          {form.environment_type === 'conda' && <><label className="block text-sm">Conda 环境名称<Input value={form.conda_env} onChange={e => edit('conda_env',e.target.value)} placeholder="rfdetr" className="mt-1"/></label><label className="block text-sm">Conda 程序路径（可选）<Input value={form.conda_executable} onChange={e => edit('conda_executable',e.target.value)} placeholder="/opt/miniconda3/bin/conda" className="mt-1"/></label></>}
          {form.environment_type === 'venv' && <label className="block text-sm">venv 目录<Input value={form.venv_path} onChange={e => edit('venv_path',e.target.value)} placeholder="/data/venvs/rfdetr" className="mt-1"/></label>}
          <label className="block text-sm">GPU 选择<select className={`${selectClass} mt-1`} value={form.gpu_mode === 'auto' ? 'auto' : String(form.gpu_id)} onChange={e => setForm(p => ({ ...p,gpu_mode:e.target.value === 'auto' ? 'auto' : 'manual',gpu_id:e.target.value === 'auto' ? null : Number(e.target.value) }))}><option value="auto">自动</option>{form.gpu_mode === 'manual' && !report?.gpus.some(g => g.index===form.gpu_id) && <option value={String(form.gpu_id)}>GPU {form.gpu_id}（未检测）</option>}{report?.gpus.map(g => <option key={g.index} value={g.index}>GPU {g.index} · {g.name}</option>)}</select></label>
          <div className="flex flex-wrap gap-2"><Button variant="outline" onClick={test}>测试连接</Button><Button onClick={save}>保存节点</Button><Button variant="ghost" onClick={() => choose(selected)}>取消</Button></div>
          {selected && <Button variant="ghost" className="text-destructive" onClick={() => { if(window.confirm('删除此节点及其本机保存的凭据？不会修改远程服务器。')) action(async () => { await api(`/${selected.id}`,'DELETE'); choose(null); await reload() }) }}>删除节点</Button>}
        </fieldset>
      </CardContent></Card>
      <div className="space-y-4">
        <Card><CardHeader><CardTitle className="flex items-center justify-between text-base">连接状态<Button size="sm" variant="ghost" disabled={busy || !selected} onClick={recheck}><RefreshCw className="size-4 mr-1"/>重新检测</Button></CardTitle></CardHeader><CardContent className="space-y-3 text-sm">
          {busy ? <p className="flex gap-2"><Loader2 className="size-4 animate-spin"/>处理中，请稍候…</p> : <p className={report?.connected ? 'text-primary' : 'text-muted-foreground'}>● {report ? report.connected ? '连接成功' : stateName[report.status] : '未检测'}</p>}
          {report?.jump && <div className="rounded-md border p-2.5">
            <p className="mb-1.5 text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">连接路径</p>
            <HopRow label="跳板机" endpoint={report.jump.endpoint} ok={report.jump.connected} />
            <HopRow label="内网服务器" endpoint={targetEndpoint} ok={report.connected} last />
          </div>}
          <p className="text-xs text-muted-foreground">最近检测：{report ? new Date(report.checked_at).toLocaleString() : '未检测'}</p>
          {report?.errors.map((e,i) => <p key={i} className="text-amber-600 break-words">{e}</p>)}
          {report?.trust_required && <div className="rounded-md border p-3 space-y-3"><p className="font-medium">核对服务器指纹（{trustHop}）</p><p className="text-xs break-all">{report.trust_required.target}<br/>{report.trust_required.algorithm}<br/>{report.trust_required.fingerprint}</p><label className="flex gap-2 text-xs"><input type="checkbox" checked={trustChecked} disabled={busy} onChange={e => setTrustChecked(e.target.checked)}/>已向管理员核对指纹</label><Button size="sm" disabled={!trustChecked || busy} onClick={() => action(async () => { await api('/trust','POST',report.trust_required); setMessage('指纹已保存，请再次测试连接'); setReport(null) })}>信任此服务器</Button></div>}
          {report?.system && <p className="break-all text-xs text-muted-foreground">{report.system}</p>}
        </CardContent></Card>
        <Card><CardHeader><CardTitle className="text-base">环境信息</CardTitle></CardHeader><CardContent className="space-y-3 text-sm">
          {([['Python',env.python],['PyTorch',env.torch ?? env.torch_error],['CUDA（PyTorch）',env.cuda],['GPU',env.gpu_available == null ? undefined : env.gpu_available ? '可用' : '不可用'],['RF-DETR',env.rfdetr ?? env.rfdetr_error],['Transformers',env.transformers]] as [string,string|undefined][]).map(([label,value]) => <div key={label} className="flex justify-between gap-3"><span className="text-muted-foreground">{label}</span><span className="text-right break-all">{value ?? '未检测'}</span></div>)}
          {env.torch_note && <p className="break-words text-xs text-amber-700 dark:text-yellow-300">{env.torch_note}</p>}
          {!!venvs.length && <div className="space-y-2 rounded-md border p-2.5">
            <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-muted-foreground">工作目录下的虚拟环境</p>
            {venvs.map(v => <div key={v.path} className="space-y-1 border-t pt-2 first:border-t-0 first:pt-0">
              <div className="flex justify-between gap-3"><span className="break-all font-medium">{v.name}{v.active ? '（当前）' : ''}</span><span className={v.torch ? 'text-emerald-700 dark:text-green-400' : 'text-muted-foreground'}>{v.torch ?? v.torch_error ?? v.python_error ?? '未检测'}</span></div>
              <div className="flex justify-between gap-3 text-xs"><span className="text-muted-foreground">CUDA · GPU</span><span className="text-right">{v.cuda ? `${v.cuda}${v.gpu_available == null ? '' : v.gpu_available ? ' · 可用' : ' · 不可用'}` : '未检测'}</span></div>
              <div className="flex justify-between gap-3 text-xs"><span className="text-muted-foreground">RF-DETR</span><span className="text-right">{v.rfdetr ?? v.rfdetr_error ?? '未安装'}</span></div>
              {v.transformers && <div className="flex justify-between gap-3 text-xs"><span className="text-muted-foreground">Transformers</span><span className="text-right">{v.transformers}</span></div>}
              <p className="break-all text-[10px] text-muted-foreground/70">{v.path}</p>
            </div>)}
          </div>}
        </CardContent></Card>
        {!report?.gpus.length && <Card><CardContent className="pt-6 text-sm text-muted-foreground">GPU：未检测</CardContent></Card>}
        {report?.gpus.map(g => <Card key={g.index}><CardHeader><CardTitle className="text-base">{g.name}</CardTitle><p className="text-xs text-muted-foreground">GPU {g.index} · {g.utilization==null ? '未检测' : g.utilization>=80 ? '高负载' : g.utilization>0 || (g.memory_used ?? 0)>100 ? '使用中' : '空闲'}</p></CardHeader><CardContent className="space-y-3 text-sm">
          <p>显存：{g.memory_used==null ? '未检测' : (g.memory_used/1024).toFixed(1)} / {g.memory_total==null ? '未检测' : (g.memory_total/1024).toFixed(1)} GiB</p>
          <div className="h-2 rounded bg-muted"><div className="h-full rounded bg-primary" style={{width:`${Math.min(100,100*(g.memory_used ?? 0)/(g.memory_total || 1))}%`}}/></div>
          <div className="flex justify-between"><span>利用率 {gpuValue(g.utilization,'%')}</span><span>温度 {gpuValue(g.temperature,'℃')}</span></div><p>计算进程：{gpuValue(g.process_count)}</p>
        </CardContent></Card>)}
      </div>
    </div>
  </div>
}
