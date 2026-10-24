'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { toast } from 'sonner'
import {
  Bot, Plus, Trash2, Loader2, CheckCircle2, AlertCircle, Save,
  PlugZap, Wand2, ChevronDown, RefreshCw,
  SlidersHorizontal, History,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { readVlxConnectionSummary, type VlxConnectionSummary } from '@/lib/lmm-shared'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const COORD_LABELS: Record<string, string> = {
  '0-1000': '0-1000 归一化（Qwen 约定，推荐）',
  'pixel': '绝对像素坐标',
  '0-1': '0-1 浮点归一化',
}

const UNMATCHED_LABELS: Record<string, string> = {
  drop: '丢弃（推荐，防止类别表膨胀）',
  keep: '保留为原样标签',
}

const PRESETS: { name: string; base_url: string; model: string; hint: string }[] = [
  { name: '通义千问 Qwen-VL（阿里云 DashScope）', base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen2.5-vl-72b-instruct', hint: '国内直连，中文标注效果好' },
  { name: 'OpenAI GPT-4o', base_url: 'https://api.openai.com/v1', model: 'gpt-4o', hint: '需要海外网络环境' },
  { name: '本地 vLLM', base_url: 'http://127.0.0.1:8001/v1', model: 'Qwen/Qwen2.5-VL-7B-Instruct', hint: '自己部署，数据不出内网' },
  { name: '本地 Ollama', base_url: 'http://127.0.0.1:11434/v1', model: 'qwen2.5vl:7b', hint: '轻量本地部署' },
]

interface Provider {
  id: string
  name: string
  base_url: string
  model: string
  api_key: string
  has_api_key?: boolean
  coord_space: string
  max_side: number
  timeout: number
  extra_instruction: string
  unmatched: string
  transport: 'direct' | 'ssh'
  node_id: string
  port: number
  api_path: string
}

interface ComputeNode {
  id: string
  name: string
  host?: string
  status?: string
}

interface VlxDraft {
  transport: 'direct' | 'ssh'
  node_id: string
  port: number
  base_url: string
  token: string
  timeout: number
  task: NonNullable<VlxConnectionSummary['task']>
  lang: NonNullable<VlxConnectionSummary['lang']>
  max_new_tokens: number
  frame_interval: number
  extra_instruction: string
}

const EMPTY_VLX_DRAFT: VlxDraft = {
  transport: 'direct', node_id: '', port: 8021, base_url: '', token: '',
  timeout: 300, task: 'detection', lang: 'zh', max_new_tokens: 256, frame_interval: 1,
  extra_instruction: '',
}

const EMPTY: Provider = {
  id: '', name: '', base_url: '', model: '', api_key: '',
  coord_space: '0-1000', max_side: 1536, timeout: 120,
  extra_instruction: '', unmatched: 'drop',
  transport: 'direct', node_id: '', port: 8000, api_path: '/v1',
}

interface Dataset {
  id: string
  name: string
  classes?: string[]
}

interface Props {
  datasets?: Dataset[]
  selectedDataset?: Dataset | null
  apiUrl?: string
}

interface TestRun {
  id: string
  at: string
  model: string
  classes: string[]
  count: number
  elapsed: number | null
  ok: boolean
  error?: string
}

export function VlmView({ datasets = [], selectedDataset, apiUrl }: Props) {
  const base = apiUrl || API

  const [providers, setProviders] = useState<Provider[]>([])
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [vlxConnection, setVlxConnection] = useState<VlxConnectionSummary | null>(null)
  const [vlxDraft, setVlxDraft] = useState<VlxDraft>(EMPTY_VLX_DRAFT)
  const [vlxLoading, setVlxLoading] = useState(true)
  const [savingVlx, setSavingVlx] = useState(false)
  const [testingVlx, setTestingVlx] = useState(false)
  const [vlxTestResult, setVlxTestResult] = useState<{ connected: boolean; status?: string; message?: string; model?: string } | null>(null)
  const [vlxTestHistory, setVlxTestHistory] = useState<{ at: string; message: string; connected: boolean }[]>([])
  const [coordSpaces, setCoordSpaces] = useState<string[]>(['0-1000', 'pixel', '0-1'])
  const [activeId, setActiveId] = useState<string | null>('vlx-seek-saved')
  const [draft, setDraft] = useState<Provider>(EMPTY)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<any>(null)
  const [history, setHistory] = useState<TestRun[]>([])
  const [availableServerModels, setAvailableServerModels] = useState<string[]>([])
  const [discoveringModels, setDiscoveringModels] = useState(false)

  /** VLX uses its dedicated API, while OpenAI-compatible models use providers. */
  const [panel, setPanel] = useState<'providers' | 'vlx'>('vlx')

  const recordTest = useCallback((patch: Partial<TestRun> & { error?: string }) => {
    setHistory(h => [{
      id: `${Date.now()}-${h.length}`,
      at: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
      model: '', classes: [], count: 0, elapsed: null, ok: false,
      ...patch,
    }, ...h].slice(0, 6))
  }, [])

  // Read classes from the active dataset only for connection tests; target
  // classes are chosen on the annotation page, not stored per provider.
  const [datasetClasses, setDatasetClasses] = useState<string[]>([])

  const activeDataset = useMemo(
    () => selectedDataset || datasets[0] || null,
    [datasets, selectedDataset]
  )

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const r = await fetch(`${base}/api/vlm/providers`)
      if (!r.ok) throw new Error('读取配置失败')
      const d = await r.json()
      setProviders(d.providers || [])
      if (d.coord_spaces?.length) setCoordSpaces(d.coord_spaces)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '读取配置失败')
    } finally {
      setLoading(false)
    }
  }, [base])

  const loadNodes = useCallback(async () => {
    try {
      const r = await fetch(`${base}/api/compute-nodes`)
      if (!r.ok) return
      const d = await r.json()
      setNodes((d.nodes || []) as ComputeNode[])
    } catch { /* the user can still configure direct API providers */ }
  }, [base])

  // The VLX endpoint is a dedicated API. Read only its saved connection summary;
  // this route does not probe the endpoint or contact the model service.
  useEffect(() => {
    let alive = true
    readVlxConnectionSummary(base)
      .then(config => {
        if (!alive) return
        setVlxConnection(config)
        setVlxDraft({
          ...EMPTY_VLX_DRAFT,
          transport: config.transport,
          node_id: config.node_id || '',
          port: config.port || 8021,
          base_url: config.base_url || '',
          timeout: config.timeout ?? 300,
          task: config.task ?? 'detection',
          lang: config.lang ?? 'zh',
          max_new_tokens: config.max_new_tokens ?? 256,
          frame_interval: config.frame_interval ?? 1,
          extra_instruction: config.extra_instruction ?? '',
        })
      })
      .finally(() => { if (alive) setVlxLoading(false) })
    return () => { alive = false }
  }, [base])

  useEffect(() => { load(); loadNodes() }, [load, loadNodes])

  // Pull the live class list for the current dataset for connection tests.
  useEffect(() => {
    let alive = true
    const fromProp = (activeDataset?.classes || []).filter(Boolean)
    setDatasetClasses(fromProp)
    if (!activeDataset?.id) return
    ;(async () => {
      try {
        const r = await fetch(`${base}/api/datasets/${activeDataset.id}`)
        if (!r.ok) return
        const d = await r.json()
        const cls: string[] = (d.dataset?.classes || d.classes || []).filter(Boolean)
        if (alive && cls.length) { setDatasetClasses(cls); return }
      } catch { /* ignore — fall through to annotation-derived classes */ }

      // Fallback — mirrors AnnotationView: when the dataset YAML declares no
      // class names, the annotate toolbar derives them from existing annotations.
      // Read from the very same place so the two lists can't drift apart.
      try {
        const r2 = await fetch(`${base}/api/datasets/${activeDataset.id}/images?limit=999999`)
        if (r2.ok) {
          const d2 = await r2.json()
          const imgs: any[] = d2.images || []
          const seen = new Set<string>()
          for (const im of imgs) {
            if (im?.class_name) seen.add(String(im.class_name))
            for (const a of im?.annotations || []) if (a?.class_name) seen.add(String(a.class_name))
          }
          if (alive) setDatasetClasses([...seen])
        }
      } catch { /* keep whatever we already have */ }
    })()
    return () => { alive = false }
  }, [activeDataset?.id, base]) // eslint-disable-line react-hooks/exhaustive-deps

  const selectProvider = (p: Provider) => {
    setPanel('providers')
    setActiveId(p.id)
    setDraft({ ...EMPTY, ...p, api_key: '' })
    setAvailableServerModels([])
    setTestResult(null)
  }

  const newProvider = (transport: 'direct' | 'ssh' = 'direct') => {
    setPanel('providers')
    setActiveId(null)
    setDraft({ ...EMPTY, transport })
    setAvailableServerModels([])
    setTestResult(null)
  }

  const selectVlx = () => {
    setPanel('vlx')
    setActiveId('vlx-seek-saved')
    setAvailableServerModels([])
    setTestResult(null)
  }

  const saveVlxConfig = async () => {
    if (!vlxConnection?.supports_persistent_prompt) {
      toast.error('RunOS API 更新尚未生效，当前不能保存 VLX 配置')
      return
    }
    if (vlxDraft.transport === 'ssh' && (!vlxDraft.node_id || !vlxDraft.port)) {
      toast.error('请选择算力节点并填写服务端口')
      return
    }
    if (vlxDraft.transport === 'direct' && !vlxDraft.base_url.trim()) {
      toast.error('请填写接口地址')
      return
    }
    setSavingVlx(true)
    try {
      const payload: Record<string, unknown> = {
        transport: vlxDraft.transport,
        node_id: vlxDraft.node_id,
        port: vlxDraft.port,
        base_url: vlxDraft.base_url.trim(),
        timeout: vlxDraft.timeout,
        task: vlxDraft.task,
        lang: vlxDraft.lang,
        max_new_tokens: vlxDraft.max_new_tokens,
        frame_interval: vlxDraft.frame_interval,
        extra_instruction: vlxDraft.extra_instruction,
      }
      // An empty token means keep the saved token; the API never returns its value.
      if (vlxDraft.token.trim()) payload.token = vlxDraft.token.trim()
      const response = await fetch(`${base}/api/lmm/vlx/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || `保存失败 (${response.status})`)
      const config = data.config as VlxConnectionSummary
      setVlxConnection(config)
      setVlxDraft(prev => ({ ...prev, token: '' }))
      toast.success('VLX-Seek 配置已保存')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '保存失败')
    } finally {
      setSavingVlx(false)
    }
  }

  const testVlxConnection = async () => {
    setTestingVlx(true)
    setVlxTestResult(null)
    try {
      // Only an explicit click contacts the VLX service. Opening the page stays read-only.
      const response = await fetch(`${base}/api/lmm/vlx/health`, { cache: 'no-store' })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(result.detail || `连接失败 (${response.status})`)
      const status = result as { connected: boolean; status?: string; message?: string; model?: string }
      setVlxTestResult(status)
      const message = status.message || (status.connected ? 'VLX 服务已连接' : status.status === 'blocked' || status.status === 'error'
        ? 'VLX 服务不可用' : '服务可达，模型尚未加载')
      setVlxTestHistory(current => [{ at: new Date().toLocaleTimeString('zh-CN', { hour12: false }), message, connected: status.connected }, ...current].slice(0, 6))
    } catch (error) {
      const message = error instanceof Error ? error.message : '连接失败'
      setVlxTestResult({ connected: false, status: 'error', message })
      setVlxTestHistory(current => [{ at: new Date().toLocaleTimeString('zh-CN', { hour12: false }), message, connected: false }, ...current].slice(0, 6))
    } finally {
      setTestingVlx(false)
    }
  }

  const applyPreset = (idx: number) => {
    const p = PRESETS[idx]
    setDraft(d => ({ ...d, transport: 'direct', node_id: '', base_url: p.base_url, model: p.model, name: d.name || p.name }))
  }

  const save = async () => {
    if (!draft.id.trim()) { toast.error('请填写配置 ID'); return }
    if (draft.transport === 'ssh' ? !draft.node_id || !draft.port : !draft.base_url.trim()) {
      toast.error(draft.transport === 'ssh' ? '请选择算力节点并填写服务端口' : '请填写接口地址'); return
    }
    if (!draft.model.trim()) { toast.error('请填写模型名'); return }
    setSaving(true)
    try {
      const r = await fetch(`${base}/api/vlm/providers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...draft, classes: [], id: draft.id.trim(), name: draft.name.trim() || draft.id.trim() }),
      })
      if (!r.ok) {
        const e = await r.json().catch(() => ({}))
        throw new Error(e.detail || '保存失败')
      }
      const d = await r.json()
      toast.success(`已保存「${d.provider?.name || draft.id}」`)
      await load()
      setActiveId(draft.id.trim())
      setDraft(prev => ({ ...prev, api_key: '' }))
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const remove = async (id: string) => {
    try {
      const r = await fetch(`${base}/api/vlm/providers/${id}`, { method: 'DELETE' })
      if (!r.ok) throw new Error('删除失败')
      toast.success('已删除配置')
      if (activeId === id) newProvider()
      await load()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '删除失败')
    }
  }

  const testConnection = async () => {
    if ((draft.transport === 'ssh' ? !draft.node_id || !draft.port : !draft.base_url.trim()) || !draft.model.trim()) {
      toast.error('请先填写服务器连接信息与模型名'); return
    }
    setTesting(true)
    setTestResult(null)
    const testClasses = datasetClasses
    try {
      // Test against the currently open dataset's first image.
      const r = await fetch(`${base}/api/vlm/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: { ...draft, classes: testClasses, id: draft.id || '__draft__' },
          dataset_id: activeDataset?.id || null,
        }),
      })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(d.detail || `测试失败 (${r.status})`)
      setTestResult(d)
      const n = d.annotations?.length ?? 0
      recordTest({
        model: draft.model, classes: testClasses.slice(), count: n,
        elapsed: d.debug?.elapsed_s ?? null, ok: true,
      })
      if (n > 0) toast.success(`连接成功，识别到 ${n} 个目标`)
      else toast.warning('连接成功，但未识别到目标；可检查标注页当前数据集类别或附加提示词')
    } catch (e) {
      const msg = e instanceof Error ? e.message : '测试失败'
      setTestResult({ error: msg })
      recordTest({ model: draft.model, classes: testClasses.slice(), ok: false, error: msg })
      toast.error(msg)
    } finally {
      setTesting(false)
    }
  }

  const discoverModels = async () => {
    if (draft.transport === 'ssh' ? !draft.node_id || !draft.port : !draft.base_url.trim()) {
      toast.error('请先填写服务器连接信息'); return
    }
    setDiscoveringModels(true)
    setAvailableServerModels([])
    try {
      const r = await fetch(`${base}/api/vlm/discover`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...draft, id: draft.id || '__discover__', classes: [] }),
      })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(d.detail || `读取模型列表失败 (${r.status})`)
      const models: string[] = d.models || []
      setAvailableServerModels(models)
      if (models.length === 0) toast.warning('服务器可访问，但没有返回模型列表；可直接填写模型 ID')
      else toast.success(`读取到 ${models.length} 个模型`)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '读取模型列表失败')
    } finally {
      setDiscoveringModels(false)
    }
  }

  return (
    <div className="flex h-full">
      {/* ── Provider list ───────────────────────────────────────────── */}
      <aside className="w-64 shrink-0 border-r border-border flex flex-col">
        <div className="p-3 border-b border-border flex items-center justify-between">
          <span className="text-xs font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            模型配置
          </span>
          <Button size="sm" variant="ghost" className="h-6 px-2 text-xs gap-1" onClick={() => newProvider()}>
            <Plus className="w-3 h-3" /> 新建
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {loading ? (
            <div className="flex items-center gap-2 p-3 text-xs text-muted-foreground">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> 加载中…
            </div>
          ) : (
            <>
              <button
                  onClick={selectVlx}
                  className={cn(
                    'w-full text-left px-2.5 py-2 rounded-md text-xs transition-colors border',
                    activeId === 'vlx-seek-saved'
                      ? 'bg-primary/10 border-primary/30 text-primary'
                      : 'border-transparent hover:bg-muted'
                  )}
                >
                  <div className="flex items-center gap-1.5 font-medium truncate">
                    <Bot className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate">VLX-Seek</span>
                  </div>
                  <div className="mt-0.5 text-xs text-muted-foreground truncate font-mono">
                    {vlxLoading ? '读取配置中…' : vlxConnection?.configured ? 'VLX-Seek-1.5-10B' : '未配置 · VLX-Seek-1.5-10B'}
                  </div>
              </button>
              {providers.length === 0 && !loading && (
                <p className="px-3 py-2 text-xs text-muted-foreground leading-relaxed">
                  暂无其他模型配置。点击「新建」可添加兼容 OpenAI 接口的模型。
                </p>
              )}
              {providers.map(p => (
                <button
                  key={p.id}
                  onClick={() => selectProvider(p)}
                  className={cn(
                    'w-full text-left px-2.5 py-2 rounded-md text-xs transition-colors border',
                    activeId === p.id
                      ? 'bg-primary/10 border-primary/30 text-primary'
                      : 'border-transparent hover:bg-muted'
                  )}
                >
                  <div className="flex items-center gap-1.5 font-medium truncate">
                    <Bot className="w-3.5 h-3.5 shrink-0" />
                    <span className="truncate">{p.name || p.id}</span>
                  </div>
                  <div className="mt-0.5 text-xs text-muted-foreground truncate font-mono">
                    {p.model || '未设置模型'}
                  </div>
                  {p.transport === 'ssh' && (
                    <div className="mt-1">
                      <Badge variant="secondary" className="text-xs px-1 py-0 h-4 font-normal">服务器 · {nodes.find(n => n.id === p.node_id)?.name || p.node_id}</Badge>
                    </div>
                  )}
                  {p.has_api_key && (
                    <div className="mt-1 flex items-center gap-1">
                      <Badge variant="secondary" className="text-xs px-1 py-0 h-4 font-normal">已设密钥</Badge>
                    </div>
                  )}
                </button>
              ))}
            </>
          )}
        </div>

      </aside>

      {/* ── Editor + preview ────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {panel === 'vlx' ? (
          <VlxConfigEditor
            config={vlxConnection}
            draft={vlxDraft}
            nodes={nodes}
            loading={vlxLoading}
            saving={savingVlx}
            testing={testingVlx}
            testResult={vlxTestResult}
            testHistory={vlxTestHistory}
            onChange={patch => setVlxDraft(current => ({ ...current, ...patch }))}
            onSave={saveVlxConfig}
            onTest={testVlxConnection}
          />
        ) : (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-6 p-5 items-start">
        <div className="max-w-2xl space-y-6 w-full">

          <div>
            <h2 className="text-sm font-medium flex items-center gap-2">
              <Bot className="w-4 h-4" />
              视觉模型配置
            </h2>
            <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
              可以接入多个服务器和模型。服务器节点走 SSH 隧道，支持兼容 OpenAI 接口的 Qwen-VL、vLLM、Ollama 等；也可配置云端 API。
              每个模型单独保存，之后可在标注和多模态检测中选择。
            </p>
          </div>

          {/* Basic */}
          <section className="space-y-3">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              接口
            </h3>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">配置 ID</Label>
                <Input
                  className="h-8 text-xs font-mono"
                  value={draft.id}
                  disabled={!!activeId}
                  placeholder="qwen-vl"
                  onChange={e => setDraft(d => ({ ...d, id: e.target.value }))}
                />
                <p className="text-xs text-muted-foreground">唯一标识，保存后不可改</p>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">显示名称</Label>
                <Input
                  className="h-8 text-xs"
                  value={draft.name}
                  placeholder="Qwen2.5-VL 72B"
                  onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">快速预设</Label>
              <div className="flex flex-wrap gap-1.5">
                {PRESETS.map((p, i) => (
                  <button
                    key={p.name}
                    onClick={() => applyPreset(i)}
                    title={`${p.base_url} · ${p.hint}`}
                    className="px-2 py-1 rounded border border-border hover:border-primary/40 hover:bg-primary/5 text-xs transition-colors"
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">接入方式</Label>
              <Select
                value={draft.transport}
                onValueChange={v => setDraft(d => ({ ...d, transport: v as Provider['transport'] }))}
              >
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ssh">算力节点 SSH 隧道</SelectItem>
                  <SelectItem value="direct">HTTP / 云端 API</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {draft.transport === 'ssh'
                  ? '复用「算力节点」中保存的 SSH 凭据，访问服务器本机模型接口。'
                  : '直接访问可达的 OpenAI 兼容接口。'}
              </p>
            </div>

            {draft.transport === 'ssh' ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label className="text-xs">算力节点</Label>
                    <Select
                      value={draft.node_id || '__none__'}
                      onValueChange={v => setDraft(d => ({ ...d, node_id: v === '__none__' ? '' : v }))}
                    >
                      <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="选择已配置节点" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__none__">请选择节点</SelectItem>
                        {nodes.map(n => <SelectItem key={n.id} value={n.id}>{n.name || n.id}{n.host ? ` · ${n.host}` : ''}</SelectItem>)}
                      </SelectContent>
                    </Select>
                    {nodes.length === 0 && <p className="text-xs text-amber-600">请先在「算力节点」中配置并测试 SSH 连接。</p>}
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs">服务端口</Label>
                    <Input className="h-8 text-xs font-mono" type="number" min={1} max={65535}
                      value={draft.port} onChange={e => setDraft(d => ({ ...d, port: Number(e.target.value) || 8000 }))} />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs">API 路径前缀</Label>
                  <Input className="h-8 text-xs font-mono" value={draft.api_path}
                    placeholder="/v1" onChange={e => setDraft(d => ({ ...d, api_path: e.target.value }))} />
                  <p className="text-xs text-muted-foreground">常用 <span className="font-mono">/v1</span>，请求会访问该节点内的模型服务，不需要填写服务器 IP。</p>
                </div>
              </>
            ) : (
              <div className="space-y-1.5">
                <Label className="text-xs">接口地址 (Base URL)</Label>
                <Input className="h-8 text-xs font-mono" value={draft.base_url}
                  placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
                  onChange={e => setDraft(d => ({ ...d, base_url: e.target.value }))} />
                <p className="text-xs text-muted-foreground">会自动补 <span className="font-mono">/chat/completions</span></p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 items-end">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <Label className="text-xs">模型名 / Model ID</Label>
                  <Button size="sm" variant="outline" className="h-6 px-2 text-xs gap-1"
                    onClick={discoverModels} disabled={discoveringModels}>
                    {discoveringModels ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                    {discoveringModels ? '读取中' : '读取服务器模型'}
                  </Button>
                </div>
                <Input
                  className="h-8 text-xs font-mono"
                  value={draft.model}
                  placeholder="从服务器读取或填写 Model ID"
                  onChange={e => setDraft(d => ({ ...d, model: e.target.value }))}
                />
                {availableServerModels.length > 0 && (
                  <div className="flex flex-wrap gap-1 pt-1">
                    {availableServerModels.map(model => (
                      <button key={model} type="button" onClick={() => setDraft(d => ({ ...d, model }))}
                        className={cn('rounded border px-1.5 py-0.5 text-xs font-mono', draft.model === model ? 'border-primary bg-primary/10 text-primary' : 'border-border hover:border-primary/40')}>
                        {model}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">API Key / 服务 Token</Label>
                <Input
                  className="h-8 text-xs font-mono"
                  type="password"
                  value={draft.api_key}
                  placeholder={activeId ? '留空则沿用已保存密钥' : '可选；本地服务通常留空'}
                  onChange={e => setDraft(d => ({ ...d, api_key: e.target.value }))}
                />
              </div>
            </div>
          </section>

          {/* Advanced */}
          <section className="space-y-3">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              高级
            </h3>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">坐标格式</Label>
                <Select
                  value={draft.coord_space}
                  onValueChange={v => setDraft(d => ({ ...d, coord_space: v }))}
                >
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {coordSpaces.map(c => (
                      <SelectItem key={c} value={c}>{COORD_LABELS[c] || c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">匹配不上的标签</Label>
                <Select
                  value={draft.unmatched}
                  onValueChange={v => setDraft(d => ({ ...d, unmatched: v }))}
                >
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.entries(UNMATCHED_LABELS).map(([v, l]) => (
                      <SelectItem key={v} value={v}>{l}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">上传前最长边 (px)</Label>
                <Input
                  className="h-8 text-xs font-mono" type="number" min={320} max={4096} step={64}
                  value={draft.max_side}
                  onChange={e => setDraft(d => ({ ...d, max_side: Number(e.target.value) || 1536 }))}
                />
                <p className="text-xs text-muted-foreground">越小越快越省，坐标不受影响</p>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">超时 (秒)</Label>
                <Input
                  className="h-8 text-xs font-mono" type="number" min={10} max={600}
                  value={draft.timeout}
                  onChange={e => setDraft(d => ({ ...d, timeout: Number(e.target.value) || 120 }))}
                />
                <p className="text-xs text-muted-foreground">大模型比检测器慢，建议 ≥ 60</p>
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">附加提示词</Label>
              <Textarea
                className="text-xs min-h-[64px]"
                value={draft.extra_instruction}
                placeholder="例如：只标注水面上的游泳者，忽略岸上的人。"
                onChange={e => setDraft(d => ({ ...d, extra_instruction: e.target.value }))}
              />
              <p className="text-xs text-muted-foreground">
                常驻附加说明；标注页的提示词输入框会在此基础上叠加。
              </p>
            </div>
          </section>

          {/* Actions */}
          <section className="space-y-3 pt-1 border-t border-border">
            <div className="flex items-center gap-2 pt-3">
              <Button size="sm" className="h-8 text-xs gap-1.5" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                保存配置
              </Button>
              <Button
                size="sm" variant="outline" className="h-8 text-xs gap-1.5"
                onClick={testConnection} disabled={testing}
              >
                {testing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <PlugZap className="w-3.5 h-3.5" />}
                {testing ? '测试中…' : '测试连接'}
              </Button>
              {activeId && (
                <Button
                  size="sm" variant="outline"
                  className="h-8 text-xs gap-1.5 ml-auto text-red-600 dark:text-red-400 hover:text-red-700"
                  onClick={() => remove(activeId)}
                >
                  <Trash2 className="w-3.5 h-3.5" /> 删除
                </Button>
              )}
            </div>

            <p className="text-xs text-muted-foreground pb-1">
              「测试连接」会用「{activeDataset?.name || activeDataset?.id || '当前数据集'}」的第一张图试跑一次，
              <span className="font-medium">不会写入标注</span>，可放心试。
            </p>
          </section>
        </div>

        {/* ── Preview / inspector ─────────────────────────────────────── */}
        <aside className="space-y-4 xl:sticky xl:top-0">

          {/* 当前配置摘要 */}
          <section className="rounded-lg border border-border bg-card/50">
            <div className="px-3 py-2 border-b border-border flex items-center gap-1.5">
              <SlidersHorizontal className="w-3.5 h-3.5 text-primary" />
              <span className="text-xs font-medium">配置摘要</span>
              {activeId
                ? <Badge variant="secondary" className="ml-auto text-xs px-1 py-0 h-4 font-normal">已保存</Badge>
                : <Badge variant="outline" className="ml-auto text-xs px-1 py-0 h-4 font-normal">未保存</Badge>}
            </div>
            <dl className="p-3 space-y-2 text-xs">
              <SummaryRow label="模型名" value={draft.model || '—'} mono />
              <SummaryRow
                label="服务器"
                value={draft.transport === 'ssh'
                  ? `${nodes.find(n => n.id === draft.node_id)?.name || draft.node_id || '未选择节点'}:${draft.port}${draft.api_path}`
                  : (draft.base_url || '—')}
                mono truncate
              />
              <SummaryRow label="接入方式" value={draft.transport === 'ssh' ? '算力节点 SSH 隧道' : 'HTTP / 云端 API'} />
              <SummaryRow
                label="密钥"
                value={draft.api_key ? '新密钥已填' : (activeId ? '沿用已存密钥' : '未填写')}
              />
              <SummaryRow label="坐标格式" value={COORD_LABELS[draft.coord_space] || draft.coord_space} />
              <SummaryRow label="上传最长边" value={`${draft.max_side} px`} mono />
              <SummaryRow label="超时" value={`${draft.timeout} s`} mono />
            </dl>
          </section>

          {/* 测试连接结果 */}
          <section className="rounded-lg border border-border bg-card/50">
            <div className="px-3 py-2 border-b border-border flex items-center gap-1.5">
              <PlugZap className="w-3.5 h-3.5 text-primary" />
              <span className="text-xs font-medium">测试结果</span>
              {testing && <Loader2 className="w-3 h-3 animate-spin ml-auto text-muted-foreground" />}
            </div>

            <div className="p-3">
              {!testResult && !testing && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  还没有测试记录。<br />
                  填好接口与密钥后点「测试连接」，这里会显示识别结果与模型原始回复。
                </p>
              )}

              {testing && (
                <p className="text-xs text-muted-foreground">正在请求模型，大模型通常需要几秒…</p>
              )}

              {testResult?.error && (
                <div className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400">
                  <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                  <span className="break-all">{testResult.error}</span>
                </div>
              )}

              {testResult && !testResult.error && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-xs">
                    <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0" />
                    <span className="font-medium">连接正常</span>
                    <span className="text-muted-foreground font-mono text-xs ml-auto">
                      {testResult.annotations?.length ?? 0} 个 · {testResult.debug?.elapsed_s ?? '?'}s
                    </span>
                  </div>

                  {(testResult.annotations || []).length > 0 ? (
                    <div className="space-y-1">
                      {(testResult.annotations || []).slice(0, 12).map((a: any, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-xs font-mono">
                          <span className="w-3 text-muted-foreground shrink-0">{i + 1}</span>
                          <span className="text-primary truncate flex-1">{a.class_name}</span>
                        </div>
                      ))}
                      {(testResult.annotations || []).length > 12 && (
                        <p className="text-xs text-muted-foreground">
                          还有 {(testResult.annotations || []).length - 12} 个未显示…
                        </p>
                      )}
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      模型返回成功但没识别到目标，可检查标注页当前数据集类别或附加提示词。
                    </p>
                  )}

                  {testResult.debug && (
                    <details className="text-xs text-muted-foreground">
                      <summary className="cursor-pointer hover:text-foreground flex items-center gap-1">
                        <ChevronDown className="w-3 h-3" /> 查看原始回复
                      </summary>
                      <pre className="mt-1.5 p-2 rounded bg-background border border-border overflow-x-auto whitespace-pre-wrap break-all max-h-56 text-xs">
                        {testResult.debug.raw}
                      </pre>
                      <p className="mt-1">
                        解析出 {testResult.debug.parsed_objects} 个 · 丢弃 {testResult.debug.dropped} 个
                        · 低于阈值 {testResult.debug.below_threshold} 个
                      </p>
                    </details>
                  )}
                </div>
              )}
            </div>
          </section>

          {/* 测试记录 */}
          <section className="rounded-lg border border-border bg-card/50">
            <div className="px-3 py-2 border-b border-border flex items-center gap-1.5">
              <History className="w-3.5 h-3.5 text-primary" />
              <span className="text-xs font-medium">测试记录</span>
              {history.length > 0 && (
                <button
                  onClick={() => setHistory([])}
                  className="ml-auto text-xs text-muted-foreground hover:text-foreground"
                >
                  清空
                </button>
              )}
            </div>
            <div className="p-3 space-y-1.5">
              {history.length === 0 ? (
                <p className="text-xs text-muted-foreground">本次会话的测试记录会列在这里。</p>
              ) : history.map(h => (
                <div key={h.id} className="flex items-center gap-2 text-xs font-mono">
                  {h.ok
                    ? <CheckCircle2 className="w-3 h-3 text-primary shrink-0" />
                    : <AlertCircle className="w-3 h-3 text-red-500 shrink-0" />}
                  <span className="text-muted-foreground shrink-0">{h.at}</span>
                  <span className="truncate flex-1 text-muted-foreground">
                    {h.ok ? `${h.model || '—'} · ${h.count} 个` : (h.error || '失败')}
                  </span>
                  {h.ok && h.elapsed != null && (
                    <span className="text-muted-foreground shrink-0">{h.elapsed}s</span>
                  )}
                </div>
              ))}
            </div>
          </section>

          {/* 接入说明 */}
          <section className="rounded-lg border border-dashed border-border p-3 space-y-2">
            <p className="text-xs font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              接入说明
            </p>
            <ul className="space-y-1.5 text-xs text-muted-foreground leading-relaxed">
              <li>· 服务器模型使用 OpenAI 兼容接口：<span className="font-mono">/chat/completions</span>，并需支持图片输入。</li>
              <li>· 一个模型配置绑定一个节点与 Model ID；可在不同节点重复添加不同模型。</li>
              <li>· SSH 模式复用「算力节点」凭据，服务只需监听服务器本机端口。</li>
              <li>· 坐标格式多数模型认 <span className="font-mono">0-1000</span>；若框全错先换这里。</li>
              <li>· 类别不匹配的标签默认丢弃，避免把 <span className="font-mono">data.yaml</span> 撑大。</li>
              <li>· 大模型比检测器慢得多，批量标注前建议先用单张试跑。</li>
            </ul>
            {draft.id && (
              <p className="pt-1 text-xs flex items-center gap-1.5">
                <Wand2 className="w-3.5 h-3.5 text-primary shrink-0" />
                <span className="text-muted-foreground">
                  {activeId ? '已保存，' : '保存后，'}到「标注」页模型下拉里选 <span className="font-mono text-foreground">{draft.name || draft.id}</span> 即可。
                </span>
              </p>
            )}
          </section>
        </aside>
        </div>
        )}
      </div>
    </div>
  )
}

function VlxConfigEditor({
  config, draft, nodes, loading, saving, testing, testResult, testHistory, onChange, onSave, onTest,
}: {
  config: VlxConnectionSummary | null
  draft: VlxDraft
  nodes: ComputeNode[]
  loading: boolean
  saving: boolean
  testing: boolean
  testResult: { connected: boolean; status?: string; message?: string; model?: string } | null
  testHistory: { at: string; message: string; connected: boolean }[]
  onChange: (patch: Partial<VlxDraft>) => void
  onSave: () => void
  onTest: () => void
}) {
  const nodeName = nodes.find(node => node.id === draft.node_id)?.name || draft.node_id || '未选择节点'
  const endpoint = draft.transport === 'ssh'
    ? `${nodeName} · 127.0.0.1:${draft.port}/infer`
    : draft.base_url ? `${draft.base_url.replace(/\/+$/, '')}/infer` : '—'

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px] gap-6 p-5 items-start">
      <div className="max-w-2xl space-y-6 w-full">
        <header>
          <h2 className="text-sm font-medium flex items-center gap-2">
            <Bot className="w-4 h-4 text-primary" /> 视觉模型配置
          </h2>
          <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
            VLX-Seek 使用专用接口。模型名固定；接口地址、密钥和任务参数在这里保存。
          </p>
        </header>

        <section className="space-y-3">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">接口</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">配置 ID</Label>
              <Input className="h-8 text-xs font-mono" value="vlx-seek" disabled />
              <p className="text-xs text-muted-foreground">VLX 专用配置</p>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">显示名称</Label>
              <Input className="h-8 text-xs" value="VLX-Seek" readOnly aria-readonly="true" />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">接入方式</Label>
            <Select value={draft.transport} onValueChange={value => onChange({ transport: value as VlxDraft['transport'] })}>
              <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="direct">HTTP / 服务器本机 API</SelectItem>
                <SelectItem value="ssh">算力节点 SSH 隧道</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {draft.transport === 'ssh' ? (
            <div className="grid grid-cols-[minmax(0,1fr)_120px] gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">算力节点</Label>
                <Select value={draft.node_id || '__none__'} onValueChange={value => onChange({ node_id: value === '__none__' ? '' : value })}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="选择已配置节点" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">请选择节点</SelectItem>
                    {nodes.map(node => <SelectItem key={node.id} value={node.id}>{node.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">服务端口</Label>
                <Input className="h-8 text-xs font-mono" type="number" min={1} max={65535} value={draft.port}
                  onChange={event => onChange({ port: Number(event.target.value) || 8021 })} />
              </div>
            </div>
          ) : (
            <div className="space-y-1.5">
              <Label className="text-xs">接口地址（Base URL）</Label>
              <Input className="h-8 text-xs font-mono" value={draft.base_url} placeholder="http://127.0.0.1:8021"
                onChange={event => onChange({ base_url: event.target.value })} />
              <p className="text-xs text-muted-foreground">VLX 专用接口会在地址后使用 <span className="font-mono">/infer</span></p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 items-end">
            <div className="space-y-1.5">
              <Label className="text-xs">模型名 / Model ID</Label>
              <Input className="h-8 text-xs font-mono" value="VLX-Seek-1.5-10B" readOnly aria-readonly="true" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">API Key / 服务 Token</Label>
              <Input className="h-8 text-xs font-mono" type="password" value={draft.token}
                placeholder={config?.token_configured ? '留空则沿用已保存密钥' : '填写服务 Token'}
                onChange={event => onChange({ token: event.target.value })} />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">专用接口使用固定模型名；已保存的密钥不会回显，留空时保留原值。</p>
        </section>

        <section className="space-y-3">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">高级</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">任务类型</Label>
              <Select value={draft.task} onValueChange={value => onChange({ task: value as VlxDraft['task'] })}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="detection">detection</SelectItem>
                  <SelectItem value="reasoning_detection">reasoning_detection</SelectItem>
                  <SelectItem value="grounding_single">grounding_single</SelectItem>
                  <SelectItem value="counting">counting</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">语言</Label>
              <Select value={draft.lang} onValueChange={value => onChange({ lang: value as VlxDraft['lang'] })}>
                <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent><SelectItem value="zh">中文</SelectItem><SelectItem value="en">English</SelectItem></SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">单次请求超时（秒）</Label>
              <Input className="h-8 text-xs font-mono" type="number" min={1} max={1800} value={draft.timeout}
                onChange={event => onChange({ timeout: Number(event.target.value) || 300 })} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">视频抽帧间隔（秒）</Label>
              <Input className="h-8 text-xs font-mono" type="number" min={0.1} step={0.1} value={draft.frame_interval}
                onChange={event => onChange({ frame_interval: Number(event.target.value) || 1 })} />
            </div>
          </div>
          <div className="space-y-1.5 max-w-[calc(50%-0.375rem)]">
            <Label className="text-xs">最大生成 Token</Label>
            <Input className="h-8 text-xs font-mono" type="number" min={1} max={8192} value={draft.max_new_tokens}
              onChange={event => onChange({ max_new_tokens: Number(event.target.value) || 256 })} />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs">附加提示词</Label>
            <Textarea className="text-xs min-h-[64px]" maxLength={2048}
              value={draft.extra_instruction}
              placeholder="例如：只检测水面上的游泳者，忽略岸上的人。"
              onChange={event => onChange({ extra_instruction: event.target.value })} />
            <p className="text-xs text-muted-foreground">常驻附加说明；运行 VLX 任务时会追加到目标描述后。</p>
          </div>
        </section>

        <section className="space-y-3 pt-1 border-t border-border">
          <div className="flex items-center gap-2 pt-3">
            <Button size="sm" className="h-8 text-xs gap-1.5" onClick={onSave} disabled={saving || loading}>
              {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
              保存配置
            </Button>
            <Button size="sm" variant="outline" className="h-8 text-xs gap-1.5" onClick={onTest} disabled={testing || loading}>
              {testing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <PlugZap className="w-3.5 h-3.5" />}
              {testing ? '测试中…' : '测试连接'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground pb-1">「测试连接」仅在点击后读取 VLX 服务状态，不上传图片或执行推理。</p>
        </section>
      </div>

      <aside className="space-y-4 xl:sticky xl:top-0">
        <section className="rounded-lg border border-border bg-card/50">
          <div className="px-3 py-2 border-b border-border flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-primary" />
            <span className="text-xs font-medium">配置摘要</span>
            <Badge variant={config?.configured ? 'secondary' : 'outline'} className="ml-auto text-xs px-1 py-0 h-4 font-normal">
              {loading ? '读取中' : config?.configured ? '已保存' : '未配置'}
            </Badge>
          </div>
          <dl className="p-3 space-y-2 text-xs">
            <SummaryRow label="模型名" value="VLX-Seek-1.5-10B" mono />
            <SummaryRow label="服务器" value={endpoint} mono truncate />
            <SummaryRow label="接入方式" value={draft.transport === 'ssh' ? '算力节点 SSH 隧道' : '直接 API'} />
            <SummaryRow label="API Key / Token" value={draft.token ? '新 Token 已填写' : config?.token_configured ? '沿用已保存 Token' : '未填写'} />
            <SummaryRow label="任务类型" value={draft.task} mono />
            <SummaryRow label="超时" value={`${draft.timeout} 秒`} mono />
            <SummaryRow label="抽帧间隔" value={`${draft.frame_interval} 秒`} mono />
          </dl>
        </section>

        <section className="rounded-lg border border-border bg-card/50">
          <div className="px-3 py-2 border-b border-border flex items-center gap-1.5">
            <PlugZap className="w-3.5 h-3.5 text-primary" />
            <span className="text-xs font-medium">测试结果</span>
          </div>
          <div className="p-3 text-xs text-muted-foreground">
            {testing ? '正在读取 VLX 服务状态…' : testResult
              ? <div className="space-y-1.5">
                  <p className={testResult.connected ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-700 dark:text-amber-400'}>
                    {testResult.message || (testResult.connected ? '连接正常' : testResult.status === 'blocked' || testResult.status === 'error' ? '连接失败' : '服务可达，模型尚未加载')}
                  </p>
                  {testResult.status && <p>服务状态：{testResult.status}</p>}
                  {testResult.model && <p>模型：{testResult.model}</p>}
                </div>
              : '还没有测试记录。点击「测试连接」后，这里会显示服务状态。'}
          </div>
        </section>

        <section className="rounded-lg border border-border bg-card/50">
          <div className="px-3 py-2 border-b border-border flex items-center gap-1.5">
            <History className="w-3.5 h-3.5 text-primary" />
            <span className="text-xs font-medium">测试记录</span>
          </div>
          <div className="p-3 space-y-1.5 text-xs text-muted-foreground">
            {testHistory.length ? testHistory.map((entry, index) => (
              <p key={`${entry.at}-${index}`} className="flex gap-2">
                <span className="shrink-0 font-mono">{entry.at}</span>
                <span className={entry.connected ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-700 dark:text-amber-400'}>{entry.message}</span>
              </p>
            )) : '本次会话的测试记录会列在这里。'}
          </div>
        </section>

        <section className="rounded-lg border border-dashed border-border p-3 space-y-2 text-xs text-muted-foreground leading-relaxed">
          <p className="font-medium text-foreground">配置说明</p>
          <p>保存操作只写入 VLX 接入参数；打开此页不会探测接口、启动模型、加载 GPU 或发起推理请求。</p>
          <p>VLX 的目标描述在具体任务中填写；保存的附加提示词会在发起任务时追加。</p>
          <p>配置保存后，可在支持 VLX 的推理入口中选择该模型。</p>
        </section>
      </aside>
    </div>
  )
}

function SummaryRow({ label, value, mono, truncate }: {
  label: string
  value: string
  mono?: boolean
  truncate?: boolean
}) {
  return (
    <div className="flex items-baseline gap-2">
      <dt className="text-muted-foreground shrink-0">{label}</dt>
      <dd className={cn(
        'ml-auto text-right text-foreground',
        mono && 'font-mono text-xs',
        truncate && 'truncate max-w-[190px]'
      )}>
        {value}
      </dd>
    </div>
  )
}
