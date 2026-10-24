'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { Bot, Loader2, Play, Radio, RefreshCw, Square, Trash2, UploadCloud, FileCheck2, CheckCircle2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
// ★告警的全部视觉表达★：整个视频窗口外围的红色呼吸闪烁（纯 CSS，不是 Canvas）
import { LmmAlertGlow } from '@/components/lmm-alert-glow'
import { LmmAlertList, type LmmEvent } from '@/components/lmm-alert-list'
// 目标描述句式 / 任务类型 / 任务说明 与「推理 -> 视频 + 大模型」共用同一份，
// 改文案只改一处，避免两个页面各写一套后行为不一致。
import { PROMPT_PRESETS, readVlxConnectionSummary, SAVED_VLX_PROVIDER, VLX_PROVIDER_ID } from '@/lib/lmm-shared'

/**
 * 多模态视觉告警页。
 *
 * 这一版把页面从「实时跟踪系统」彻底收成「告警系统」，页面上只剩三样东西：
 *
 *   1. **视频区域** —— 纯画面，不画任何框、不叠任何调试信息；
 *   2. **告警区域** —— 视频窗口外围的红色呼吸闪烁（有确认告警时）；
 *   3. **告警记录** —— 时间 / 类别 / 描述 / 一张证据图（触发告警的那一次
 *      检测帧），点图看原图。
 *
 * 已经**删除**（不是隐藏）的东西，以及为什么：
 *   · Tracker FPS / Age / Delay / Tracking 数量 / Lost / Drift / Track ID：
 *     系统里没有任何跟踪器，这些数字要么恒 0 要么是合成值；
 *   · Frame ID / Media Time / PTS / 推理耗时 / 更新频率 / 模型状态 / 算法状态：
 *     是开发期的观测手段，对"有没有事发生"这个问题零贡献；
 *   · 「最近使用的视频」「高级配置」「调试面板」：入口本身就是误导
 *     （高级配置里的参数已由后端配置统一下发，页面上的输入框改了不生效）。
 *
 * RTSP 帧由本机采集器送到预览通道；在线模型只接收抽样帧，凭据始终留在后端。
 */

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const SOURCE_TYPES = [
  { value: 'file', label: '图片／视频文件' },
  { value: 'rtsp', label: 'RTSP 视频流' },
  { value: 'camera', label: '本地摄像头' },
]

interface LmmTask {
  id: string
  status: string
  status_zh?: string
  message?: string
  /** 失败原因（后端在建源/加载模型失败时写入，与 message 同值但语义更明确） */
  error?: string
  labels?: string[]
  display_name?: string
  prompt?: string
  processed?: number
  total_frames?: number
  detections?: number
  /** R1 执行模式：realtime_latest（采集/推理解耦）或 legacy_sync（旧行为） */
  mode?: string
  /** 输入源。RTSP 时是完整 `rtsp://host:port/<path>`。 */
  source?: string
  source_type?: string
  events_count?: number
  model_path?: string
  detector?: string
  warnings?: string[]
}

interface TaskOptions {
  imgsz: number
  confidence: number
  iou: number
  frame_stride: number
  hit_frames: number
  cooldown_seconds: number
  min_targets: number
  engine: 'vlm' | 'vlx'
  vision_provider_id: string
  qwen_interval_seconds: number
  vlx_seek_frame_interval: number
}

interface VisionProvider {
  config_id: string
  display_name: string
  model: string
  kind_label?: string
}

const DEFAULT_OPTIONS: TaskOptions = {
  imgsz: 1280,
  confidence: 0.2,
  iou: 0.45,
  frame_stride: 1,
  hit_frames: 3,
  cooldown_seconds: 30,
  min_targets: 1,
  engine: 'vlm',
  vision_provider_id: '',
  qwen_interval_seconds: 2,
  vlx_seek_frame_interval: 1,
}

/** ★RTSP 画面状态★：给状态页用，与播放器内部的连接状态一一对应。 */
type RtspPhase = 'idle' | 'connecting' | 'live' | 'down'

export function LmmView({ apiUrl }: { apiUrl?: string }) {
  const base = apiUrl || API

  const [prompt, setPrompt] = useState('水面上未穿救生衣的人员')

  const [sourceType, setSourceType] = useState('file')
  const [source, setSource] = useState('')

  const [options, setOptions] = useState<TaskOptions>(DEFAULT_OPTIONS)
  const [visionProviders, setVisionProviders] = useState<VisionProvider[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploaded, setUploaded] = useState<{name: string; kind: string; size: number; preview_url: string} | null>(null)
  const [imageBusy, setImageBusy] = useState(false)
  const [imageResult, setImageResult] = useState<{detections: {class_name: string; bbox: number[]}[]} | null>(null)
  const uploadRef = useRef<HTMLInputElement>(null)
  const uploadRequest = useRef<XMLHttpRequest | null>(null)
  useEffect(() => () => uploadRequest.current?.abort(), [])

  const uploadFile = useCallback((file: File) => {
    if (file.size > 512 * 1024 * 1024) { toast.error('文件不能超过 512 MB'); return }
    uploadRequest.current?.abort()
    setUploading(true); setUploadProgress(0); setUploaded(null); setSource(''); setImageResult(null);setFrameB64(null);setTask(null);setEvents([])
    const xhr = new XMLHttpRequest(); uploadRequest.current = xhr
    xhr.open('POST', `${base}/api/lmm/upload-media`)
    xhr.upload.onprogress = e => { if (e.lengthComputable) setUploadProgress(Math.round(e.loaded / e.total * 100)) }
    xhr.onload = () => {
      if (uploadRequest.current !== xhr) return
      setUploading(false); uploadRequest.current = null
      try {
        const result = JSON.parse(xhr.responseText)
        if (xhr.status >= 400) throw new Error(result.detail || '文件上传失败')
        setSource(result.source); setUploaded(result); setSourceType('file')
        toast.success('文件已上传，请点击开始检测')
      } catch (error) { toast.error(error instanceof Error ? error.message : '文件上传失败') }
    }
    xhr.onerror = () => { if (uploadRequest.current === xhr) { setUploading(false); uploadRequest.current = null; toast.error('上传连接中断，请重试') } }
    const form = new FormData(); form.append('file', file); xhr.send(form)
  }, [base])

  const [task, setTask] = useState<LmmTask | null>(null)
  const [frameB64, setFrameB64] = useState<string | null>(null)
  const [events, setEvents] = useState<LmmEvent[]>([])
  const [channel, setChannel] = useState<'idle' | 'ws' | 'poll'>('idle')

  // ── ★告警光晕★ ─────────────────────────────────────────────────────────
  /** 当前是否正在闪。这是画面上**唯一**的动态元素：没有它，画面就是纯视频。 */
  const [glowing, setGlowing] = useState(false)
  const [rtspPhase, setRtspPhase] = useState<RtspPhase>('idle')
  const rtspTimeoutRef = useRef<number | null>(null)
  /** 闪烁时长（毫秒）。由后端 `pulse_ms` 下发，前端不写死。 */
  const pulseMsRef = useRef(2600)
  const glowTimerRef = useRef<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<number | null>(null)
  const aliveRef = useRef(true)
  const eventsTsRef = useRef(0)

  // ── 初始化：只读取已保存模型与连接配置，不探测模型服务 ───────────────────

  useEffect(() => {
    aliveRef.current = true
    ;(async () => {
      try {
        const r = await fetch(`${base}/api/lmm/config`)
        const d = await r.json()
        const c = d?.config || {}
        if (!aliveRef.current) return
        //: ★参数仍然从后端配置读全★：页面上不再暴露这些输入框（高级配置已删除），
        //: 但下发时必须带上后端生效值，否则会用前端默认值悄悄覆盖掉后端配置。
        setOptions((prev) => {
          const next = { ...prev }
          for (const k of Object.keys(DEFAULT_OPTIONS) as (keyof TaskOptions)[]) {
            if (c[k] !== undefined && c[k] !== null) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              ;(next as any)[k] = c[k]
            }
          }
          return next
        })
        const [qr, vlx] = await Promise.all([
          fetch(`${base}/api/lmm/configured-vision-models`),
          readVlxConnectionSummary(base),
        ])
        const qd = qr.ok ? await qr.json() : {}
        const providers = (Array.isArray(qd?.providers) ? qd.providers as VisionProvider[] : [])
        if (vlx.configured && !providers.some((p) => p.config_id === VLX_PROVIDER_ID)) {
          providers.push(SAVED_VLX_PROVIDER)
        }
        if (!aliveRef.current) return
        setVisionProviders(providers)
        const saved = String(c.vision_provider_id || c.qwen_provider_id || qd?.default_config_id || '')
        const selectedProvider = providers.some((p) => p.config_id === saved)
          ? saved : providers[0]?.config_id || ''
        setOptions((prev) => ({
          ...prev,
          engine: selectedProvider === VLX_PROVIDER_ID ? 'vlx' : 'vlm',
          vision_provider_id: selectedProvider,
        }))
      } catch { /* 用默认值 */ }
    })()
    return () => { aliveRef.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base])

  // ── 告警 → 整窗口红色闪烁 ────────────────────────────────────────────────

  /**
   * 触发一次整窗口闪烁。
   *
   * 用「自调度 `setTimeout` 回收」而不是 `setInterval`：告警是稀疏事件，
   * 常驻定时器会在 0 告警时长跑空转。
   *
   * ★为什么不用 Canvas 画光晕★：Canvas 靠 `requestAnimationFrame` 驱动，而 rAF
   * 在窗口最小化 / 被完全遮挡时**完全停摆** → 光晕会静默消失（实测 `glow=0`）。
   * 现在改成纯 CSS 动画，由合成器驱动，不依赖 rAF。
   */
  const flashAlert = useCallback(() => {
    setGlowing(true)
    if (glowTimerRef.current) window.clearTimeout(glowTimerRef.current)
    glowTimerRef.current = window.setTimeout(
      () => setGlowing(false),
      pulseMsRef.current || 2600,
    )
  }, [])

  const applyPayload = useCallback((p: Record<string, unknown>) => {
    const type = String(p?.type ?? '')

    // ── R1：preview_frame —— **只更新画面**，与 AI 结果完全无关 ──────────
    if (type === 'preview_frame') {
      if (p?.frame_b64) {
        setFrameB64(String(p.frame_b64))
        setRtspPhase('live')
        if (rtspTimeoutRef.current) window.clearTimeout(rtspTimeoutRef.current)
        rtspTimeoutRef.current = window.setTimeout(() => setRtspPhase('down'), 6000)
      }
      return
    }

    // ── R1：detection —— 只取「闪烁时长」，**其余一律不消费** ──────────────
    // 帧号 / 媒体时间 / 推理耗时 / 各类频率都不再进 state：进了 state 就一定会
    // 有人再把它渲染回页面上，那正是这一版要清理的东西。
    if (type === 'detection') {
      if (p?.pulse_ms != null) pulseMsRef.current = Number(p.pulse_ms)
      if (p?.frame_b64) {
        setFrameB64(String(p.frame_b64))
        setRtspPhase('live')
        if (rtspTimeoutRef.current) window.clearTimeout(rtspTimeoutRef.current)
        rtspTimeoutRef.current = window.setTimeout(() => setRtspPhase('down'), 6000)
      }
      return
    }

    // ── 前端**不再消费 `track_update`** ──────────────────────────────────
    // 系统里已经没有任何跟踪器，后端也不再发这条消息。这里留一个显式分支：
    // 万一将来有人把旧后端接回来，前端会**忽略**它，而不是把它的框画上去
    // （那正是"画面上出现静止大框"的老问题）。
    if (type === 'track_update') {
      return
    }

    // ── legacy_sync：一帧里同时带画面与任务状态（旧模式保留，但只取画面） ──
    if (p?.frame_b64) {
      setFrameB64(String(p.frame_b64))
      setRtspPhase('live')
      if (rtspTimeoutRef.current) window.clearTimeout(rtspTimeoutRef.current)
      rtspTimeoutRef.current = window.setTimeout(() => setRtspPhase('down'), 6000)
    }
    if (p?.task) setTask(p.task as LmmTask)
  }, [])

  const stopChannel = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.close() } catch { /* ignore */ }
      wsRef.current = null
    }
    if (pollRef.current) {
      window.clearTimeout(pollRef.current)
      pollRef.current = null
    }
    setChannel('idle')
  }, [])

  /** 轮询兜底：WS 连不上时用它。自调度 setTimeout，避免请求堆积。 */
  const startPolling = useCallback((taskId: string) => {
    const tick = async () => {
      if (!aliveRef.current) return
      try {
        const r = await fetch(`${base}/api/lmm/tasks/${taskId}/snapshot`)
        const d = await r.json()
        if (!aliveRef.current) return
        applyPayload(d)
        if (Array.isArray(d?.events)) {
          const list = d.events as LmmEvent[]
          if (list.length) {
            setEvents(list.slice(-300))
            eventsTsRef.current = Math.max(...list.map((e) => e.ts || 0))
          }
        }
        const st = d?.task?.status
        if (st === 'stopped' || st === 'failed') {
          setChannel('idle')
          return
        }
      } catch { /* 下一轮再试 */ }
      if (aliveRef.current) pollRef.current = window.setTimeout(tick, 700)
    }
    setChannel('poll')
    pollRef.current = window.setTimeout(tick, 300)
  }, [base, applyPayload])

  const channelRef = useRef(channel)
  useEffect(() => { channelRef.current = channel }, [channel])
  const taskRef = useRef(task)
  useEffect(() => { taskRef.current = task }, [task])

  const startChannel = useCallback((taskId: string) => {
    stopChannel()
    let opened = false
    void (async () => {
      try {
        const sessionResponse = await fetch('/api/runos-auth/session', { cache: 'no-store' })
        if (!sessionResponse.ok) throw new Error('RunOS account session is unavailable')
        const session = await sessionResponse.json() as { backend_port: number }
        const wsBase = `${window.location.protocol}//${window.location.hostname}:${session.backend_port}`
        const wsUrl = new URL(`/api/lmm/ws/tasks/${encodeURIComponent(taskId)}`, wsBase)
        wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:'
        const ws = new WebSocket(wsUrl.toString())
        if (!aliveRef.current) { ws.close(); return }
        wsRef.current = ws
        ws.onopen = () => {
          opened = true
          setChannel('ws')
        }
        ws.onmessage = (ev) => {
          try {
            const p = JSON.parse(ev.data)
            if (p?.type === 'event' && p.event) {
              // 事件单独推送：**立刻闪 + 立刻上屏**，不用等下一帧
              const e = p.event as LmmEvent
              setEvents((prev) => (
                prev.some((x) => x.id === e.id) ? prev : [...prev, e].slice(-300)
              ))
              eventsTsRef.current = Math.max(eventsTsRef.current, e.ts || 0)
              flashAlert()
            } else if (p?.type === 'evidence_update' && p.event) {
              // ★证据逐张 ready★：**更新同一条告警**（同一个 id），不新增、不再闪。
              // 告警是"发生了什么"，证据是"物证补上了"，是两件事。
              const e = p.event as LmmEvent
              setEvents((prev) => prev.map((x) => (x.id === e.id ? e : x)))
            } else {
              applyPayload(p)
            }
          } catch { /* 忽略坏帧 */ }
        }
        ws.onerror = () => {
          // 不弹错：自动降级，用户不该因为 WS 不可用就用不了功能
          if (!opened) startPolling(taskId)
        }
        ws.onclose = () => {
          if (aliveRef.current && channelRef.current === 'ws') {
            const st = taskRef.current?.status
            if (st !== 'stopped' && st !== 'failed') startPolling(taskId)
          }
        }
      } catch {
        startPolling(taskId)
      }
    })()
  }, [applyPayload, flashAlert, startPolling, stopChannel])

  const eventsTimerRef = useRef<number | null>(null)

  /** 事件增量拉取（WS 模式下也兜一层，防止丢包漏事件） */
  useEffect(() => {
    if (!task?.id || channel === 'idle') return
    const tick = async () => {
      if (!aliveRef.current) return
      try {
        const r = await fetch(
          `${base}/api/lmm/tasks/${task.id}/events?since=${eventsTsRef.current}&limit=100`)
        const d = await r.json()
        if (!aliveRef.current) return
        const list = (d?.events || []) as LmmEvent[]
        if (list.length) {
          setEvents((prev) => {
            const seen = new Set(prev.map((x) => x.id))
            return [...prev, ...list.filter((x) => !seen.has(x.id))].slice(-300)
          })
          eventsTsRef.current = Math.max(...list.map((e) => e.ts || 0))
          // 轮询兜底时也要闪：只有**刚到的**（2 秒内）才闪，否则补历史会闪成一片
          if (list.some((e) => (Date.now() / 1000 - (e.ts || 0)) < 2)) flashAlert()
        }
      } catch { /* 下一轮再试 */ }
      if (aliveRef.current) eventsTimerRef.current = window.setTimeout(tick, 1500)
    }
    eventsTimerRef.current = window.setTimeout(tick, 800)
    return () => {
      if (eventsTimerRef.current) window.clearTimeout(eventsTimerRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.id, channel, base])

  useEffect(() => () => {
    aliveRef.current = false
    stopChannel()
    if (glowTimerRef.current) window.clearTimeout(glowTimerRef.current)
    if (rtspTimeoutRef.current) window.clearTimeout(rtspTimeoutRef.current)
  }, [stopChannel])

  // ── 任务控制 ─────────────────────────────────────────────────────────────

  const startTask = useCallback(async () => {
    if (!source.trim()) {
      toast.error('请先指定输入源')
      return
    }
    if (!prompt.trim()) {
      toast.error('请先输入目标描述')
      return
    }
    if (options.engine === 'vlm' && !options.vision_provider_id) {
      toast.error('尚无已保存的视觉大模型，请先在「大模型配置」中保存模型')
      return
    }
    try {
      if (sourceType === 'file' && uploaded?.kind === 'image') {
        stopChannel(); setTask(null); setEvents([])
        setImageBusy(true)
        const response = await fetch(`${base}/api/lmm/detect-uploaded-image`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({source, provider_id: options.engine === 'vlx' ? 'vlx-seek' : options.vision_provider_id, prompt}),
        })
        const result = await response.json()
        if (!response.ok) throw new Error(result.detail || '图片检测失败')
        setImageResult(result); setFrameB64(result.annotated_b64)
        toast.success(`图片检测完成，发现 ${result.detections.length} 个目标`)
        return
      }
      // 画面由当前机器本地采集；在线引擎仅收到按间隔抽样的帧。
      const r = await fetch(`${base}/api/lmm/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          source_type: sourceType,
          source,
          options: { ...options },
        }),
      })
      const t = (await r.json()) as LmmTask & { detail?: string }
      if (!r.ok) throw new Error(t?.detail || '创建任务失败')
      if (t.status === 'failed') {
        setTask(t)
        toast.error(t.message || '任务创建失败')
        return
      }
      eventsTsRef.current = 0
      setFrameB64(null)
      setRtspPhase(sourceType === 'rtsp' ? 'connecting' : 'idle')
      setEvents([])
      setTask(t)

      const s = await fetch(`${base}/api/lmm/tasks/${t.id}/start`, { method: 'POST' })
      const st = await s.json()
      if (!s.ok) throw new Error(st?.detail || '启动失败')
      setTask(st as LmmTask)
      startChannel(t.id)
      toast.success('检测已启动')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '启动失败')
    }
    finally { setImageBusy(false) }
  }, [base, options, prompt, source, sourceType, startChannel, stopChannel, uploaded])

  const recoverAlertFrame = useCallback(async (event: LmmEvent) => {
    const taskId = event.task_id
    const sourcePath = task?.id === taskId ? (task.source || source) : source
    const r = await fetch(`${base}/api/lmm/tasks/${taskId}/alerts/${event.id}/recover-evidence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: sourcePath, event }),
    })
    const data = await r.json()
    if (!r.ok) throw new Error(data?.detail || '恢复原始帧失败')
    const updated = data?.event as LmmEvent | undefined
    if (!updated) throw new Error('恢复接口未返回告警图像')
    setEvents((prev) => prev.map((e) => e.id === updated.id ? updated : e))
    return updated
  }, [base, source, task?.id, task?.source])

  const stopTask = useCallback(async () => {
    if (!task?.id) return
    try {
      const r = await fetch(`${base}/api/lmm/tasks/${task.id}/stop`, { method: 'POST' })
      setTask((await r.json()) as LmmTask)
      toast.success('已请求停止')
    } catch {
      toast.error('停止失败')
    }
  }, [base, task?.id])

  const deleteTask = useCallback(async () => {
    if (!task?.id) return
    stopChannel()
    try {
      await fetch(`${base}/api/lmm/tasks/${task.id}`, { method: 'DELETE' })
      setTask(null)
      setFrameB64(null)
      setRtspPhase('idle')
      setEvents([])
      setGlowing(false)
      toast.success('任务已删除')
    } catch {
      toast.error('删除失败')
    }
  }, [base, task?.id, stopChannel])

  // 任务切换 / 重开时清掉上一轮的告警闪烁与 RTSP 连接状态。
  useEffect(() => {
    setGlowing(false)
    if (glowTimerRef.current) window.clearTimeout(glowTimerRef.current)
    if (rtspTimeoutRef.current) window.clearTimeout(rtspTimeoutRef.current)
    if (task?.status === 'running' && (task?.source_type || sourceType) === 'rtsp') {
      setRtspPhase('connecting')
    } else {
      setRtspPhase('idle')
    }
  }, [task?.id, task?.status, sourceType])

  // ── 渲染 ─────────────────────────────────────────────────────────────────

  const running = task?.status === 'running'
  const failed = task?.status === 'failed'
  const activeQuery = task?.prompt || prompt

  // 画面通道分叉：本地视频文件由浏览器播放；RTSP / 摄像头使用本机采集的 JPEG 预览。
  const isRealtime = task?.mode === 'realtime_latest'
  const taskSourceType = task?.source_type || sourceType
  const useVideoChannel = isRealtime && taskSourceType === 'file' && !!task?.id
  const mediaUrl = task?.id ? `${base}/api/lmm/tasks/${task.id}/media` : ''
  const isRtspStage = isRealtime && taskSourceType === 'rtsp' && !!task?.id

  /** 没有画面时的提示文案（任务未开始 / 失败 / 无源）。 */
  const stageHint =
    !task
      ? '填写目标描述与输入源后，点击「开始检测」'
      : failed
        ? `任务失败，未产生画面：${task.error || task.message || '未知错误'}`
        : !frameB64 && (task.status === 'stopped' || task.status === 'ready')
          ? `任务未产生画面（${task.status_zh || task.status}）：${task.message || ''}`
          : '正在等待第一帧…'

  return (
    <div className="flex h-full min-h-0">
      {/* ── 左：任务配置 ─────────────────────────────────────────────── */}
      <aside className="flex w-[340px] shrink-0 flex-col overflow-y-auto border-r border-border">
        <div className="flex items-center gap-2 border-b border-border p-3">
          <Bot className="h-4 w-4 text-muted-foreground" />
          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            多模态目标检测
          </span>
        </div>

        {/* 目标描述 */}
        <div className="space-y-2 border-b border-border p-3">
          <Label className="text-xs">目标描述</Label>
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="例如：水面上未穿救生衣的人员"
            className="min-h-[64px] text-xs"
          />
          <div className="flex flex-wrap gap-1">
            {PROMPT_PRESETS.map((p) => (
              <button
                key={p}
                className="rounded-full border border-border px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
                onClick={() => setPrompt(p)}
              >
                {p}
              </button>
            ))}
          </div>
          {/*
            这里原本是「解析目标」按钮 + 解析结果卡片（目标 / 环境 / 排除 / 判断逻辑 +
            person / swimmer / water 关键词 chips）。已整块删除：
            在线 Qwen-VL 直接消费整句中文，把「水面上 / 未穿 / 救生衣 / 人员」拆成英文关键词
            只会丢掉它们之间的语义关系，还多绕一次模型调用。
          */}
        </div>

        <div className="space-y-2 border-b border-border p-3">
          <div className="flex items-center justify-between gap-2">
            <Label className="text-xs">视觉大模型</Label>
            <Select
              value={options.vision_provider_id ? `vlm:${options.vision_provider_id}` : ''}
              onValueChange={(value) => setOptions({
                ...options,
                engine: value.slice(4) === VLX_PROVIDER_ID ? 'vlx' : 'vlm',
                vision_provider_id: value.slice(4),
              })}>
              <SelectTrigger className="h-8 w-[220px] text-xs">
                <SelectValue placeholder="请选择已保存的模型" />
              </SelectTrigger>
              <SelectContent>
                {visionProviders.map((p) => (
                  <SelectItem key={p.config_id} value={`vlm:${p.config_id}`} className="text-xs">
                    {p.display_name} · {p.model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {!visionProviders.length && (
            <p className="text-[10px] text-amber-700 dark:text-yellow-300">
              请先在「大模型配置」中保存视觉模型。
            </p>
          )}
          <NumberField label="抽帧间隔（秒）" value={options.engine === 'vlx' ? options.vlx_seek_frame_interval : options.qwen_interval_seconds}
            onChange={(v) => setOptions(options.engine === 'vlx'
              ? { ...options, vlx_seek_frame_interval: v }
              : { ...options, qwen_interval_seconds: v })}
            min={0.5} max={60} step={0.5} />
        </div>

        {/* 输入源 */}
        <div className="space-y-2 border-b border-border p-3">
          <Label className="text-xs">输入源</Label>
          <Select value={sourceType} onValueChange={setSourceType}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {SOURCE_TYPES.map((s) => (
                <SelectItem key={s.value} value={s.value} className="text-xs">{s.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {sourceType === 'file' && <div
            className="rounded-xl border border-dashed border-blue-200 bg-gradient-to-br from-blue-50 to-white p-4 text-center"
            onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault(); if (!running && !imageBusy && e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0])}}>
            <input ref={uploadRef} type="file" className="hidden" accept="video/*,.mkv,.ts,image/jpeg,image/png,image/webp,image/bmp,image/tiff"
              onChange={e => {if (e.target.files?.[0]) uploadFile(e.target.files[0]); e.target.value = ''}} />
            {uploaded ? <FileCheck2 className="mx-auto mb-2 h-7 w-7 text-blue-500" /> : <UploadCloud className="mx-auto mb-2 h-7 w-7 text-blue-500" />}
            <p className="truncate text-xs font-medium" title={uploaded?.name}>{uploaded?.name || '拖放图片或视频到这里'}</p>
            <p className="my-2 text-[10px] text-muted-foreground">JPG / PNG 图片 ≤64 MB，视频 ≤512 MB</p>
            <Button variant="outline" size="sm" className="h-8 text-xs" disabled={uploading || running || imageBusy} onClick={() => uploadRef.current?.click()}>
              {uploading ? `上传中 ${uploadProgress}%` : uploaded ? '更换文件' : '选择文件'}
            </Button>
            {uploading && <div role="progressbar" aria-valuenow={uploadProgress} aria-valuemin={0} aria-valuemax={100} className="mt-3 h-1.5 overflow-hidden rounded bg-blue-100"><div style={{width: `${uploadProgress}%`}} className="h-full bg-blue-500" /></div>}
          </div>}
          {sourceType !== 'file' && <Input
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder={sourceType === 'file' ? '视频文件绝对路径' : sourceType === 'rtsp' ? 'rtsp://…' : '0'}
            className="h-8 font-mono text-xs"
          />}
          {/*
            这里原本是「最近使用的视频」列表（读 `/api/lmm/recent-videos` 后列出
            最近 4 个文件名，点击填入）。已删除：输入源是一次性配置，而这个列表
            在换机器/换部署后指向的全是失效路径，点一下就得到"任务失败"。
          */}
        </div>

        {/* 控制按钮 */}
        <div className="space-y-2 p-3">
          <div className="flex gap-2">
            <Button size="sm" className="h-8 flex-1 gap-1 text-xs"
              onClick={startTask} disabled={running || uploading || imageBusy}>
              {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
              {task ? '重新开始' : '开始检测'}
            </Button>
            <Button size="sm" variant="outline" className="h-8 gap-1 text-xs"
              onClick={stopTask} disabled={!running}>
              <Square className="h-3 w-3" />停止
            </Button>
          </div>
          {task && (
            <Button size="sm" variant="ghost" className="h-7 w-full gap-1 text-xs text-muted-foreground"
              onClick={deleteTask}>
              <Trash2 className="h-3 w-3" />删除任务
            </Button>
          )}
          {task?.warnings?.map((w, i) => (
            <p key={i} className="text-[10px] text-amber-700 dark:text-yellow-300">{w}</p>
          ))}
          {imageResult && <div className="rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs">
            <p className="font-medium">检测结果 · {imageResult.detections.length} 个目标</p>
            {imageResult.detections.map((item, i) => <p key={i} className="mt-1">{i + 1}. {item.class_name}</p>)}
            {!imageResult.detections.length && <p className="mt-1">未发现描述中的目标。</p>}
          </div>}
        </div>
      </aside>

      {/* ── 中：实时画面（**纯视频**，只有告警时整个窗口外围才闪） ────────── */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-border px-4 py-2">
          <span className="min-w-0 truncate text-xs font-medium text-foreground"
            title={activeQuery}>
            {activeQuery || '未设置目标'}
          </span>
          <div className="ml-auto flex shrink-0 items-center gap-3 text-[10px] text-muted-foreground">
            {/* 连接状态：实时通道 + RTSP 画面流。其余指标一律不显示。 */}
            <span className="inline-flex items-center gap-1" title="实时通道">
              {imageResult ? <><CheckCircle2 className="h-3 w-3 text-emerald-600" />图片检测完成</> : channel === 'ws'
                ? <><Radio className="h-3 w-3 text-emerald-700 dark:text-green-400" />WebSocket</>
                : channel === 'poll'
                  ? <><RefreshCw className="h-3 w-3 text-amber-700 dark:text-yellow-300" />轮询</>
                  : <><Radio className="h-3 w-3" />未连接</>}
            </span>
            {isRtspStage && (
              <span className="inline-flex items-center gap-1" title="RTSP 画面流">
                {rtspPhase === 'live'
                  ? <><span className="live-dot text-emerald-600" />已连接</>
                  : rtspPhase === 'connecting'
                    ? <><Loader2 className="h-3 w-3 animate-spin" />连接中</>
                    : rtspPhase === 'down'
                      ? <><span className="h-1.5 w-1.5 rounded-full bg-red-600" />已断开</>
                      : <><span className="h-1.5 w-1.5 rounded-full bg-muted-foreground" />未启动</>}
              </span>
            )}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-4">
          {/* ★外圈闪在 overflow-hidden 之外★：容器裁圆角，光晕挂在外层，
              这样"外围"才真的是外围，而不是被裁掉一圈的边框。 */}
          <div className="relative">
            <div className="relative overflow-hidden rounded-lg border border-border bg-black">
              {uploaded?.kind === 'image' && !imageResult ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={`${base}${uploaded.preview_url}`} alt={uploaded.name} className="block h-auto max-h-[calc(100vh-180px)] min-h-[320px] w-full bg-black object-contain" />
              ) : uploaded?.kind === 'video' && !task ? (
                <video src={`${base}${uploaded.preview_url}`} controls className="block h-auto w-full bg-black" />
              ) : isRtspStage ? (
                frameB64 ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`data:image/jpeg;base64,${frameB64}`} alt="本地 RTSP 预览"
                    className="block h-auto max-h-[calc(100vh-180px)] min-h-[320px] w-full bg-black object-contain" />
                ) : (
                  <div className="flex min-h-[320px] items-center justify-center bg-black p-6">
                    <p className="text-center text-xs text-white/65">{stageHint}</p>
                  </div>
                )
              ) : useVideoChannel ? (
                /* 视频通道：画面由浏览器连续播放原始视频，与 AI 结果无关 */
                <video
                  src={mediaUrl}
                  className="block h-auto w-full bg-black"
                  autoPlay
                  muted
                  playsInline
                  controls
                />
              ) : frameB64 ? (
                /* legacy 预览通道：只有一张 JPEG，直接当纯画面显示，不叠任何框 */
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`data:image/jpeg;base64,${frameB64}`}
                  alt="实时画面"
                  className="block h-auto w-full"
                />
              ) : (
                <div className="flex min-h-[320px] items-center justify-center p-6">
                  <p className="text-center text-xs text-muted-foreground">{stageHint}</p>
                </div>
              )}

              {/* RTSP 预览与检测共用本机采集帧；云端只接收抽样 JPEG。 */}
              {isRtspStage && rtspPhase === 'connecting' && !frameB64 && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/80">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">本机正在连接 RTSP 并等待首帧…</p>
                </div>
              )}
              {isRtspStage && rtspPhase === 'down' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-background/90 px-6 text-center">
                  <span className="text-3xl leading-none" role="img" aria-label="视频流断开">📡</span>
                  <p className="text-sm font-medium text-foreground">视频流连接中断</p>
                  <p className="text-[11px] text-muted-foreground">正在等待重新连接…</p>
                </div>
              )}
              {isRtspStage && rtspPhase === 'idle' && !running && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/80 px-6">
                  <p className="text-center text-xs text-muted-foreground">{stageHint}</p>
                </div>
              )}
            </div>

            {/* ★告警的全部视觉表达★：整窗口外围红色呼吸闪烁，不画任何框 */}
            <LmmAlertGlow active={glowing} />
          </div>

          {task?.message && (
            <p className={failed
              ? 'mt-2 text-[11px] text-red-600 dark:text-red-400'
              : 'mt-2 text-[11px] text-muted-foreground'}>
              {failed ? `任务失败：${task.message}` : task.message}
            </p>
          )}

          {/*
            这里原本有四组性能卡片（后端 metrics / R1 实时 perf / WebRTC / VLX 新指标），
            以及一行 VLX 调试信息（task / query / proposal / 推理耗时）。全部删除：
            它们是开发期的观测手段，与"有没有事发生"无关，而且会让人以为画面上的框
            还在被跟踪。页面最终只留：视频区域 + 告警闪烁 + 右侧告警记录。
          */}
        </div>
      </main>

      {/* ── 右：告警记录 ─────────────────────────────────────────────── */}
      <aside className="flex w-[320px] shrink-0 flex-col border-l border-border">
        <div className="flex items-center justify-between border-b border-border p-3">
          <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            告警记录
          </span>
          <span className="font-mono text-[10px] text-muted-foreground">
            {events.length} 条
          </span>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          <LmmAlertList events={events} apiUrl={base} onClear={() => setEvents([])}
            canRecoverLocalVideo={task?.source_type === 'file'}
            onRecoverLocalFrame={recoverAlertFrame} />
        </div>
      </aside>
    </div>
  )
}

function NumberField({ label, value, onChange, min, max, step }: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
}) {
  return (
    <div className="space-y-1">
      <Label className="text-[10px] text-muted-foreground">{label}</Label>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = Number(e.target.value)
          if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)))
        }}
        className="h-7 text-xs"
      />
    </div>
  )
}
