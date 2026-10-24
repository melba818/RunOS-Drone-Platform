"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { FolderInferencePanel } from "@/components/folder-inference-panel"
import { CameraPanel } from "@/components/camera-panel"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
// 目标描述句式 / 任务类型 / 任务说明 与「多模态目标检测」共用同一份
import {
  PROMPT_PRESETS, readVlxConnectionSummary, SAVED_VLX_PROVIDER, VLX_PROVIDER_ID,
} from "@/lib/lmm-shared"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Camera, Upload, Video, Play, Square, Download, Cpu, ZapIcon, Eye,
  RefreshCw, AlertCircle, Crosshair, Wand2, X, ChevronDown, ChevronUp,
  FileText, Braces, RotateCcw, Activity, ScanSearch, Loader2, Bot,
} from "lucide-react"

// ── Interfaces ────────────────────────────────────────────────────────────────

interface ModelInfo {
  id: string; name: string; path: string; model_type: string; status: string
}

interface ModelClass { id: number; name: string }

interface OnlineVisionProvider {
  config_id: string
  display_name: string
  model: string
  kind_label?: string
}

import { hasDetectorConfidence } from "@/lib/model-capabilities"

interface Detection {
  class_name: string; confidence: number; confidence_source?: string
  bbox?: [number, number, number, number]
  mask_points?: [number, number][]
  track_id?: number
}

async function callVlxImage(
  apiUrl: string,
  image: Blob,
  prompt: string,
): Promise<{ annotated_b64: string; detections: Detection[] }> {
  const form = new FormData()
  form.append("image", image, "frame.jpg")
  form.append("prompt", prompt.trim())
  form.append("task", "detection")
  form.append("lang", "zh")
  const response = await fetch(`${apiUrl}/api/lmm/vlx/infer-image`, { method: "POST", body: form })
  const result = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(result.detail || response.statusText || "VLX-Seek 识别失败")

  const bitmap = await createImageBitmap(image)
  const canvas = document.createElement("canvas")
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const ctx = canvas.getContext("2d")
  if (!ctx) {
    bitmap.close()
    throw new Error("无法绘制 VLX-Seek 识别结果")
  }
  ctx.drawImage(bitmap, 0, 0)
  bitmap.close()

  const detections: Detection[] = (Array.isArray(result.boxes) ? result.boxes : []).flatMap((box: any) => {
    const bbox: [number, number, number, number] = [Number(box.x1), Number(box.y1), Number(box.x2), Number(box.y2)]
    if (!bbox.every(Number.isFinite)) return []
    ctx.strokeStyle = "#2563eb"
    ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 400))
    ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
    const label = String(box.label || box.display_name || prompt.trim())
    ctx.font = `${Math.max(14, Math.round(Math.min(canvas.width, canvas.height) / 45))}px sans-serif`
    const labelY = Math.max(18, bbox[1] - 5)
    ctx.fillStyle = "#2563eb"
    ctx.fillRect(bbox[0], labelY - 17, Math.min(ctx.measureText(label).width + 8, canvas.width - bbox[0]), 20)
    ctx.fillStyle = "#fff"
    ctx.fillText(label, bbox[0] + 4, labelY - 2)
    return [{ class_name: label, confidence: 1, confidence_source: "synthetic", bbox }]
  })
  return { annotated_b64: canvas.toDataURL("image/jpeg", 0.9).split(",", 2)[1], detections }
}

/** 「视频 + 大模型」任务（在线视觉模型，后端 /api/lmm/* 的任务结构） */
interface VlxVideoJob {
  id: string; status: string; status_zh?: string
  progress?: number; message?: string; error?: string | null
  processed?: number; total_frames?: number
  detections?: number; events_count?: number
  detector?: string
  /** 按「最少连续出现次数」过滤掉的零星框个数（未过滤时为 0） */
  sporadic_dropped?: number; video_min_hits?: number
  /** 完成后才有值：结果视频 / 检出数据的下载地址由任务 id 拼出 */
  result_video?: string | null; result_json?: string | null
}

/** 「视频推理记录」列表项（GET /api/lmm/video-tasks）。
 *
 *  只用**本页上传过视频**的任务：`/api/lmm/tasks` 的全量列表还混着多模态页的实时
 *  摄像头任务（没有结果视频、没有终点），铺在这里就是"记录跟操作对不上"。 */
interface VlxVideoRecord {
  id: string; prompt?: string; status: string; status_zh?: string; message?: string
  created_at?: number; finished_at?: number | null
  processed?: number; total_frames?: number; detections?: number
  detector?: string; sample_stride?: number
  has_video?: boolean; has_json?: boolean
  video_min_hits?: number | null; sporadic_dropped?: number | null
  source_name?: string
}

/** 一条**面向用户**的运行日志（GET /api/lmm/tasks/{id}/logs）。 */
interface VlxLogLine { ts: number; level?: string; text: string }

/* 「视频 + 大模型」刻意**不做**轨迹 / 告警参数：
   这一页的用途是「跑推理 → 下载结果视频」，而 hit_frames / cooldown / trackless_* /
   tracker 只影响告警事件，视频画的是每个抽样帧的全部框，一个都不受它们影响
   （min_targets 更彻底：原文透传后 conditions 为空，它从不参与判定）。
   与其摆一排点了看不出差别的控件，不如不给。要调这些参数去「多模态目标检测」页。 */

/** 数字输入统一在这里收敛：空串 / NaN / 越界都退化成默认值，别把 NaN 或 0 发给后端。 */
function clampNumber(raw: string, min: number, max: number, fallback: number, integer = false) {
  const n = Number(raw)
  if (!Number.isFinite(n)) return fallback
  const v = Math.min(max, Math.max(min, n))
  return integer ? Math.round(v) : v
}

/** 「视频推理记录」里 lmm 任务状态的中文。**不要**复用 YOLO 视频任务的 VIDEO_STATUS_ZH：
 *  两套状态枚举不一样（那边是 completed/running/…，这边是 stopped/running/…），
 *  混用会把"已完成"的任务显示成英文原样。 */
const VLX_RECORD_STATUS_ZH: Record<string, string> = {
  created: "已创建", parsing: "解析中", ready: "待开始", running: "识别中",
  stopping: "停止中", stopped: "已完成", failed: "失败",
}

/** 日志行的时钟时间。同一次任务跨天的概率极低，日期只会挤占版面。 */
function logClock(ts: number) {
  try {
    return new Date(Number(ts) * 1000).toLocaleTimeString("zh-CN", { hour12: false })
  } catch { return "" }
}

interface VideoJob {
  logs?: string[]
  id: string; status: string; progress: number; message: string; error?: string
  frames?: number; track_count?: number; total_detections?: number; has_tracks?: boolean
  processed?: number | null; total_frames?: number | null
  eta?: number | null; proc_fps?: number | null
  reid_enabled?: boolean; reid_reason?: string; reid_label?: string
  reid_calls?: number; reid_crops?: number; reid_native?: boolean
}

// Status → Chinese label. The wire value stays English (it is compared against
// literals in the polling logic); only the badge text is localised.
const VIDEO_STATUS_ZH: Record<string, string> = {
  starting:  "启动中",
  running:   "运行中",
  completed: "已完成",
  failed:    "失败",
}

interface ReidOfficial {
  name: string; kind: "pt" | "onnx"; path: string
  status: "downloaded" | "not_downloaded" | "downloading" | "error"
  progress: number; size_mb: number | null; error?: string | null
}

// One entry inside a model zone. A zone holds files of a single *role*, and the
// role is what the pipeline consumes — never a specific architecture.
interface ZoneModel {
  id: string; name: string; path: string; zone?: string
  kind?: string | null; arch?: string | null
  status?: string; message?: string
  size_mb?: number | null; out_dim?: number | null; in_dim?: number | null
  source?: string | null; official?: boolean
  tracker_type?: string | null
  // true when this is a raw transformer trunk we can rebuild, which is what makes
  // multi-layer feature aggregation possible.
  aggregatable?: boolean
  meta?: HeadMeta | null
}

// Optional training metadata shipped next to a projection head (<name>.json).
// Dimensions alone cannot tell you how the head was trained, so this is where the
// original feature strategy (if known) is recorded for future use.
interface HeadMeta {
  type?: string; backbone?: string; feature_strategy?: string
  input_dim?: number; output_dim?: number; normalize?: boolean; source?: string
}

interface TrackerOption { value: string; label: string; reid: boolean }

// A backbone shipped with the app (torchvision / torch.hub / torchreid), as
// opposed to one the user imported. `value` is opaque to the UI — it is handed
// straight to /compose, which knows how to build it.
interface BackboneOption {
  value: string; label: string; family: string
  out_dim: number | null; status: string; message: string; note: string
}

interface ModelZoneCatalog {
  onnxruntime: boolean
  official: ReidOfficial[]
  trackers: TrackerOption[]
  backbones: BackboneOption[]
  // Declared server-side (reid_encoders.FEATURE_STRATEGIES) so the UI never owns a
  // second copy of the multiplier table.
  feature_strategies: {
    default: string
    options: { value: string; label: string; mult: number; note: string }[]
    supports: Record<string, boolean>
  }
  zones: {
    detector: ZoneModel[]; tracker: ZoneModel[]; backbone: ZoneModel[]
    reid: ZoneModel[]; head: ZoneModel[]; other: ZoneModel[]
  }
}

interface InferenceViewProps { apiUrl?: string }

// Video jobs run server-side; polling faster than the render cadence only adds
// request churn, so 400ms is the sweet spot for a smooth bar.
const VIDEO_POLL_MS = 400

function formatEta(sec: number): string {
  const s = Math.max(0, Math.round(sec))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return m < 60 ? `${m}m${String(s % 60).padStart(2, "0")}s` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`
}

const TASK_OPTIONS = [
  { value: "detect",  label: "目标检测" },
  { value: "segment", label: "分割" },
  { value: "pose",    label: "姿态 / 关键点" },
  { value: "track",   label: "多目标跟踪" },
]

// Fallback mirror of the backend tracker registry (routers/inference.py TRACKERS);
// the live list is delivered by /api/infer/models/catalog as `trackers`.
const TRACKER_OPTIONS: TrackerOption[] = [
  { value: "bytetrack",  label: "ByteTrack",      reid: false },
  { value: "botsort",    label: "BoT-SORT",       reid: true  },
  { value: "ocsort",     label: "OC-SORT",        reid: false },
  { value: "deepocsort", label: "Deep OC-SORT",   reid: true  },
  { value: "fasttrack",  label: "FastTracker",    reid: false },
  { value: "tracktrack", label: "TrackTrack",     reid: true  },
]

// Import targets per role. `accept` documents what each role expects.
const ZONE_ACCEPT: Record<string, string> = {
  detector: ".pt,.pth,.onnx,.torchscript,.engine,.bin",
  tracker:  ".yaml,.yml",
  backbone: ".pt,.pth,.onnx,.torchscript,.ts,.engine,.bin",
  reid:     ".pt,.pth,.onnx,.torchscript,.engine,.bin",
  head:     ".pt,.pth,.bin,.torchscript,.onnx",
  other:    ".pt,.pth,.onnx,.torchscript,.engine,.bin,.yaml,.yml",
}

const STATUS_COLORS: Record<string, string> = {
  loaded:         "text-emerald-700 dark:text-green-400 border-green-400/40",
  downloading:    "text-blue-700 dark:text-blue-400 border-blue-400/40",
  downloaded:     "text-amber-700 dark:text-yellow-400 border-yellow-400/40",
  not_downloaded: "text-muted-foreground border-border/40",
}

// ── Model zone card ────────────────────────────────────────────────────────────
// A zone is one *role* in the pipeline. Most zones hold a single selectable file;
// a zone may expose several slots when its role naturally pairs two artifacts
// (an appearance encoder with an optional projection head).

// Whether a zone's selection is actually reaching the inference pipeline, plus
// the reason it is not, so the UI never silently ignores an imported model.
interface ZoneStatus {
  live: boolean
  label: string
  reason?: string
}

interface ZoneSlot {
  label?: string                 // shown only when the zone has more than one slot
  badge?: string                 // per-slot state, e.g. which sub-module is driving
  placeholder: string
  items: ZoneModel[]
  value: string
  onSelect: (path: string) => void
  onRemove?: () => void
  optional?: boolean
  simple?: boolean                // plain option row: no import/path/remove controls
  disabled?: boolean
  // Optional because a `simple` row has no import / path controls at all.
  onPickFile?: () => void
  showPath?: boolean
  pathValue?: string
  onPathChange?: (v: string) => void
  onAddPath?: () => void
  onTogglePath?: () => void
  message?: string               // resolved description of the current selection
}

interface ModelZoneProps {
  title: string
  icon: React.ReactNode
  hint: string
  busy?: boolean
  slots: ZoneSlot[]
  status?: ZoneStatus
  notice?: string
  error?: string
  success?: string
  extra?: React.ReactNode
  headerExtra?: React.ReactNode   // controls rendered on the title row (e.g. a mode switch)
}

function ModelZone({
  title, icon, hint, busy, slots, status, notice, error, success, extra, headerExtra,
}: ModelZoneProps) {
  const total = slots.reduce((n, s) => n + s.items.length, 0)
  const multi = slots.length > 1
  return (
    <div className={`rounded-lg border bg-muted/10 p-2.5 space-y-2 ${
      status?.live ? "border-emerald-500/40" : "border-border/60"}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          {icon}
          <span className="text-sm font-medium truncate">{title}</span>
          <Badge variant="outline" className="text-[10px] shrink-0">{total}</Badge>
          {status && (
            <Badge variant="outline" className={`text-[10px] shrink-0 ${
              status.live
                ? "text-emerald-700 dark:text-green-400 border-emerald-500/40"
                : "text-muted-foreground border-border/40"}`}>
              {status.label}
            </Badge>
          )}
        </div>
        {headerExtra ? <div className="shrink-0">{headerExtra}</div> : null}
      </div>

      {slots.map((slot, i) => {
        const selected = slot.items.find(m => m.path === slot.value)
        return (
          <div key={slot.label ?? i} className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <span className={multi ? "text-[11px] font-medium text-muted-foreground" : "hidden"}>
                {slot.label}{slot.optional ? <span className="font-normal"> (optional)</span> : null}
              </span>
              {slot.badge ? (
                <Badge variant="outline" className="text-[10px] shrink-0 text-muted-foreground border-border/40">
                  {slot.badge}
                </Badge>
              ) : null}
              {slot.simple ? null : (
                <div className="flex items-center gap-1 shrink-0">
                <button type="button" onClick={slot.onPickFile} disabled={busy}
                  className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-50">
                  <Upload className="w-3 h-3" />{busy ? "导入中…" : "导入文件"}
                </button>
                <button type="button" onClick={slot.onTogglePath}
                  className="px-1.5 py-0.5 rounded text-[10px] text-muted-foreground hover:text-foreground hover:bg-muted transition-colors">
                  路径
                </button>
                {slot.onRemove && selected && selected.source && !selected.official && (
                  <button type="button" onClick={slot.onRemove} title="移出列表"
                    className="px-1.5 py-0.5 rounded text-[10px] text-muted-foreground hover:text-destructive transition-colors">
                    <X className="w-3 h-3" />
                  </button>
                )}
                </div>
              )}
            </div>

            <Select value={slot.value || "__none__"} disabled={slot.disabled || busy}
              onValueChange={v => slot.onSelect(v === "__none__" ? "" : v)}>
              <SelectTrigger className="h-8 text-xs">
                <SelectValue placeholder={slot.placeholder} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">未选择</SelectItem>
                {slot.items.map(m => (
                  <SelectItem key={m.id || m.path} value={m.path}>
                    <span className="truncate max-w-[180px]">{m.name}</span>
                    {m.out_dim ? <Badge variant="outline" className="ml-1 text-[10px]">{m.out_dim}d</Badge> : null}
                    {m.arch && m.arch !== m.name ? (
                      <Badge variant="outline" className="ml-1 text-[10px]">{m.arch}</Badge>
                    ) : null}
                    {m.status && m.status !== "ready" && m.status !== "downloaded" ? (
                      <Badge variant="outline" className="ml-1 text-[10px] text-amber-700 dark:text-amber-400">
                        {m.status}
                      </Badge>
                    ) : null}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {slot.showPath && (
              <div className="flex gap-1.5">
                <Input value={slot.pathValue ?? ""} onChange={e => slot.onPathChange?.(e.target.value)}
                  placeholder="粘贴绝对路径，如 D:\models\model.pt" className="h-7 text-[11px] font-mono" />
                <Button size="sm" className="h-7 px-2 text-[11px]" onClick={slot.onAddPath}>添加</Button>
              </div>
            )}

            {slot.message && (
              <p className="text-[10px] text-muted-foreground leading-snug">{slot.message}</p>
            )}
          </div>
        )
      })}

      {status?.reason && (
        <p className="text-[10px] text-amber-700 dark:text-amber-400 leading-snug">{status.reason}</p>
      )}
      {error && (
        <p className="text-[10px] text-red-600 dark:text-red-400 leading-snug">{error}</p>
      )}
      {success && (
        <p className="text-[10px] text-emerald-700 dark:text-green-400 leading-snug">{success}</p>
      )}
      {notice && (
        <p className="text-[10px] text-blue-700 dark:text-blue-400 leading-snug">{notice}</p>
      )}
      {extra}
      {!total && !extra && <p className="text-[10px] text-muted-foreground">{hint}</p>}
    </div>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────

export function InferenceView({ apiUrl = "http://localhost:8000" }: InferenceViewProps) {
  const [computeNodeId, setComputeNodeId] = useState("local")
  const [computeNodes, setComputeNodes] = useState<{id:string; name:string; status:string; connection_type:string}[]>([])
  const [nodesError, setNodesError] = useState("")
  const executionUrl = computeNodeId === "local" ? apiUrl : `${apiUrl}/api/compute-inference/${computeNodeId}`
  async function refreshComputeNodes() {
    try {
      const res = await fetch(`${apiUrl}/api/compute-nodes`)
      if (!res.ok) throw new Error("算力节点读取失败")
      const data = await res.json(); setComputeNodes(data.nodes); setNodesError("")
    } catch(e) { setNodesError(e instanceof Error ? e.message : "算力节点读取失败") }
  }
  useEffect(() => { refreshComputeNodes() }, [apiUrl])


  // ── Model source ─────────────────────────────────────────────────────────────
  const [modelSource, setModelSource] = useState<"yolo" | "rfdetr" | "smart">("yolo")

  // ── YOLO state ───────────────────────────────────────────────────────────────
  const [yoloModels, setYoloModels]       = useState<ModelInfo[]>([])
  const [yoloSelected, setYoloSelected] = useState("")
  const [rfSelected, setRfSelected] = useState("")
  const selectedModel = modelSource === "rfdetr" ? rfSelected : yoloSelected
  const setSelectedModel = modelSource === "rfdetr" ? setRfSelected : setYoloSelected
  const [yoloPath, setYoloPath] = useState("")
  const [rfPath, setRfPath] = useState("")
  const manualPath = modelSource === "rfdetr" ? rfPath : yoloPath
  const setManualPath = modelSource === "rfdetr" ? setRfPath : setYoloPath
  const [task, setTask]                   = useState("detect")
  // Tracker reference: a bundled algorithm name, or the path of an imported tracker
  // config (whose algorithm is resolved by the backend).
  const [tracker, setTracker]             = useState("bytetrack")
  // ── Appearance module (optional): two interchangeable ways to get embeddings ──
  // 1) OSNet alone (a dedicated ReID trunk), 2) Backbone + Projection Head (any trunk + a head).
  // They are mutually exclusive: picking one clears the other, and each one is a
  // complete, runnable appearance model on its own.
  const [reidMode, setReidMode]           = useState<"osnet" | "backbone">("backbone")
  const [osnetModel, setOsnetModel]       = useState("")
  const [reidModel, setReidModel]         = useState("")   // backbone of the backbone+head path
  const [reidHead, setReidHead]           = useState("")
  const [featureStrategy, setFeatureStrategy] = useState<string>("")  // "" = not chosen yet
  const [reidComposed, setReidComposed]   = useState<{ path: string; out_dim?: number | null; message?: string } | null>(null)
  const [reidPairErr, setReidPairErr]     = useState("")
  const [otherModel, setOtherModel]       = useState("")   // "其它" zone (not used by inference)
  const [trackerNotice, setTrackerNotice] = useState("")
  const [newTrackerType, setNewTrackerType] = useState("botsort")  // type declared when importing a tracker
  const [showTrackAdvanced, setShowTrackAdvanced] = useState(false)
  const [trackBuffer, setTrackBuffer]     = useState(30)
  const [trackHighThresh, setTrackHighThresh] = useState(0.25)
  const [newTrackThresh, setNewTrackThresh]   = useState(0.25)
  const [matchThresh, setMatchThresh]         = useState(0.8)
  const [iou, setIou]                     = useState(0.45)
  const [imgsz, setImgsz]                 = useState(640)
  const [classData, setClassData] = useState<{
    key: string; options: ModelClass[]; selected: number[]; error: string | null
  }>({ key: "", options: [], selected: [], error: null })
  const [classSearch, setClassSearch] = useState("")
  const [classReload, setClassReload] = useState(0)

  // ── Smart model state ─────────────────────────────────────────────────────────
  const [smartImgsz, setSmartImgsz]     = useState(640)
  const [localVisionProviders, setLocalVisionProviders] = useState<OnlineVisionProvider[]>([])
  const [localVisionProviderId, setLocalVisionProviderId] = useState("")
  const [localVisionPrompt, setLocalVisionPrompt] = useState("")
  const [localVisionLoading, setLocalVisionLoading] = useState(false)
  const [localVisionError, setLocalVisionError] = useState("")

  // ── Shared ────────────────────────────────────────────────────────────────────
  const [confidence, setConfidence] = useState(0.25)

  // Ref always holds live params for webcam interval
  const activeModelPath = manualPath.trim() || selectedModel
  const classKey = `${apiUrl}\n${activeModelPath}`
  // Never send IDs from a previously selected model, even before its effect runs.
  const selectedClasses = classData.key === classKey ? classData.selected : []
  const modelClasses = classData.key === classKey ? classData.options : []
  const classError = classData.key === classKey ? classData.error : null
  const classesLoading = modelSource !== "smart" && !!activeModelPath && classData.key !== classKey
  const paramsRef = useRef({
    modelSource, task, confidence, iou, imgsz,
    activeModelPath, tracker, smartImgsz, selectedClasses, classesLoading,
    computeNodeId, localVisionProviderId, localVisionPrompt,
    reidMode, osnetModel, reidModel, reidHead,
    trackBuffer, trackHighThresh, newTrackThresh, matchThresh,
    reidActive: false, reidEffPath: "",   // resolved below; refreshed on every render
  })
  useEffect(() => {
    paramsRef.current = {
      modelSource, task, confidence, iou, imgsz,
      activeModelPath, tracker, smartImgsz, selectedClasses, classesLoading,
      computeNodeId, localVisionProviderId, localVisionPrompt,
      reidMode, osnetModel, reidModel, reidHead,
      trackBuffer, trackHighThresh, newTrackThresh, matchThresh,
      reidActive, reidEffPath,
    }
  })

  // ── Model zones: one list per role in the pipeline ────────────────────────────
  const [zones, setZones]                 = useState<ModelZoneCatalog | null>(null)
  const [zonePath, setZonePath]           = useState<Record<string, string>>({})
  const [openPathZone, setOpenPathZone]   = useState<string | null>(null)
  const [zoneBusy, setZoneBusy]           = useState<string | null>(null)
  const [uploadZone, setUploadZone]       = useState<string | null>(null)
  const zoneFileRef = useRef<HTMLInputElement>(null)

  async function refreshZones() {
    try {
      const r = await fetch(`${apiUrl}/api/infer/models/catalog`)
      if (r.ok) setZones(await r.json())
    } catch {}
  }

  useEffect(() => { refreshZones() }, [apiUrl])

  // ── Restore the ReID selection across reloads ──────────────────────────────
  // Nothing else in this panel is persisted today, so this keeps to its own key and
  // to the existing `cvdm_` prefix. Missing/unknown keys fall back to their defaults,
  // which is what makes configs saved before `feature_strategy` existed still load.
  const REID_STORE_KEY = "cvdm_infer_reid"
  useEffect(() => {
    try {
      const raw = localStorage.getItem(REID_STORE_KEY)
      if (!raw) return
      const s = JSON.parse(raw) as Partial<{
        mode: "osnet" | "backbone"; osnet: string; backbone: string
        head: string; feature_strategy: string
      }>
      if (s.mode === "osnet" || s.mode === "backbone") setReidMode(s.mode)
      if (s.osnet) setOsnetModel(s.osnet)
      if (s.backbone) setReidModel(s.backbone)
      if (s.head) setReidHead(s.head)
      if (s.feature_strategy) setFeatureStrategy(s.feature_strategy)
    } catch {}
  }, [])
  useEffect(() => {
    try {
      localStorage.setItem(REID_STORE_KEY, JSON.stringify({
        mode: reidMode, osnet: osnetModel, backbone: reidModel,
        head: reidHead, feature_strategy: featureStrategy,
      }))
    } catch {}
  }, [reidMode, osnetModel, reidModel, reidHead, featureStrategy])

  async function importModel(zone: string, file: File | null, path: string | null) {
    setZoneBusy(zone)
    try {
      const fd = new FormData()
      fd.append("zone", zone)
      if (file) fd.append("file", file)
      if (path) fd.append("path", path)
      // A tracker config does not always declare its algorithm, so state it explicitly.
      if (zone === "tracker") fd.append("tracker_type", newTrackerType)
      const r = await fetch(`${apiUrl}/api/infer/models/import`, { method: "POST", body: fd })
      const data = await r.json().catch(() => ({})) as
        { detail?: string; ok?: boolean; error?: string | null
          entry?: { path?: string; status?: string; message?: string; tracker_type?: string
            kind?: string | null; arch?: string | null; out_dim?: number | null } | null }
      if (!r.ok || data.ok === false) {
        setError(data.detail || data.error || `导入失败（HTTP ${r.status}）`)
      } else {
        setError(null)
        if (path) setZonePath(p => ({ ...p, [zone]: "" }))
        const entry = data.entry
        if (entry?.status && entry.status !== "ready") {
          setError(entry.message || `导入未生效（${entry.status}）`)
        }
        // A newly imported artifact should be usable right away, not merely listed.
        // ReID imports land in whichever sub-module the file actually belongs to.
        if ((zone === "backbone" || zone === "reid") && entry?.path) {
          const asItem: ZoneModel = { id: `new:${entry.path}`, name: "", path: entry.path,
            kind: entry.kind ?? "unknown", arch: entry.arch ?? undefined,
            out_dim: entry.out_dim ?? null, zone }
          useReidSubModule(isOsnet(asItem) ? "osnet" : "backbone", entry.path)
        }
        else if (zone === "detector" && entry?.path) { setSelectedModel(entry.path); setManualPath("") }
        else if (zone === "head" && entry?.path) pickReidHead(entry.path)
        else if (zone === "tracker" && entry?.path) {
          setTracker(entry.path)
          setTrackerNotice("")
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setZoneBusy(null)
      refreshZones()
    }
  }

  async function removeZoneEntry(id: string) {
    try {
      await fetch(`${apiUrl}/api/infer/models/${id}`, { method: "DELETE" })
    } catch {}
    refreshZones()
  }

  const zoneItems = (name: keyof ModelZoneCatalog["zones"]) => {
    const items = zones?.zones[name] ?? []
    if (name !== "detector") return items
    return items.filter(m => modelSource === "rfdetr" ? m.kind === "rfdetr" : m.kind !== "rfdetr")
  }

  // ── Tracker role: an algorithm name, or an imported config with a declared type ──
  const trackerOptions: TrackerOption[] = zones?.trackers?.length ? zones.trackers : TRACKER_OPTIONS
  const trackerEntry = zoneItems("tracker").find(m => m.path === tracker)
  const trackerType = trackerEntry?.tracker_type
    || (trackerOptions.some(o => o.value === tracker) ? tracker : "bytetrack")
  const trackerMeta = trackerOptions.find(t => t.value === trackerType)
  const trackerSupportsReid = trackerMeta?.reid ?? false

  // ── Appearance role: backbone (+ optional projection head) ──
  // The backbone zone holds the feature extractor. Older builds imported encoders
  // into `reid`; the backend mirrors them into `backbone`, but keep a local union
  // so an entry is reachable even while an older backend is running.
  // Built-in trunks first (OSNet family, then the generic backbones), then
  // whatever the user imported — one dropdown covering every way to get features.
  const backboneBuiltins: ZoneModel[] = (zones?.backbones ?? []).map(b => ({
    id: `builtin:${b.value}`, name: `${b.label}（内置）`, path: b.value,
    kind: "builtin", arch: b.family, zone: "backbone",
    out_dim: b.out_dim, status: b.status,
    // Built-ins are either ready (weights already on disk, or downloadable) or not;
    // there is no license-gated state any more — the DINOv3 trunks ship as local files.
    message: b.status === "ready"
      ? `${b.note} · 输出 ${b.out_dim} 维。${b.message}`
      : `${b.label} 暂不可用：${b.message}`,
  }))
  const backboneItems: ZoneModel[] = [...backboneBuiltins]
  {
    // `reid` is the legacy alias of the `backbone` zone, so the same weights can be
    // registered in both (each zone keeps its own copy on disk). Identity is therefore
    // the role + display name + width, not the path — otherwise one model would be
    // offered twice, and a head saved here must never be offered as a backbone at all.
    const seen = new Set<string>()
    for (const m of [...zoneItems("backbone"), ...zoneItems("reid")]) {
      const key = `${m.kind}|${m.name}|${m.out_dim ?? ""}`
      if (m.kind === "head" || seen.has(key)) continue
      seen.add(key)
      backboneItems.push(m)
    }
  }
  // OSNet is its own sub-module: a purpose-built ReID trunk that already outputs a
  // ready-to-use embedding, so it never needs a projection head. Everything else
  // belongs to the Backbone + Projection Head sub-module.
  const isOsnet = (m: ZoneModel) =>
    m.kind === "osnet" || m.arch === "OSNet" || /osnet/i.test(`${m.name ?? ""} ${m.arch ?? ""}`)
  const osnetItems = backboneItems.filter(isOsnet)
  const encItems = backboneItems.filter(m => !isOsnet(m))

  // Whichever sub-module is active supplies the encoder; the other is kept empty so
  // there is never any ambiguity about what actually runs.
  const reidModePath = reidMode === "osnet" ? osnetModel.trim() : reidModel.trim()
  const isBuiltinBackbone = reidModePath.startsWith("builtin:")
  const reidEncEntry = backboneItems.find(m => m.path === reidModePath)
  const osnetEntry = osnetItems.find(m => m.path === osnetModel)
  const reidHeadEntry = zoneItems("head").find(m => m.path === reidHead)
  const headMeta: HeadMeta | undefined = reidHeadEntry?.meta ?? undefined
  const headInDim = reidHeadEntry?.in_dim ?? null

  // ── Feature aggregation ───────────────────────────────────────────────────
  // The numbers here come from the backend table (mult), so the UI can never quote a
  // different width than the exporter does.
  const strategyOptions = zones?.feature_strategies?.options ?? []
  const strategyDefault = zones?.feature_strategies?.default ?? "cls"
  const activeStrategy = featureStrategy.trim() || strategyDefault
  const strategySpec = strategyOptions.find(s => s.value === activeStrategy)
  // Only a transformer trunk we rebuild can be aggregated; anything else (CNN, OSNet,
  // an already-exported module) keeps whatever it already outputs.
  const aggregationSupported = reidMode === "backbone" && !!reidModePath && (
    isBuiltinBackbone
      ? zones?.feature_strategies?.supports?.[reidModePath] ?? false
      : reidEncEntry?.aggregatable ?? /vit/i.test(`${reidEncEntry?.arch ?? ""} ${reidEncEntry?.name ?? ""}`)
  )
  const baseDim = reidEncEntry?.out_dim ?? null
  const aggDim = baseDim && strategySpec && aggregationSupported
    ? baseDim * strategySpec.mult
    : baseDim
  const strategyItems: ZoneModel[] = strategyOptions.map(s => ({
    id: s.value, name: s.label, path: s.value,
    out_dim: baseDim && aggregationSupported ? baseDim * s.mult : null,
    message: s.note,
  }))
  // One strategy whose width feeds the selected head; suggested, never silently forced.
  const suggestedStrategy = headInDim && baseDim && aggregationSupported
    ? strategyOptions.find(s => baseDim * s.mult === headInDim)?.value : undefined

  // The head only participates in the backbone+head sub-module.
  const headPath = reidMode === "backbone" ? reidHead.trim() : ""

  // Compatibility is decided on the *aggregated* width — that is what the head sees.
  const pairCheckable = typeof aggDim === "number" && typeof headInDim === "number"
  const pairMismatch = pairCheckable && aggDim !== headInDim
  const reidPairErrFinal = pairMismatch
    ? `Projection Head 需要输入 ${headInDim} 维特征，当前特征聚合方式输出 ${aggDim} 维，两者不匹配，无法串接。`
    : reidPairErr
  // Human-readable pipeline, e.g. 384d → 1536d → 128d.
  const featureChain = (() => {
    if (!baseDim) return null
    const parts = [`${baseDim}d`]
    if (aggregationSupported && aggDim !== baseDim && strategySpec) parts.push(`${aggDim}d`)
    if (!reidHead.trim() || headInDim !== aggDim) return null
    if (typeof reidHeadEntry?.out_dim === "number") parts.push(`${reidHeadEntry.out_dim}d`)
    return parts.length > 1 ? parts.join(" → ") : null
  })()
  // What actually gets handed to the tracker: the encoder alone, or the composed pair.
  // OSNet files are already complete models, so composing only kicks in for built-ins.
  const reidEffPath = reidComposed ? reidComposed.path : reidModePath
  // A non-default read-out changes the embedding itself, so it has to be baked into a
  // file even when no projection head is attached.
  const needsCompose = !!reidModePath && (isBuiltinBackbone || !!headPath
    || (aggregationSupported && activeStrategy !== strategyDefault))
  // A built-in has no file on disk until it is composed, so wait for that.
  const reidReady = !!reidModePath && !reidPairErrFinal
    && (!needsCompose || !!reidComposed)
  const reidActive = computeNodeId === "local" && reidReady && task === "track" && trackerSupportsReid


  // Which roles are actually driving inference right now, and why not when they aren't.
  const detectorStatus: ZoneStatus = activeModelPath
    ? { live: true, label: "使用中" }
    : { live: false, label: "未选择" }

  // A tracker only runs for tracking tasks: on detect/segment/pose the selection is
  // remembered but nothing is associated, so it must not claim to be in use.
  const trackerStatus: ZoneStatus = task !== "track"
    ? { live: false, label: "未生效",
        reason: "跟踪器只在「任务类型 = 多目标跟踪」时参与计算。" }
    : { live: true,
        label: trackerEntry ? `${trackerEntry.tracker_type ?? "自定义"} · 使用中` : "使用中" }

  // One status for the whole ReID module: whichever sub-module is chosen reports here.
  const reidStatus: ZoneStatus = !reidModePath
    ? { live: false, label: "未选择",
        reason: "不挂外观特征时跟踪只用运动信息；任选一种方式即可启用（OSNet 或 Backbone + Projection Head）。" }
    : !reidActive
      ? { live: false, label: "未生效", reason:
          task !== "track"
            ? "外观特征只在「任务类型 = 多目标跟踪」时参与计算。"
            : !trackerSupportsReid
              ? `当前跟踪器（${trackerMeta?.label ?? trackerType}）不支持外观特征；改用支持外观的跟踪器即可启用。`
              : reidPairErrFinal }
      : { live: true, label: reidMode === "osnet" ? "使用中 · OSNet" : "使用中 · Backbone + Head" }

  // Per-sub-module hints, so the inactive one can explain why it is not driving.
  const osnetSubStatus: ZoneStatus = !osnetModel.trim()
    ? { live: false, label: "未选择" }
    : reidMode !== "osnet"
      ? { live: false, label: "未启用", reason: "当前使用「Backbone + Projection Head」方式；选一个 OSNet 即切回本方式。" }
      : { live: reidActive, label: reidActive ? "使用中" : "已就绪" }

  const backboneSubStatus: ZoneStatus = !reidModel.trim()
    ? { live: false, label: "未选择" }
    : reidMode !== "backbone"
      ? { live: false, label: "未启用", reason: "当前使用 OSNet 方式；选一个 Backbone 即切回本方式。" }
      : typeof aggDim === "number"
        ? { live: reidActive, label: reidActive ? "使用中" : "已就绪" }
        : { live: false, label: "维度未知", reason: "无法读出该 Backbone 的输出维度，因此不能与 Projection Head 配对校验。" }

  const aggregatorStatus: ZoneStatus = !aggregationSupported
    ? { live: false, label: "不适用",
        reason: reidModePath
          ? "该 Backbone 不是可重建的 Transformer 主干，直接使用它自身的输出。" : undefined }
    : pairCheckable && !pairMismatch
      ? { live: reidActive, label: "维度兼容" }
      : { live: false, label: `${aggDim ?? "?"}d` }

  const otherStatus: ZoneStatus = zoneItems("other").length
    ? { live: false, label: "不参与推理", reason: "「其它」区的文件只做集中存放，不会被加载运行。" }
    : { live: false, label: "未选择" }

  // Picking an appearance model is itself the intent to use it: switch the module to
  // that sub-module, clear the other, and — when the active tracker cannot use
  // appearance — move to a tracker that can, so the selection is never inert.
  function useReidSubModule(mode: "osnet" | "backbone", path: string) {
    setReidMode(mode)
    setTrackerNotice("")
    // The two sub-modules never run together: the one not chosen is emptied.
    if (mode === "osnet") { setOsnetModel(path); setReidModel(""); setReidHead("") }
    else { setReidModel(path); setOsnetModel("") }
    setReidComposed(null)
    setReidPairErr("")
    const bi = backboneBuiltins.find(b => b.path === path)
    if (bi && bi.status !== "ready") { setError(bi.message || "该模型暂不可用"); return }
    if (!path || trackerSupportsReid) return
    const fallback = trackerOptions.find(t => t.reid)
    if (fallback) {
      setTracker(fallback.value)
      setTrackerNotice(`当前跟踪器不支持外观特征，已自动切换为 ${fallback.label}。`)
    }
  }
  const pickOsnetModel = (path: string) => useReidSubModule("osnet", path)
  const pickFeatureStrategy = (value: string) => {
    setFeatureStrategy(value)
    setReidComposed(null)
    setReidPairErr("")
  }
  // Choosing a backbone or a head resets the read-out suggestion, because both change
  // what width would be a match. Nothing here is permanent and nothing is hidden: the
  // picked strategy stays visible in the dropdown and can always be changed by hand.
  const pickReidModel = (path: string) => {
    setFeatureStrategy("")
    useReidSubModule("backbone", path)
  }
  // Flipping the switch only changes which sub-module is being configured; a
  // sub-module with no selection simply means appearance features are off.
  const switchReidMode = (mode: "osnet" | "backbone") => {
    setReidMode(mode)
    setReidComposed(null)
    setReidPairErr("")
  }
  // Touching the head implies the backbone+head way of working.
  const pickReidHead = (path: string) => {
    setReidHead(path)
    if (path) { setReidMode("backbone"); setOsnetModel("") }
    setFeatureStrategy("")   // let the read-out be re-suggested for this head
  }

  // Suggest an aggregation strategy only while the user has not picked one (`""`):
  // a head that declares how it was trained wins, otherwise the width that lines up
  // with its input is offered. An explicit choice is never overwritten.
  useEffect(() => {
    if (featureStrategy.trim() || !aggregationSupported || !headInDim || !baseDim) return
    const declared = headMeta?.feature_strategy
    if (declared && strategyOptions.some(s => s.value === declared)) {
      setFeatureStrategy(declared)
      return
    }
    if (suggestedStrategy) setFeatureStrategy(suggestedStrategy)
  }, [featureStrategy, aggregationSupported, headInDim, baseDim, strategyOptions,
      suggestedStrategy, headMeta])

  // Materialise the active sub-module into one loadable file. A built-in has no file
  // until it is built; an imported model is already a file and only needs re-export
  // when a head is chained onto it (which never happens for OSNet).
  useEffect(() => {
    if (!reidModePath) {
      setReidComposed(null)
      setReidPairErr("")
      return
    }
    const head = headPath
    // A built-in must be materialised, an aggregation other than the plain read-out
    // must be baked in, and a head has to be folded onto the encoder. Only an
    // imported model with the default read-out is already a usable file as it stands.
    if (!needsCompose) {
      setReidComposed(null)
      setReidPairErr("")
      return
    }
    if (pairMismatch) {
      setReidComposed(null)
      setReidPairErr("")
      return
    }
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/infer/models/compose`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ encoder: reidModePath, head,
            // "" lets the backend use its default, which keeps old configs working.
            feature_strategy: aggregationSupported ? activeStrategy : undefined }),
        })
        const d = await r.json().catch(() => ({})) as
          { detail?: string; path?: string; out_dim?: number | null; message?: string }
        if (cancelled) return
        if (!r.ok || !d.path) { setReidComposed(null); setReidPairErr(d.detail || "串接失败") }
        else { setReidComposed(d as { path: string; out_dim?: number | null; message?: string }); setReidPairErr("") }
      } catch (e) {
        if (!cancelled) { setReidComposed(null); setReidPairErr(e instanceof Error ? e.message : String(e)) }
      }
    }, 200)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [apiUrl, reidMode, reidModePath, reidHead, pairMismatch, needsCompose,
      aggregationSupported, activeStrategy, strategyDefault])

  function trackerParams() {
    return {
      tracker,
      with_reid: reidActive,
      reid_model: reidActive && reidEffPath ? reidEffPath : null,
      track_high_thresh: trackHighThresh,
      new_track_thresh: newTrackThresh,
      match_thresh: matchThresh,
      track_buffer: trackBuffer,
    }
  }

  // ── Source tab ────────────────────────────────────────────────────────────────
  const [sourceTab, setSourceTab] = useState<"image" | "video" | "webcam" | "camera">("image")

  const [imageMode, setImageMode] = useState<"single" | "folder">("single")

  // ── Image state ───────────────────────────────────────────────────────────────
  const [imageFile, setImageFile]       = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [inferring, setInferring]       = useState(false)
  const [resultImg, setResultImg]       = useState<string | null>(null)
  const [detections, setDetections]     = useState<Detection[]>([])
  const [error, setError]               = useState<string | null>(null)

  // ── Video state ───────────────────────────────────────────────────────────────
  // Progress is polled from a background job, so whatever the server reports
  // arrives in discrete samples. `videoProgress` is a client-side interpolated
  // value that eases toward the latest sample, which keeps the bar gliding
  // instead of hopping once per poll.
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoJob, setVideoJob]   = useState<VideoJob | null>(null)
  const [videoHistory, setVideoHistory] = useState<{node_id:string;job_id:string;created_at:number;snapshot:VideoJob}[]>([])
  const [restoreVideoId, setRestoreVideoId] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState("")
  const [videoSubmitting, setVideoSubmitting] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStage, setUploadStage] = useState("")
  const [remoteUpload, setRemoteUpload] = useState<{stage:string;sent:number;total:number} | null>(null)
  const [videoPollError, setVideoPollError] = useState("")
  const [videoProgress, setVideoProgress] = useState(0)
  const videoTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null)
  const videoInFlightRef  = useRef(false)
  const videoJobIdRef     = useRef<string | null>(null)
  const videoTargetRef    = useRef(0)

  const stopVideoPolling = useCallback(() => {
    if (videoTimerRef.current) { clearTimeout(videoTimerRef.current); videoTimerRef.current = null }
    videoInFlightRef.current = false
    videoJobIdRef.current = null
  }, [])

  // Ease the displayed value toward the last reported sample while a job runs.
  // Keyed on id/status so it survives the per-poll object identity change.
  useEffect(() => {
    const st = videoJob?.status
    if (!st || st === "completed" || st === "failed") return
    const id = setInterval(() => {
      const target = videoTargetRef.current
      setVideoProgress(cur => (Math.abs(target - cur) < 0.05 ? target : cur + (target - cur) * 0.22))
    }, 70)
    return () => clearInterval(id)
  }, [videoJob?.id, videoJob?.status])

  useEffect(() => stopVideoPolling, [stopVideoPolling])

  // ── 视频 + 大模型（在线视觉模型）──────────────────────────────────────────
  // 视频上传到本机后端落盘，抽样帧送到选定服务器模型，检出框画好写成 mp4 供下载。
  const [vlxFile, setVlxFile]             = useState<File | null>(null)
  const [vlxPrompt, setVlxPrompt]         = useState("")
  const [videoVisionProviders, setVideoVisionProviders] = useState<OnlineVisionProvider[]>([])
  const [videoVisionProviderId, setVideoVisionProviderId] = useState("")
  const [videoVisionLoading, setVideoVisionLoading] = useState(false)
  const [videoVisionError, setVideoVisionError] = useState("")
  const [videoVisionInterval, setVideoVisionInterval] = useState("2")
  // 在线模型的推理速度差异较大，允许按秒控制抽帧间隔。
  // 「最少连续出现次数」：可压掉模型偶发的零星框。1 = 不过滤
  const [vlxMinHits, setVlxMinHits]       = useState("1")
  const [vlxJob, setVlxJob]               = useState<VlxVideoJob | null>(null)
  const [vlxSubmitting, setVlxSubmitting] = useState(false)
  const [vlxUploadPct, setVlxUploadPct]   = useState(0)
  const [vlxError, setVlxError]           = useState("")
  // 「识别日志」：后端按任务记的**面向用户**的运行轨迹，增量拉取后只做 append。
  // 「视频推理记录」：只列本页上传过视频的任务（见 VlxVideoRecord 的说明）。
  const [vlxLogs, setVlxLogs]             = useState<VlxLogLine[]>([])
  const [vlxHistory, setVlxHistory]       = useState<VlxVideoRecord[]>([])
  const vlxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const vlxLogsSinceRef = useRef(0)
  const vlxLogBoxRef = useRef<HTMLDivElement | null>(null)
  const vlxRunning  = !!vlxJob && vlxJob.status !== "stopped" && vlxJob.status !== "failed"

  const stopVlxPolling = useCallback(() => {
    if (vlxTimerRef.current) { clearTimeout(vlxTimerRef.current); vlxTimerRef.current = null }
  }, [])

  // 只从「大模型配置」读取已保存的视觉模型元数据，不探测服务；响应不含 endpoint 或密钥。
  const refreshLocalVisionProviders = useCallback(async () => {
    setLocalVisionLoading(true)
    try {
      const [r, vlx] = await Promise.all([
        fetch(`${apiUrl}/api/lmm/configured-vision-models`),
        readVlxConnectionSummary(apiUrl),
      ])
      if (!r.ok) throw new Error("读取视觉大模型失败")
      const d = await r.json()
      const providers = Array.isArray(d?.providers) ? d.providers as OnlineVisionProvider[] : []
      if (vlx.configured && !providers.some(p => p.config_id === VLX_PROVIDER_ID)) {
        providers.push(SAVED_VLX_PROVIDER)
      }
      setLocalVisionProviders(providers)
      setLocalVisionError("")
      setLocalVisionProviderId(current => providers.some(p => p.config_id === current)
        ? current
        : providers.find(p => p.config_id === d?.default_config_id)?.config_id || providers[0]?.config_id || "")
    } catch (e) {
      setLocalVisionError(e instanceof Error ? e.message : "无法读取视觉大模型")
    } finally {
      setLocalVisionLoading(false)
    }
  }, [apiUrl])

  useEffect(() => {
    if (modelSource === "smart" && sourceTab !== "video") {
      void refreshLocalVisionProviders()
    }
  }, [modelSource, sourceTab, refreshLocalVisionProviders])

  // ── 可用于视频识别的已保存服务器模型 ───────────────────────────────────────
  const refreshVideoVisionProviders = useCallback(async () => {
    setVideoVisionLoading(true)
    try {
      const [r, vlx] = await Promise.all([
        fetch(`${apiUrl}/api/lmm/configured-vision-models`),
        readVlxConnectionSummary(apiUrl),
      ])
      if (!r.ok) throw new Error("读取服务器模型失败")
      const d = await r.json()
      const providers = Array.isArray(d?.providers) ? d.providers as OnlineVisionProvider[] : []
      if (vlx.configured && !providers.some(p => p.config_id === VLX_PROVIDER_ID)) {
        providers.push(SAVED_VLX_PROVIDER)
      }
      setVideoVisionProviders(providers)
      setVideoVisionError("")
      setVideoVisionProviderId(current => providers.some(p => p.config_id === current)
        ? current
        : providers.find(p => p.config_id === d?.default_config_id)?.config_id || providers[0]?.config_id || "")
    } catch (e) {
      setVideoVisionError(e instanceof Error ? e.message : "无法读取服务器模型")
    } finally {
      setVideoVisionLoading(false)
    }
  }, [apiUrl])

  useEffect(() => { void refreshVideoVisionProviders() }, [refreshVideoVisionProviders])

  // ── 「识别日志」：增量拉取 + 只 append ────────────────────────────────────────
  // 用 since 游标而不是整份重取：进度日志在两次轮询之间可能成对产生，
  // 整份重取会丢掉第一条；而 append 天然保序。level 只影响颜色，不筛行。
  const fetchVlxLogs = useCallback(async (taskId: string, reset = false) => {
    if (reset) { vlxLogsSinceRef.current = 0; setVlxLogs([]) }
    const since = reset ? 0 : vlxLogsSinceRef.current
    try {
      const r = await fetch(`${apiUrl}/api/lmm/tasks/${taskId}/logs?limit=400${since ? `&since=${since}` : ""}`)
      if (!r.ok) return
      const d = await r.json()
      const lines = (d?.logs || []) as VlxLogLine[]
      if (!lines.length) return
      vlxLogsSinceRef.current = Math.max(
        vlxLogsSinceRef.current, ...lines.map(l => Number(l.ts) || 0))
      setVlxLogs(prev => (reset ? lines : [...prev, ...lines]))
    } catch { /* 日志只是辅助信息，取不到不能打断识别主流程 */ }
  }, [apiUrl])

  // ── 「视频推理记录」：只列本页上传过视频的任务 ────────────────────────────────
  const fetchVlxHistory = useCallback(async () => {
    try {
      const r = await fetch(`${apiUrl}/api/lmm/video-tasks?limit=20`)
      if (!r.ok) return
      const d = await r.json()
      setVlxHistory((d?.tasks || []) as VlxVideoRecord[])
    } catch { /* 同上：记录列表失败不影响识别 */ }
  }, [apiUrl])

  // 进「视频 + 大模型」就把记录列表拉一遍：用户可能在别处跑过、后来又切回来。
  useEffect(() => {
    if (modelSource === "smart" && sourceTab === "video") void fetchVlxHistory()
  }, [modelSource, sourceTab, fetchVlxHistory])

  // 自调度的 setTimeout 链：setInterval 在请求慢于间隔时会乱序返回，进度条会倒退。
  const pollVlxJob = useCallback((taskId: string) => {
    stopVlxPolling()
    const tick = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/lmm/tasks/${taskId}`)
        // 后端重启、任务被清理都是 404。继续轮询只会重复报同一个错，必须终止并告知。
        if (res.status === 404) { setVlxError("视频任务已不存在（后端可能已重启）"); return }
        if (!res.ok) throw new Error("任务状态读取失败")
        const task: VlxVideoJob = await res.json()
        setVlxJob(task)
        setVlxError("")
        // 日志与状态同频拉取：两者分开轮询会出现"进度 100% 但日志还停在 60%"的错位
        void fetchVlxLogs(taskId)
        if (task.status === "stopped" || task.status === "failed") {
          void fetchVlxLogs(taskId)   // 抓最后一批（收尾那几行）
          void fetchVlxHistory()
          return
        }
      } catch (e) {
        setVlxError(e instanceof Error ? e.message : String(e))
      }
      vlxTimerRef.current = setTimeout(tick, 1000)
    }
    void tick()
  }, [apiUrl, stopVlxPolling, fetchVlxLogs, fetchVlxHistory])

  useEffect(() => stopVlxPolling, [stopVlxPolling])

  /** 点历史记录：把那条任务重新显示在面板上（含它的日志）。
   *  还在跑的任务顺带恢复轮询；已结束的**不轮询** —— 它们多半是上一次后端进程的，
   *  轮询只会拿到 404 然后弹一句没用的报错。 */
  function openVlxRecord(row: VlxVideoRecord) {
    stopVlxPolling()
    setVlxError("")
    const pct = (row.total_frames && row.processed) ? row.processed / row.total_frames * 100 : 0
    setVlxJob({
      id: row.id, status: row.status, status_zh: row.status_zh, message: row.message,
      progress: pct, processed: row.processed, total_frames: row.total_frames,
      detections: row.detections, detector: row.detector,
      sporadic_dropped: row.sporadic_dropped ?? undefined,
      video_min_hits: row.video_min_hits ?? undefined,
      // 只是"有/没有"的开关：真正的下载地址由 id 拼出，不需要在这里搬路径
      result_video: row.has_video ? "1" : null,
      result_json: row.has_json ? "1" : null,
    })
    void fetchVlxLogs(row.id, true)
    if (row.status === "running" || row.status === "starting" || row.status === "stopping") {
      pollVlxJob(row.id)
    }
  }

  // 日志面板始终停在最新一行：用户盯的是"现在到哪了"，不是"从哪开始"。
  useEffect(() => {
    const el = vlxLogBoxRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [vlxLogs])

  async function startVlxVideo() {
    if (!vlxFile || !vlxPrompt.trim()) return
    if (!videoVisionProviderId) {
      setVlxError("请先在「大模型配置」接入并选择一个服务器视觉模型")
      return
    }
    setVlxSubmitting(true); setVlxError(""); setVlxJob(null); setVlxUploadPct(0)
    try {
      const fd = new FormData()
      fd.append("video", vlxFile)
      fd.append("prompt", vlxPrompt.trim())
      fd.append("vision_provider_id", videoVisionProviderId)
      fd.append("engine", videoVisionProviderId === VLX_PROVIDER_ID ? "vlx" : "vlm")
      fd.append("interval", String(clampNumber(videoVisionInterval, 0.25, 60, 2)))
      fd.append("min_hits", String(clampNumber(vlxMinHits, 1, 10, 1, true)))
      // 不传轨迹 / 告警参数：它们只影响告警事件，不影响结果视频（详见文件上方的说明），
      // 后端会用默认值，需要调整时去「多模态目标检测」页。
      // 用 XHR 而不是 fetch：只有它给得到上传进度，而视频动辄几百 MB。
      const task = await new Promise<VlxVideoJob>((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open("POST", `${apiUrl}/api/lmm/video-tasks`)
        xhr.upload.onprogress = e => { if (e.lengthComputable) setVlxUploadPct(e.loaded / e.total * 100) }
        xhr.onload = () => {
          let body: { task?: VlxVideoJob; detail?: string } = {}
          try { body = JSON.parse(xhr.responseText) } catch { /* 非 JSON 响应按空处理 */ }
          if (xhr.status >= 200 && xhr.status < 300 && body.task) resolve(body.task)
          else reject(new Error(body.detail || `上传失败（HTTP ${xhr.status}）`))
        }
        xhr.onerror = () => reject(new Error("上传失败：无法连接后端"))
        xhr.send(fd)
      })
      setVlxJob(task)
      // 新任务的日志要**从空开始**：否则上一个任务的"已写入结果视频"会留在面板上，
      // 看起来像这次已经跑完了。
      void fetchVlxLogs(task.id, true)
      void fetchVlxHistory()
      if (task.status !== "failed") pollVlxJob(task.id)
    } catch (e) {
      setVlxError(e instanceof Error ? e.message : String(e))
    } finally { setVlxSubmitting(false) }
  }

  async function stopVlxVideo() {
    if (!vlxJob?.id) return
    try { await fetch(`${apiUrl}/api/lmm/tasks/${vlxJob.id}/stop`, { method: "POST" }) } catch { /* 取消失败不影响轮询收尾 */ }
  }

  // ── Webcam state ──────────────────────────────────────────────────────────────
  const [webcamActive, setWebcamActive] = useState(false)
  const [fps, setFps]                   = useState(0)
  const [sessionId, setSessionId]       = useState("")
  const webcamSessionRef = useRef<{ url: string; id: string } | null>(null)
  const videoRef          = useRef<HTMLVideoElement>(null)
  const canvasRef         = useRef<HTMLCanvasElement>(null)
  const streamRef         = useRef<MediaStream | null>(null)
  const webcamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fpsCounterRef     = useRef<number[]>([])
  const inFlightRef       = useRef(false)   // frame-dropout: skip when previous still running

  // ── Derived ───────────────────────────────────────────────────────────────────
  const smartInferenceReady = !!localVisionProviderId && !!localVisionPrompt.trim()

  // ── Load YOLO models ──────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${apiUrl}/api/infer/models`)
      .then(r => r.json())
      .then(d => setYoloModels(d.models || []))
      .catch(() => {})
  }, [apiUrl])

  // Read model.names on model changes; debounce pasted paths and discard stale responses.
  useEffect(() => {
    setClassData({ key: "", options: [], selected: [], error: null })
    setClassSearch("")
    if (modelSource === "smart" || !activeModelPath) return
    const controller = new AbortController()
    let disposed = false
    let timeout: ReturnType<typeof setTimeout> | undefined
    const timer = setTimeout(async () => {
      timeout = setTimeout(() => controller.abort(), 30000)
      try {
        // We upload this local checkpoint for remote inference; its names are identical.
        const res = await fetch(`${apiUrl}/api/infer/classes?model_path=${encodeURIComponent(activeModelPath)}`, {
          signal: controller.signal,
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || "无法加载模型类别")
        if (!controller.signal.aborted) {
          setClassData({ key: classKey, options: data.classes, selected: [], error: null })
        }
      } catch (e) {
        if (!disposed) {
          setClassData({ key: classKey, options: [], selected: [], error: controller.signal.aborted ? "类别读取超时，请重试" : e instanceof Error ? e.message : String(e) })
        }
      } finally {
        clearTimeout(timeout)
      }
    }, 300)
    return () => { disposed = true; clearTimeout(timer); clearTimeout(timeout); controller.abort() }
  }, [apiUrl, activeModelPath, modelSource, classKey, classReload])

  // ── Image inference ───────────────────────────────────────────────────────────
  function handleImageFile(f: File) {
    setImageFile(f); setResultImg(null); setDetections([]); setError(null)
    setImagePreview(URL.createObjectURL(f))
  }

  async function requestImageInference(file: File, signal?: AbortSignal) {
      const fd = new FormData()
      fd.append("image", file)
      let url: string
      if (modelSource === "smart") {
        if (localVisionProviderId === VLX_PROVIDER_ID) {
          return await callVlxImage(apiUrl, file, localVisionPrompt)
        }
        fd.append("prompt", localVisionPrompt.trim())
        fd.append("provider_id", localVisionProviderId)
        url = `${apiUrl}/api/lmm/vision/infer-image`
      } else {
        fd.append("model_path", activeModelPath)
        fd.append("task", task)
        fd.append("confidence", String(confidence))
        fd.append("iou", String(iou))
        fd.append("imgsz", String(imgsz))
        fd.append("classes", JSON.stringify(selectedClasses))
        const tp = trackerParams()
        fd.append("tracker", tp.tracker)
        fd.append("with_reid", String(tp.with_reid))
        if (tp.reid_model) fd.append("reid_model", tp.reid_model)
        fd.append("track_high_thresh", String(tp.track_high_thresh))
        fd.append("new_track_thresh", String(tp.new_track_thresh))
        fd.append("match_thresh", String(tp.match_thresh))
        fd.append("track_buffer", String(tp.track_buffer))
        url = `${executionUrl}/api/infer/image`
      }
      const res = await fetch(url, { method: "POST", body: fd, signal })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
      const data = await res.json()
      return data
  }

  async function runImageInference() {
    if (!imageFile) return
    if (modelSource !== "smart" && (!activeModelPath || classesLoading)) return
    if (modelSource === "smart" && (!localVisionProviderId || !localVisionPrompt.trim())) return
    setInferring(true); setError(null)
    try {
      const data = await requestImageInference(imageFile)
      setResultImg(data.annotated_b64); setDetections(data.detections || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setInferring(false) }
  }

  function downloadResult() {
    if (!resultImg) return
    const a = document.createElement("a")
    a.href = `data:image/jpeg;base64,${resultImg}`; a.download = "result.jpg"; a.click()
  }

  // ── Video inference ────────────────────────────────────────────────────────────
  async function startVideoInference() {
    if (!videoFile || !activeModelPath || classesLoading || videoSubmitting) return
    setVideoSubmitting(true)
    setUploadProgress(0); setUploadStage("上传到本机"); setRemoteUpload(null); setVideoJob(null); setVideoPollError("")
    setError(null)
    const fd = new FormData()
    fd.append("model_path", activeModelPath); fd.append("task", task)
    fd.append("confidence", String(confidence)); fd.append("iou", String(iou))
    fd.append("imgsz", String(imgsz))
    const tp = trackerParams()
    fd.append("tracker", tp.tracker)
    fd.append("with_reid", String(tp.with_reid))
    if (tp.reid_model) fd.append("reid_model", tp.reid_model)
    fd.append("track_high_thresh", String(tp.track_high_thresh))
    fd.append("new_track_thresh", String(tp.new_track_thresh))
    fd.append("match_thresh", String(tp.match_thresh))
    fd.append("track_buffer", String(tp.track_buffer))
    fd.append("classes", JSON.stringify(selectedClasses))
    fd.append("video", videoFile)
    const uploadId = crypto.randomUUID()
    let transferTimer: ReturnType<typeof setTimeout> | undefined
    let transferDone = false
    const checkTransfer = async () => {
      try {
        const res = await fetch(`${executionUrl}/uploads/${uploadId}`, { signal: AbortSignal.timeout(10000) })
        if (res.ok && !transferDone) setRemoteUpload(await res.json())
      } catch {} finally {
        if (!transferDone) transferTimer = setTimeout(checkTransfer, 1000)
      }
    }
    if (computeNodeId !== "local") void checkTransfer()
    try {
      const { job_id } = await new Promise<{job_id:string}>((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open("POST", `${executionUrl}/api/infer/video?upload_id=${uploadId}`)
        xhr.upload.onprogress = e => { if (e.lengthComputable) setUploadProgress(Math.min(100, e.loaded / e.total * 100)) }
        xhr.upload.onload = () => { setUploadProgress(100); setUploadStage(computeNodeId === "local" ? "本机接收完成，准备推理" : "本机接收完成，等待服务器") }
        xhr.onerror = () => reject(new Error("上传连接中断，请检查后端及网络"))
        xhr.onabort = () => reject(new Error("上传已取消"))
        xhr.onload = () => {
          let data: {job_id:string;detail?:string}
          try { data = JSON.parse(xhr.responseText) } catch { reject(new Error("服务器响应无效")); return }
          if (xhr.status < 200 || xhr.status >= 300) reject(new Error(data.detail || `上传失败 (${xhr.status})`))
          else resolve(data)
        }
        xhr.send(fd)
      })
      stopVideoPolling()
      videoJobIdRef.current = job_id
      videoTargetRef.current = 0
      void rememberVideo({ id: job_id, status: "starting", progress: 0, message: "启动中…" })
      setVideoProgress(0)
      setVideoJob({ id: job_id, status: "starting", progress: 0, message: "启动中…" })
      pollVideoJob(job_id)   // poll immediately; waiting a full interval looks like a freeze
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { transferDone = true; clearTimeout(transferTimer); setVideoSubmitting(false) }
  }

  // Self-scheduling poll. A fixed setInterval fires regardless of whether the
  // previous request returned, so slow responses can land out of order and drag
  // the bar backwards; chaining the next tick from the finished one cannot.
  const pollVideoJob = useCallback(async (job_id: string) => {
    if (videoInFlightRef.current) return
    videoInFlightRef.current = true
    let settled = false
    try {
      const res  = await fetch(`${executionUrl}/api/infer/video/${job_id}/status`)
      // 4xx 是**终态**，不是"暂时取不到"：任务不存在（后端重启 / 被清理）或代理打到
      // 了不匹配的端点（曾观察到成片的 422），再轮询一万次也是同一个结果。
      // 必须终止 —— 否则这条任务永远停在"运行中"，而"运行中的视频任务"会把
      // 「模型与设置」的模型来源 tab 一直锁住（用户只会看到"点了没反应"）。
      if (res.status === 404 || res.status === 410 || res.status === 422) {
        settled = true
        stopVideoPolling()
        videoJobIdRef.current = null
        setVideoPollError("推理任务已不存在（后端可能已重启或被清理），请重新提交。")
        setVideoJob(prev => prev ? {
          ...prev, status: "failed", message: "任务已不存在",
          error: "任务已不存在（后端可能已重启或被清理）",
        } : prev)
        return
      }
      if (!res.ok) throw new Error(res.statusText)
      const data: VideoJob = await res.json()
      if (computeNodeId === "local") void rememberVideo(data)
      setVideoPollError("")
      if (videoJobIdRef.current !== job_id) return   // a newer job took over
      settled = data.status === "completed" || data.status === "failed"
      const pct = Number.isFinite(data.progress) ? data.progress : 0
      videoTargetRef.current = Math.max(videoTargetRef.current, pct)
      setVideoJob(prev => ({
        ...data,
        progress: Math.max(prev?.id === data.id ? prev.progress : 0, pct),
      }))
      if (settled) {
        if (data.status === "completed") { videoTargetRef.current = 100; setVideoProgress(100) }
        stopVideoPolling()
      }
    } catch {
      setVideoPollError("暂时无法获取进度，正在重连；不会重复提交任务。")
      // Transient failure — keep the loop alive rather than freezing the bar.
    } finally {
      videoInFlightRef.current = false
      if (!settled && videoJobIdRef.current === job_id) {
        videoTimerRef.current = setTimeout(() => pollVideoJob(job_id), VIDEO_POLL_MS)
      }
    }
  }, [executionUrl, stopVideoPolling])

  async function rememberVideo(job: VideoJob) {
    try {
      const res = await fetch(`${apiUrl}/api/compute-inference/history`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({node_id:computeNodeId,job_id:job.id,snapshot:job})})
      if (!res.ok) throw new Error("推理记录保存失败")
      const row = await res.json()
      setVideoHistory(prev => [row,...prev.filter(r=>!(r.node_id===row.node_id && r.job_id===row.job_id))])
    } catch { setHistoryError("推理记录保存失败，请保留当前页面。") }
  }

  function restoreVideo(row: {node_id:string;job_id:string;snapshot:VideoJob}) {
    stopVideoPolling()
    setComputeNodeId(row.node_id); setSourceTab("video")
    setVideoFile(null); setVideoPollError(""); setError(null)
    if (modelSource === "smart") setModelSource("yolo")
    setVideoJob({...row.snapshot,id:row.job_id})
    setVideoProgress(row.snapshot.progress || 0); videoTargetRef.current=row.snapshot.progress || 0
    setRestoreVideoId(row.job_id)
  }

  useEffect(() => {
    let disposed=false
    fetch(`${apiUrl}/api/compute-inference/history`).then(async res=>{
      if(!res.ok) throw new Error("history")
      const data=await res.json()
      if(disposed)return
      setVideoHistory(data.jobs || [])
      if(data.jobs?.length) restoreVideo(data.jobs[0])
    }).catch(()=>{if(!disposed)setHistoryError("历史任务读取失败")})
    return ()=>{disposed=true}
  }, [apiUrl])

  useEffect(() => {
    if (!restoreVideoId) return
    videoJobIdRef.current=restoreVideoId
    void pollVideoJob(restoreVideoId)
    setRestoreVideoId(null)
  }, [restoreVideoId, pollVideoJob])

  // ── Webcam (with frame dropout) ────────────────────────────────────────────────
  async function startWebcam() {
    const p = paramsRef.current
    if (p.modelSource !== "smart" && !p.activeModelPath)  { setError("请先选择模型"); return }
    const useConfiguredVision = p.modelSource === "smart"
    if (useConfiguredVision
      && (!p.localVisionProviderId || !p.localVisionPrompt.trim())) {
      setError("请选择视觉大模型并填写目标描述"); return
    }
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      streamRef.current = stream
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
      setWebcamActive(true)
      const sid = sessionId || str8(); setSessionId(sid)
      webcamSessionRef.current = { url: executionUrl, id: sid }
      inFlightRef.current = false

      webcamIntervalRef.current = setInterval(async () => {
        // Frame dropout: skip this tick if previous request is still in flight
        if (inFlightRef.current) return
        if (!videoRef.current || !canvasRef.current) return
        const v = videoRef.current
        if (v.videoWidth === 0) return

        const cp = paramsRef.current
        if (cp.modelSource !== "smart" && cp.classesLoading) return
        const ctx = canvasRef.current.getContext("2d")!
        // For smart models, downscale frame before encoding to reduce latency
        let cw = v.videoWidth, ch = v.videoHeight
        if (cp.modelSource === "smart" && cp.smartImgsz < Math.max(cw, ch)) {
          const scale = cp.smartImgsz / Math.max(cw, ch)
          cw = Math.round(cw * scale); ch = Math.round(ch * scale)
        }
        canvasRef.current.width = cw; canvasRef.current.height = ch
        ctx.drawImage(v, 0, 0, cw, ch)
        const frame = canvasRef.current.toDataURL("image/jpeg", 0.7).split(",")[1]

        inFlightRef.current = true
        try {
          if (cp.modelSource === "smart" && cp.localVisionProviderId === VLX_PROVIDER_ID) {
            const image = await fetch(`data:image/jpeg;base64,${frame}`).then(r => r.blob())
            const data = await callVlxImage(apiUrl, image, cp.localVisionPrompt)
            setResultImg(data.annotated_b64); setDetections(data.detections)
          } else {
            let res: Response
            if (cp.modelSource !== "smart") {
            res = await fetch(`${executionUrl}/api/infer/frame`, {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_path: cp.activeModelPath, frame_b64: frame,
                task: cp.task, confidence: cp.confidence, iou: cp.iou, imgsz: cp.imgsz,
                tracker: cp.tracker, session_id: sid,
                with_reid: cp.reidActive,
                reid_model: cp.reidActive && cp.reidEffPath ? cp.reidEffPath : null,
                track_high_thresh: cp.trackHighThresh,
                new_track_thresh: cp.newTrackThresh,
                match_thresh: cp.matchThresh,
                track_buffer: cp.trackBuffer,
                classes: cp.selectedClasses,
              }),
            })
            } else {
              res = await fetch(`${apiUrl}/api/lmm/vision/infer-frame`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  provider_id: cp.localVisionProviderId,
                  prompt: cp.localVisionPrompt,
                  frame_b64: frame,
                }),
              })
            }
            if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
            const data = await res.json()
            setResultImg(data.annotated_b64); setDetections(data.detections || [])
          }
          const now = Date.now()
          fpsCounterRef.current.push(now)
          fpsCounterRef.current = fpsCounterRef.current.filter(t => now - t < 1000)
          setFps(fpsCounterRef.current.length)
        } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
        finally { inFlightRef.current = false }
      }, useConfiguredVision ? 1500 : 50) // 配置式视觉大模型请求需限速
    } catch (e) {
      setError(`摄像头出错：${e instanceof Error ? e.message : e}`)
    }
  }

  function stopWebcam() {
    if (webcamIntervalRef.current) { clearInterval(webcamIntervalRef.current); webcamIntervalRef.current = null }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (videoRef.current) videoRef.current.srcObject = null
    const activeSession = webcamSessionRef.current
    webcamSessionRef.current = null
    if (activeSession) fetch(`${activeSession.url}/api/infer/session/${activeSession.id}`, { method: "DELETE", keepalive: true }).catch(() => {})
    setWebcamActive(false); setSessionId(""); setFps(0); inFlightRef.current = false
  }

  useEffect(() => () => { stopWebcam() }, [])

  function str8() { return Math.random().toString(36).slice(2, 10) }
  const taskColor = (t: string) =>
    t === "detect" ? "text-blue-600 dark:text-blue-400" : t === "segment" ? "text-purple-600 dark:text-purple-400" : "text-emerald-700 dark:text-green-400"

  // ── Render ─────────────────────────────────────────────────────────────────────
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">推理</h1>
        <p className="text-muted-foreground mt-1">
          在图像、视频或摄像头上运行 YOLO、RF-DETR，或选择「大模型配置」中已保存的视觉模型进行语义识别。
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Config panel ─────────────────────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" /> 模型与设置
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">算力节点</label>
              <Select value={computeNodeId} disabled={webcamActive || inferring || videoSubmitting || (!!videoJob && !["completed","failed"].includes(videoJob.status))} onValueChange={id => { setComputeNodeId(id); setResultImg(null); setDetections([]); setVideoJob(null); setError(null) }}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent><SelectItem value="local">本地电脑</SelectItem>{computeNodes.filter(n => n.connection_type === "ssh").map(n => <SelectItem key={n.id} value={n.id}>{n.name} · {n.status === "online" ? "最近检测在线" : "待检测"}</SelectItem>)}</SelectContent>
              </Select>
              <div className="flex gap-3 text-xs"><button type="button" onClick={refreshComputeNodes} className="text-primary">刷新节点</button><a href="/compute-nodes" className="text-primary">管理节点</a></div>
              {nodesError && <p className="text-xs text-destructive">{nodesError}</p>}
              {computeNodeId !== "local" && <p className="text-xs text-muted-foreground">远程 YOLO / RF-DETR / 大模型推理；模型与输入自动上传到该节点。摄像头由本机采集。</p>}
            </div>
            {/* Source toggle */}
            <div className="flex gap-1 bg-muted/30 p-1 rounded-lg">
              {(["yolo", "rfdetr", "smart"] as const).map(src => (
                <button key={src} onClick={() => { if (webcamActive || inferring || videoJob?.status === "running") return; setModelSource(src); setError(null); if (src === "rfdetr") { setTask("detect"); setImgsz(576); if (!rfSelected) setRfSelected(zones?.zones.detector.find(m => m.kind === "rfdetr")?.path || "") } }}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                    modelSource === src ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                  }`}>
                  {src === "yolo" ? <Cpu className="w-3.5 h-3.5" /> : src === "rfdetr" ? <ScanSearch className="w-3.5 h-3.5" /> : <Wand2 className="w-3.5 h-3.5" />}
                  {src === "yolo" ? "YOLO" : src === "rfdetr" ? "RF-DETR" : "大模型"}
                </button>
              ))}
            </div>

            {/* ── YOLO: model loading split into zones ── */}
            {modelSource !== "smart" && (<>
              {modelSource === "rfdetr" && <p className="text-xs text-muted-foreground">RF-DETR · 默认 GPU。输入尺寸需为 32 的倍数；支持图片、视频和摄像头。ReID 请明确选择 OSNet 或 Backbone + Head。</p>}
              <div className="space-y-2.5">
                <p className="text-xs font-medium text-muted-foreground">模型加载</p>

                <ModelZone
                  title="检测器" icon={<Cpu className="w-3.5 h-3.5 text-primary" />}
                  hint={modelSource === "rfdetr" ? "导入 RF-DETR .pth 权重，自动读取架构和类别。默认使用 GPU。" : "导入检测 / 分割 / 姿态权重，任务类型由权重自身决定。"}
                  busy={zoneBusy === "detector"}
                  status={detectorStatus}
                  slots={[{
                    placeholder: "选择检测器…",
                    items: zoneItems("detector"),
                    value: selectedModel,
                    onSelect: v => { setSelectedModel(v); setManualPath("") },
                    onPickFile: () => { setUploadZone("detector"); zoneFileRef.current?.click() },
                    showPath: openPathZone === "detector",
                    pathValue: zonePath.detector ?? "",
                    onPathChange: v => setZonePath(p => ({ ...p, detector: v })),
                    onAddPath: () => importModel("detector", null, (zonePath.detector ?? "").trim() || null),
                    onTogglePath: () => setOpenPathZone(z => z === "detector" ? null : "detector"),
                    onRemove: () => { const it = zoneItems("detector").find(m => m.path === selectedModel); if (it) removeZoneEntry(it.id) },
                    message: zoneItems("detector").find(m => m.path === selectedModel)?.message,
                  }]}
                />

                <ModelZone
                  title="跟踪器" icon={<Activity className="w-3.5 h-3.5 text-primary" />}
                  hint="选择内置跟踪器，或导入自定义跟踪器配置并声明它的算法类型。"
                  busy={zoneBusy === "tracker"}
                  status={trackerStatus}
                  notice={trackerNotice}
                  slots={[{
                    placeholder: "选择跟踪器…",
                    items: [
                      ...trackerOptions.map(t => ({
                        id: `builtin:${t.value}`, name: `${t.label}（内置）`, path: t.value,
                        status: "ready", arch: t.reid ? "支持外观" : null,
                      } as ZoneModel)),
                      ...zoneItems("tracker"),
                    ],
                    value: tracker,
                    onSelect: v => { setTracker(v); setTrackerNotice("") },
                    onPickFile: () => { setUploadZone("tracker"); zoneFileRef.current?.click() },
                    showPath: openPathZone === "tracker",
                    pathValue: zonePath.tracker ?? "",
                    onPathChange: v => setZonePath(p => ({ ...p, tracker: v })),
                    onAddPath: () => importModel("tracker", null, (zonePath.tracker ?? "").trim() || null),
                    onTogglePath: () => setOpenPathZone(z => z === "tracker" ? null : "tracker"),
                    onRemove: () => { if (trackerEntry) removeZoneEntry(trackerEntry.id) },
                    message: trackerEntry?.message
                      ?? (trackerMeta ? `决定轨迹关联方式${trackerMeta.reid ? "，并使用下方外观特征" : "，不使用外观特征"}。` : undefined),
                  }]}
                  extra={
                    <div className="space-y-1.5">
                      <p className="text-[10px] text-muted-foreground">
                        导入跟踪器配置时，声明它实现的算法类型：
                      </p>
                      <Select value={newTrackerType} onValueChange={setNewTrackerType}>
                        <SelectTrigger className="h-7 text-[11px]"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {trackerOptions.map(o => (
                            <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  }
                />

                <ModelZone
                  title="重识别（ReID）" icon={<Eye className="w-3.5 h-3.5 text-primary" />}
                  hint="两种重识别方式任选其一：OSNet（专用 ReID 主干，选中即用），或 Backbone + Projection Head（任意 Backbone 接一个 Projection Head）。选中一种，另一种自动让位。"
                  busy={zoneBusy === "backbone" || zoneBusy === "reid" || zoneBusy === "head"}
                  status={reidStatus}
                  error={reidPairErrFinal || undefined}
                  success={featureChain && !reidPairErrFinal
                    ? `特征链路：${featureChain} · 维度兼容${headMeta?.feature_strategy ? "（元数据一致）" : ""}`
                    : undefined}
                  notice={reidHead && reidComposed ? reidComposed.message : undefined}
                  headerExtra={
                    <div className="flex items-center gap-0.5 rounded-md border border-border/60 p-0.5">
                      {(["osnet", "backbone"] as const).map(m => (
                        <button key={m} type="button" onClick={() => switchReidMode(m)}
                          className={`px-1.5 py-0.5 rounded text-[10px] transition-colors ${
                            reidMode === m
                              ? "bg-primary/10 text-foreground"
                              : "text-muted-foreground hover:text-foreground"}`}>
                          {m === "osnet" ? "OSNet" : "Backbone + Head"}
                        </button>
                      ))}
                    </div>
                  }
                  slots={reidMode === "osnet" ? [{
                    placeholder: "选择 OSNet 模型…",
                    badge: osnetSubStatus.label,
                    items: osnetItems,
                    value: osnetModel,
                    onSelect: pickOsnetModel,
                    onPickFile: () => { setUploadZone("backbone"); zoneFileRef.current?.click() },
                    showPath: openPathZone === "backbone",
                    pathValue: zonePath.backbone ?? "",
                    onPathChange: v => setZonePath(p => ({ ...p, backbone: v })),
                    onAddPath: () => importModel("backbone", null, (zonePath.backbone ?? "").trim() || null),
                    onTogglePath: () => setOpenPathZone(z => z === "backbone" ? null : "backbone"),
                    onRemove: osnetEntry && !osnetModel.startsWith("builtin:")
                      ? () => removeZoneEntry(osnetEntry.id) : undefined,
                    message: osnetEntry?.message
                      ?? (osnetItems.length
                        ? "OSNet 是专用的重识别主干，本身就能输出可用的特征向量，不需要 Projection Head。"
                        : "还没有 OSNet 模型；导入 .pth 会自动重建网络并导出 TorchScript。"),
                  }] : [
                    {
                      label: "Backbone",
                      badge: backboneSubStatus.label,
                      placeholder: "选择 Backbone…",
                      items: encItems,
                      value: reidModel,
                      onSelect: pickReidModel,
                      onPickFile: () => { setUploadZone("backbone"); zoneFileRef.current?.click() },
                      showPath: openPathZone === "backbone",
                      pathValue: zonePath.backbone ?? "",
                      onPathChange: v => setZonePath(p => ({ ...p, backbone: v })),
                      onAddPath: () => importModel("backbone", null, (zonePath.backbone ?? "").trim() || null),
                      onTogglePath: () => setOpenPathZone(z => z === "backbone" ? null : "backbone"),
                      onRemove: reidEncEntry && !isBuiltinBackbone
                        ? () => removeZoneEntry(reidEncEntry.id) : undefined,
                      message: reidEncEntry?.message
                        ?? (typeof aggDim === "number"
                          ? `输出 ${baseDim} 维特征；聚合后的向量才是 Projection Head 的输入。`
                          : undefined),
                    },
                    {
                      label: "特征聚合方式",
                      simple: true,
                      disabled: !aggregationSupported || strategyItems.length === 0,
                      badge: aggregationSupported
                        ? (typeof aggDim === "number" ? `${aggDim}d` : "—")
                        : "不适用",
                      placeholder: "选择特征聚合方式…",
                      items: strategyItems,
                      value: activeStrategy,
                      onSelect: pickFeatureStrategy,
                      showPath: false,
                      pathValue: "",
                      onPathChange: () => {},
                      onAddPath: () => {},
                      onTogglePath: () => {},
                      message: !aggregationSupported
                        ? "该 Backbone 不是可重建的 Transformer 主干，直接使用它自身的输出。"
                        : headMeta?.feature_strategy
                          ? `Projection Head 元数据声明训练时用的是「${headMeta.feature_strategy}」，已按此选择；可随时切换。`
                          : suggestedStrategy && suggestedStrategy !== activeStrategy
                            ? `存在能让维度串接的方式：${strategyOptions.find(s => s.value === suggestedStrategy)?.label}（${baseDim}×${strategyOptions.find(s => s.value === suggestedStrategy)?.mult}）。`
                            : strategySpec?.note,
                    },
                    {
                      label: "Projection Head",
                      optional: true,
                      placeholder: "选择 Projection Head…",
                      items: zoneItems("head"),
                      value: reidHead,
                      onSelect: pickReidHead,
                      onPickFile: () => { setUploadZone("head"); zoneFileRef.current?.click() },
                      showPath: openPathZone === "head",
                      pathValue: zonePath.head ?? "",
                      onPathChange: v => setZonePath(p => ({ ...p, head: v })),
                      onAddPath: () => importModel("head", null, (zonePath.head ?? "").trim() || null),
                      onTogglePath: () => setOpenPathZone(z => z === "head" ? null : "head"),
                      onRemove: () => { if (reidHeadEntry) removeZoneEntry(reidHeadEntry.id) },
                      message: reidHeadEntry?.message
                        ?? (typeof aggDim === "number"
                          ? `留空则直接用聚合后的 ${aggDim} 维；挂 Projection Head 时需要它输入 ${aggDim} 维。`
                          : undefined),
                    },
                  ]}
                />

                <ModelZone
                  title="其它" icon={<FileText className="w-3.5 h-3.5 text-primary" />}
                  hint="导入其它用途的模型文件（暂不参与推理）。"
                  busy={zoneBusy === "other"}
                  status={otherStatus}
                  slots={[{
                    placeholder: "选择其它模型…",
                    items: zoneItems("other"),
                    value: otherModel,
                    onSelect: setOtherModel,
                    onPickFile: () => { setUploadZone("other"); zoneFileRef.current?.click() },
                    showPath: openPathZone === "other",
                    pathValue: zonePath.other ?? "",
                    onPathChange: v => setZonePath(p => ({ ...p, other: v })),
                    onAddPath: () => importModel("other", null, (zonePath.other ?? "").trim() || null),
                    onTogglePath: () => setOpenPathZone(z => z === "other" ? null : "other"),
                    onRemove: () => { const it = zoneItems("other").find(m => m.path === otherModel); if (it) removeZoneEntry(it.id) },
                    message: zoneItems("other").find(m => m.path === otherModel)?.message,
                  }]}
                />

                <input ref={zoneFileRef} type="file" className="hidden"
                  accept={uploadZone === "detector" && modelSource === "rfdetr" ? ".pth" : ZONE_ACCEPT[uploadZone ?? "other"]}
                  onChange={e => {
                    const f = e.target.files?.[0]
                    const z = uploadZone
                    e.target.value = ""
                    if (f && z) importModel(z, f, null)
                  }} />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">或粘贴模型路径</label>
                <Input placeholder="/path/to/model.pt" value={manualPath}
                  onChange={e => { setManualPath(e.target.value); setSelectedModel("") }}
                  className="text-xs font-mono" />
              </div>
              {activeModelPath && <p className="text-[10px] font-mono text-primary truncate">{activeModelPath}</p>}
              <fieldset className="space-y-2" aria-label="类别筛选 / 类别">
                <legend className="text-sm font-medium">类别筛选 / 类别</legend>
                {!activeModelPath ? (
                  <p className="text-xs text-muted-foreground">请先选择模型以加载其类别。</p>
                ) : classesLoading ? (
                  <p className="text-xs text-muted-foreground flex items-center gap-1.5" role="status">
                    <RefreshCw className="w-3 h-3 animate-spin" /> 正在加载模型类别…
                  </p>
                ) : classError ? (
                  <div className="space-y-1">
                    <p className="text-xs text-red-600 dark:text-red-400 break-words" role="alert">{classError}</p>
                    <Button type="button" variant="outline" size="sm" onClick={() => setClassReload(v => v + 1)}>
                      重试加载类别
                    </Button>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between gap-2">
                      <Badge variant="secondary">{selectedClasses.length ? `${selectedClasses.length} 个已选` : "全部类别"}</Badge>
                      <Button type="button" variant="ghost" size="sm" className="h-7 text-xs"
                        disabled={selectedClasses.length === 0}
                        onClick={() => setClassData(prev => ({ ...prev, selected: [] }))}>
                        清除（全部）
                      </Button>
                    </div>
                    <Input aria-label="搜索类别" placeholder="搜索类别…" value={classSearch}
                      onChange={e => setClassSearch(e.target.value)} className="h-8 text-xs" />
                    <div className="max-h-44 overflow-y-auto rounded-md border border-border/50 p-2 space-y-1">
                      {modelClasses.filter(c => c.name.toLowerCase().includes(classSearch.trim().toLowerCase())).map(c => (
                        <label key={c.id} className="flex items-center gap-2 rounded px-1 py-1.5 text-xs cursor-pointer hover:bg-muted/30">
                          <Checkbox aria-label={c.name} checked={selectedClasses.includes(c.id)}
                            onCheckedChange={checked => setClassData(prev => ({
                              ...prev,
                              selected: checked === true
                                ? [...new Set([...prev.selected, c.id])]
                                : prev.selected.filter(id => id !== c.id),
                            }))} />
                          <span className="break-words">{c.name}</span>
                        </label>
                      ))}
                      {!modelClasses.some(c => c.name.toLowerCase().includes(classSearch.trim().toLowerCase())) && (
                        <p className="text-xs text-muted-foreground py-1">没有匹配的类别。</p>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground break-words" aria-live="polite">
                      {selectedClasses.length
                        ? `仅：${modelClasses.filter(c => selectedClasses.includes(c.id)).map(c => c.name).join(", ")}`
                        : "不选择 = 检测全部类别。"}
                    </p>
                  </>
                )}
              </fieldset>
              <div className="space-y-2">
                <label className="text-sm font-medium">任务类型</label>
                <Select value={task} onValueChange={setTask}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {TASK_OPTIONS.filter(o => modelSource !== "rfdetr" || ["detect", "track"].includes(o.value)).map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              {task === "track" && (
                <div className="space-y-3 rounded-lg border border-border/50 bg-muted/20 p-3">
                  {/* Tracker + appearance module are chosen in 「模型加载」 above, so this
                      block only reports the resolved pipeline and exposes runtime knobs. */}
                  <div className="space-y-1">
                    <p className="text-sm font-medium">跟踪流水线</p>
                    <p className="text-[10px] text-muted-foreground break-words">
                      跟踪器：<span className="font-mono">{trackerEntry?.name ?? trackerMeta?.label ?? tracker}</span>
                      {trackerEntry?.tracker_type ? `（${trackerEntry.tracker_type}）` : ""}
                      {" ← 检测器："}
                      <span className="font-mono">{activeModelPath ? activeModelPath.split(/[\\/]/).pop() : "未选择"}</span>
                    </p>
                    <p className="text-[10px] text-muted-foreground break-words">
                      外观特征：{reidActive
                        ? <span className="text-emerald-700 dark:text-green-400">
                            已启用 · <span className="font-mono">{reidEffPath.split(/[\\/]/).pop()}</span>
                            {reidMode === "osnet" ? "（OSNet）" : reidHead ? "（Backbone + Projection Head）" : "（Backbone）"}
                          </span>
                        : !reidModePath
                          ? "未启用（可在上方「重识别」区任选 OSNet 或 Backbone + Projection Head）"
                          : !trackerSupportsReid
                            ? `未启用（当前跟踪器 ${trackerMeta?.label ?? trackerType} 不使用外观特征）`
                            : "未启用（任务类型不是「多目标跟踪」）"}
                    </p>
                  </div>

                  {/* ── Advanced tracker knobs ── */}
                  <div className="pt-1 border-t border-border/40">
                    <button type="button"
                      onClick={() => setShowTrackAdvanced(v => !v)}
                      className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
                      <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showTrackAdvanced ? "rotate-180" : ""}`} />
                      高级参数
                    </button>

                    {showTrackAdvanced && (
                      <div className="space-y-3 pt-3">
                        <div className="space-y-1.5">
                          <div className="flex justify-between">
                            <label className="text-xs">高置信度阈值</label>
                            <span className="text-xs text-muted-foreground">{trackHighThresh.toFixed(2)}</span>
                          </div>
                          <Slider value={[trackHighThresh]} onValueChange={([v]) => setTrackHighThresh(v)}
                            min={0.05} max={0.95} step={0.05} />
                          <p className="text-[10px] text-muted-foreground">第一阶段关联的检测分数门槛；调高轨迹更干净。</p>
                        </div>

                        <div className="space-y-1.5">
                          <div className="flex justify-between">
                            <label className="text-xs">新轨迹阈值</label>
                            <span className="text-xs text-muted-foreground">{newTrackThresh.toFixed(2)}</span>
                          </div>
                          <Slider value={[newTrackThresh]} onValueChange={([v]) => setNewTrackThresh(v)}
                            min={0.05} max={0.95} step={0.05} />
                          <p className="text-[10px] text-muted-foreground">未匹配时新建轨迹所需的最低分数；调高可减少误检轨迹。</p>
                        </div>

                        <div className="space-y-1.5">
                          <div className="flex justify-between">
                            <label className="text-xs">匹配阈值</label>
                            <span className="text-xs text-muted-foreground">{matchThresh.toFixed(2)}</span>
                          </div>
                          <Slider value={[matchThresh]} onValueChange={([v]) => setMatchThresh(v)}
                            min={0.1} max={0.99} step={0.05} />
                          <p className="text-[10px] text-muted-foreground">轨迹与检测的关联相似度门槛；数值越高匹配越严格。</p>
                        </div>

                        <div className="space-y-1.5">
                          <div className="flex justify-between">
                            <label className="text-xs">轨迹缓冲帧数</label>
                            <span className="text-xs text-muted-foreground">{trackBuffer} 帧</span>
                          </div>
                          <Slider value={[trackBuffer]} onValueChange={([v]) => setTrackBuffer(v)}
                            min={5} max={150} step={5} />
                          <p className="text-[10px] text-muted-foreground">目标丢失后保留轨迹的帧数；调大可抗遮挡，但会增加 ID 切换风险。</p>
                        </div>

                        <Button variant="ghost" size="sm" className="h-7 text-xs gap-1"
                          onClick={() => {
                            setTrackHighThresh(0.25); setNewTrackThresh(0.25)
                            setMatchThresh(0.8); setTrackBuffer(30)
                          }}>
                          <RotateCcw className="w-3 h-3" /> 恢复默认
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              )}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">IoU 阈值{modelSource === "rfdetr" ? "（RF-DETR 无需 NMS，不生效）" : ""}</label>
                  <span className="text-sm text-muted-foreground">{iou.toFixed(2)}</span>
                </div>
                <Slider disabled={modelSource === "rfdetr"} value={[iou]} onValueChange={([v]) => setIou(v)} min={0.1} max={0.95} step={0.05} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">图像尺寸</label>
                  <span className="text-sm text-muted-foreground">{imgsz}px</span>
                </div>
                <Slider value={[imgsz]} onValueChange={([v]) => setImgsz(v)} min={320} max={1280} step={32} />
              </div>
            </>)}

            {/* 图像与摄像头统一使用「大模型配置」里保存的视觉模型。 */}
            {modelSource === "smart" && sourceTab !== "video" && (<>
              <p className="text-xs text-muted-foreground">
                图像、视频和摄像头共用「大模型配置」中的已保存视觉模型；此处只读取配置，不探测或加载模型。
              </p>
              <div className="space-y-3 rounded-lg border border-border/50 p-3">
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-sm font-medium">视觉大模型</label>
                    <button type="button" onClick={() => void refreshLocalVisionProviders()}
                      disabled={localVisionLoading}
                      className="inline-flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground disabled:opacity-50">
                      <RefreshCw className={cn("h-3 w-3", localVisionLoading && "animate-spin")} />
                      {localVisionLoading ? "读取中…" : "刷新模型列表"}
                    </button>
                  </div>
                  <Select value={localVisionProviderId || "__none__"}
                    onValueChange={value => setLocalVisionProviderId(value === "__none__" ? "" : value)}
                    disabled={localVisionLoading || inferring || webcamActive}>
                    <SelectTrigger><SelectValue placeholder="选择大模型配置中的视觉模型" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__">请选择模型</SelectItem>
                      {localVisionProviders.map(provider => (
                    <SelectItem key={provider.config_id} value={provider.config_id}
                      className="group data-[highlighted]:bg-primary data-[highlighted]:text-primary-foreground data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground">
                        <span className="flex items-center gap-2">
                          <Bot className="h-3.5 w-3.5 shrink-0 text-primary group-data-[highlighted]:text-primary-foreground group-data-[state=checked]:text-primary-foreground" />
                          <span className="truncate">{provider.display_name} · {provider.model}</span>
                        </span>
                      </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {localVisionError ? (
                    <p className="text-[10px] text-red-600 dark:text-red-400">{localVisionError}</p>
                  ) : localVisionProviders.length === 0 ? (
                    <p className="text-[10px] text-amber-700 dark:text-yellow-300">
                      暂无已保存的视觉模型，请先到「大模型配置」中添加。
                    </p>
                  ) : (
                    <p className="text-[10px] text-muted-foreground">模型地址和访问凭据由大模型配置统一管理。</p>
                  )}
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">目标描述</label>
                  <Textarea value={localVisionPrompt} onChange={e => setLocalVisionPrompt(e.target.value)}
                    placeholder="例如：识别画面中的无人机和人员"
                    disabled={inferring || webcamActive} className="min-h-[72px] text-sm" />
                  <div className="flex flex-wrap gap-1.5">
                    {PROMPT_PRESETS.map(p => (
                      <button key={p} type="button" disabled={inferring || webcamActive}
                        onClick={() => setLocalVisionPrompt(p)}
                        className="rounded-full border border-border/60 px-2 py-1 text-[10px] text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground disabled:opacity-50">
                        {p}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium">摄像头帧尺寸</label>
                    <span className="text-sm text-muted-foreground">{smartImgsz}px</span>
                  </div>
                  <Slider value={[smartImgsz]} onValueChange={([v]) => setSmartImgsz(v)} min={160} max={1280} step={32} />
                </div>
              </div>
            </>)}

            {/* ── 大模型（视频）：选择服务器视觉模型并配置抽帧间隔 ── */}
            {modelSource === "smart" && sourceTab === "video" && (<>
              <div className="space-y-2">
                <label className="text-sm font-medium">目标描述</label>
                <Textarea value={vlxPrompt} onChange={e => setVlxPrompt(e.target.value)}
                  placeholder="例如：水面上未穿救生衣的人员"
                  disabled={vlxRunning || vlxSubmitting} className="min-h-[72px] text-sm" />
                <div className="flex flex-wrap gap-1.5">
                  {PROMPT_PRESETS.map(p => (
                    <button key={p} type="button" disabled={vlxRunning || vlxSubmitting}
                      onClick={() => setVlxPrompt(p)}
                      className="text-[10px] px-2 py-1 rounded-full border border-border/60 text-muted-foreground hover:border-primary/60 hover:text-foreground transition-colors disabled:opacity-50">
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <label className="text-sm font-medium">视觉大模型</label>
                  <button type="button"
                    className="inline-flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground disabled:opacity-50"
                    onClick={() => void refreshVideoVisionProviders()} disabled={videoVisionLoading}>
                    <RefreshCw className={cn("w-3 h-3", videoVisionLoading && "animate-spin")} />
                    {videoVisionLoading ? "读取中…" : "刷新模型列表"}
                  </button>
                </div>
                <Select value={videoVisionProviderId || "__none__"}
                  onValueChange={value => setVideoVisionProviderId(value === "__none__" ? "" : value)}
                  disabled={vlxRunning || vlxSubmitting || videoVisionLoading}>
                  <SelectTrigger><SelectValue placeholder="选择大模型配置中的视觉模型" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">请选择模型</SelectItem>
                    {videoVisionProviders.map(provider => (
                      <SelectItem key={provider.config_id} value={provider.config_id}>
                        {provider.display_name} · {provider.model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {videoVisionError ? (
                  <p className="text-[10px] text-red-600 dark:text-red-400">{videoVisionError}</p>
                ) : videoVisionProviders.length === 0 ? (
                    <p className="text-[10px] text-amber-700 dark:text-yellow-300">
                      暂无已保存的视觉模型，请先到「大模型配置」中添加。
                    </p>
                  ) : (
                    <p className="text-[10px] text-muted-foreground">
                      图像、视频和摄像头共用「大模型配置」中的模型连接与凭据。
                  </p>
                )}
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium">抽帧间隔（秒）</label>
                <Input type="number" min={0.25} max={60} step={0.25} value={videoVisionInterval}
                  onChange={e => setVideoVisionInterval(e.target.value)}
                  disabled={vlxRunning || vlxSubmitting} className="text-xs" />
                <p className="text-[10px] text-muted-foreground">服务器模型较慢时可调大间隔；视频仍按原帧率输出。</p>
              </div>

              {/* 想压掉只闪一次的零星框，可按连续出现次数过滤；>1 时后端会多读一遍视频。 */}
              <div className="space-y-1.5">
                <label className="text-sm font-medium">最少连续出现次数</label>
                <Input type="number" min={1} max={10} step={1} value={vlxMinHits}
                  onChange={e => setVlxMinHits(e.target.value)}
                  disabled={vlxRunning || vlxSubmitting} className="text-xs" />
                <p className="text-[10px] leading-snug text-muted-foreground">
                  同一个目标至少连续出现这么多帧才画进结果视频，<b>1 = 不过滤</b>。
                  设为 2 能压掉只闪一次的框，代价是多读一遍视频、耗时增加。
                </p>
              </div>
            </>)}

            {/* 检测器使用输出分数；当前视觉语言模型不支持置信度阈值。 */}
            {modelSource !== "smart" && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">置信度</label>
                  <span className="text-sm text-muted-foreground">{confidence.toFixed(2)}</span>
                </div>
                <Slider value={[confidence]} onValueChange={([v]) => setConfidence(v)} min={0.01} max={0.99} step={0.01} />
              </div>
            )}

            {error && (
              <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-600 dark:text-red-400 flex gap-2">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span className="break-all">{error}</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Input / Output panel ─────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Source tabs */}
          <div className="flex gap-1 bg-muted/30 p-1 rounded-lg w-fit">
            {(["image", "video", "webcam", "camera"] as const).map(tab => (
              <button key={tab}
                onClick={() => { if (tab !== "webcam" && webcamActive) stopWebcam(); setSourceTab(tab); setResultImg(null); setDetections([]) }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  sourceTab === tab ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                }`}>
                {tab === "image" ? <Upload className="w-3.5 h-3.5" /> :
                 tab === "video" ? <Video  className="w-3.5 h-3.5" /> :
                                   <Camera className="w-3.5 h-3.5" />}
                {tab === "camera" ? "实时摄像头" : tab === "image" ? "图像" : tab === "video" ? "视频" : "网络摄像头"}
              </button>
            ))}
          </div>

          {sourceTab === "camera" && <CameraPanel key={`${modelSource}:${computeNodeId}`} apiUrl={apiUrl}
            enabled={modelSource !== "smart" && !!activeModelPath && !classesLoading}
            options={{ compute_node_id: computeNodeId === "local" ? null : computeNodeId, model_path: activeModelPath, task, confidence, iou, imgsz, classes: selectedClasses, ...trackerParams() }}/>}

          {/* ── Image tab ─────────────────────────────────────────────────── */}
          {sourceTab === "image" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-4">
                <div className="flex gap-2">
                  <Button size="sm" variant={imageMode === "single" ? "default" : "outline"}
                    onClick={() => { setImageMode("single"); setError(null) }}>单张图像</Button>
                  <Button size="sm" variant={imageMode === "folder" ? "default" : "outline"}
                    onClick={() => { setImageMode("folder"); setError(null) }}>图像文件夹</Button>
                </div>
                {imageMode === "folder" ? (
                  <FolderInferencePanel key={`${modelSource}:${computeNodeId}`}
                    enabled={(modelSource !== "smart" && !!activeModelPath && !classesLoading)
                      || (modelSource === "smart" && smartInferenceReady)}
                    infer={(file, signal) => requestImageInference(file, signal)}
                  />
                ) : (<>

                <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border/50 rounded-lg cursor-pointer hover:border-primary/50 transition-colors">
                  <Upload className="w-6 h-6 text-muted-foreground mb-2" />
                  <span className="text-sm text-muted-foreground">{imageFile ? imageFile.name : "点击上传图像"}</span>
                  <input type="file" accept="image/*" className="hidden"
                    onChange={e => e.target.files?.[0] && handleImageFile(e.target.files[0])} />
                </label>

                <div className="flex gap-2">
                  <Button onClick={runImageInference}
                    disabled={!imageFile || inferring || (modelSource !== "smart" ? !activeModelPath || classesLoading : !smartInferenceReady)}
                    className="gap-2">
                    {inferring ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
                    {inferring ? "运行中…" : "运行推理"}
                  </Button>
                  {resultImg && (
                    <Button variant="outline" onClick={downloadResult} className="gap-2">
                      <Download className="w-4 h-4" /> 保存结果
                    </Button>
                  )}
                </div>

                {(imagePreview || resultImg) && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {imagePreview && (
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">
                          输入
                        </p>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={imagePreview} alt="输入图像" className="w-full rounded border border-border/40" />
                      </div>
                    )}
                    {resultImg && (
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">结果</p>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`data:image/jpeg;base64,${resultImg}`} alt="推理结果"
                          className="w-full rounded border border-border/40" />
                      </div>
                    )}
                  </div>
                )}
                </>)}
              </CardContent>
            </Card>
          )}

          {/* ── Video tab ─────────────────────────────────────────────────── */}
          {/* 「视频推理记录」两套来源，绝不能混：
              · YOLO / RF-DETR → 算力节点上的 /api/infer/video 任务（videoHistory）
              · 大模型 → 本页上传的 VLX 任务（vlxHistory，= 开了 save_video 的 lmm 任务）
              混在一起就会出现"记录跟自己的操作对不上"：大模型面板里挂着一条
              2026/09/21 的远程节点 YOLO 任务，既点不动也说明不了任何事。 */}
          {modelSource !== "smart" && (videoHistory.length > 0 || historyError) && <Card className="border-border/50 bg-card/50">
            <CardContent className="pt-4 space-y-2">
              <p className="text-sm font-medium">视频推理记录</p>
              {historyError && <p className="text-xs text-destructive">{historyError}</p>}
              <div className="max-h-48 overflow-auto space-y-1">
                {videoHistory.map(row=><button key={`${row.node_id}:${row.job_id}`} type="button"
                  disabled={videoSubmitting || webcamActive}
                  onClick={()=>restoreVideo(row)} className="block w-full text-left rounded px-2 py-2 text-xs hover:bg-muted">
                  {new Date(row.created_at*1000).toLocaleString()} · {row.node_id === "local" ? "本机" : computeNodes.find(n=>n.id===row.node_id)?.name || "远程节点"} · {row.job_id} · {VIDEO_STATUS_ZH[(videoJob?.id === row.job_id && computeNodeId === row.node_id ? videoJob.status : row.snapshot?.status)] || "待查询"}
                </button>)}
              </div>
            </CardContent>
          </Card>}
          {modelSource === "smart" && sourceTab === "video" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">视频推理记录</p>
                  <button type="button" onClick={() => void fetchVlxHistory()}
                    className="text-[10px] text-muted-foreground hover:text-foreground">刷新</button>
                </div>
                {vlxHistory.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    还没有跑过视频识别。上传视频并点「开始识别」后，这里会列出每一条任务
                    （时间 · 目标描述 · 进度 · 能否下载）。
                  </p>
                ) : (
                  <div className="max-h-48 overflow-auto space-y-1">
                    {vlxHistory.map(row => (
                      <button key={row.id} type="button" onClick={() => openVlxRecord(row)}
                        className={cn("block w-full text-left rounded px-2 py-2 text-xs hover:bg-muted",
                          vlxJob?.id === row.id && "bg-muted/60")}>
                        <span className="block truncate">
                          {row.created_at ? new Date(row.created_at * 1000).toLocaleString() : "—"}
                          {" · "}{row.prompt || "（无目标描述）"}
                        </span>
                        <span className="mt-0.5 block text-[10px] text-muted-foreground">
                          {VLX_RECORD_STATUS_ZH[row.status] || row.status}
                          {row.total_frames
                            ? ` · ${row.processed ?? 0}/${row.total_frames} 帧`
                            : ""}
                          {row.source_name ? ` · ${row.source_name}` : ""}
                          {row.has_video ? " · 可下载结果视频" : ""}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          {sourceTab === "video" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-4">
                {modelSource === "smart" ? (<>
                  {/* 设置（目标描述 / 服务器模型 / 抽帧间隔 / 最少连续出现次数）在左栏「模型与设置」里，
                      这一侧只负责选文件、看进度、拿结果。 */}
                  <p className="text-xs text-muted-foreground">
                    目标描述与检测参数在左侧「模型与设置」；这里上传视频后点「开始识别」。
                    视频帧会按设置的间隔发给选定服务器模型，结果视频保留原始帧率。
                  </p>

                  <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border/50 rounded-lg cursor-pointer hover:border-primary/50 transition-colors">
                    <Video className="w-6 h-6 text-muted-foreground mb-2" />
                    <span className="text-sm text-muted-foreground">{vlxFile ? vlxFile.name : "点击上传视频"}</span>
                    <input type="file" accept="video/*" className="hidden"
                      disabled={vlxRunning || vlxSubmitting}
                      onChange={e => {
                        if (e.target.files?.[0]) setVlxFile(e.target.files[0])
                        stopVlxPolling(); setVlxJob(null); setVlxError("")
                      }} />
                  </label>

                  <div className="flex flex-wrap gap-2">
                    <Button onClick={startVlxVideo}
                      disabled={vlxSubmitting || vlxRunning || !vlxFile || !vlxPrompt.trim() || !videoVisionProviderId}
                      className="gap-2">
                      <Play className="w-4 h-4" />
                      {vlxSubmitting ? "上传中…" : vlxRunning ? "识别中…" : "开始识别"}
                    </Button>
                    {vlxRunning && (
                      <Button variant="outline" className="gap-2" onClick={stopVlxVideo}>
                        <Square className="w-4 h-4" /> 停止
                      </Button>
                    )}
                    {vlxJob?.status === "stopped" && vlxJob.result_video && (
                      <Button variant="outline" className="gap-2"
                        onClick={() => window.open(`${apiUrl}/api/lmm/tasks/${vlxJob.id}/result-video`, "_blank")}>
                        <Download className="w-4 h-4" /> 下载结果视频
                      </Button>
                    )}
                    {vlxJob?.status === "stopped" && vlxJob.result_json && (
                      <Button variant="outline" className="gap-2"
                        onClick={() => window.open(`${apiUrl}/api/lmm/tasks/${vlxJob.id}/result-json`, "_blank")}>
                        <Braces className="w-4 h-4" /> 检出数据
                      </Button>
                    )}
                  </div>

                  {vlxSubmitting && <div className="space-y-2 text-xs text-muted-foreground" aria-live="polite">
                    <div className="flex justify-between"><span>上传视频</span><span>{vlxUploadPct.toFixed(0)}%</span></div>
                    <Progress value={vlxUploadPct} />
                  </div>}

                  {vlxJob && (
                    <div className="space-y-2" aria-live="polite">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>{vlxJob.message || vlxJob.status_zh || "处理中"}</span>
                        <span>{Math.round(vlxJob.progress || 0)}%</span>
                      </div>
                      <Progress value={vlxJob.progress || 0} />
                      <p className="text-[10px] text-muted-foreground">
                        已处理 {vlxJob.processed ?? 0} / {vlxJob.total_frames ?? 0} 帧 · 检出 {vlxJob.detections ?? 0} 个目标
                      </p>
                    </div>
                  )}

                  {vlxJob?.status === "failed" && (
                    <p className="text-xs text-destructive">{vlxJob.error || vlxJob.message || "识别失败"}</p>
                  )}
                  {vlxJob?.status === "stopped" && (
                    <p className="text-sm text-primary">识别已完成。点「下载结果视频」保存到电脑；本机全程未加载任何检测模型。</p>
                  )}
                  {vlxJob?.status === "stopped" && (vlxJob.sporadic_dropped ?? 0) > 0 && (
                    <p className="text-[10px] text-muted-foreground">
                      已按「最少连续出现 {(vlxJob.video_min_hits ?? 1)} 次」过滤掉 {vlxJob.sporadic_dropped} 个零星框；
                      检出数据里仍保留过滤前的全部结果。
                    </p>
                  )}
                  {vlxError && <p className="text-xs text-destructive">{vlxError}</p>}
                </>) : (<>
                  <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border/50 rounded-lg cursor-pointer hover:border-primary/50 transition-colors">
                    <Video className="w-6 h-6 text-muted-foreground mb-2" />
                    <span className="text-sm text-muted-foreground">{videoFile ? videoFile.name : "点击上传视频"}</span>
                    <input type="file" accept="video/*" className="hidden"
                      disabled={videoSubmitting || videoJob?.status === "running" || videoJob?.status === "starting"}
                      onChange={e => {
                        if (e.target.files?.[0]) setVideoFile(e.target.files[0])
                        stopVideoPolling(); setVideoJob(null)
                        videoTargetRef.current = 0; setVideoProgress(0)
                      }} />
                  </label>
                  <div className="flex gap-2">
                    <Button onClick={startVideoInference}
                      disabled={videoSubmitting || !videoFile || !activeModelPath || classesLoading || videoJob?.status === "running" || videoJob?.status === "starting"}
                      className="gap-2">
                      <Play className="w-4 h-4" />
                      {videoSubmitting ? "上传中…" : videoJob?.status === "running" ? "处理中…" : "处理视频"}
                    </Button>
                    {videoJob?.status === "completed" && (
                      <Button variant="outline" className="gap-2"
                        onClick={() => window.open(`${executionUrl}/api/infer/video/${videoJob.id}/result`, "_blank")}>
                        <Download className="w-4 h-4" /> 下载结果视频
                      </Button>
                    )}
                    {videoJob?.status === "completed" && videoJob.has_tracks && (
                      <>
                        <Button variant="outline" className="gap-2"
                          onClick={() => window.open(`${executionUrl}/api/infer/video/${videoJob.id}/tracks/mot`, "_blank")}>
                          <FileText className="w-4 h-4" /> 轨迹 MOT
                        </Button>
                        <Button variant="outline" className="gap-2"
                          onClick={() => window.open(`${executionUrl}/api/infer/video/${videoJob.id}/tracks/json`, "_blank")}>
                          <Braces className="w-4 h-4" /> 轨迹 JSON
                        </Button>
                      </>
                    )}
                  </div>
                  {videoSubmitting && <div className="space-y-2 text-xs text-muted-foreground" aria-live="polite">
                    <div className="flex justify-between"><span>{uploadStage}</span><span>{uploadProgress.toFixed(0)}%</span></div>
                    <Progress value={uploadProgress} />
                    {computeNodeId !== "local" && <>
                      <div className="flex justify-between"><span>{remoteUpload?.stage || "等待本机接收完成"}</span><span>{remoteUpload?.total ? `${Math.min(100, remoteUpload.sent / remoteUpload.total * 100).toFixed(0)}%` : "等待中"}</span></div>
                      {remoteUpload && remoteUpload.total > 0 && <Progress value={Math.min(100, remoteUpload.sent / remoteUpload.total * 100)} />}
                    </>}
                  </div>}
                  {videoJob?.status === "completed" && <p className="text-sm text-primary">推理已完成。点击“下载结果视频”保存到电脑，文件由浏览器下载管理器接收。</p>}
                  {videoPollError && <p className="text-xs text-amber-600">{videoPollError}</p>}
                  {videoJob?.status === "completed" && videoJob.has_tracks && (
                    <p className="text-[10px] text-muted-foreground">
                      共追踪 {videoJob.track_count ?? 0} 条轨迹 / {videoJob.total_detections ?? 0} 次检测。
                      MOT 为 MOTChallenge 格式（frame,id,x,y,w,h,conf,cls,-1）。
                    </p>
                  )}
                  {/* Whether appearance matching really ran is separate from whether it was
                      configured: a tracker that ignores embeddings would otherwise look
                      identical to one that used them. */}
                  {videoJob && videoJob.status !== "failed" && (
                    <p className={`flex items-start gap-1.5 text-[10px] ${
                      !videoJob.reid_enabled
                        ? "text-amber-700 dark:text-yellow-300"
                        : videoJob.reid_native || (videoJob.reid_crops ?? 0) > 0
                          ? "text-emerald-700 dark:text-green-400"
                          : "text-amber-700 dark:text-yellow-300"
                    }`}>
                      <span className="shrink-0">外观模型（ReID）：</span>
                      <span className="min-w-0 break-all">
                        {!videoJob.reid_enabled
                          ? (videoJob.reid_reason || "未启用")
                          : videoJob.status !== "completed"
                            ? `已启用 — ${videoJob.reid_label || "外观编码器"}`
                            : videoJob.reid_native
                              ? `已参与匹配 — ${videoJob.reid_label || "检测器原生特征"}`
                              : (videoJob.reid_crops ?? 0) > 0
                                ? `已参与匹配 — ${videoJob.reid_label || "外观编码器"}，`
                                  + `实际编码 ${videoJob.reid_crops} 个 crop（${videoJob.reid_calls} 次调用）`
                                : `已启用但未被调用（${videoJob.reid_label || "外观编码器"}）`
                                  + " — 本视频没有检测到目标，或跟踪器未走到外观匹配分支"}
                      </span>
                    </p>
                  )}
                  {videoJob && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                        <span className="truncate tabular-nums">{videoJob.message}</span>
                        <span className="flex items-center gap-2 shrink-0">
                          <span className="tabular-nums">
                            {videoProgress.toFixed(0)}%
                            {typeof videoJob.proc_fps === "number" && videoJob.proc_fps > 0 && (
                              <> · {videoJob.proc_fps.toFixed(1)} fps</>
                            )}
                            {typeof videoJob.eta === "number" && videoJob.eta > 0 && videoJob.status === "running" && (
                              <> · 剩余 {formatEta(videoJob.eta)}</>
                            )}
                          </span>
                          <Badge variant="outline" className={`min-w-[3.25rem] justify-center shrink-0 ${
                            videoJob.status === "completed" ? "text-emerald-700 dark:text-green-400 border-green-400/30" :
                            videoJob.status === "failed"    ? "text-red-600 dark:text-red-400  border-red-400/30"  :
                                                              "text-blue-600 dark:text-blue-400 border-blue-400/30"
                          }`}>{VIDEO_STATUS_ZH[videoJob.status] ?? videoJob.status}</Badge>
                        </span>
                      </div>
                      <Progress value={videoProgress} className="h-1.5" />
                      <div className="rounded-md border border-border/50 p-3 space-y-2">
                        <p className="text-sm">{computeNodeId === "local" ? "本机推理日志" : "服务器推理日志"}</p>
                        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-xs font-mono">{videoJob.logs?.length ? videoJob.logs.join("\n") : videoJob.message || "等待服务器日志…"}</pre>
                      </div>
                    </div>
                  )}
                </>)}
              </CardContent>
            </Card>
          )}

          {/* ── 识别日志（视频 + 大模型）─────────────────────────────────────
              与 YOLO 那条链路的「服务器推理日志」对应：那一边 YOLO 的日志来自算力节点
              的任务快照（videoJob.logs），这一边来自后端按任务记的日志
              （GET /api/lmm/tasks/{id}/logs，增量 append）。 */}
          {sourceTab === "video" && modelSource === "smart" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-medium">识别日志</p>
                  <div className="flex items-center gap-3">
                    {vlxJob?.id && (
                      <span className="font-mono text-[10px] text-muted-foreground">{vlxJob.id}</span>
                    )}
                    <button type="button" disabled={!vlxLogs.length}
                      onClick={() => { setVlxLogs([]); vlxLogsSinceRef.current = 0 }}
                      className="text-[10px] text-muted-foreground hover:text-foreground disabled:opacity-40">
                      清空
                    </button>
                  </div>
                </div>
                {/* 按 level 上色：正常是灰、完成是绿、警告是黄、失败是红。
                    刻意不用 <pre> —— 一行一个 <p> 才能单独上色。 */}
                <div ref={vlxLogBoxRef}
                  className="max-h-64 overflow-auto space-y-0.5 rounded-md border border-border/50 bg-muted/20 p-3 font-mono text-xs">
                  {vlxLogs.length === 0 ? (
                    <p className="text-muted-foreground">
                      还没有日志 —— 上传视频后点「开始识别」，这里会按时间显示每一步
                      （打开视频 / 服务状态 / 抽样策略 / 进度 / 落盘 / 完成）。
                    </p>
                  ) : vlxLogs.map((l, i) => (
                    <p key={`${l.ts}-${i}`}
                      className={cn("break-all",
                        l.level === "error" ? "text-red-600 dark:text-red-400"
                          : l.level === "warn" ? "text-amber-700 dark:text-yellow-300"
                            : l.level === "ok" ? "text-emerald-700 dark:text-green-400"
                              : "text-muted-foreground")}>
                      <span className="tabular-nums">{logClock(l.ts)}</span>
                      {"  "}{l.text}
                    </p>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* ── Webcam tab ────────────────────────────────────────────────── */}
          {sourceTab === "webcam" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  {!webcamActive ? (
                    <Button onClick={startWebcam}
                      disabled={modelSource !== "smart" ? !activeModelPath || classesLoading : !smartInferenceReady}
                      className="gap-2">
                      <Camera className="w-4 h-4" /> 启动网络摄像头
                    </Button>
                  ) : (
                    <Button variant="destructive" onClick={stopWebcam} className="gap-2">
                      <Square className="w-4 h-4" /> 停止
                    </Button>
                  )}
                  {webcamActive && (
                    <Badge variant="outline" className="text-emerald-700 dark:text-green-400 border-green-400/30 gap-1">
                      <ZapIcon className="w-3 h-3" /> {fps} FPS
                    </Badge>
                  )}
                  {resultImg && webcamActive && (
                    <Button variant="outline" size="sm" onClick={downloadResult} className="gap-1 h-7 text-xs ml-auto">
                      <Download className="w-3 h-3" /> 保存当前帧
                    </Button>
                  )}
                </div>

                <video ref={videoRef} className="hidden" muted playsInline />
                <canvas ref={canvasRef} className="hidden" />

                {resultImg ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`data:image/jpeg;base64,${resultImg}`} alt="实时推理"
                    className="w-full rounded border border-border/40" />
                ) : webcamActive ? (
                  <div className="h-60 rounded border border-border/40 bg-muted/40 flex items-center justify-center">
                    <p className="text-sm text-muted-foreground">等待第一帧…</p>
                  </div>
                ) : (
                  <div className="h-60 rounded border-2 border-dashed border-border/30 flex flex-col items-center justify-center gap-2">
                    <Camera className="w-8 h-8 text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">启动网络摄像头以开始实时推理</p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* ── Detections ────────────────────────────────────────────────── */}
          {detections.length > 0 && !(sourceTab === "image" && imageMode === "folder") && (
            <Card className="border-border/50 bg-card/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Eye className="w-4 h-4 text-primary" /> 检测结果
                  <Badge variant="secondary" className="ml-auto">{detections.length}</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-48">
                  <div className="space-y-1">
                    {detections.map((d, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs p-1.5 rounded hover:bg-muted/30">
                        {d.track_id !== undefined && (
                          <span className="font-mono text-[10px] bg-primary/20 text-primary px-1 py-0.5 rounded shrink-0">#{d.track_id}</span>
                        )}
                        <span className={`font-semibold ${taskColor(task)}`}>{d.class_name}</span>
                        {hasDetectorConfidence(d) && <span className="text-muted-foreground">{(d.confidence * 100).toFixed(1)}%</span>}
                        {d.mask_points && (
                          <Badge variant="outline" className="text-[9px] text-purple-600 dark:text-purple-400 border-purple-400/30">mask</Badge>
                        )}
                        {d.bbox && (
                          <span className="font-mono text-muted-foreground/60 ml-auto text-[10px]">
                            [{d.bbox.map(v => Math.round(v)).join(", ")}]
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
