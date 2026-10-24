"use client"

import { useState, useEffect, useRef } from "react"
import trainingArchitectures from "@/lib/training-architectures.json"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import {
  Play,
  Pause,
  Square,
  Cpu,
  Settings2,
  Terminal,
  TrendingUp,
  TrendingDown,
  HardDrive,
  Activity,
  Download,
  ChevronDown,
  LineChart as LineChartIcon,
  FileDown,
  Package,
  Trash2,
  FolderOpen,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One row of per-epoch training data from the backend */
interface EpochRecord {
  epoch: number
  train_box_loss: number
  train_cls_loss: number
  train_dfl_loss: number
  val_box_loss: number
  val_cls_loss: number
  val_dfl_loss: number
  mAP50: number
  mAP50_95: number
  precision: number
  recall: number
  speed_ms: number
  its?: number
  gpu_mem_gb?: number
  // segmentation extras
  train_seg_loss?: number
  val_seg_loss?: number
  mAP50_seg?: number
}

interface LogEntry {
  timestamp: string
  message: string
  type: "info" | "warning" | "error" | "success"
}

interface BatchStats {
  epoch: number
  batch: number
  total_batches: number
  its: number
  losses: Record<string, number>
}

/** Shape returned by GET /api/train/:id/status */
interface TrainingStatus {
  remote_connection?: string | null
  remote_pid?: number | null
  id: string
  name: string
  status: "starting" | "running" | "pausing" | "completed" | "failed" | "stopped" | "paused" | "interrupted"
  progress: number
  current_epoch: number
  total_epochs: number
  metrics: Record<string, number>
  epoch_history: EpochRecord[]
  device_info: string | null
  gpu_mem_gb: number | null
  started_at: string
  model_path: string | null
  batch_stats: BatchStats | null
  logs: string[]
  error?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_ARCHITECTURES = trainingArchitectures

/** Derive the backend model_type from the selected architecture value. */
function getModelType(archValue: string): string {
  if (archValue.startsWith("rtdetrv2")) return "rtdetrv2"
  if (archValue.startsWith("rtdetr"))   return "rtdetr"
  if (archValue.startsWith("rfdetr"))   return "rfdetr"
  if (archValue.includes("-seg"))     return "segmentation"
  if (archValue.includes("-cls"))     return "classification"
  return "yolo"
}

const TASK_TYPES = [
  { value: "detection",      label: "Object Detection（目标检测）" },
  { value: "segmentation",   label: "Instance Segmentation（实例分割）" },
  { value: "classification", label: "Classification（分类）" },
]

/** 内部状态 key 保持英文（用于逻辑判断），仅展示时映射为中文 */
const STATUS_LABEL_ZH: Record<string, string> = {
  "Starting…": "启动中…",
  "Preparing dataset…": "准备数据集…",
  "Running": "训练中",
  "Finishing epoch…": "正在结束当前 epoch…",
  "Stopping…": "停止中…",
  "Completed": "已完成",
  "Failed": "失败",
  "Stopped": "已停止",
  "Paused": "已暂停",
  "Interrupted": "已中断",
}

/** 后端返回的原始 run status → 中文 */
const RUN_STATUS_ZH: Record<string, string> = {
  starting: "启动中",
  running: "训练中",
  pausing: "暂停中",
  paused: "已暂停",
  completed: "已完成",
  failed: "失败",
  stopped: "已停止",
  interrupted: "已中断",
}

const POLL_INTERVAL_MS = 2000

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DatasetInfo {
  id: string
  name: string
  path: string
  format?: string
  task_type?: string
}

interface TrainingViewProps {
  datasets?: DatasetInfo[]
  apiUrl?: string
  assistantDraftId?:string
  workflowId?:string
  initialDatasetId?:string
}

/** Classify a dataset's task domain based on format and task_type fields. */
function datasetDomain(ds: DatasetInfo): "classification" | "segmentation" | "detection" {
  const fmt = (ds.format  ?? "").toLowerCase()
  const tt  = (ds.task_type ?? "").toLowerCase()
  if (fmt === "classification-folder" || fmt === "classification_folder" || tt === "classification") return "classification"
  if (fmt === "cityscapes" || fmt === "ade20k" || fmt === "yolo-seg" || tt === "segmentation") return "segmentation"
  return "detection"
}

/** Return null if compatible, or an object with error / warning strings. */
function checkCompatibility(
  dsDomain: "classification" | "segmentation" | "detection",
  modelTaskType: string,   // "detection" | "segmentation" | "classification"
): { error?: string; warning?: string } | null {
  if (dsDomain === "classification" && modelTaskType !== "classification")
    return { error: "分类数据集只能训练 Classification 模型。" }
  if (dsDomain === "segmentation" && modelTaskType === "classification")
    return { error: "分割数据集不能训练 Classification 模型。" }
  if (dsDomain === "detection" && modelTaskType === "segmentation")
    return { error: "检测数据集不含多边形掩码。请使用分割格式的数据集，或改为训练 Object Detection 模型。" }
  if (dsDomain === "detection" && modelTaskType === "classification")
    return { error: "检测数据集不能训练 Classification 模型，请使用 classification-folder 格式的数据集。" }
  if (dsDomain === "segmentation" && modelTaskType === "detection")
    return { warning: "该数据集包含分割掩码。进行 Detection 训练时，多边形会自动转换为边界框。" }
  return null
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function getNewLogLines(previous: string[], incoming: string[]): string[] {
  // The API returns a sliding window of the last 50 lines, not an ever-growing list.
  for (let overlap = Math.min(previous.length, incoming.length); overlap > 0; overlap--) {
    if (previous.slice(-overlap).every((line, index) => line === incoming[index])) {
      return incoming.slice(overlap)
    }
  }
  return incoming
}

// Upload progress lines are throttled and get re-sent as the window slides, so
// keeping every historical percentage would bury the epoch summaries. Drop the
// older lines of the same stage and keep only the newest one.
function collapseProgressLines(lines: string[]): string[] {
  const stage = (line: string) => {
    const hit = line.match(/^(数据集快照|初始权重（[^）]*）|训练引擎（[^）]*）)\s/)
    return hit ? hit[1] : null
  }
  const lastIndexOf = new Map<string, number>()
  lines.forEach((line, i) => {
    const key = stage(line)
    if (key) lastIndexOf.set(key, i)
  })
  return lines.filter((line, i) => {
    const key = stage(line)
    return !key || lastIndexOf.get(key) === i
  })
}

export function TrainingView({ datasets = [], apiUrl = "http://localhost:8000",assistantDraftId,workflowId,initialDatasetId }: TrainingViewProps) {
  // Config state
  const [selectedDatasetId, setSelectedDatasetId] = useState("")
  const [modelArch, setModelArch]     = useState("yolov8n")
  const [taskType, setTaskType]       = useState("detection")
  const [epochs, setEpochs]           = useState(80)
  const [batchSize, setBatchSize]     = useState(1)
  const [imgSize, setImgSize]         = useState(640)
  const [trainingDevice,setTrainingDevice] = useState('auto')
  const [usePretrained, setUsePretrained] = useState(true)
  const [weightsMode, setWeightsMode] = useState("builtin")
  const [customWeights, setCustomWeights] = useState("")
  const [weightModels, setWeightModels] = useState<{ name: string; kind?: string; architecture?: string }[]>([])
  const [weightsLoading, setWeightsLoading] = useState(false)
  const [weightsError, setWeightsError] = useState<string | null>(null)
  const [weightsRefresh, setWeightsRefresh] = useState(0)
  const [computeNodeId, setComputeNodeId] = useState("local")
  const [computeNodes, setComputeNodes] = useState<{id:string; name:string; status:string; connection_type:string}[]>([])
  const [remoteConnection, setRemoteConnection] = useState<string | null>(null)
  const [remotePid, setRemotePid] = useState<number | null>(null)
  const [nodesError, setNodesError] = useState("")
  async function refreshComputeNodes() {
    try {
      const res = await fetch(`${apiUrl}/api/compute-nodes`)
      if (!res.ok) throw new Error("算力节点列表读取失败")
      const data = await res.json(); setComputeNodes(data.nodes); setNodesError("")
    } catch (e) { setNodesError(e instanceof Error ? e.message : "节点读取失败") }
  }
  useEffect(() => { refreshComputeNodes() }, [apiUrl])

  const selectedModelType = weightsMode === "custom"
    ? (weightModels.find(m => m.name === customWeights)?.kind === "rfdetr" ? "rfdetr" : taskType === "detection" ? "yolo" : taskType)
    : getModelType(modelArch)

  useEffect(() => {
    if (weightsMode !== "custom") return
    const controller = new AbortController()
    setWeightModels([])
    setWeightsLoading(true)
    setWeightsError(null)
    fetch(`${apiUrl}/api/train/initial-weights`, { signal: controller.signal })
      .then(async res => {
        if (!res.ok) throw new Error("加载自定义权重失败，请刷新列表。")
        return res.json()
      })
      .then(data => {
        if (!controller.signal.aborted) {
          const models = data.models ?? []
          setWeightModels(models)
          setCustomWeights(current => models.some((model: { name: string }) => model.name === current) ? current : "")
        }
      })
      .catch(err => { if (!controller.signal.aborted) setWeightsError(err.message) })
      .finally(() => { if (!controller.signal.aborted) setWeightsLoading(false) })
    return () => controller.abort()
  }, [apiUrl, weightsMode, weightsRefresh])

  // Save-period
  const [savePeriod, setSavePeriod]       = useState(10)  // 0 = disabled

  // Advanced hyperparams
  const [lr0, setLr0]                     = useState(0.001)
  const [lrf, setLrf]                     = useState(0.01)
  const [optimizer, setOptimizer]         = useState("AdamW")
  const [patience, setPatience]           = useState(20)
  const [cosLr, setCosLr]                 = useState(true)
  const [warmupEpochs, setWarmupEpochs]   = useState(3)
  const [weightDecay, setWeightDecay]     = useState(0.0005)
  const [mosaic, setMosaic]               = useState(0.5)
  const [hsvH, setHsvH]                   = useState(0.015)
  const [hsvS, setHsvS]                   = useState(0.5)
  const [hsvV, setHsvV]                   = useState(0.3)
  const [flipud, setFlipud]               = useState(0.0)
  const [fliplr, setFliplr]               = useState(0.5)
  const [amp, setAmp]                     = useState(false)
  const [dropout, setDropout]             = useState(0.0)
  const [advancedOpen, setAdvancedOpen]   = useState(false)

  // Export state
  const [isExporting, setIsExporting]     = useState(false)
  const [exportFormat, setExportFormat]   = useState("onnx")

  // GPU install state
  const [gpuWarning, setGpuWarning]       = useState<string | null>(null)
  const [installingCuda, setInstallingCuda] = useState(false)

  // Run name
  const [runName, setRunName]         = useState("swimmer_all_v1_640_b1_e80_AdamW_noAMP_v2")

  // Runtime state
  const [trainingId, setTrainingId]   = useState<string | null>(null)
  const [isTraining, setIsTraining]   = useState(false)
  const [isPaused, setIsPaused]       = useState(false)   // true when training is paused on backend
  const [isPausing, setIsPausing]     = useState(false)   // true while finishing current epoch before pause
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(100)
  const [progress, setProgress]       = useState(0)
  const [metrics, setMetrics]         = useState<EpochRecord[]>([])
  const [logs, setLogs]               = useState<LogEntry[]>([])
  const [statusLabel, setStatusLabel] = useState<string>("")
  const [modelPath, setModelPath]     = useState<string | null>(null)
  const [error, setError]             = useState<string | null>(null)

  // Past runs — all jobs from backend, refreshed on mount and after training ends
  interface RunSummary {
    compute_node_id?: string | null
    id: string; name: string; status: string; progress: number
    current_epoch: number; total_epochs: number; model_type: string
    started_at: string; model_path: string | null; last_checkpoint?: string | null
  }
  interface CheckpointInfo { name: string; path: string; size_mb: number }
  const [pastRuns, setPastRuns]           = useState<RunSummary[]>([])
  const nodeRuns = pastRuns.filter(run => (run.compute_node_id || "local") === computeNodeId)
  const [checkpoints, setCheckpoints]         = useState<CheckpointInfo[]>([])
  const [checkpointRunId, setCheckpointRunId] = useState<string | null>(null)
  const [deviceInfo, setDeviceInfo]   = useState<string | null>(null)
  const [liveGpuMem, setLiveGpuMem]   = useState<number | null>(null)
  const [batchStats, setBatchStats]   = useState<BatchStats | null>(null)

  // Real system stats — polled from /api/system while training
  const [cpuUsage, setCpuUsage]         = useState(0)
  const [gpuUsage, setGpuUsage]         = useState<number | null>(null)
  const [memoryUsage, setMemoryUsage]   = useState(0)
  const [gpuName, setGpuName]           = useState<string | null>(null)
  const systemPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const pollRef          = useRef<ReturnType<typeof setInterval> | null>(null)
  const previousLogWindow = useRef<string[]>([])
  const logsEndRef       = useRef<HTMLDivElement>(null)
  const failedPollCount  = useRef(0)   // consecutive polls returning "failed" — stop after threshold

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function inferLogType(message: string, fallback: LogEntry["type"]): LogEntry["type"] {
    const lc = message.toLowerCase()
    // Remote-training progress lines are Chinese, so the log type has to be inferred
    // from both languages — otherwise every upload line falls back to plain grey.
    if (lc.includes("error") || lc.includes("failed") || lc.includes("exception")
        || lc.includes("失败") || lc.includes("错误") || lc.includes("异常") || lc.includes("无法")) return "error"
    if (lc.includes("warning") || lc.includes("warn")
        || lc.includes("警告") || lc.includes("中断") || lc.includes("重连")
        || lc.includes("停止") || lc.includes("超时")) return "warning"
    if (lc.includes("completed") || lc.includes("saved") || lc.includes("success")
        || lc.includes("完成") || lc.includes("成功") || lc.includes("已确认")) return "success"
    return fallback
  }

  function addLog(message: string, type: LogEntry["type"]) {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, { timestamp, message, type }])
  }

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  async function handleInstallCudaTorch() {
    setInstallingCuda(true)
    addLog("正在安装 CUDA 版 PyTorch…可能需要几分钟。", "info")
    try {
      const res = await fetch(`${apiUrl}/api/install-cuda-torch`, { method: "POST" })
      const data = await res.json()
      if (data.success) {
        addLog("✓ CUDA PyTorch 安装完成。请重启后端服务，然后重新开始训练。", "success")
        setGpuWarning(null)
      } else {
        addLog(`安装失败：${data.message}`, "error")
      }
    } catch (err) {
      addLog(`安装请求失败：${err instanceof Error ? err.message : err}`, "error")
    } finally {
      setInstallingCuda(false)
    }
  }

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // -------------------------------------------------------------------------
  // System stats polling — always active so CPU/RAM/GPU are shown even when idle
  // -------------------------------------------------------------------------
  useEffect(() => {
    const pollSystem = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/system`)
        if (!res.ok) return
        const data = await res.json()
        setCpuUsage(data.cpu_percent ?? 0)
        if (data.gpu_percent !== null && data.gpu_percent !== undefined) {
          setGpuUsage(data.gpu_percent)
          if (data.gpu_name) setGpuName(data.gpu_name)
        } else {
          setGpuUsage(null)
        }
        setMemoryUsage(data.gpu_memory_percent ?? data.ram_percent ?? 0)
      } catch {
        // Backend unreachable — leave stats at last known value
      }
    }

    pollSystem()
    // Poll every 3 s while training, every 10 s while idle
    const interval = isTraining ? 3000 : 10000
    systemPollRef.current = setInterval(pollSystem, interval)
    return () => {
      if (systemPollRef.current) {
        clearInterval(systemPollRef.current)
        systemPollRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, apiUrl])

  // -------------------------------------------------------------------------
  // Persist trainingId to localStorage so it survives page refreshes
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (trainingId) localStorage.setItem('cvdm_training_id', trainingId)

  }, [trainingId])

  // -------------------------------------------------------------------------
  const openRunVersion = useRef(0)
  const [runViewVersion, setRunViewVersion] = useState(0)
  function switchComputeNode(id: string) {
    ++openRunVersion.current
    stopPolling()
    setComputeNodeId(id); setTrainingId(null)
    setIsTraining(false); setIsPaused(false); setIsPausing(false)
    setLogs([]); previousLogWindow.current=[]; setMetrics([]); setError(null)
    setProgress(0); setCurrentEpoch(0); setTotalEpochs(epochs); setStatusLabel("")
    setModelPath(null); setDeviceInfo(null); setLiveGpuMem(null); setBatchStats(null); setGpuWarning(null)
    setCheckpoints([]); setCheckpointRunId(null)
    setRemoteConnection(null); setRemotePid(null)
    const matching = pastRuns.filter(run => (run.compute_node_id || "local") === id)
    const run = matching.find(r => ["running","starting","pausing"].includes(r.status)) || matching[0]
    if (run) void openTrainingRun(run)
  }
  async function openTrainingRun(active: RunSummary) {
    const version = ++openRunVersion.current
    setRunViewVersion(v=>v+1)
    stopPolling()
    setLogs([]); previousLogWindow.current=[]; setMetrics([]); setError(null)
    setModelPath(active.model_path || null)
    setDeviceInfo(null); setLiveGpuMem(null); setBatchStats(null); setGpuWarning(null)
    setCheckpoints([]); setCheckpointRunId(null)
    setComputeNodeId(active.compute_node_id || "local")
    setRemoteConnection(null); setRemotePid(null)
    failedPollCount.current=0
    try {
          setTrainingId(active.id)
          setIsTraining(['running','starting','pausing'].includes(active.status))
          setIsPaused(false)
          setIsPausing(active.status === 'pausing')
          setProgress(active.progress ?? 0)
          setCurrentEpoch(active.current_epoch ?? 0)
          setTotalEpochs(active.total_epochs ?? 100)
          setStatusLabel(active.status === 'pausing' ? 'Finishing epoch…' : 'Running')
          addLog(`已打开训练任务「${active.name}」（id: ${active.id}）`, 'info')
          // Restore the submitted configuration as well as the progress panel.
          // The report/status endpoints are read-only; this does not submit training.
          setRunName(active.name)
          const reportResponse = await fetch(`${apiUrl}/api/train/${active.id}/report`)
          if (reportResponse.ok) {
            const report = await reportResponse.json()
            if (openRunVersion.current !== version) return
            const config = report.config || {}
            setComputeNodeId(config.compute_node_id || "local")
            let datasetId = config.dataset_id || String(report.model_path || "").replace(/\\/g, "/").match(/\/datasets\/([^/]+)\//)?.[1]
            if (!datasetId) {
              const statusResponse = await fetch(`${apiUrl}/api/train/${active.id}/status`)
              if (statusResponse.ok) {
                const status = await statusResponse.json()
                const dataLog = (status.logs || []).find((line: string) => line.startsWith("Data config:"))
                datasetId = String(dataLog || "").replace(/\\/g, "/").match(/\/datasets\/([^/]+)\//)?.[1]
              }
            }
            if (datasetId) setSelectedDatasetId(datasetId)
            setTaskType(report.model_type === "segmentation" ? "segmentation" : report.model_type === "classification" ? "classification" : "detection")
            if (config.initial_weights) setWeightsMode(config.initial_weights)
            if (config.custom_weights) setCustomWeights(config.custom_weights)
            const architecture = String(config.base_model || "").match(/yolo(?:v)?\d+[nslmx]/i)?.[0]
            if (architecture) setModelArch(architecture.toLowerCase())
            if (config.pretrained != null) setUsePretrained(config.pretrained)
            if (config.epochs != null) setEpochs(config.epochs)
            if (config.batch_size != null) setBatchSize(config.batch_size)
            if (config.image_size != null) setImgSize(config.image_size)
            if (config.save_period != null) setSavePeriod(config.save_period)
            if (config.lr0 != null) setLr0(config.lr0)
            if (config.lrf != null) setLrf(config.lrf)
            if (config.optimizer != null) setOptimizer(config.optimizer)
            if (config.patience != null) setPatience(config.patience)
            if (config.cos_lr != null) setCosLr(config.cos_lr)
            if (config.warmup_epochs != null) setWarmupEpochs(config.warmup_epochs)
            if (config.weight_decay != null) setWeightDecay(config.weight_decay)
            if (config.mosaic != null) setMosaic(config.mosaic)
            if (config.hsv_h != null) setHsvH(config.hsv_h)
            if (config.hsv_s != null) setHsvS(config.hsv_s)
            if (config.hsv_v != null) setHsvV(config.hsv_v)
            if (config.flipud != null) setFlipud(config.flipud)
            if (config.fliplr != null) setFliplr(config.fliplr)
            if (config.amp != null) setAmp(config.amp)
            if (config.dropout != null) setDropout(config.dropout)
          } else {
            addLog("无法恢复配置；后端训练按原配置继续进行。", "warning")
          }
    } catch { addLog("任务配置读取失败，请重新打开该任务。", "warning") }
  }

  // Restore monitoring only; opening history never resumes or submits training.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const restore = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/train/jobs`)
        if (!res.ok) return
        const data = await res.json()
        const jobs: RunSummary[] = data.jobs || []

        // Always populate the past-runs list
        setPastRuns(jobs)

        // Prefer active work, then the last viewed job, then the latest saved run.
        const savedId = localStorage.getItem('cvdm_training_id')
        const active = jobs.find(j => j.status === 'running' || j.status === 'starting' || j.status === 'pausing')
          || jobs.find(j => j.id === savedId) || jobs[0]
        if (active && !new URLSearchParams(location.search).get("assistant_draft")) await openTrainingRun(active)

      } catch { /* backend not reachable yet */ }
    }
    restore()
  }, [apiUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Polling — starts whenever trainingId is set, pauses if isPaused
  // -------------------------------------------------------------------------

  useEffect(() => {
    // Stop polling only when fully paused/stopped — NOT while still finishing an epoch
    if (!trainingId || isPaused) {
      stopPolling()
      return
    }

    let disposed = false
    const viewVersion = openRunVersion.current
    async function poll() {
      try {
        const res = await fetch(`${apiUrl}/api/train/${trainingId}/status${previousLogWindow.current.length === 0 ? "?full_logs=true" : ""}`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: TrainingStatus = await res.json()
        if (disposed || viewVersion !== openRunVersion.current) return
        setRemoteConnection(data.remote_connection || null); setRemotePid(data.remote_pid || null)

        setCurrentEpoch(data.current_epoch)
        setTotalEpochs(data.total_epochs)
        setProgress(data.progress)

        // Sync device info, live GPU memory, and batch stats
        if (data.device_info) setDeviceInfo(data.device_info)
        if (data.gpu_mem_gb != null) setLiveGpuMem(data.gpu_mem_gb)
        if (data.batch_stats) setBatchStats(data.batch_stats)

        // Replace epoch history wholesale (backend is authoritative)
        if (data.epoch_history && data.epoch_history.length > 0) {
          setMetrics(data.epoch_history)
        }

        // Append only new log lines, auto-detecting type from content
        const incomingLogs = collapseProgressLines(data.logs || [])
        const newLogs = getNewLogLines(previousLogWindow.current, incomingLogs)
        previousLogWindow.current = [...incomingLogs]
        newLogs.forEach(msg => {
          addLog(msg, inferLogType(msg, "info"))
          if (msg.includes("Install CUDA-enabled PyTorch") || msg.includes("CPU-only")) {
            setGpuWarning(msg)
          }
        })

        if (data.status === "pausing") {
          failedPollCount.current = 0
          setIsPausing(true)
          setStatusLabel("Finishing epoch…")
          // keep polling — backend will transition to "paused" when the epoch completes
        } else if (data.status === "completed") {
          failedPollCount.current = 0
          setIsTraining(false); setIsPausing(false); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          addLog("训练成功完成！模型已保存。", "success")
          setStatusLabel("Completed")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
          fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
        } else if (data.status === "failed") {
          // Poll a couple more times to flush any final log lines, then stop.
          failedPollCount.current += 1
          setIsTraining(false); setIsPausing(false); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          setError(data.error ?? "Unknown error")
          setStatusLabel("Failed")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          if (failedPollCount.current >= 3) {
            stopPolling()
            fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
          }
        } else if (data.status === "stopped") {
          failedPollCount.current = 0
          setIsTraining(false); setIsPausing(false); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          addLog("训练已停止。", "warning")
          setStatusLabel("Stopped")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
          fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
        } else if (data.status === "paused") {
          failedPollCount.current = 0
          setIsTraining(false); setIsPausing(false); setIsPaused(true); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          addLog("本轮 Epoch 已完成 — 训练已暂停。点击「继续」以恢复。", "info")
          setStatusLabel("Paused")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
          fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
        } else if (data.status === "interrupted") {
          failedPollCount.current = 0
          setIsTraining(false)
          if (data.model_path) setModelPath(data.model_path)
          setStatusLabel("Interrupted")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
        } else {
          // status is "running" or "starting" — re-sync isTraining in case the
          // component was remounted (navigation) or the state drifted
          failedPollCount.current = 0
          setIsTraining(true)
          if (data.status === "starting") {
            setStatusLabel("Preparing dataset…")
          } else {
            setStatusLabel(prev => (prev === "Preparing dataset…" || prev === "Starting…") ? "Running" : prev)
          }
        }
      } catch (err) {
        addLog(`轮询出错：${err instanceof Error ? err.message : err}`, "error")
      }
    }

    poll()  // fire immediately, then every POLL_INTERVAL_MS
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS)
    return () => { disposed = true; stopPolling() }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingId, isPaused, apiUrl, runViewVersion])

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------

  function handleDownloadReport() {
    if (!trainingId) return
    window.open(`${apiUrl}/api/train/${trainingId}/report`, "_blank")
  }

  const [assistantDraft,setAssistantDraft] = useState<any>(null)
  const [assistantNotice,setAssistantNotice] = useState("")
  const [submitting,setSubmitting] = useState(false)
  const submitGuard = useRef(false)
  const loadedDraft = useRef("")
  async function assistantCall(path:string,method="GET",body?:unknown) {
    const r=await fetch(`${apiUrl}/api/assistant/${path}`,{method,headers:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body),cache:"no-store"})
    const data=await r.json()
    if(!r.ok)throw Error(typeof data.detail==="string"?data.detail:data.detail?.message||"参数草稿操作未完成。")
    return data
  }
  function formData() { return {
          compute_node_id: computeNodeId === "local" ? null : computeNodeId,
          name: runName.trim(),
          dataset_id: selectedDatasetId,
          model_type: selectedModelType,
          model_arch: modelArch,
          epochs,
          batch_size: batchSize,
          image_size: imgSize,
          pretrained: weightsMode === "custom" ? true : usePretrained,
          initial_weights: weightsMode,
          custom_weights: weightsMode === "custom" ? customWeights : null,
          device: trainingDevice,
          save_period: savePeriod,
          // Advanced hyperparams
          lr0,
          lrf,
          optimizer,
          patience,
          cos_lr: cosLr,
          warmup_epochs: warmupEpochs,
          weight_decay: weightDecay,
          mosaic,
          hsv_h: hsvH,
          hsv_s: hsvS,
          hsv_v: hsvV,
          flipud,
          fliplr,
          amp,
          dropout,
        } }
  function applyAssistantForm(c:Record<string,any>) {
    if(c.device!=null)setTrainingDevice(c.device)
    if (c.dataset_id != null) setSelectedDatasetId(c.dataset_id)
    if (c.name != null) setRunName(c.name)
    if (c.model_arch != null) setModelArch(c.model_arch)
    if (c.epochs != null) setEpochs(c.epochs)
    if (c.batch_size != null) setBatchSize(c.batch_size)
    if (c.image_size != null) setImgSize(c.image_size)
    if (c.pretrained != null) setUsePretrained(c.pretrained)
    if (c.initial_weights != null) setWeightsMode(c.initial_weights)
    if (c.custom_weights != null) setCustomWeights(c.custom_weights)
    if (c.save_period != null) setSavePeriod(c.save_period)
    if (c.lr0 != null) setLr0(c.lr0)
    if (c.lrf != null) setLrf(c.lrf)
    if (c.optimizer != null) setOptimizer(c.optimizer)
    if (c.patience != null) setPatience(c.patience)
    if (c.cos_lr != null) setCosLr(c.cos_lr)
    if (c.warmup_epochs != null) setWarmupEpochs(c.warmup_epochs)
    if (c.weight_decay != null) setWeightDecay(c.weight_decay)
    if (c.mosaic != null) setMosaic(c.mosaic)
    if (c.hsv_h != null) setHsvH(c.hsv_h)
    if (c.hsv_s != null) setHsvS(c.hsv_s)
    if (c.hsv_v != null) setHsvV(c.hsv_v)
    if (c.flipud != null) setFlipud(c.flipud)
    if (c.fliplr != null) setFliplr(c.fliplr)
    if (c.amp != null) setAmp(c.amp)
    if (c.dropout != null) setDropout(c.dropout)
    setComputeNodeId(c.compute_node_id || "local")
    setTaskType(c.model_type==="segmentation"?"segmentation":c.model_type==="classification"?"classification":"detection")
  }
  useEffect(()=>{
    const id=assistantDraftId||(!workflowId?new URLSearchParams(location.search).get("assistant_draft"):"")||""
    if(!/^[a-f0-9]{32}$/.test(id)||loadedDraft.current===id)return
    loadedDraft.current=id
    let live=true
    void assistantCall(`drafts/${id}`).then(d=>{
      if(!live)return
      if(d.kind!=="training")throw Error("该草稿不是训练参数。")
      if(!sessionStorage.getItem(`runos-form-backup:${id}`))sessionStorage.setItem(`runos-form-backup:${id}`,JSON.stringify(formData()))
      setAssistantDraft(d);applyAssistantForm(d.data)
      setAssistantNotice("助手参数已填入。可调整全部参数，核对后确认执行。原表单参数已保留。")
      if(d.status==="submitted"&&d.result?.training_id){setTrainingId(d.result.training_id);setIsTraining(true)}
    }).catch(e=>{if(live)setAssistantNotice(e.message)})
    return()=>{live=false}
  },[apiUrl,assistantDraftId,workflowId]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(()=>{if(initialDatasetId&&!assistantDraft)setSelectedDatasetId(initialDatasetId)},[initialDatasetId,assistantDraft])
  async function handleConfirmTraining() {
    if(submitGuard.current)return
    submitGuard.current=true;setSubmitting(true)
    try{await handleStartTraining()}finally{submitGuard.current=false;setSubmitting(false)}
  }
  function restoreBeforeAssistant(){
    const old=sessionStorage.getItem(`runos-form-backup:${assistantDraft?.id}`)
    if(old){applyAssistantForm(JSON.parse(old));setAssistantDraft(null);setAssistantNotice("已恢复回填前的表单参数。");history.replaceState(null,"",location.pathname)}
  }

  async function handleStartTraining() {
    if(workflowId&&!assistantDraft){setError('请先通过助手选择模型与参数并生成训练草稿，再确认执行。');return}
    if (!selectedDatasetId) {
      addLog("请先选择数据集再开始训练。", "error")
      return
    }

    if (weightsMode === "custom" && (weightsLoading || weightsError || !weightModels.some(m => m.name === customWeights))) {
      setError("请先选择一个可用的自定义 .pt 模型再开始训练。")
      return
    }

    // Dataset / model compatibility check (client-side early warning)
    const selectedDs = datasets.find(d => d.id === selectedDatasetId)
    if (selectedDs) {
      const dsDomain  = datasetDomain(selectedDs)
      const modelTask = selectedModelType   // "yolo" | "rtdetr" | "rfdetr" | "segmentation" | "classification"
      // Map backend model_type to simplified task domain for check
      const modelDomain = modelTask === "segmentation" ? "segmentation"
                        : modelTask === "classification" ? "classification"
                        : "detection"
      const compat = checkCompatibility(dsDomain, modelDomain)
      if (compat?.error) {
        addLog(compat.error, "error")
        setError(compat.error)
        return
      }
    }

    // Clear trainingId FIRST so that when isPaused becomes false the polling
    // effect sees !trainingId and stops immediately, preventing a race where
    // the previous job is re-polled and restores its stale epoch/metrics.
    setTrainingId(null)
    setMetrics([])
    setLogs([])
    setError(null)
    setModelPath(null)
    setCurrentEpoch(0)
    setTotalEpochs(epochs)
    setProgress(0)
    setDeviceInfo(null)
    setLiveGpuMem(null)
    previousLogWindow.current = []
    failedPollCount.current = 0
    setIsPaused(false)
    setIsPausing(false)
    setIsTraining(false)
    setBatchStats(null)
    setStatusLabel("Starting…")

    if (selectedModelType === "rfdetr" && taskType !== "detection") {
      setIsTraining(false); setStatusLabel("请选择目标检测任务"); addLog("RF-DETR 当前支持目标检测任务", "error"); return
    }
    if (computeNodeId !== "local" && (!computeNodes.some(n => n.id === computeNodeId) || taskType !== "detection" || !["yolo", "rfdetr"].includes(selectedModelType))) {
      setError("请选择可用 SSH 节点和 YOLO / RF-DETR 目标检测模型"); return
    }
    const backendModelType = selectedModelType

    try {
      let data: any
      if (assistantDraft) {
        if(assistantDraft.status==='submitted'){const existing=await assistantCall(`drafts/${assistantDraft.id}`);setTrainingId(existing.result?.training_id||null);setIsTraining(true);setAssistantNotice('已定位已有训练任务。重新训练需重新准备参数，避免重复提交。');return}
        const saved = await assistantCall(`drafts/${assistantDraft.id}`, "PUT", {kind:"training",data:formData(),revision:assistantDraft.revision})
        setAssistantDraft(saved)
        const submitted = await assistantCall(`drafts/${saved.id}/execute`, "POST", {revision:saved.revision})
        setAssistantDraft(submitted)
        data = submitted.result
      } else {
        const res = await fetch(`${apiUrl}/api/train`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(formData())})
        data = await res.json()
        if (!res.ok) throw Error(typeof data.detail === "string" ? data.detail : data.detail?.message || "训练参数校验未通过。")
      }
      const id: string = data.training_id

      addLog(`训练任务已启动（id: ${id}）`, "info")
      addLog(`模型：${weightsMode === "custom" ? customWeights : MODEL_ARCHITECTURES.find(m => m.value === modelArch)?.label ?? modelArch}`, "info")
      addLog(`数据集：${datasets.find(d => d.id === selectedDatasetId)?.name ?? selectedDatasetId}`, "info")

      setTrainingId(id)
      setIsTraining(true)
      setStatusLabel("Running")
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addLog(`启动训练失败：${msg}`, "error")
      setError(msg)
    }
  }

  async function handleStopTraining() {
    if (!trainingId) return
    try {
      await fetch(`${apiUrl}/api/train/${trainingId}/stop`, { method: "POST" })
      addLog("已发送停止信号 — 训练将在当前 Epoch 结束后停止。", "warning")
      setIsTraining(computeNodeId !== "local")
      setIsPaused(false)
      setStatusLabel("Stopping…")
      if (computeNodeId === "local") stopPolling()
    } catch (err: unknown) {
      addLog(`停止请求失败：${err instanceof Error ? err.message : err}`, "error")
    }
  }

  async function handlePauseResume() {
    if (!trainingId) return
    if (!isPaused) {
      // Pause: signal backend — keep polling so we see "pausing" → "paused" transition
      try {
        const res = await fetch(`${apiUrl}/api/train/${trainingId}/pause`, { method: "POST" })
        if (res.ok) {
          setIsPausing(true)
          setStatusLabel("Finishing epoch…")
          // Do NOT stop polling or set isPaused here — the poll handler does that
          // when the backend responds with status "paused"
        } else {
          addLog("暂停失败。", "error")
        }
      } catch (err: unknown) {
        addLog(`暂停请求失败：${err instanceof Error ? err.message : err}`, "error")
      }
    } else {
      await doResume(trainingId)
    }
  }

  async function doResume(id: string) {
    try {
      const res = await fetch(`${apiUrl}/api/train/${id}/resume`, { method: "POST" })
      if (res.ok) {
        setTrainingId(id)
        setIsPaused(false)
        setIsPausing(false)
        setIsTraining(true)
        setStatusLabel("Running")
        previousLogWindow.current = []   // re-stream all logs from this job
        failedPollCount.current = 0
        addLog("正在从 Checkpoint 恢复训练…", "info")
      } else {
        const err = await res.json().catch(() => ({ detail: "恢复失败" }))
        addLog(`恢复失败：${err.detail}`, "error")
      }
    } catch (err: unknown) {
      addLog(`恢复请求失败：${err instanceof Error ? err.message : err}`, "error")
    }
  }

  async function handleDeleteRun(id: string) {
    if (!confirm("从列表中删除该次训练？（模型权重文件仍保留在磁盘上。）")) return
    try {
      const res = await fetch(`${apiUrl}/api/train/${id}`, { method: "DELETE" })
      if (res.ok) {
        setPastRuns(prev => prev.filter(r => r.id !== id))
        if (trainingId === id) {
          setTrainingId(null)
          setIsPaused(false)
          setStatusLabel("")
        }
      } else {
        const err = await res.json().catch(() => ({ detail: "删除失败" }))
        addLog(`删除失败：${err.detail}`, "error")
      }
    } catch (e) {
      addLog(`删除请求失败：${e instanceof Error ? e.message : e}`, "error")
    }
  }

  async function handleLoadCheckpoints(id: string) {
    if (checkpointRunId === id) { setCheckpointRunId(null); return }
    try {
      const res = await fetch(`${apiUrl}/api/train/${id}/checkpoints`)
      if (res.ok) {
        const data = await res.json()
        setCheckpoints(data.checkpoints || [])
        setCheckpointRunId(id)
      }
    } catch { /* ignore */ }
  }

  function handleExportModel() {
    if (!trainingId) return
    window.open(`${apiUrl}/api/train/${trainingId}/export`, "_blank")
  }

  async function handleExportFormat(format: string) {
    if (!trainingId) return
    setIsExporting(true)
    try {
      const res = await fetch(`${apiUrl}/api/train/${trainingId}/export-format?format=${format}`, { method: "POST" })
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `model.${format === 'onnx' ? 'onnx' : format === 'tflite' ? 'tflite' : format === 'coreml' ? 'mlpackage' : 'engine'}`
        a.click()
        URL.revokeObjectURL(url)
      } else {
        // Fallback: direct download link
        window.open(`${apiUrl}/api/train/${trainingId}/export?format=${format}`, "_blank")
      }
    } catch {
      window.open(`${apiUrl}/api/train/${trainingId}/export`, "_blank")
    } finally {
      setIsExporting(false)
    }
  }

  // -------------------------------------------------------------------------
  // Derived display values
  // -------------------------------------------------------------------------

  const latestMetrics = metrics[metrics.length - 1]
  const prevMetrics   = metrics[metrics.length - 2]

  const etaDisplay = (() => {
    if (!isTraining || isPaused || isPausing || currentEpoch === 0) return null
    const remaining = totalEpochs - currentEpoch
    let secsPerEpoch = 30
    // Live: batch_stats has real it/s and total_batches → most accurate
    if (batchStats && batchStats.its > 0 && batchStats.total_batches > 0) {
      secsPerEpoch = batchStats.total_batches / batchStats.its
    } else if (latestMetrics?.its) {
      const nb = batchStats?.total_batches ?? 100
      secsPerEpoch = nb / latestMetrics.its
    }
    const total = remaining * secsPerEpoch
    const h = Math.floor(total / 3600)
    const m = Math.floor((total % 3600) / 60)
    return h > 0 ? `${h}h ${m}m` : `${m}m`
  })()

  // 展示用的中文状态（内部状态仍为英文 key）
  const statusDisplay = STATUS_LABEL_ZH[statusLabel] ?? statusLabel

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">训练</h1>
          <p className="text-muted-foreground mt-1">使用本地或远程算力训练模型</p>
        </div>
        <div className="flex items-center gap-2">
          {isTraining ? (
            <>
              <Button
                variant="outline"
                onClick={handlePauseResume}
                disabled={isPausing || selectedModelType === "rfdetr" || computeNodeId !== "local"}
                className="gap-2"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPausing ? "正在结束当前 epoch…" : isPaused ? "继续" : "暂停"}
              </Button>
              <Button variant="destructive" onClick={handleStopTraining} className="gap-2">
                <Square className="w-4 h-4" />
                停止
              </Button>
            </>
          ) : (
            <>
              {modelPath && (
              <>
                <Button variant="outline" onClick={handleExportModel} className="gap-2">
                  <Download className="w-4 h-4" />
                  下载 {modelPath.split(/[\\/]/).pop()}
                </Button>
                <Select value={exportFormat} onValueChange={setExportFormat}>
                  <SelectTrigger className="w-28 h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="onnx">ONNX</SelectItem>
                    <SelectItem value="tflite">TFLite</SelectItem>
                    <SelectItem value="coreml">CoreML</SelectItem>
                    <SelectItem value="engine">TensorRT</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  onClick={() => handleExportFormat(exportFormat)}
                  disabled={isExporting}
                  className="gap-2"
                >
                  <FileDown className="w-4 h-4" />
                  {isExporting ? "导出中…" : `导出 ${exportFormat.toUpperCase()}`}
                </Button>
              </>
            )}
            {trainingId && metrics.length > 0 && (
              <Button variant="outline" onClick={handleDownloadReport} className="gap-2">
                <Package className="w-4 h-4" />
                报告
              </Button>
            )}
              <Button onClick={handleConfirmTraining} disabled={submitting||!!workflowId&&!assistantDraft} className="gap-2">
                <Play className="w-4 h-4" />
                {submitting?"正在提交…":assistantDraft?"确认并执行训练":"开始训练"}
              </Button>
            </>
          )}
        </div>
      </div>

      {assistantNotice&&<div role="status" className="mb-4 rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">{assistantNotice} {assistantDraft&&<button type="button" onClick={restoreBeforeAssistant} className="ml-3 text-primary underline">恢复原表单</button>}</div>}
      {workflowId&&<div className="mb-4 rounded-lg border p-3 text-sm"><p className="font-medium">执行前核对</p><dl className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3"><div><dt className="text-muted-foreground">划分数据集</dt><dd>{datasets.find(d=>d.id===selectedDatasetId)?.name||'等待选择'}</dd></div><div><dt className="text-muted-foreground">模型</dt><dd>{MODEL_ARCHITECTURES.find(m=>m.value===modelArch)?.label||modelArch}</dd></div><div><dt className="text-muted-foreground">初始权重</dt><dd>{weightsMode==='custom'?customWeights:'内置预训练权重'}</dd></div><div><dt className="text-muted-foreground">轮次 / 批次 / 图像尺寸</dt><dd>{epochs} / {batchSize} / {imgSize}</dd></div><div><dt className="text-muted-foreground">执行设备</dt><dd>{computeNodeId==='local'?trainingDevice:computeNodes.find(n=>n.id===computeNodeId)?.name||computeNodeId}</dd></div></dl><p className="mt-2 text-xs text-muted-foreground">确认后创建训练任务。下方表单仍可修改，结果在当前页查看。</p></div>}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Config */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Settings2 className="w-5 h-5 text-primary" />
              训练配置
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">

            <div className="space-y-2">
              <label className="text-sm font-medium">算力节点</label>
              <label className="text-xs text-muted-foreground">训练设备（auto / cpu / GPU 编号）</label><Input value={trainingDevice} onChange={e=>setTrainingDevice(e.target.value)} placeholder="auto" disabled={isTraining} />
              <Select value={computeNodeId} onValueChange={switchComputeNode} disabled={isTraining}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="local">本地电脑</SelectItem>
                  {computeNodeId !== "local" && !computeNodes.some(n => n.id === computeNodeId) && <SelectItem value={computeNodeId} disabled>节点不可用</SelectItem>}
                  {computeNodes.filter(n => n.connection_type === "ssh").map(n => <SelectItem key={n.id} value={n.id}>{n.name} · {n.status === "online" ? "最近检测在线" : "待检测"}</SelectItem>)}
                </SelectContent>
              </Select>
              <div className="flex gap-3 text-xs"><button type="button" disabled={isTraining} onClick={refreshComputeNodes} className="text-primary">刷新节点</button><a href="/compute-nodes" className="text-primary">管理节点</a></div>
              {nodesError && <p className="text-xs text-destructive">{nodesError}</p>}
              {computeNodeId !== "local" && <p className="text-xs text-muted-foreground">SSH 远程训练：YOLO / RF-DETR 目标检测。自动上传数据与权重，完成后取回模型。暂不支持暂停。</p>}
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">数据集</label>
              {datasets.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">
                  暂无数据集。请先到「数据集」标签页导入。
                </p>
              ) : (
                <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId} disabled={isTraining}>
                  <SelectTrigger><SelectValue placeholder="选择数据集…" /></SelectTrigger>
                  <SelectContent>
                    {datasets.map(ds => (
                      <SelectItem key={ds.id} value={ds.id}>
                        <span>{ds.name}</span>
                        {ds.format && (
                          <Badge variant="outline" className="ml-2 text-[10px]">{ds.format}</Badge>
                        )}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {/* Compatibility notice */}
              {(() => {
                const ds = datasets.find(d => d.id === selectedDatasetId)
                if (!ds) return null
                const dsDomain  = datasetDomain(ds)
                const modelTask = selectedModelType
                const modelDomain = modelTask === "segmentation" ? "segmentation"
                                  : modelTask === "classification" ? "classification"
                                  : "detection"
                const compat = checkCompatibility(dsDomain, modelDomain)
                if (!compat) return null
                if (compat.error) return (
                  <p className="text-[11px] text-red-600 dark:text-red-400 leading-snug">{compat.error}</p>
                )
                if (compat.warning) return (
                  <p className="text-[11px] text-amber-700 dark:text-yellow-400 leading-snug">{compat.warning}</p>
                )
                return null
              })()}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">运行名称</label>
              <Input
                placeholder="例如 cityscapes-run-1"
                value={runName}
                onChange={e => setRunName(e.target.value)}
                disabled={isTraining}
                className="text-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">任务类型</label>
              <Select
                value={taskType}
                onValueChange={v => {
                  setTaskType(v)
                  // Auto-select the first arch for the chosen task
                  const first = MODEL_ARCHITECTURES.find(a => a.taskType === v)
                  if (first) setModelArch(first.value)
                }}
                disabled={isTraining}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {TASK_TYPES.map(t => (
                    <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">初始权重 / 微调来源</label>
              <Select value={weightsMode} onValueChange={setWeightsMode} disabled={isTraining}>
                <SelectTrigger aria-label="初始权重 / 微调来源"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="builtin">内置预训练权重</SelectItem>
                  <SelectItem value="custom">自定义模型权重</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {weightsMode === "custom" && (
              <div className="space-y-2">
                <label className="text-sm font-medium">自定义模型权重</label>
                <Select value={customWeights} onValueChange={setCustomWeights} disabled={isTraining || weightsLoading}>
                  <SelectTrigger aria-label="自定义模型权重"><SelectValue placeholder={weightsLoading ? "加载模型中…" : "选择 YOLO .pt / RF-DETR .pth 模型"} /></SelectTrigger>
                  <SelectContent>
                    {weightModels.map(model => <SelectItem key={model.name} value={model.name}>{model.kind === "rfdetr" ? `RF-DETR · ${model.name}` : model.name}</SelectItem>)}
                  </SelectContent>
                </Select>
                <Button type="button" variant="ghost" size="sm" disabled={isTraining || weightsLoading} onClick={() => setWeightsRefresh(v => v + 1)}>刷新模型列表</Button>
                {weightsError && <p role="alert" className="text-xs text-destructive">{weightsError}</p>}
                {!weightsLoading && !weightsError && weightModels.length === 0 && <p className="text-xs text-muted-foreground">未找到 YOLO .pt 或 RF-DETR .pth 模型。请先在「模型」中导入，然后刷新。</p>}
                <p className="text-xs text-muted-foreground">支持 YOLO .pt / RF-DETR .pth，自动适配类别。</p>
              </div>
            )}

            {selectedModelType === "rfdetr" && <p className="text-xs text-muted-foreground">RF-DETR：目标检测 · GPU 优先 · 支持 HSV / Flip，暂不支持暂停续训。</p>}

            {weightsMode === "builtin" && <div className="space-y-2">
              <label className="text-sm font-medium">模型架构</label>
              <Select
                value={modelArch}
                onValueChange={v => {
                  setModelArch(v)
                  // Sync task type to match the chosen arch
                  const m = MODEL_ARCHITECTURES.find(a => a.value === v)
                  if (m && m.taskType !== taskType) setTaskType(m.taskType)
                }}
                disabled={isTraining}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {MODEL_ARCHITECTURES.filter(m => m.taskType === taskType).map(model => (
                    <SelectItem key={model.value} value={model.value}>
                      <span>{model.label}</span>
                      <Badge variant="outline" className="ml-1 text-[10px]">{model.arch}</Badge>
                      <Badge variant="secondary" className="ml-1 text-[10px]">{model.params}</Badge>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {modelArch.startsWith("rfdetr") && (
                <p className="text-[11px] text-muted-foreground leading-snug">
                  RF-DETR 支持当前 YOLO 数据集及 Roboflow COCO 数据集。
                </p>
              )}
            </div>}

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Epochs（训练轮数）</label>
                <span className="text-sm text-muted-foreground">{epochs}</span>
              </div>
              <Slider value={[epochs]} onValueChange={([v]) => setEpochs(v)} min={10} max={500} step={10} disabled={isTraining} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Batch Size（批大小）</label>
                <span className="text-sm text-muted-foreground">{batchSize}</span>
              </div>
              <Slider value={[batchSize]} onValueChange={([v]) => setBatchSize(v)} min={1} max={64} step={1} disabled={isTraining} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Image Size（图像尺寸）</label>
                <span className="text-sm text-muted-foreground">{imgSize}px</span>
              </div>
              <Slider value={[imgSize]} onValueChange={([v]) => setImgSize(v)} min={320} max={1280} step={32} disabled={isTraining} />
            </div>

            {weightsMode === "builtin" && <div className="flex items-center justify-between">
              <label className="text-sm font-medium">使用预训练权重</label>
              <Switch checked={usePretrained} onCheckedChange={setUsePretrained} disabled={isTraining} />
            </div>}

            {/* Advanced Hyperparameters */}
            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full flex items-center justify-between px-0 h-8">
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <Settings2 className="w-3.5 h-3.5" /> 高级超参数
                  </span>
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3 pt-2">
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">lr0（初始学习率）</label>
                    <Input type="number" value={lr0} onChange={e => setLr0(+e.target.value)} step={0.001} min={0.0001} max={1} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">lrf（最终学习率）</label>
                    <Input type="number" value={lrf} onChange={e => setLrf(+e.target.value)} step={0.001} min={0.0001} max={1} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Optimizer（优化器）</label>
                  <Select value={optimizer} onValueChange={setOptimizer} disabled={isTraining}>
                    <SelectTrigger className="h-7 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['SGD','Adam','AdamW','NAdam','RAdam','RMSProp','auto'].map(o => <SelectItem key={o} value={o} className="text-xs">{o}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Patience（早停轮数）</label>
                    <Input type="number" value={patience} onChange={e => setPatience(+e.target.value)} min={0} max={300} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Warmup Epochs（预热轮数）</label>
                    <Input type="number" value={warmupEpochs} onChange={e => setWarmupEpochs(+e.target.value)} min={0} max={10} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Weight Decay（权重衰减）</label>
                    <Input type="number" value={weightDecay} onChange={e => setWeightDecay(+e.target.value)} step={0.0001} min={0} max={0.1} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                  <div className={`space-y-1 ${selectedModelType === "rfdetr" ? "opacity-50 grayscale cursor-not-allowed" : ""}`}>
                    <label className="text-xs text-muted-foreground">Dropout（随机失活）</label>
                    <Input type="number" value={dropout} onChange={e => setDropout(+e.target.value)} step={0.05} min={0} max={0.9} disabled={isTraining || selectedModelType === "rfdetr"} className="h-7 text-xs" />
                  </div>
                </div>
                <div className={`space-y-1 ${selectedModelType === "rfdetr" ? "opacity-50 grayscale cursor-not-allowed" : ""}`}>
                  <div className="flex justify-between"><label className="text-xs text-muted-foreground">Mosaic（马赛克增强）</label><span className="text-xs">{mosaic.toFixed(1)}</span></div>
                  <Slider value={[mosaic]} onValueChange={([v]) => setMosaic(v)} min={0} max={1} step={0.1} disabled={isTraining || selectedModelType === "rfdetr"} />
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground" title={selectedModelType === "rfdetr" ? "RF-DETR：色相幅度，使用 ColorJitter，与 YOLO 实现不同" : undefined}>HSV-H</label><span>{hsvH.toFixed(3)}</span></div>
                    <Slider value={[hsvH]} onValueChange={([v]) => setHsvH(v)} min={0} max={0.1} step={0.001} disabled={isTraining} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground" title={selectedModelType === "rfdetr" ? "RF-DETR：饱和度幅度，使用 ColorJitter，与 YOLO 实现不同" : undefined}>HSV-S</label><span>{hsvS.toFixed(2)}</span></div>
                    <Slider value={[hsvS]} onValueChange={([v]) => setHsvS(v)} min={0} max={1} step={0.05} disabled={isTraining} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground" title={selectedModelType === "rfdetr" ? "RF-DETR：亮度幅度，使用 ColorJitter，与 YOLO 实现不同" : undefined}>HSV-V</label><span>{hsvV.toFixed(2)}</span></div>
                    <Slider value={[hsvV]} onValueChange={([v]) => setHsvV(v)} min={0} max={1} step={0.05} disabled={isTraining} />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground" title={selectedModelType === "rfdetr" ? "RF-DETR：水平翻转概率" : undefined}>FlipLR</label><span>{fliplr.toFixed(1)}</span></div>
                    <Slider value={[fliplr]} onValueChange={([v]) => setFliplr(v)} min={0} max={1} step={0.1} disabled={isTraining} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground" title={selectedModelType === "rfdetr" ? "RF-DETR：垂直翻转概率" : undefined}>FlipUD</label><span>{flipud.toFixed(1)}</span></div>
                    <Slider value={[flipud]} onValueChange={([v]) => setFlipud(v)} min={0} max={1} step={0.1} disabled={isTraining} />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">Cosine LR 学习率调度</label>
                  <Switch checked={cosLr} onCheckedChange={setCosLr} disabled={isTraining} />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">混合精度 (AMP)</label>
                  <Switch checked={amp} onCheckedChange={setAmp} disabled={isTraining} />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <label className="text-xs text-muted-foreground">每 N 个 Epoch 保存 Checkpoint</label>
                    <span className="text-xs">{savePeriod === 0 ? "关闭" : `每 ${savePeriod} 轮`}</span>
                  </div>
                  <Slider value={[savePeriod]} onValueChange={([v]) => setSavePeriod(v)} min={0} max={50} step={1} disabled={isTraining} />
                  <p className="text-[10px] text-muted-foreground/60">0 = 仅保存 best 与 last。设为大于 0 后将生成 epoch5.pt、epoch10.pt 等文件。</p>
                </div>
              </CollapsibleContent>
            </Collapsible>

            {gpuWarning && computeNodeId === "local" && (
              <div className="rounded-md bg-yellow-500/10 border border-yellow-500/30 p-3 text-xs text-yellow-700 dark:text-yellow-300 space-y-2">
                <p className="font-semibold">检测到 GPU，但未安装 CUDA 版 PyTorch</p>
                <p className="text-yellow-700/90 dark:text-yellow-400/80">训练将运行在 CPU 上（速度较慢）。安装 CUDA 版 PyTorch 以使用 GPU。</p>
                <Button
                  size="sm"
                  variant="outline"
                  className="w-full h-7 text-xs border-yellow-500/40 text-yellow-700 dark:text-yellow-300 hover:bg-yellow-500/10"
                  onClick={handleInstallCudaTorch}
                  disabled={installingCuda}
                >
                  {installingCuda ? "安装中…（可能需要几分钟）" : "自动安装 CUDA 版 PyTorch"}
                </Button>
              </div>
            )}

            {error && (
              <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-600 dark:text-red-400 break-all">
                {error}
              </div>
            )}

            {modelPath && (
              <div className={`rounded-md p-3 text-xs break-all space-y-1 ${
                statusLabel === "Failed"
                  ? "bg-yellow-500/10 border border-yellow-500/30 text-yellow-700 dark:text-yellow-400"
                  : "bg-green-500/10 border border-green-500/30 text-green-700 dark:text-green-400"
              }`}>
                <p className="font-semibold">
                  {statusLabel === "Failed" ? "可用的部分权重" : "模型已保存"}
                </p>
                <p className="font-mono opacity-80">{modelPath}</p>
                <p>点击上方「下载 .pt」按钮保存模型文件。</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right column */}
        <div className="lg:col-span-2 space-y-6">

          {/* Progress */}
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                训练进度
                <div className="ml-auto flex items-center gap-2">
                  {deviceInfo && (
                    <Badge variant="outline" className="text-xs font-mono gap-1">
                      <Cpu className="w-3 h-3" />{deviceInfo.length > 30 ? deviceInfo.slice(0, 30) + "…" : deviceInfo}
                    </Badge>
                  )}
                  {liveGpuMem != null && (
                    <Badge variant="outline" className="text-xs font-mono gap-1">
                      <HardDrive className="w-3 h-3" />{liveGpuMem.toFixed(2)} GB VRAM
                    </Badge>
                  )}
                  {statusLabel && (
                    <Badge variant="secondary" className="text-xs">{statusDisplay}</Badge>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    Epoch {currentEpoch} / {totalEpochs}
                    {batchStats && isTraining && (
                      <span className="ml-2 text-xs text-muted-foreground/60">
                        · Batch {batchStats.batch} / {batchStats.total_batches}
                      </span>
                    )}
                  </span>
                  <span className="text-sm text-muted-foreground flex items-center gap-3">
                    {batchStats && isTraining && batchStats.its > 0 ? (
                      <span className="font-mono text-xs">⚡ {batchStats.its.toFixed(1)} it/s</span>
                    ) : latestMetrics?.its ? (
                      <span className="font-mono text-xs">⚡ {latestMetrics.its.toFixed(1)} it/s</span>
                    ) : null}
                    {etaDisplay && `ETA: ${etaDisplay}`}
                    {isPaused && "训练已暂停 — Checkpoint 已保存"}
                  </span>
                </div>
                <Progress value={progress} className="h-2" />

                {/* Live batch losses during training */}
                {batchStats && isTraining && Object.keys(batchStats.losses).length > 0 && (
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono text-muted-foreground/80">
                    <span className="text-muted-foreground/40">实时：</span>
                    {Object.entries(batchStats.losses).map(([key, val]) => (
                      <span key={key}>
                        {key.replace(/_loss$/, "")}
                        <span className="ml-1 text-foreground/60">{val.toFixed(4)}</span>
                      </span>
                    ))}
                  </div>
                )}

                {/* Train losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">训练损失</p>
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard label="Box Loss"  value={latestMetrics?.train_box_loss} prev={prevMetrics?.train_box_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="Cls Loss"  value={latestMetrics?.train_cls_loss} prev={prevMetrics?.train_cls_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="DFL Loss"  value={latestMetrics?.train_dfl_loss} prev={prevMetrics?.train_dfl_loss} lowerIsBetter decimals={4} />
                  </div>
                </div>

                {/* Val losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">验证损失</p>
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard label="Val Box"  value={latestMetrics?.val_box_loss} prev={prevMetrics?.val_box_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="Val Cls"  value={latestMetrics?.val_cls_loss} prev={prevMetrics?.val_cls_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="Val DFL"  value={latestMetrics?.val_dfl_loss} prev={prevMetrics?.val_dfl_loss} lowerIsBetter decimals={4} />
                  </div>
                </div>

                {/* mAP + P/R */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">验证指标</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <MetricCard label="mAP@50"    value={latestMetrics?.mAP50}     prev={prevMetrics?.mAP50}     lowerIsBetter={false} decimals={4} />
                    <MetricCard label="mAP@50-95" value={latestMetrics?.mAP50_95}  prev={prevMetrics?.mAP50_95}  lowerIsBetter={false} decimals={4} />
                    <MetricCard label="Precision" value={latestMetrics?.precision} prev={prevMetrics?.precision} lowerIsBetter={false} decimals={4} />
                    <MetricCard label="Recall"    value={latestMetrics?.recall}    prev={prevMetrics?.recall}    lowerIsBetter={false} decimals={4} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Training Curves */}
          {metrics.length > 1 && (
            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <LineChartIcon className="w-5 h-5 text-primary" />
                  训练曲线
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                {/* Train losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">训练损失</p>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" />
                      <Tooltip contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11, color: 'var(--popover-foreground)' }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="train_box_loss" name="Box" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="train_cls_loss" name="Cls" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="train_dfl_loss" name="DFL" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Val losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">验证损失</p>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" />
                      <Tooltip contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11, color: 'var(--popover-foreground)' }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="val_box_loss" name="Val Box" stroke="#60a5fa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                      <Line type="monotone" dataKey="val_cls_loss" name="Val Cls" stroke="#fbbf24" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                      <Line type="monotone" dataKey="val_dfl_loss" name="Val DFL" stroke="#a78bfa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* mAP chart */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">mAP 随 Epoch 变化</p>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11, color: 'var(--popover-foreground)' }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="mAP50"    name="mAP@50"    stroke="#10b981" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="mAP50_95" name="mAP@50-95" stroke="#06b6d4" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Precision / Recall chart */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Precision &amp; Recall</p>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }} stroke="var(--muted-foreground)" domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11, color: 'var(--popover-foreground)' }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="precision" name="Precision" stroke="#f97316" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="recall"    name="Recall"    stroke="#ec4899" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* System stats */}
          {computeNodeId === "local" ? <>
          <div className="grid grid-cols-3 gap-4">
            <SystemStat
              icon={<Cpu className="w-5 h-5 text-primary" />}
              label="CPU"
              value={`${cpuUsage.toFixed(0)}%`}
              progress={cpuUsage}
            />
            <SystemStat
              icon={<HardDrive className="w-5 h-5 text-primary" />}
              label={liveGpuMem != null ? "VRAM 已用" : gpuUsage !== null ? "GPU 显存" : "RAM"}
              value={liveGpuMem != null
                ? `${liveGpuMem.toFixed(2)} GB`
                : `${memoryUsage.toFixed(0)}%`}
              progress={liveGpuMem == null ? memoryUsage : undefined}
            />
            <SystemStat
              icon={<Activity className="w-5 h-5 text-primary" />}
              label={gpuUsage !== null ? `GPU${gpuName ? ` (${gpuName.split(" ").pop()})` : ""}` : "GPU"}
              value={gpuUsage !== null ? `${gpuUsage.toFixed(0)}%` : "—"}
              progress={gpuUsage ?? undefined}
            />
          </div>

          </> : <Card><CardContent className="pt-6 space-y-2 text-sm"><p>远程设备：{deviceInfo || computeNodes.find(n=>n.id===computeNodeId)?.name || "未检测"}</p><p>连接状态：{remoteConnection === "reconnecting" ? "正在重连" : remoteConnection === "connected" ? "最近通信正常" : computeNodes.find(n=>n.id===computeNodeId)?.status === "online" ? "最近检测在线" : "待检测"}{remotePid ? ` · PID ${remotePid}` : ""}</p><p>训练显存：{liveGpuMem == null ? "等待训练进程上报" : `${liveGpuMem.toFixed(2)} GB`}</p><a href="/compute-nodes" className="text-primary">查看算力节点状态</a></CardContent></Card>}

          {/* Logs */}
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Terminal className="w-5 h-5 text-primary" />
                日志
                <span className="ml-auto text-xs font-normal text-muted-foreground">{logs.length} 行</span>
              </CardTitle>
              <p className="text-xs text-muted-foreground">每轮验证后会输出 Epoch 摘要；单个 Epoch 可能需要数分钟。</p>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[280px] w-full rounded border border-border/50 bg-muted/40 p-3">
                {logs.length === 0 ? (
                  <p className="text-sm text-muted-foreground font-mono">训练日志将在此处显示…</p>
                ) : (
                  <div className="space-y-0.5 font-mono text-[11px] leading-5">
                    {logs.map((log, i) => (
                      <div key={i} className={
                        log.type === "error"   ? "text-red-600 dark:text-red-400"       :
                        log.type === "warning" ? "text-amber-700 dark:text-yellow-300" :
                        log.type === "success" ? "text-emerald-700 dark:text-green-400": "text-foreground/75"
                      }>
                        <span className="text-muted-foreground/60 select-none">[{log.timestamp}]</span>{" "}{log.message}
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Past Runs */}
      {nodeRuns.length === 0 && <p className="text-sm text-muted-foreground">当前算力节点暂无训练记录。</p>}
      {nodeRuns.length > 0 && (
        <Collapsible defaultOpen>
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-muted/20 rounded-t-lg pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Package className="w-4 h-4 text-primary" />
                  历史训练
                  <Badge variant="outline" className="ml-1 text-xs">{nodeRuns.length}</Badge>
                  <ChevronDown className="w-4 h-4 ml-auto text-muted-foreground" />
                </CardTitle>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="pt-0 space-y-2">
                {nodeRuns.map(run => (
                  <div key={run.id} className="rounded-lg border border-border/40 bg-background/40 overflow-hidden">
                    <div className="flex items-center gap-3 p-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium truncate">{run.name}</span>
                          <Badge
                            variant="outline"
                            className={`text-[10px] shrink-0 ${
                              run.status === "completed"   ? "text-emerald-700 border-emerald-600/30 dark:text-green-400 dark:border-green-400/30" :
                              run.status === "failed"      ? "text-red-600 border-red-500/30 dark:text-red-400 dark:border-red-400/30" :
                              run.status === "paused"      ? "text-amber-700 border-amber-500/30 dark:text-yellow-400 dark:border-yellow-400/30" :
                              run.status === "interrupted" ? "text-orange-600 border-orange-500/30 dark:text-orange-400 dark:border-orange-400/30" :
                              run.status === "running" || run.status === "pausing" ? "text-blue-700 border-blue-500/30 dark:text-blue-400 dark:border-blue-400/30" :
                              "text-muted-foreground"
                            }`}
                          >
                            {RUN_STATUS_ZH[run.status] ?? run.status}
                          </Badge>
                          <span className="text-[10px] text-muted-foreground shrink-0">{run.model_type}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="flex-1 h-1 rounded bg-muted overflow-hidden">
                            <div className="h-full bg-primary/60 rounded" style={{ width: `${run.progress}%` }} />
                          </div>
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {run.current_epoch}/{run.total_epochs} ep
                          </span>
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {new Date(run.started_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={()=>void openTrainingRun(run)}>查看任务</Button>
                        {/* Download best model */}
                        {run.model_path && (
                          <Button size="sm" variant="ghost" className="h-7 gap-1 text-xs" title="下载模型权重"
                            onClick={() => window.open(`${apiUrl}/api/train/${run.id}/export`, "_blank")}>
                            <Download className="w-3.5 h-3.5" /> 下载 {run.model_path?.split(/[\\/]/).pop()}
                          </Button>
                        )}
                        {/* Show checkpoints */}
                        {(run.status === "completed" || run.status === "paused" || run.status === "stopped") && (
                          <Button size="sm" variant="ghost"
                            className={`h-7 w-7 p-0 ${checkpointRunId === run.id ? "text-primary" : ""}`}
                            title="查看 Checkpoint"
                            onClick={() => handleLoadCheckpoints(run.id)}>
                            <FolderOpen className="w-3.5 h-3.5" />
                          </Button>
                        )}
                        {/* Load & Resume */}
                        {(run.status === "paused" || run.status === "interrupted") && run.id !== trainingId && (
                          <Button size="sm" variant="outline" className="gap-1 h-7 text-xs"
                            onClick={() => {
                              setTrainingId(run.id)
                              setIsPaused(true)
                              setIsPausing(false)
                              setIsTraining(false)
                              setProgress(run.progress)
                              setCurrentEpoch(run.current_epoch)
                              setTotalEpochs(run.total_epochs)
                              setStatusLabel(run.status === "interrupted" ? "Interrupted" : "Paused")
                              setMetrics([])
                              setLogs([])
                              previousLogWindow.current = []
                              if (run.model_path) setModelPath(run.model_path)
                            }}>
                            <Play className="w-3 h-3" />
                            继续
                          </Button>
                        )}
                        {/* Delete (only for finished runs) */}
                        {run.status !== "running" && run.status !== "starting" && run.status !== "pausing" && (
                          <Button size="sm" variant="ghost" className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                            title="删除该次训练" onClick={() => handleDeleteRun(run.id)}>
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        )}
                      </div>
                    </div>
                    {/* Checkpoint list inline */}
                    {checkpointRunId === run.id && checkpoints.length > 0 && (
                      <div className="border-t border-border/30 px-3 py-2 bg-muted/20">
                        <p className="text-[10px] text-muted-foreground mb-1.5 font-medium">Checkpoints（检查点）</p>
                        <div className="flex flex-wrap gap-1.5">
                          {checkpoints.map(cp => (
                            <button key={cp.name}
                              className="text-[10px] px-2 py-0.5 rounded bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20 font-mono"
                              onClick={() => window.open(`${apiUrl}/api/train/${run.id}/checkpoint/${cp.name}`, "_blank")}
                              title={`${cp.size_mb} MB`}>
                              {cp.name}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {checkpointRunId === run.id && checkpoints.length === 0 && (
                      <div className="border-t border-border/30 px-3 py-2 text-[10px] text-muted-foreground">
                        未找到 Epoch Checkpoint。训练前请将「每 N 个 Epoch 保存 Checkpoint」设为大于 0。
                      </div>
                    )}
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricCard({
  label, value, prev, lowerIsBetter, decimals = 4,
}: {
  label: string
  value?: number
  prev?: number
  lowerIsBetter: boolean
  decimals?: number
}) {
  const improved =
    value != null && prev != null
      ? lowerIsBetter ? value < prev : value > prev
      : null

  return (
    <div className="text-center p-3 rounded-lg bg-muted/30">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className="text-lg font-semibold flex items-center justify-center gap-1">
        {value != null ? value.toFixed(decimals) : "—"}
        {improved === true  && <TrendingDown className="w-3 h-3 text-green-500" />}
        {improved === false && <TrendingUp   className="w-3 h-3 text-red-400"   />}
      </p>
    </div>
  )
}

function SystemStat({
  icon, label, value, progress,
}: {
  icon: React.ReactNode
  label: string
  value: string
  progress?: number
}) {
  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur">
      <CardContent className="pt-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
            {icon}
          </div>
          <div>
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-lg font-semibold">{value}</p>
          </div>
        </div>
        {progress !== undefined && (
          <Progress value={progress} className="h-1 mt-3" />
        )}
      </CardContent>
    </Card>
  )
}
