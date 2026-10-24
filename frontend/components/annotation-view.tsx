'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
// VLX-Seek 的常用描述——与多模态页共用同一份，两边句式一致
import { PROMPT_PRESETS, readVlxConnectionSummary, VLX_PROVIDER_ID } from '@/lib/lmm-shared'
import { Button } from '@/components/ui/button'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
  SelectGroup, SelectLabel, SelectSeparator,
} from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  MousePointer2, Square, Pentagon, Wand2, Save, Trash2,
  ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw,
  AlertCircle, Loader2, Undo2, FolderOpen, Tag,
  Brush, Crosshair, ChevronsRight, Keyboard,
  Download, CheckCircle2, X, Plus, Pause, Play, StopCircle, Bot,
  ChevronDown,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { supportsConfidenceThreshold } from '@/lib/model-capabilities'
import { AutoAnnotateToggle } from '@/components/auto-annotate-toggle'
import { ReviewWorkspace } from '@/components/review-workspace'
import type { Dataset, ImageData, Annotation, ImageCache } from '@/app/page'
import { useSettings } from '@/lib/settings-context'
import { toast } from 'sonner'
import { assistantCall } from '@/lib/assistant-workflow'

const PAGE_SIZE = 50
const PRELOAD_AHEAD = 3

interface BatchJobState {
  job_id: string
  status: 'running' | 'paused' | 'done' | 'error' | 'cancelled' | 'interrupted' | 'undone'
  paused: boolean
  progress: number
  total: number
  processed: number
  annotated: number
  failed: number
  total_annotations: number
  error?: string
  text_prompt: string
  // P2A 追加：新目标结构。★只加不改★——历史 job 没有这几个字段也能照常渲染，
  // 不做任何迁移 / 改名 / 删除，text_prompt 继续保留用于历史兼容。
  target?: { label?: string; description?: string; extra_prompt?: string; confidence?: number }
  target_mode?: string
  target_executed?: boolean
  model_id?: string
  // ★双模型追加★：只加不改，历史 job 没有这两个字段也照常渲染
  verify_model_id?: string
  dual?: { accept: number; review: number }
  tile_pass?: boolean
  tile_progress?: { phase?: string; completed?: number; total?: number; image?: string; coverage_complete?: boolean }
  review_images?: Array<{ image: string; image_id: string; count: number; key: string }>
  review_summary?: { pending: number; accepted: number; rejected: number; total: number }
  review_readonly?: boolean
  teachers_down?: string[]
  confidence_threshold?: number
  started_at: string | number
  dataset_id: string
  recent_images?: Array<{ filename: string; path: string; image_id: string; annotation_count: number }>
}

interface AnnotationViewProps {
  selectedDataset: Dataset | null
  apiUrl: string
  imageCache: ImageCache
  updateImageCache: (datasetId: string, images: ImageData[]) => void
  initialImageId?: string | null
  onInitialImageConsumed?: () => void
  onOpenVlmConfig?: () => void
  assistantDraftId?:string
  onAssistantDraftChange?:(id:string)=>void
  assistantAction?:{action:'jobs'|'review'|'annotate';version:number}|null
  registerBeforeLeave?:(handler:()=>Promise<void>)=>void
}

type Tool = 'select' | 'bbox' | 'polygon' | 'keypoint' | 'brush' | 'sam'

export function AnnotationView({ selectedDataset, apiUrl, imageCache, updateImageCache, initialImageId, onInitialImageConsumed, onOpenVlmConfig,assistantDraftId,onAssistantDraftChange,assistantAction,registerBeforeLeave }: AnnotationViewProps) {
  const { settings } = useSettings()
  // Image list state
  const [allImages, setAllImages] = useState<ImageData[]>([])
  const [filteredImages, setFilteredImages] = useState<ImageData[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isLoadingList, setIsLoadingList] = useState(false)
  const [availableSplits, setAvailableSplits] = useState<string[]>([])
  const [availableClasses, setAvailableClasses] = useState<string[]>([])
  const [selectedSplit, setSelectedSplit] = useState('all')
  const [selectedClass, setSelectedClass] = useState('all')

  // Current image state
  const [currentImage, setCurrentImage] = useState<ImageData | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [isImageLoading, setIsImageLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Tool state
  const [activeTool, setActiveTool] = useState<Tool>('select')
  const [activeClass, setActiveClass] = useState<string>('')
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [spacePressed, setSpacePressed] = useState(false)
  const [isPanning, setIsPanning] = useState(false)
  const viewportRef = useRef({ zoom: 1, pan: { x: 0, y: 0 } })
  const spaceRef = useRef(false)
  const panGestureRef = useRef<{ id: number; x: number; y: number } | null>(null)
  const editingRef = useRef(false)
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [tempBox, setTempBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [polygonPoints, setPolygonPoints] = useState<{ x: number; y: number }[]>([])
  const [history, setHistory] = useState<Annotation[][]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  // Brush state
  const [brushSize, setBrushSize] = useState(20)
  const [brushPoints, setBrushPoints] = useState<{ x: number; y: number }[]>([])
  const [isBrushing, setIsBrushing] = useState(false)

  // Drag state for moving selected annotation in select mode
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState<{ x: number; y: number; origBbox: number[] } | null>(null)

  // Add new class state
  const [addingClass, setAddingClass] = useState(false)
  const [newClassName, setNewClassName] = useState('')
  // Local copy of dataset classes so we can add to it without mutating the prop
  const [localClasses, setLocalClasses] = useState<string[]>(selectedDataset?.classes || [])

  // SAM click queue state
  type SamClick = { id: string; nx: number; ny: number; status: 'pending' | 'processing' }
  const [samPendingClicks, setSamPendingClicks] = useState<SamClick[]>([])
  const samPendingClicksRef = useRef<SamClick[]>([])
  const samProcessingRef = useRef(false)
  const selectedModelRef = useRef('')
  const confidenceRef = useRef(0.5)

  // Keypoint state
  const [keypointList, setKeypointList] = useState<{ x: number; y: number; label: string }[]>([])

  // Batch ops state
  const [copyToNext, setCopyToNext] = useState(1)
  const [showBatchPanel, setShowBatchPanel] = useState(false)
  const [showShortcutHint, setShowShortcutHint] = useState(false)

  // Auto-annotate state
  const [models, setModels] = useState<{ id: string; name: string; type: string; downloaded: boolean; loaded: boolean; supports_confidence_threshold?: boolean }[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [confidence, set置信度] = useState(0.5)
  const [isAutoAnnotating, setIsAutoAnnotating] = useState(false)
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null)
  // SAM3 text prompt — tag-chip model
  const [promptTags, setPromptTags] = useState<string[]>([])
  const [promptInput, setPromptInput] = useState('')
  // ── P2A：类别 / 目标描述 / 高级约束 三者分离 ──────────────────────────────
  // 只有 targetLabel 允许成为 data.yaml 的类别；description / extra_prompt
  // 只是给模型看的语义提示，绝不写进 data.yaml。
  const [targetLabel, setTargetLabel] = useState('')
  const [targetDescription, setTargetDescription] = useState('')
  const [targetExtra, setTargetExtra] = useState('')
  const [showExtraPrompt, setShowExtraPrompt] = useState(false)
  const [addingTargetClass, setAddingTargetClass] = useState(false)
  const [newTargetClass, setNewTargetClass] = useState('')
  // Batch job tracking: job_id → BatchJobState
  const [batchJobs, setBatchJobs] = useState<Record<string, BatchJobState>>({})
  const [isStartingBatch, setIsStartingBatch] = useState(false)
  const [assistantDraft,setAssistantDraft]=useState<any>(null)
  const batchSubmitGuard=useRef(false),assistantLoaded=useRef('')
  // Right sidebar tab: 'annotate' = auto-annotate settings, 'jobs' = batch job monitor,
  // 'autolabel' = Auto Label V1（VLX + Qwen 双 Teacher 并行检测 / 双向校验）
  const [sidebarTab, setSidebarTab] = useState<'annotate' | 'jobs' | 'review'>('annotate')
  // ── 双模型：主模型 = selectedModel（标注），第二个模型 = verifyModel（校验）──
  // 关掉就是原来的单模型流程，VLX 之类需要连服务器的模型保留可选但默认不选。
  const [tileEnabled, setTileEnabled] = useState(false)
  const [tileSize, setTileSize] = useState(640)
  const [tileOverlap, setTileOverlap] = useState(128)
  const [tilePreview, setTilePreview] = useState<{ job_id: string; status: string; phase?: string; progress?: { completed?: number; total?: number } } | null>(null)
  const [tileSummary, setTileSummary] = useState<any>(null)
  const [tileReview, setTileReview] = useState<{ imageId: string; datasetId: string; jobId: string; items: any[] } | null>(null)
  const [reviewPending, setReviewPending] = useState(0)
  const [reviewJobId, setReviewJobId] = useState<string | null>(null)
  const [reviewImageKey, setReviewImageKey] = useState<string | null>(null)
  const previewJobRef = useRef<string | null>(null)
  const [dualEnabled, setDualEnabled] = useState(false)
  const [verifyModel, setVerifyModel] = useState('')
  const [dualResult, setDualResult] = useState<{ accept: number; review: number; teachers_down: string[] } | null>(null)
  useEffect(()=>{
    if(!assistantDraftId){setAssistantDraft(null);assistantLoaded.current='';return}
    if(!assistantDraftId||!selectedDataset||assistantLoaded.current===assistantDraftId)return
    let live=true
    void assistantCall(apiUrl,`drafts/${assistantDraftId}`).then(d=>{
      if(!live||d.kind!=='annotation'||d.data.dataset_id!==selectedDataset.id)return
      assistantLoaded.current=assistantDraftId;setAssistantDraft(d)
      const c=d.data;setSelectedModel(c.model_id);set置信度(c.confidence_threshold??.5);setTargetLabel(c.label);setActiveClass(c.label);setTargetDescription(c.description||c.label);setTargetExtra(c.extra_prompt||'');setPromptTags((c.text_prompt||c.label).split(',').map((v:string)=>v.trim()).filter(Boolean));setDualEnabled(!!c.verify_model_id);setVerifyModel(c.verify_model_id||'');setTileEnabled(!!c.tile_pass);setTileSize(c.tile_size||640);setTileOverlap(c.tile_overlap??128);setSidebarTab('annotate')
      if(d.result?.job_id){const id=d.result.job_id;void fetch(`${apiUrl}/api/auto-annotate/text-batch/${id}/status`).then(r=>r.ok?r.json():null).then(job=>{if(!job||!live)return;setBatchJobs(prev=>({...prev,[id]:job}));if(['running','paused'].includes(job.status))_pollJob(id)})}
    }).catch(err=>{if(live)toast.error(err.message)})
    return()=>{live=false}
  },[assistantDraftId,selectedDataset?.id,apiUrl]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(()=>{if(assistantAction)setSidebarTab(assistantAction.action)},[assistantAction?.version])

  // Resume-progress banner state
  const [resumeBanner, setResumeBanner] = useState<{
    imageId: string
    imageName: string
    imageIndex: number
    split: string
    classFilter: string
  } | null>(null)
  const [resumeRequestVersion, setResumeRequestVersion] = useState(0)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const imageLoadVersionRef = useRef(0)
  const imageListLoadVersionRef = useRef(0)
  const scaleRef = useRef(1)
  editingRef.current = isDrawing || isDragging || isBrushing

  const stopPan = useCallback(() => {
    const gesture = panGestureRef.current
    panGestureRef.current = null
    if (gesture && containerRef.current?.hasPointerCapture(gesture.id)) {
      containerRef.current.releasePointerCapture(gesture.id)
    }
    setIsPanning(false)
  }, [])

  const resetViewport = useCallback(() => {
    if (editingRef.current) return
    const ratio = 1 / viewportRef.current.zoom
    setPolygonPoints(points => points.map(p => ({ x: p.x * ratio, y: p.y * ratio })))
    viewportRef.current = { zoom: 1, pan: { x: 0, y: 0 } }
    setZoom(1)
    setPan({ x: 0, y: 0 })
    stopPan()
  }, [stopPan])

  const zoomAt = useCallback((requested: number, clientX?: number, clientY?: number) => {
    if (!imageRef.current || editingRef.current || panGestureRef.current) return
    const container = containerRef.current
    if (!container) return
    const previous = viewportRef.current
    const nextZoom = Math.max(0.25, Math.min(8, requested))
    const ratio = nextZoom / previous.zoom
    const rect = container.getBoundingClientRect()
    const x = (clientX ?? rect.left + rect.width / 2) - rect.left - rect.width / 2
    const y = (clientY ?? rect.top + rect.height / 2) - rect.top - rect.height / 2
    const nextPan = { x: x - (x - previous.pan.x) * ratio, y: y - (y - previous.pan.y) * ratio }
    viewportRef.current = { zoom: nextZoom, pan: nextPan }
    setPan(nextPan)
    setZoom(nextZoom)
    // Unfinished polygons use display pixels; completed annotations stay in image coordinates.
    setPolygonPoints(points => points.map(p => ({ x: p.x * ratio, y: p.y * ratio })))
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const wheel = (e: WheelEvent) => {
      if (!imageRef.current) return
      e.preventDefault()
      if (e.deltaY !== 0) zoomAt(viewportRef.current.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), e.clientX, e.clientY)
    }
    container.addEventListener('wheel', wheel, { passive: false })
    return () => container.removeEventListener('wheel', wheel)
  }, [currentImage, isImageLoading, isLoadingList, selectedDataset?.id, zoomAt])

  useEffect(() => {
    const editable = (target: EventTarget | null) => target instanceof HTMLElement &&
      (target.isContentEditable || !!target.closest('input, textarea, select, [role="textbox"], [role="combobox"], [role="listbox"], [role="menu"]'))
    const clear = () => { spaceRef.current = false; setSpacePressed(false); stopPan() }
    const down = (e: KeyboardEvent) => {
      if (sidebarTab === 'review') return
      if (editable(e.target)) return
      if ((e.ctrlKey || e.metaKey) && e.key === '0') { e.preventDefault(); resetViewport(); return }
      if (e.code !== 'Space' || e.ctrlKey || e.metaKey || e.altKey || !imageRef.current) return
      e.preventDefault()
      spaceRef.current = true
      setSpacePressed(true)
    }
    const up = (e: KeyboardEvent) => { if (e.code === 'Space' && spaceRef.current) { e.preventDefault(); clear() } }
    const visibility = () => { if (document.hidden) clear() }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', clear)
    document.addEventListener('visibilitychange', visibility)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', clear)
      document.removeEventListener('visibilitychange', visibility)
      spaceRef.current = false
      panGestureRef.current = null
    }
  }, [resetViewport, stopPan, sidebarTab])

  useEffect(() => {
    viewportRef.current = { zoom: 1, pan: { x: 0, y: 0 } }
    setZoom(1)
    setPan({ x: 0, y: 0 })
    spaceRef.current = false
    setSpacePressed(false)
    stopPan()
  }, [currentImage?.id, selectedDataset?.id, stopPan])

  const handlePanStart = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!spaceRef.current || e.button !== 0 || !imageRef.current || editingRef.current) return
    e.preventDefault()
    e.stopPropagation()
    panGestureRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY }
    e.currentTarget.setPointerCapture(e.pointerId)
    setIsPanning(true)
  }
  const handlePanMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const gesture = panGestureRef.current
    if (!gesture || gesture.id !== e.pointerId) return
    e.preventDefault()
    e.stopPropagation()
    const next = { x: viewportRef.current.pan.x + e.clientX - gesture.x, y: viewportRef.current.pan.y + e.clientY - gesture.y }
    viewportRef.current.pan = next
    gesture.x = e.clientX
    gesture.y = e.clientY
    setPan(next)
  }
  const handlePanEnd = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!panGestureRef.current) return
    e.preventDefault()
    e.stopPropagation()
    stopPan()
  }
  // Always-current refs so polling intervals can read latest values without closure stale issues
  const currentImageRef = useRef<ImageData | null>(null)
  const selectedDatasetRef = useRef<Dataset | null>(null)
  const annotationsRef = useRef<Annotation[]>([])
  // Track active poll intervals so we never start two for the same job
  const pollIntervalsRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})
  // When true, the allImages useEffect skips the reset-to-index-0 behaviour
  const silentRefreshRef = useRef(false)
  // When set, the allImages useEffect jumps to this image id instead of index 0
  const initialImageIdRef = useRef<string | null>(null)
  const pendingResumeRef = useRef<{ datasetId: string; imageId: string; imageIndex: number } | null>(null)
  const loadedImagesDatasetIdRef = useRef<string | null>(null)

  // ── Keep refs in sync with latest state ─────────────────────────────────────
  useEffect(() => { currentImageRef.current = currentImage }, [currentImage])
  useEffect(() => { selectedDatasetRef.current = selectedDataset }, [selectedDataset])
  useEffect(() => { annotationsRef.current = annotations }, [annotations])
  useEffect(() => { samPendingClicksRef.current = samPendingClicks }, [samPendingClicks])
  useEffect(() => { selectedModelRef.current = selectedModel }, [selectedModel])
  useEffect(() => { confidenceRef.current = confidence }, [confidence])
  // Sync localClasses when dataset changes
  useEffect(() => { setLocalClasses(selectedDataset?.classes || []) }, [selectedDataset?.id])
  // If the dataset YAML has no class names, populate localClasses from the
  // annotation data once images have loaded (availableClasses is derived from annotations)
  useEffect(() => {
    if (availableClasses.length > 0 && localClasses.length === 0) {
      setLocalClasses(availableClasses)
      setActiveClass(prev => prev || availableClasses[0] || '')
    }
  }, [availableClasses]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Clear all poll intervals on unmount ───────────────────────────────────────
  useEffect(() => {
    return () => { Object.values(pollIntervalsRef.current).forEach(clearInterval) }
  }, [])

  // ── SAM queue: redraw canvas when pending clicks change ───────────────────────
  useEffect(() => {
    if (imageRef.current) {
      // drawCanvasWithData reads samPendingClicksRef directly, so just trigger a redraw
      const img = imageRef.current
      drawCanvasWithData(img, annotations, selectedAnnotation, tempBox, polygonPoints, zoom, keypointList)
    }
  }, [samPendingClicks]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── SAM queue: process clicks one at a time ───────────────────────────────────
  useEffect(() => {
    if (samPendingClicks.length === 0 || samProcessingRef.current) return
    const processNext = async () => {
      const queue = samPendingClicksRef.current
      if (queue.length === 0 || samProcessingRef.current) return
      samProcessingRef.current = true
      const click = queue[0]
      // Mark as processing (yellow dot)
      const updated = samPendingClicksRef.current.map(c =>
        c.id === click.id ? { ...c, status: 'processing' as const } : c
      )
      samPendingClicksRef.current = updated
      setSamPendingClicks(updated)

      const modelId = selectedModelRef.current
      const conf = confidenceRef.current
      const ds = selectedDatasetRef.current
      const img = currentImageRef.current
      if (modelId && ds && img) {
        try {
          const params = new URLSearchParams({
            model_id: modelId,
            confidence_threshold: String(conf),
            point_x: String(click.nx),
            point_y: String(click.ny),
          })
          if (img.path) params.set('image_path_hint', img.path)
          const resp = await fetch(
            `${apiUrl}/api/auto-annotate/${ds.id}/single/${img.id}?${params}`,
            { method: 'POST' }
          )
          if (resp.ok) {
            const data = await resp.json()
            if (data.annotations?.length) {
              const next = [...annotationsRef.current, ...data.annotations]
              setAnnotations(next)
              saveToHistory(next)
              await saveAnnotations(false, next)
            } else {
              // Fallback: standard auto-annotate (no point)
              const fbParams = new URLSearchParams({ model_id: modelId, confidence_threshold: String(conf) })
              if (img.path) fbParams.set('image_path_hint', img.path)
              const fbResp = await fetch(
                `${apiUrl}/api/auto-annotate/${ds.id}/single/${img.id}?${fbParams}`,
                { method: 'POST' }
              )
              if (fbResp.ok) {
                const fbData = await fbResp.json()
                if (fbData.annotations?.length) {
                  const next = [...annotationsRef.current, ...fbData.annotations]
                  setAnnotations(next)
                  saveToHistory(next)
                  await saveAnnotations(false, next)
                }
              }
            }
          }
        } catch { /* silently skip failed individual clicks */ }
      }

      // Remove processed click
      const remaining = samPendingClicksRef.current.filter(c => c.id !== click.id)
      samPendingClicksRef.current = remaining
      setSamPendingClicks(remaining)
      samProcessingRef.current = false
    }
    processNext()
  }, [samPendingClicks]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Persist batch jobs to localStorage ───────────────────────────────────────
  useEffect(() => {
    try { localStorage.setItem('cvdm_batchJobs', JSON.stringify(batchJobs)) } catch {}
  }, [batchJobs])

  // ── Restore batch jobs from backend on first mount ───────────────────────────
  // Backend persists jobs to disk; on restart they survive with updated statuses.
  // We also fall back to localStorage for jobs the backend doesn't know about.
  useEffect(() => {
    const restore = async () => {
      try {
        // Try backend first (authoritative after restart)
        const r = await fetch(`${apiUrl}/api/auto-annotate/jobs`)
        if (r.ok) {
          const data = await r.json()
          const backendJobs: Record<string, BatchJobState> = {}
          for (const job of (data.jobs || [])) {
            backendJobs[job.job_id] = job
          }
          if (Object.keys(backendJobs).length > 0) {
            setBatchJobs(backendJobs)
            // Poll any still-running jobs
            Object.values(backendJobs).forEach(job => {
              if (job.status === 'running' || job.status === 'paused') {
                _pollJob(job.job_id)
              }
            })
            return
          }
        }
      } catch {}
      // Fallback: localStorage
      try {
        const saved = localStorage.getItem('cvdm_batchJobs')
        if (!saved) return
        const jobs: Record<string, BatchJobState> = JSON.parse(saved)
        if (Object.keys(jobs).length === 0) return
        setBatchJobs(jobs)
        Object.values(jobs).forEach(job => {
          if (job.status === 'running' || job.status === 'paused') {
            _pollJob(job.job_id)
          }
        })
      } catch {}
    }
    restore()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Save annotation progress to localStorage ─────────────────────────────────
  useEffect(() => {
    // Keep the saved cursor intact while the resume choice is visible or being applied.
    if (!selectedDataset || !currentImage || resumeBanner || pendingResumeRef.current) return
    try {
      localStorage.setItem(`cvdm_annotate_${selectedDataset.id}`, JSON.stringify({
        imageId: currentImage.id,
        imageName: currentImage.filename,
        imageIndex: currentIndex,
        split: selectedSplit,
        classFilter: selectedClass,
      }))
    } catch {}
  }, [currentIndex, currentImage?.id, selectedSplit, selectedClass, resumeBanner, selectedDataset?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load image list ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedDataset) return
    // Check for saved progress (only when not externally directed to a specific image)
    setResumeBanner(null)
    if (!initialImageId) {
      try {
        const raw = localStorage.getItem(`cvdm_annotate_${selectedDataset.id}`)
        if (raw) {
          const saved = JSON.parse(raw)
          if (saved.imageId && saved.imageIndex > 0) {
            setResumeBanner(saved)
          }
        }
      } catch {}
    }
    // Label deletions and class edits must be read from the server on each entry.
    // Retain external image targets / resume behavior, but do not restore cached boxes.
    loadImageList()
    loadModels()
    if (selectedDataset.classes?.length) setActiveClass(selectedDataset.classes[0])
    setLocalClasses(selectedDataset.classes || [])
  }, [selectedDataset])

  useEffect(() => {
    let imgs = allImages
    if (selectedSplit !== 'all') imgs = imgs.filter(i => i.split === selectedSplit)
    if (selectedClass !== 'all') {
      imgs = imgs.filter(i =>
        i.class_name === selectedClass ||
        (i.annotations || []).some(a => a.class_name === selectedClass)
      )
    }
    setFilteredImages(imgs)
    const resumeRequest = pendingResumeRef.current
    if (resumeRequest) {
      if (loadedImagesDatasetIdRef.current !== resumeRequest.datasetId) return
      pendingResumeRef.current = null
      if (imgs.length > 0) {
        const exactIndex = imgs.findIndex(img => img.id === resumeRequest.imageId)
        const fallbackIndex = Math.min(Math.max(resumeRequest.imageIndex, 0), imgs.length - 1)
        const idx = exactIndex >= 0 ? exactIndex : fallbackIndex
        setCurrentIndex(idx)
        loadImageFile(imgs[idx])
      }
      return
    }
    // Silent refresh (from batch polling): don't reset position or reload the canvas
    if (silentRefreshRef.current) {
      silentRefreshRef.current = false
      return
    }
    if (imgs.length > 0) {
      const targetId = initialImageIdRef.current
      if (targetId) {
        initialImageIdRef.current = null
        onInitialImageConsumed?.()
        const idx = Math.max(0, imgs.findIndex(img => img.id === targetId))
        setCurrentIndex(idx)
        loadImageFile(imgs[idx])
      } else {
        setCurrentIndex(0)
        loadImageFile(imgs[0])
      }
    }
  }, [selectedSplit, selectedClass, allImages, resumeRequestVersion])

  const initFromImages = (images: ImageData[], fromStart = false) => {
    // Set ref BEFORE setAllImages so the allImages useEffect can consume it
    loadedImagesDatasetIdRef.current = selectedDataset?.id ?? null
    if (!fromStart && initialImageId) initialImageIdRef.current = initialImageId
    setAllImages(images)
    const splits = [...new Set(images.map(i => i.split).filter(Boolean) as string[])]
    const classes = [...new Set(
      images.flatMap(i => [
        i.class_name,
        ...(i.annotations || []).map(a => a.class_name)
      ]).filter(Boolean) as string[]
    )]
    setAvailableSplits(splits)
    setAvailableClasses(classes)
    setFilteredImages(images)
    // currentIndex + image loading are handled by the allImages useEffect below
  }

  const loadImageList = async (fromStart = false) => {
    if (!selectedDataset) return
    const datasetId = selectedDataset.id
    const requestVersion = ++imageListLoadVersionRef.current
    setIsLoadingList(true)
    setError(null)
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${datasetId}/images?limit=999999&bust_cache=true`)
      if (!resp.ok) throw new Error('加载图片失败')
      const data = await resp.json()
      if (requestVersion !== imageListLoadVersionRef.current || selectedDatasetRef.current?.id !== datasetId) return
      const images: ImageData[] = data.images || []
      updateImageCache(datasetId, images)
      if (fromStart) {
        silentRefreshRef.current = false
        initialImageIdRef.current = null
        pendingResumeRef.current = null
      }
      initFromImages(images, fromStart)
    } catch (err) {
      if (requestVersion === imageListLoadVersionRef.current && selectedDatasetRef.current?.id === datasetId) {
        setError(err instanceof Error ? err.message : '加载图片失败')
      }
    } finally {
      if (requestVersion === imageListLoadVersionRef.current) setIsLoadingList(false)
    }
  }

  /**
   * Fetch a fresh image list then navigate to a specific image by stem id.
   * Used by batch-job thumbnail clicks and "View Annotations" buttons.
   */
  const navigateToAnnotatedImage = async (imageId: string) => {
    if (!selectedDataset) return
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images?limit=999999&bust_cache=true`)
      if (!resp.ok) return
      const data = await resp.json()
      const images: ImageData[] = data.images || []
      updateImageCache(selectedDataset.id, images)
      // Set target before initFromImages so the allImages useEffect navigates there
      initialImageIdRef.current = imageId
      setAllImages(images)
    } catch {}
  }

  /**
   * Refresh the image list and cache WITHOUT resetting the current canvas view.
   * Used by batch-job polling so annotations from the job become visible
   * in the list/gallery without kicking the user back to image #0.
   * Also updates annotations for the currently displayed image if new ones exist.
   */
  const refreshImageListSilent = async () => {
    const ds = selectedDatasetRef.current
    if (!ds) return
    const original=annotationsRef.current
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${ds.id}/images?limit=999999&bust_cache=true`)
      if (!resp.ok) return
      const data = await resp.json()
      const images: ImageData[] = data.images || []
      updateImageCache(ds.id, images)
      // Set flag BEFORE setAllImages so the useEffect sees it synchronously
      silentRefreshRef.current = true
      setAllImages(images)
      // Sync current image annotations if they differ from what's on canvas
      const img = currentImageRef.current
      if (img) {
        const updated = images.find(i => i.id === img.id)
        if (updated && annotationsRef.current===original&&!editingRef.current&&JSON.stringify(updated.annotations||[])!==JSON.stringify(original)) {
          setAnnotations(updated.annotations || [])
        }
      }
    } catch {}
  }

  const loadModels = async () => {
    try {
      // The VLX summary endpoint only reads saved connection settings; it does
      // not probe or contact the model service. Keep it selectable even while
      // runtime inference remains disabled.
      const [r, vlxConfig] = await Promise.all([
        fetch(`${apiUrl}/api/models`),
        readVlxConnectionSummary(apiUrl),
      ])
      if (r.ok) {
        const d = await r.json()
        const nextModels = (d.models || []).map((m: any) => ({
          id: m.id,
          name: m.name,
          type: m.type || 'unknown',
          downloaded: !!(m.downloaded || m.loaded),
          loaded: !!m.loaded,
        }))
        if (vlxConfig.configured) {
          if (!nextModels.some((m: any) => m.id === VLX_PROVIDER_ID)) {
            nextModels.unshift({
              id: VLX_PROVIDER_ID,
              name: 'VLX-Seek',
              type: 'vlx',
              // VLX is a remote service entry, never a local model download.
              downloaded: false,
              loaded: false,
            })
          }
        }
        setModels(nextModels)
      }
    } catch {}
  }

  const downloadModel = async (model: { id: string; name: string; type: string }) => {
    setDownloadingModelId(model.id)
    toast.info(`正在下载 ${model.name}… 可能需要一分钟`, { duration: 120000, id: `dl-${model.id}` })
    try {
      const fd = new FormData()
      fd.append('model_type', model.type)
      fd.append('pretrained', model.id)
      const r = await fetch(`${apiUrl}/api/models/download`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error('启动下载失败')

      await new Promise<void>((resolve, reject) => {
        const poll = setInterval(async () => {
          try {
            const s = await fetch(`${apiUrl}/api/models/download-status/${model.id}`)
            if (!s.ok) return
            const status = await s.json()
            if (status.phase === 'loading_memory') {
              toast.info(`正在将 ${model.name} 载入内存… 请稍候`, { duration: 120000, id: `dl-${model.id}` })
            }
            if (status.status === 'done') { clearInterval(poll); resolve() }
            else if (status.status === 'error') { clearInterval(poll); reject(new Error(status.error ?? 'Download failed')) }
          } catch {}
        }, 1000)
      })

      toast.dismiss(`dl-${model.id}`)
      await loadModels()
      setSelectedModel(model.id)
      toast.success(`${model.name} 已就绪`)
    } catch (err) {
      toast.dismiss(`dl-${model.id}`)
      toast.error(`下载失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setDownloadingModelId(null)
    }
  }

  // Auto-switch to SAM wand when a SAM model is selected
  useEffect(() => {
    if (!selectedModel) return
    const m = models.find(m => m.id === selectedModel)
    if (m && (m.type === 'sam' || m.type === 'sam2' || m.type === 'sam3')) {
      setActiveTool('sam')
    }
  }, [selectedModel])

  // ── Load a single image file (fast — just one HTTP request) ─────────────────
  const loadImageFile = useCallback(async (imgData: ImageData) => {
    const requestVersion = ++imageLoadVersionRef.current
    setIsImageLoading(true)
    setError(null)
    setCurrentImage(imgData)
    setAnnotations(imgData.annotations || [])
    setSelectedAnnotation(null)
    setPolygonPoints([])
    setTempBox(null)
    setHistory([])
    setHistoryIndex(-1)
    imageRef.current = null

    return new Promise<void>((resolve) => {
      const img = new Image()
      img.onload = () => {
        if (requestVersion !== imageLoadVersionRef.current) { resolve(); return }
        imageRef.current = img
        setIsImageLoading(false)
        requestAnimationFrame(() => {
          if (requestVersion === imageLoadVersionRef.current && imageRef.current === img) {
            drawCanvasWithData(img, imgData.annotations || [], null, null, [])
          }
        })
        resolve()
      }
      img.onerror = () => {
        if (requestVersion === imageLoadVersionRef.current) {
          setIsImageLoading(false)
          setError('加载图片失败')
        }
        resolve()
      }
      img.src = `${apiUrl}/api/datasets/${selectedDataset?.id}/image-file/${imgData.path}`
    })
  }, [apiUrl, selectedDataset])

  const navigateImage = useCallback(async (dir: 'prev' | 'next') => {
    const newIdx = dir === 'prev'
      ? Math.max(0, currentIndex - 1)
      : Math.min(filteredImages.length - 1, currentIndex + 1)
    if (newIdx !== currentIndex && filteredImages[newIdx]) {
      // Auto-save current annotations before moving if the setting is on
      if (settings.autoSave && selectedDataset && currentImage && annotations.length > 0) {
        try {
          await fetch(
            `${apiUrl}/api/datasets/${selectedDataset.id}/image/${currentImage.id}/annotations`,
            { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: currentImage.id, annotations }) }
          )
          if (settings.notifications) toast.success('标注已自动保存')
        } catch {
          if (settings.notifications) toast.error('自动保存失败')
        }
      }
      setCurrentIndex(newIdx)
      await loadImageFile(filteredImages[newIdx])
    }
  }, [currentIndex, filteredImages, loadImageFile, settings.autoSave, settings.notifications, selectedDataset, currentImage, annotations, apiUrl])

  // ── Canvas drawing ───────────────────────────────────────────────────────────
  const drawCanvasWithData = useCallback((
    img: HTMLImageElement,
    anns: Annotation[],
    selectedIdx: number | null,
    tBox: { x: number; y: number; w: number; h: number } | null,
    polyPts: { x: number; y: number }[],
    zoomLevel = viewportRef.current.zoom,
    kpList: { x: number; y: number; label: string }[] = []
  ) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cw = container.clientWidth - 32
    const ch = container.clientHeight - 32
    const scale = Math.min(cw / img.width, ch / img.height) * zoomLevel
    scaleRef.current = scale
    canvas.width = img.width * scale
    canvas.height = img.height * scale
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const COLORS = ['#00D4B4', '#22c55e', '#ab00ff', '#ab00ff', '#ab00ff', '#00D4B4']

    const hexToRgba = (hex: string | null | undefined, alpha: number) => {
      const value = typeof hex === 'string' ? hex.trim() : ''
      const validHex = /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(value) ? value : '#00D4B4'
      const safeHex = validHex.length === 4
        ? '#' + validHex.slice(1).split('').map(c => c + c).join('')
        : validHex
      const r = parseInt(safeHex.slice(1, 3), 16)
      const g = parseInt(safeHex.slice(3, 5), 16)
      const b = parseInt(safeHex.slice(5, 7), 16)
      return `rgba(${r},${g},${b},${alpha})`
    }

    const drawLabel = (ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) => {
      ctx.font = 'bold 12px sans-serif'
      const tw = ctx.measureText(text).width + 10
      ctx.fillStyle = color
      ctx.fillRect(x, y - 20, tw, 20)
      ctx.fillStyle = '#fff'
      ctx.fillText(text, x + 5, y - 6)
    }

    anns.forEach((ann, idx) => {
      const isSel = idx === selectedIdx
      // Unknown/invalid class IDs (for example indexOf returning -1) have no palette entry.
      const classId = ann.class_id ?? -1
      const color = isSel ? '#22c55e' : (
        Number.isInteger(classId) && classId >= 0 ? COLORS[classId % COLORS.length] : undefined
      ) ?? '#00D4B4'
      ctx.strokeStyle = color
      ctx.lineWidth = isSel ? 3 : 2
      ctx.fillStyle = hexToRgba(color, isSel ? 0.25 : 0.125)

      if (ann.bbox && ann.bbox.length === 4) {
        const [x, y, w, h] = ann.bbox
        ctx.fillRect(x * scale, y * scale, w * scale, h * scale)
        ctx.strokeRect(x * scale, y * scale, w * scale, h * scale)
        drawLabel(ctx, ann.class_name || 'unknown', x * scale, y * scale, color)
      } else if (ann.x_center !== undefined && ann.width !== undefined) {
        const cx = ann.x_center * img.width * scale
        const cy = (ann.y_center || 0) * img.height * scale
        const w = ann.width * img.width * scale
        const h = (ann.height || 0) * img.height * scale
        const x = cx - w / 2, y = cy - h / 2
        ctx.fillRect(x, y, w, h)
        ctx.strokeRect(x, y, w, h)
        drawLabel(ctx, ann.class_name || 'unknown', x, y, color)
      } else if (ann.points && ann.points.length >= 4) {
        const pts = ann.points
        // Points may be normalized (0-1) — detect by checking if all <= 1
        const isNorm = ann.normalized || pts.every(p => p >= 0 && p <= 1)
        const px = (v: number, dim: number) => isNorm ? v * dim * scale : v * scale
        ctx.beginPath()
        ctx.moveTo(px(pts[0], img.width), px(pts[1], img.height))
        for (let i = 2; i < pts.length; i += 2) {
          ctx.lineTo(px(pts[i], img.width), px(pts[i + 1], img.height))
        }
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
        drawLabel(ctx, ann.class_name || 'unknown', px(pts[0], img.width), px(pts[1], img.height), color)
      }
    })

    if (tBox) {
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.strokeRect(tBox.x, tBox.y, tBox.w, tBox.h)
      ctx.setLineDash([])
    }

    if (polyPts.length > 0) {
      ctx.fillStyle = '#f59e0b'; ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2; ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(polyPts[0].x, polyPts[0].y)
      for (let i = 1; i < polyPts.length; i++) ctx.lineTo(polyPts[i].x, polyPts[i].y)
      ctx.stroke()
      polyPts.forEach((pt, i) => {
        ctx.beginPath()
        ctx.arc(pt.x, pt.y, i === 0 ? 7 : 5, 0, Math.PI * 2)
        ctx.fillStyle = i === 0 ? '#ef4444' : '#f59e0b'
        ctx.fill()
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke()
        ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2
      })
      ctx.fillStyle = 'rgba(0,0,0,0.75)'
      ctx.fillRect(10, canvas.height - 34, 270, 28)
      ctx.fillStyle = '#fff'; ctx.font = '12px sans-serif'
      ctx.fillText(`${polyPts.length} 个点 — 点击添加 · Enter/双击 完成`, 16, canvas.height - 14)
    }

    // Draw in-progress keypoints
    if (kpList.length > 0) {
      kpList.forEach((kp, i) => {
        const px = kp.x * img.width * scale
        const py = kp.y * img.height * scale
        ctx.beginPath()
        ctx.arc(px, py, 6, 0, Math.PI * 2)
        ctx.fillStyle = '#a855f7'
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2
        ctx.stroke()
        // Draw cross
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(px - 9, py); ctx.lineTo(px + 9, py)
        ctx.moveTo(px, py - 9); ctx.lineTo(px, py + 9)
        ctx.stroke()
        // Label
        ctx.font = 'bold 10px sans-serif'
        ctx.fillStyle = '#a855f7'
        ctx.fillText(kp.label || String(i + 1), px + 8, py - 4)
      })
    }

    // Draw SAM pending click markers
    const samClicks = samPendingClicksRef.current
    samClicks.forEach(click => {
      const cx = click.nx * img.width * scale
      const cy = click.ny * img.height * scale
      const isProcessing = click.status === 'processing'
      const color = isProcessing ? '#f59e0b' : '#ef4444'
      // Crosshair lines
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(cx - 18, cy); ctx.lineTo(cx + 18, cy)
      ctx.moveTo(cx, cy - 18); ctx.lineTo(cx, cy + 18)
      ctx.stroke()
      // Outer ring
      ctx.beginPath()
      ctx.arc(cx, cy, 11, 0, Math.PI * 2)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.stroke()
      // Inner fill dot
      ctx.beginPath()
      ctx.arc(cx, cy, 5, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      // Status label
      ctx.font = 'bold 10px sans-serif'
      ctx.fillStyle = color
      ctx.fillText(isProcessing ? '…' : '+', cx + 13, cy - 6)
    })
  }, [zoom])

  const drawCanvas = useCallback(() => {
    if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, tempBox, polygonPoints, zoom, keypointList)
  }, [annotations, selectedAnnotation, tempBox, polygonPoints, zoom, keypointList, drawCanvasWithData])

  useEffect(() => {
    if (sidebarTab === 'review' || isLoadingList || isImageLoading || !imageRef.current) return
    // Review and list loading unmount the canvas. Paint again after its replacement is mounted.
    const frame = requestAnimationFrame(drawCanvas)
    return () => cancelAnimationFrame(frame)
  }, [drawCanvas, currentImage?.id, sidebarTab, isLoadingList, isImageLoading])

  // ── Canvas interaction ───────────────────────────────────────────────────────
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height)
    }
  }

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 || spaceRef.current || panGestureRef.current) return
    const coords = getCanvasCoords(e)
    if (activeTool === 'bbox') {
      setIsDrawing(true); setDrawStart(coords)
    } else if (activeTool === 'polygon') {
      const newPts = [...polygonPoints, coords]
      setPolygonPoints(newPts)
      if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, tempBox, newPts, zoom)
    } else if (activeTool === 'brush') {
      setIsBrushing(true)
      setBrushPoints([coords])
    } else if (activeTool === 'keypoint') {
      const scale = scaleRef.current
      const img = imageRef.current
      if (!img) return
      const kp = { x: coords.x / scale / img.width, y: coords.y / scale / img.height, label: activeClass || 'point' }
      const newKps = [...keypointList, kp]
      setKeypointList(newKps)
      // Redraw with keypoints shown
      if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, tempBox, polygonPoints, zoom, newKps)
    } else if (activeTool === 'sam') {
      enqueueSamClick(coords)
    } else {
      // In select mode: if clicking on already-selected bbox, start drag
      if (activeTool === 'select' && selectedAnnotation !== null) {
        const ann = annotations[selectedAnnotation]
        if (ann?.bbox?.length === 4) {
          const scale = scaleRef.current
          const [x, y, w, h] = ann.bbox
          const imgX = coords.x / scale, imgY = coords.y / scale
          if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) {
            setIsDragging(true)
            setDragStart({ x: coords.x, y: coords.y, origBbox: [...ann.bbox] })
            return
          }
        }
      }
      handleAnnotationClick(coords)
    }
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (panGestureRef.current || (spaceRef.current && !editingRef.current)) return
    if (activeTool === 'brush' && isBrushing) {
      const coords = getCanvasCoords(e)
      setBrushPoints(prev => [...prev, coords])
      // Draw brush stroke on canvas
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (ctx) {
        ctx.beginPath()
        ctx.arc(coords.x, coords.y, brushSize, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(59,130,246,0.35)'
        ctx.fill()
      }
      return
    }
    // Drag selected annotation
    if (isDragging && dragStart && selectedAnnotation !== null) {
      const coords = getCanvasCoords(e)
      const scale = scaleRef.current
      const dx = (coords.x - dragStart.x) / scale
      const dy = (coords.y - dragStart.y) / scale
      const [ox, oy, ow, oh] = dragStart.origBbox
      const next = annotations.map((ann, i) =>
        i === selectedAnnotation ? { ...ann, bbox: [ox + dx, oy + dy, ow, oh] } : ann
      )
      setAnnotations(next)
      return
    }
    if (!isDrawing || !drawStart || activeTool !== 'bbox') return
    const c = getCanvasCoords(e)
    const box = { x: Math.min(drawStart.x, c.x), y: Math.min(drawStart.y, c.y), w: Math.abs(c.x - drawStart.x), h: Math.abs(c.y - drawStart.y) }
    setTempBox(box)
    if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, box, polygonPoints, zoom, keypointList)
  }

  const handleCanvasMouseUp = () => {
    if (panGestureRef.current) return
    if (isDragging) {
      setIsDragging(false)
      setDragStart(null)
      saveToHistory(annotations)
      return
    }
    if (activeTool === 'brush' && isBrushing) {
      setIsBrushing(false)
      if (brushPoints.length > 3) {
        // Convert brush stroke to a polygon annotation
        const scale = scaleRef.current
        const img = imageRef.current
        if (!img) { setBrushPoints([]); return }
        // Simplify brush points to convex hull approx
        const pts = brushPoints.flatMap(p => [p.x / scale / img.width, p.y / scale / img.height])
        const ann: Annotation = {
          type: 'polygon',
          class_id: localClasses.indexOf(activeClass),
          class_name: activeClass || 'brush_mask',
          points: pts,
          normalized: true
        }
        const next = [...annotations, ann]
        setAnnotations(next); saveToHistory(next)
      }
      setBrushPoints([])
      return
    }
    if (!isDrawing || !tempBox || activeTool !== 'bbox') {
      setIsDrawing(false); setDrawStart(null); setTempBox(null); return
    }
    const scale = scaleRef.current
    const bbox = [tempBox.x / scale, tempBox.y / scale, tempBox.w / scale, tempBox.h / scale]
    if (bbox[2] > 5 && bbox[3] > 5) {
      const ann: Annotation = {
        type: 'bbox',
        class_id: localClasses.indexOf(activeClass),
        class_name: activeClass || 'unknown',
        bbox
      }
      const next = [...annotations, ann]
      setAnnotations(next)
      saveToHistory(next)
    }
    setIsDrawing(false); setDrawStart(null); setTempBox(null)
  }

  const enqueueSamClick = (coords: { x: number; y: number }) => {
    if (!selectedDataset || !currentImage || !selectedModel) {
      toast.error('请先选择一个支持 SAM 的模型')
      return
    }
    const scale = scaleRef.current
    const img = imageRef.current
    if (!img) return
    const nx = coords.x / scale / img.width
    const ny = coords.y / scale / img.height
    const id = Math.random().toString(36).slice(2)
    const newClick = { id, nx, ny, status: 'pending' as const }
    const updated = [...samPendingClicksRef.current, newClick]
    samPendingClicksRef.current = updated
    setSamPendingClicks(updated)
  }

  const commitKeypoints = () => {
    if (keypointList.length === 0) return
    const pts = keypointList.flatMap(kp => [kp.x, kp.y])
    const ann: Annotation = {
      type: 'keypoints',
      class_id: localClasses.indexOf(activeClass),
      class_name: activeClass || 'skeleton',
      points: pts,
      normalized: true
    }
    const next = [...annotations, ann]
    setAnnotations(next); saveToHistory(next)
    setKeypointList([])
    toast.success(`已添加 ${keypointList.length} 个关键点`)
  }

  const copyAnnotationsToNext = async (count: number) => {
    if (!selectedDataset || !currentImage || annotations.length === 0) {
      toast.error('没有可复制的标注')
      return
    }
    let copied = 0
    for (let i = 1; i <= count; i++) {
      const targetIdx = currentIndex + i
      if (targetIdx >= filteredImages.length) break
      const target = filteredImages[targetIdx]
      try {
        await fetch(
          `${apiUrl}/api/datasets/${selectedDataset.id}/image/${target.id}/annotations`,
          { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: target.id, annotations }) }
        )
        copied++
      } catch {}
    }
    toast.success(`标注已复制到后续 ${copied} 张图片`)
  }

  const completePolygon = useCallback(() => {
    if (polygonPoints.length < 3) return
    const scale = scaleRef.current
    const points = polygonPoints.flatMap(pt => [pt.x / scale, pt.y / scale])
    const ann: Annotation = {
      type: 'polygon',
      class_id: localClasses.indexOf(activeClass),
      class_name: activeClass || 'unknown',
      points
    }
    const next = [...annotations, ann]
    setAnnotations(next)
    saveToHistory(next)
    setPolygonPoints([])
  }, [polygonPoints, annotations, activeClass, selectedDataset])

  const handleAnnotationClick = (coords: { x: number; y: number }) => {
    const scale = scaleRef.current
    const imgX = coords.x / scale, imgY = coords.y / scale
    for (let i = annotations.length - 1; i >= 0; i--) {
      const ann = annotations[i]
      if (ann.bbox && ann.bbox.length === 4) {
        const [x, y, w, h] = ann.bbox
        if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) { setSelectedAnnotation(i); return }
      } else if (ann.x_center !== undefined) {
        const img = imageRef.current
        const iw = img?.width || 1, ih = img?.height || 1
        const cx = ann.x_center * iw, cy = (ann.y_center || 0) * ih
        const w = ann.width! * iw, h = (ann.height || 0) * ih
        if (imgX >= cx - w/2 && imgX <= cx + w/2 && imgY >= cy - h/2 && imgY <= cy + h/2) { setSelectedAnnotation(i); return }
      }
    }
    setSelectedAnnotation(null)
  }

  const saveToHistory = (anns: Annotation[]) => {
    const h = history.slice(0, historyIndex + 1)
    h.push([...anns])
    setHistory(h)
    setHistoryIndex(h.length - 1)
  }

  const undo = () => {
    if (historyIndex > 0) { setHistoryIndex(historyIndex - 1); setAnnotations([...history[historyIndex - 1]]) }
    else if (historyIndex === 0 && currentImage) { setAnnotations(currentImage.annotations || []); setHistoryIndex(-1) }
  }

  const deleteAnnotation = () => {
    if (selectedAnnotation === null) return
    const next = annotations.filter((_, i) => i !== selectedAnnotation)
    setAnnotations(next); saveToHistory(next); setSelectedAnnotation(null)
  }

  const saveAnnotations = async (andAdvance = false, explicitAnns?: Annotation[]) => {
    if (!selectedDataset || !currentImage) return
    const annsToSave = explicitAnns ?? annotations
    setIsSaving(true)
    try {
      const resp = await fetch(
        `${apiUrl}/api/datasets/${selectedDataset.id}/image/${currentImage.id}/annotations`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: currentImage.id, annotations: annsToSave }) }
      )
      if (!resp.ok) throw new Error('保存失败')
      window.dispatchEvent(new Event('visos-datasets-changed'))
      // Update global image cache entry — set silent flag so useEffect doesn't reset to image 0
      const updated = allImages.map(img => img.id === currentImage.id ? { ...img, annotations: annsToSave } : img)
      silentRefreshRef.current = true
      setAllImages(updated)
      updateImageCache(selectedDataset.id, updated)
      setCurrentImage(current=>current?.id===currentImage.id?{...current,annotations:annsToSave}:current)

      if (settings.notifications) toast.success('标注已保存')
      if (andAdvance && currentIndex < filteredImages.length - 1) {
        await navigateImage('next')
      }
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存失败')
      if (settings.notifications) toast.error('保存失败')
      return false
    } finally {
      setIsSaving(false)
    }
  }
  useEffect(()=>{
    registerBeforeLeave?.(async()=>{
      if(isSaving||isAutoAnnotating||editingRef.current)throw Error('请等待当前标注操作完成后再继续。')
      if(currentImage&&JSON.stringify(currentImage.annotations||[])!==JSON.stringify(annotationsRef.current)){
        const saved=await saveAnnotations(false,annotationsRef.current);if(!saved)throw Error('当前图片的标注未保存，请保存后再继续。')
      }
    })
    return()=>registerBeforeLeave?.(async()=>{})
  },[currentImage,annotations,isSaving,isAutoAnnotating,registerBeforeLeave]) // eslint-disable-line react-hooks/exhaustive-deps

  const [isModelLoading, setIsModelLoading] = useState(false)

  const autoAnnotate = async () => {
    if (!selectedDataset || !currentImage || !selectedModel) return
    // Check if model is downloaded but not loaded into memory — first call will load it
    const modelInfo = models.find(m => m.id === selectedModel)
    const needsLoad = modelInfo?.type !== 'vlx' && modelInfo?.downloaded && !modelInfo?.loaded
    if (needsLoad) {
      setIsModelLoading(true)
      toast.info('正在将模型载入内存 — 首次标注可能需要 10–30 秒…', { duration: 20000, id: 'model-load' })
    }
    setIsAutoAnnotating(true)
    try {
      const params = new URLSearchParams({
        model_id: selectedModel,
      })
      if (supportsConfidenceThreshold(models.find(m => m.id === selectedModel))) params.set('confidence_threshold', String(confidence))
      // Pass text prompt for zero-shot models (YOLO-World, GroundingDINO)
      if (combinedPrompt) params.set('text_prompt', combinedPrompt)
      if (currentImage.path) params.set('image_path_hint', currentImage.path)
      if (tileEnabled) {
        params.set('tile_pass','true');params.set('tile_size',String(tileSize));params.set('tile_overlap',String(tileOverlap))
        if (dualEnabled && verifyModel && verifyModel !== selectedModel) params.set('verify_model_id',verifyModel)
        setTileReview(null);setTileSummary(null)
      }
      const startedImage=currentImage.id
      const startedDataset=selectedDataset.id
      const resp = await fetch(
        `${apiUrl}/api/auto-annotate/${selectedDataset.id}/single/${currentImage.id}?${params}`,
        { method: 'POST' }
      )
      if (needsLoad) toast.dismiss('model-load')
      if (!resp.ok) throw new Error('自动标注失败')
      let data = await resp.json()
      if (data.preview && data.job_id) {
        previewJobRef.current = data.job_id
        setTilePreview(data)
        while (true) {
          await new Promise(resolve => setTimeout(resolve, 1200))
          const status = await fetch(`${apiUrl}/api/auto-annotate/tile-preview/${data.job_id}/status`)
          if (!status.ok) throw new Error('读取切片预览进度失败')
          const job = await status.json()
          setTilePreview(job)
          if (job.status === 'error') throw new Error(job.error || '切片预览失败')
          if (job.status === 'cancelled') { toast.info('切片预览已取消'); return }
          if (job.status === 'done') {
            data = job.result
            if (currentImageRef.current?.id !== startedImage || selectedDatasetRef.current?.id !== startedDataset) {
              toast.info('预览已完成，图片已切换，结果未写入；请回到原图重新预览')
              return
            }
            setTileSummary(data.tile_pass)
            setTileReview({ imageId: startedImage, datasetId: startedDataset, jobId: data.review_job_id || previewJobRef.current!, items: data.review_annotations || [] })
            refreshReviewIndex()
            break
          }
        }
      }
      if (data.review_job_id && data.review_summary?.pending > 0) {
        setTileReview({ imageId: startedImage, datasetId: startedDataset, jobId: data.review_job_id, items: data.review_annotations || [] })
        refreshReviewIndex()
      }
      if (data.annotations?.length) {
        const next = [...annotationsRef.current, ...data.annotations]
        setAnnotations(next)
        saveToHistory(next)
        // Save to dataset immediately so annotations persist
        await saveAnnotations(false, next)
      } else {
        toast.info('未检测到目标')
      }
      // Refresh model list so the model shows as loaded now
      if (needsLoad) loadModels()
    } catch (err) {
      setError(err instanceof Error ? err.message : '自动标注失败')
    } finally {
      setIsAutoAnnotating(false)
      setTilePreview(null)
      previewJobRef.current=null
      setIsModelLoading(false)
    }
  }

  // nameOverride 供 P2A 的「类别」下拉旁的新增入口复用，避免再造一套类别系统。
  const submitNewClass = async (nameOverride?: string) => {
    const name = (nameOverride ?? newClassName).trim()
    if (!name || !selectedDataset) return
    setAddingClass(false)
    setNewClassName('')
    // Optimistic update — show the class immediately without waiting for the slow backend parse
    if (!localClasses.includes(name)) {
      setLocalClasses(prev => [...prev, name])
      setAvailableClasses(prev => [...prev, name])
    }
    setActiveClass(name)
    toast.success(`已添加类别「${name}」`)
    try {
      const resp = await fetch(
        `${apiUrl}/api/datasets/${selectedDataset.id}/add-classes`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_id: selectedDataset.id, new_classes: [name], use_model: false }) }
      )
      if (!resp.ok) throw new Error('添加类别失败')
      const data = await resp.json()
      const updatedClasses: string[] = data.updated_dataset?.classes || []
      if (updatedClasses.length > 0) {
        setLocalClasses(updatedClasses)
        setAvailableClasses(updatedClasses)
      }
    } catch {
      // Revert optimistic update on failure
      setLocalClasses(prev => prev.filter(c => c !== name))
      setAvailableClasses(prev => prev.filter(c => c !== name))
      toast.error('保存类别到后端失败')
    }
  }

  const combinedPrompt = promptTags.join(', ')

  const addPromptTag = (raw: string) => {
    const tags = raw.split(',').map(t => t.trim()).filter(Boolean)
    setPromptTags(prev => [...prev, ...tags.filter(t => !prev.includes(t))])
    setPromptInput('')
  }

  const removePromptTag = (tag: string) =>
    setPromptTags(prev => prev.filter(t => t !== tag))

  /**
   * Poll a single batch job until it finishes.
   * Uses refs so the interval stays valid even after component remounts.
   * Every 5 polls it re-fetches the current image's annotations so they
   * appear live while the job is running.
   * Deduplicates: calling this twice for the same job_id cancels the old interval.
   */
  const _pollJob = (job_id: string) => {
    // Cancel any existing interval for this job to avoid duplicates
    if (pollIntervalsRef.current[job_id]) {
      clearInterval(pollIntervalsRef.current[job_id])
    }
    let tick = 0
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/status`)
        if (!r.ok) { clearInterval(iv); delete pollIntervalsRef.current[job_id]; return }
        const data = await r.json()
        setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], ...data } }))

        // Every 5 ticks while running, refresh annotations for whichever image
        // is currently on screen so batch results appear without navigating away.
        tick++
        if (tick % 5 === 0 && (data.status === 'running' || data.status === 'paused')) {
          const ds = selectedDatasetRef.current
          const img = currentImageRef.current
          if (ds && img) {
            const before=annotationsRef.current
            try {
              const ir = await fetch(`${apiUrl}/api/datasets/${ds.id}/image/${img.id}`)
              if (ir.ok) {
                const imgData = await ir.json()
                const fresh: Annotation[] = imgData.annotations || []
                // Preserve edits made while the status request was in flight.
                if (annotationsRef.current===before&&currentImageRef.current?.id===img.id&&selectedDatasetRef.current?.id===ds.id&&!editingRef.current&&JSON.stringify(fresh)!==JSON.stringify(before)&&JSON.stringify(before)===JSON.stringify(img.annotations||[])) {
                  setAnnotations(fresh)
                  setCurrentImage(prev=>prev?.id===img.id?{...prev,annotations:fresh}:prev)
                }
              }
            } catch {}
          }
        }

        const done = data.status === 'done' || data.status === 'error' || data.status === 'cancelled'
        if (done) {
          clearInterval(iv)
          delete pollIntervalsRef.current[job_id]
          if (data.status === 'done') {
            toast.success(
              `批量完成 — 已标注 ${data.annotated} 张图片，` +
              `创建 ${data.total_annotations ?? 0} 个标注`
            )
          } else if (data.status === 'error') {
            toast.error(`批量任务失败：${data.error}`)
          }
          // Silent refresh: update image list / annotations without resetting canvas view
          refreshImageListSilent()
          refreshReviewIndex()
        }
        // 'paused' keeps polling so stats stay live
      } catch { clearInterval(iv); delete pollIntervalsRef.current[job_id] }
    }, 1500)
    pollIntervalsRef.current[job_id] = iv
  }

  // When switching to annotate tab, refresh the current image's annotations
  // from the backend so batch results are immediately visible.
  useEffect(() => {
    if (sidebarTab !== 'annotate') return
    const ds = selectedDatasetRef.current
    const img = currentImageRef.current
    if (!ds || !img) return
    const originalAnnotations = annotationsRef.current
    ;(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/datasets/${ds.id}/image/${img.id}`)
        if (r.ok) {
          const data = await r.json()
          if (selectedDatasetRef.current?.id !== ds.id || currentImageRef.current?.id !== img.id) return
          const fresh: Annotation[] = data.annotations || []
          // Apply removals and same-count edits too; preserve edits made while this request was pending.
          if (annotationsRef.current === originalAnnotations && JSON.stringify(fresh) !== JSON.stringify(originalAnnotations)) {
            setAnnotations(fresh)
          }
        }
      } catch {}
    })()
  }, [sidebarTab]) // eslint-disable-line react-hooks/exhaustive-deps

  const controlTilePreview = async (action: 'pause' | 'resume' | 'cancel') => {
    const id = previewJobRef.current
    if (!id) return
    const response = await fetch(`${apiUrl}/api/auto-annotate/tile-preview/${id}/${action}`, { method: 'POST' })
    if (!response.ok) toast.error('任务操作失败')
  }

  const refreshReviewIndex = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/reviews/jobs`)
      if (!response.ok) return
      const data = await response.json()
      const jobs = data.jobs || []
      setReviewPending(jobs.filter((j: any) => !j.readonly && (!selectedDataset || j.dataset_id === selectedDataset.id)).reduce((n: number,j: any) => n + j.summary.pending,0))
      setBatchJobs(prev => {
        const next = { ...prev }
        for (const job of jobs) if (job.kind === 'batch' && next[job.job_id]) next[job.job_id] = { ...next[job.job_id], review_summary: job.summary, review_readonly: job.readonly }
        return next
      })
      setTileReview(prev => {
        if (!prev) return prev
        const job = jobs.find((j: any) => j.job_id === prev.jobId)
        return job?.summary.pending === 0 ? null : prev
      })
    } catch {}
  }
  useEffect(() => { refreshReviewIndex() }, [apiUrl,selectedDataset?.id,sidebarTab]) // eslint-disable-line react-hooks/exhaustive-deps
  const openReview = (jobId?: string, imageKey?: string) => {
    setReviewJobId(jobId || null);setReviewImageKey(imageKey || null);setSidebarTab('review')
  }
  const returnFromReview = (tab: 'annotate' | 'jobs') => {
    // Start over with current server labels; a silent list refresh would retain an unpainted canvas.
    imageLoadVersionRef.current += 1
    initialImageIdRef.current = null
    pendingResumeRef.current = null
    silentRefreshRef.current = false
    loadedImagesDatasetIdRef.current = null
    imageRef.current = null
    currentImageRef.current = null
    annotationsRef.current = []
    setResumeBanner(null)
    setSelectedSplit('all')
    setSelectedClass('all')
    setCurrentIndex(0)
    setCurrentImage(null)
    setAnnotations([])
    setAllImages([])
    setFilteredImages([])
    setIsImageLoading(false)
    setSelectedAnnotation(null)
    setIsDrawing(false)
    setIsDragging(false)
    setIsBrushing(false)
    setPolygonPoints([])
    setTempBox(null)
    setBrushPoints([])
    setKeypointList([])
    viewportRef.current = { zoom: 1, pan: { x: 0, y: 0 } }
    setZoom(1)
    setPan({ x: 0, y: 0 })
    stopPan()
    setSidebarTab(tab)
    void loadImageList(true)
    void refreshReviewIndex()
  }

  const textAnnotateSingle = async () => {
    const selModel = models.find(m => m.id === selectedModel)
    const isTargetModel = selModel?.type === 'vlx' || selModel?.type === 'vlm'
    // VLX / Qwen / 豆包统一按「类别 + 目标描述 + 高级约束」提交。
    const prompt = isTargetModel
      ? [targetDescription.trim(), targetExtra.trim()].filter(Boolean).join('\n')
      : combinedPrompt
    if (!selectedDataset || !currentImage || !selectedModel || !prompt) return
    if (isTargetModel && !targetLabel.trim()) { toast.error('请先选择类别'); return }
    setIsAutoAnnotating(true)
    try {
      // VLX has a dedicated inference endpoint rather than an OpenAI-compatible
      // provider. Keep this single-image action usable on the currently running
      // legacy API without asking that process to reload its model catalog.
      if (selectedModel === VLX_PROVIDER_ID && !tileEnabled && !dualEnabled) {
        const classId = localClasses.indexOf(targetLabel.trim())
        if (classId < 0) throw new Error('请先选择数据集已有类别')
        if (!currentImage.path) throw new Error('当前图片缺少数据集路径')
        const imageResponse = await fetch(
          `${apiUrl}/api/datasets/${selectedDataset.id}/image-file/${currentImage.path}`
        )
        if (!imageResponse.ok) throw new Error('读取当前图片失败')
        const form = new FormData()
        form.append('image', await imageResponse.blob(), currentImage.path.split('/').pop() || 'image.jpg')
        form.append('prompt', prompt)
        form.append('task', 'detection')
        form.append('lang', 'zh')
        const response = await fetch(`${apiUrl}/api/lmm/vlx/infer-image`, { method: 'POST', body: form })
        const result = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(result.detail || 'VLX-Seek 标注失败')
        const generated: Annotation[] = (Array.isArray(result.boxes) ? result.boxes : [])
          .flatMap((box: any) => {
            const [x1, y1, x2, y2] = [box.x1, box.y1, box.x2, box.y2].map(Number)
            if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return []
            return [{
              type: 'bbox', class_id: classId, class_name: targetLabel.trim(),
              confidence: 1, confidence_source: "synthetic", bbox: [x1, y1, x2 - x1, y2 - y1],
            }]
          })
        if (generated.length) {
          const next = [...annotations, ...generated]
          setAnnotations(next)
          saveToHistory(next)
          await saveAnnotations(false, next)
        } else {
          toast.info(`这张图上没有找到类别「${targetLabel}」`)
        }
        return
      }

      const params = new URLSearchParams({
        model_id: selectedModel,
        text_prompt: prompt,
      })
      if (supportsConfidenceThreshold(selModel)) params.set('confidence_threshold', String(confidence))
      // ── P2A：追加新目标结构（与 text_prompt 并存，后端按开关取用）──
      if (isTargetModel) {
        params.set('label', targetLabel.trim())
        params.set('description', targetDescription.trim())
        if (targetExtra.trim()) params.set('extra_prompt', targetExtra.trim())
      }
      if (currentImage.path) params.set('image_path_hint', currentImage.path)
      // ── 双模型：第二个模型做交叉校验（不传就是原来的单模型行为）──
      const dualOn = dualEnabled && !!verifyModel && verifyModel !== selectedModel
      if (dualOn) params.set('verify_model_id', verifyModel)
      if (tileEnabled) {
        params.set('tile_pass', 'true')
        params.set('tile_size', String(tileSize))
        params.set('tile_overlap', String(tileOverlap))
        setTileReview(null); setTileSummary(null)
      }
      const startedImage = currentImage.id
      const startedDataset = selectedDataset.id
      const resp = await fetch(
        `${apiUrl}/api/auto-annotate/${selectedDataset.id}/single/${currentImage.id}?${params}`,
        { method: 'POST' }
      )
      if (!resp.ok) {
        let detail = '文本标注失败'
        try { detail = (await resp.json()).detail ?? detail } catch {}
        throw new Error(detail)
      }
      let data = await resp.json()
      if (data.preview && data.job_id) {
        previewJobRef.current = data.job_id
        setTilePreview(data)
        while (true) {
          await new Promise(resolve => setTimeout(resolve, 1200))
          const status = await fetch(`${apiUrl}/api/auto-annotate/tile-preview/${data.job_id}/status`)
          if (!status.ok) throw new Error('读取切片预览进度失败')
          const job = await status.json()
          setTilePreview(job)
          if (job.status === 'error') throw new Error(job.error || '切片预览失败')
          if (job.status === 'cancelled') { toast.info('切片预览已取消'); return }
          if (job.status === 'done') {
            data = job.result
            if (currentImageRef.current?.id !== startedImage || selectedDatasetRef.current?.id !== startedDataset) {
              toast.info('预览已完成，图片已切换，结果未写入；请回到原图重新预览')
              return
            }
            setTileSummary(data.tile_pass)
            setTileReview({ imageId: startedImage, datasetId: startedDataset, jobId: data.review_job_id || previewJobRef.current!, items: data.review_annotations || [] })
            refreshReviewIndex()
            break
          }
        }
      }
      if (data.review_job_id && data.review_summary?.pending > 0) {
        setTileReview({ imageId: startedImage, datasetId: startedDataset, jobId: data.review_job_id, items: data.review_annotations || [] })
        refreshReviewIndex()
      }
      if (data.annotations?.length) {
        const next = [...annotations, ...data.annotations]
        setAnnotations(next)
        saveToHistory(next)
        await saveAnnotations(false, next)
        if (data.dual) {
          setDualResult({
            accept: data.dual.accept ?? 0,
            review: data.dual.review ?? 0,
            teachers_down: data.dual.teachers_down ?? [],
          })
          toast.success(
            `双模型：采纳 ${data.dual.accept ?? 0} / 待复核 ${data.dual.review ?? 0}` +
            ((data.dual.teachers_down ?? []).length
              ? `（${(data.dual.teachers_down as string[]).join('、')} 不可用）`
              : '')
          )
        } else {
          setDualResult(null)
          toast.success(`针对「${prompt}」找到 ${data.annotations.length} 个标注`)
        }
      } else if (tileEnabled) {
        setDualResult({ accept: data.dual?.accept || 0, review: data.dual?.review || 0, teachers_down: data.dual?.teachers_down || [] })
        toast.info(`切片预览：采纳 ${data.dual?.accept || 0} / 待复核 ${data.dual?.review || 0}`)
      } else {
        toast.info(selModel?.type === 'vlx' || selModel?.type === 'vlm'
          ? `这张图上没有找到类别「${targetLabel}」`
          : '该图片上没有与文本提示匹配的目标')
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '文本标注失败')
    } finally {
      setIsAutoAnnotating(false)
      setTilePreview(null)
      previewJobRef.current = null
    }
  }

  const textAnnotateBatch = async () => {
    if (!selectedDataset || !selectedModel || isStartingBatch||batchSubmitGuard.current) return
    if(assistantDraftId&&!assistantDraft){toast.error('标注参数正在加载，请稍后重试。');return}
    if(assistantDraft&&assistantDraft.data.dataset_id!==selectedDataset.id){toast.error('当前数据集与流程不一致，请返回流程数据集或暂停引导后操作。');return}
    if(assistantDraft?.status==='uncertain'){toast.error('请先检查批量任务状态，勿重复提交。');return}
    if(assistantDraft?.result?.job_id&&['running','paused'].includes(batchJobs[assistantDraft.result.job_id]?.status)){toast.info('批量任务仍在进行，请在任务面板查看。');return}
    const selModel = models.find(m => m.id === selectedModel)
    const isTextModel = selModel && ['sam3', 'yoloworld', 'groundingdino', 'owlvit', 'vlx', 'vlm'].includes(selModel.type)
    const isTargetModel = selModel?.type === 'vlx' || selModel?.type === 'vlm'
    // 云端视觉模型共用同一套目标表单；约束与描述一起送入推理提示词。
    const prompt = isTargetModel
      ? [targetDescription.trim(), targetExtra.trim()].filter(Boolean).join('\n')
      : combinedPrompt
    if (isTextModel && !prompt) return
    if (isTargetModel && !targetLabel.trim()) { toast.error('请先选择类别'); return }
    setIsStartingBatch(true)
    batchSubmitGuard.current=true
    try {
      const params = new URLSearchParams({
        model_id: selectedModel,
      })
      if (supportsConfidenceThreshold(selModel)) params.set('confidence_threshold', String(confidence))
      if (prompt) params.set('text_prompt', prompt)
      // ── P2A：追加新目标结构（job 里会多存一个 target 字段，历史 job 不受影响）──
      if (isTargetModel) {
        params.set('label', targetLabel.trim())
        params.set('description', targetDescription.trim())
        if (targetExtra.trim()) params.set('extra_prompt', targetExtra.trim())
      }
      // ── 双模型：批量同样走两个模型交叉校验 ──
      const dualOn = dualEnabled && !!verifyModel && verifyModel !== selectedModel
      if (dualOn) params.set('verify_model_id', verifyModel)
      if (tileEnabled) {
        params.set('tile_pass', 'true')
        params.set('tile_size', String(tileSize))
        params.set('tile_overlap', String(tileOverlap))
      }
      let job_id:string
      if(assistantDraft){
        const config={...assistantDraft.data,dataset_id:selectedDataset.id,model_id:selectedModel,label:isTargetModel?targetLabel.trim():activeClass||assistantDraft.data.label,description:isTargetModel?targetDescription.trim():assistantDraft.data.description,extra_prompt:isTargetModel?targetExtra.trim():assistantDraft.data.extra_prompt,text_prompt:prompt||targetLabel||assistantDraft.data.text_prompt,confidence_threshold:confidence,verify_model_id:dualOn?verifyModel:null,tile_pass:tileEnabled,tile_size:tileSize,tile_overlap:tileOverlap}
        let saved=assistantDraft.status==='prepared'?await assistantCall(apiUrl,`drafts/${assistantDraft.id}`,'PUT',{kind:'annotation',revision:assistantDraft.revision,data:config}):await assistantCall(apiUrl,'drafts','POST',{kind:'annotation',data:config})
        setAssistantDraft(saved)
        saved=await assistantCall(apiUrl,`drafts/${saved.id}/execute`,'POST',{revision:saved.revision});setAssistantDraft(saved);job_id=saved.result.job_id
        onAssistantDraftChange?.(saved.id)
      }else{
        const resp=await fetch(`${apiUrl}/api/auto-annotate/${selectedDataset.id}/text-batch?${params}`,{method:'POST'})
        if(!resp.ok)throw Error('启动批量标注失败');({job_id}=await resp.json())
      }

      // Register job in state immediately so the monitor shows it right away
      setBatchJobs(prev => ({
        ...prev,
        [job_id]: {
          job_id,
          status: 'running',
          paused: false,
          progress: 0,
          total: 0,
          processed: 0,
          annotated: 0,
          failed: 0,
          total_annotations: 0,
          text_prompt: prompt || '',
          // 双模型：追加字段，历史 job 没有也不影响
          tile_pass: tileEnabled,
          verify_model_id: dualOn ? verifyModel : undefined,
          dual: dualOn ? { accept: 0, review: 0 } : undefined,
          // P2A：新目标结构随 job 一起记录（追加字段，历史 job 没有也不影响）
          target: isTargetModel ? {
            label: targetLabel.trim(),
            description: targetDescription.trim(),
            extra_prompt: targetExtra.trim(),
          } : undefined,
          started_at: new Date().toISOString(),
          dataset_id: selectedDataset.id,
          recent_images: [],
        },
      }))
      toast.success(`批量任务已启动${prompt ? `（${prompt}）` : ''}${dualOn ? ' · 双模型' : ''}`)
      setSidebarTab('jobs')
      _pollJob(job_id)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '批量标注失败')
    } finally {
      setIsStartingBatch(false)
      batchSubmitGuard.current=false
    }
  }

  const deleteAllAnnotations = async () => {
    if (!selectedDataset) return
    if (!confirm(`确定要删除「${selectedDataset.name}」中的全部标注吗？此操作无法撤销。`)) return
    if (!confirm(`再次确认？该数据集中的所有标注标签都将被永久清除。`)) return
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/annotations`, { method: 'DELETE' })
      if (!resp.ok) throw new Error('删除标注失败')
      setAnnotations([])
      // Invalidate image cache so annotation counts refresh
      updateImageCache(selectedDataset.id, [])
      toast.success('全部标注已删除')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '删除标注失败')
    }
  }

  const pauseJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/pause`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'paused', paused: true } }))
    } catch { toast.error('暂停任务失败') }
  }

  const resumeJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/resume`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'running', paused: false } }))
    } catch { toast.error('继续任务失败') }
  }

  const cancelJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/cancel`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'cancelled' } }))
    } catch { toast.error('取消任务失败') }
  }

  const undoJob = async (job_id: string) => {
    try {
      const resp = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/undo`, { method: 'POST' })
      if (!resp.ok) throw new Error((await resp.json()).detail ?? '撤销失败')
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'undone' } }))
      await refreshImageListSilent()
      await refreshReviewIndex()
      toast.success('批量标注已撤销 — 数据集已恢复到任务前状态')
    } catch (err) { toast.error(err instanceof Error ? err.message : '撤销失败') }
  }

  // ── Keyboard shortcuts ───────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = async (e: KeyboardEvent) => {
      if (sidebarTab === 'review') return
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 's' || e.key === 'S') { e.preventDefault(); await saveAnnotations(true) }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotation !== null) { e.preventDefault(); deleteAnnotation() }
      if (e.key === 'ArrowLeft' && e.altKey) { e.preventDefault(); navigateImage('prev') }
      if (e.key === 'ArrowRight' && e.altKey) { e.preventDefault(); navigateImage('next') }
      if (e.key === 'Escape') { setSelectedAnnotation(null); setPolygonPoints([]); setTempBox(null); setIsDrawing(false); setKeypointList([]) }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo() }
      if (e.key === 'Enter') {
        e.preventDefault()
        if (polygonPoints.length >= 3) completePolygon()
        else if (keypointList.length > 0) commitKeypoints()
      }
      // Tool shortcuts
      if (e.key === 'v') setActiveTool('select')
      if (e.key === 'b') setActiveTool('bbox')
      if (e.key === 'p') { setActiveTool('polygon'); setPolygonPoints([]) }
      if (e.key === 'k') setActiveTool('keypoint')
      if (e.key === 'r') setActiveTool('brush')
      if (e.key === 'q') setActiveTool('sam')
      // 1-9 class assignment shortcuts
      const num = parseInt(e.key)
      if (!isNaN(num) && num >= 1 && num <= 9 && localClasses.length > 0) {
        const cls = localClasses[num - 1]
        if (cls) {
          setActiveClass(cls)
          // Also update selected annotation's class if one is selected
          if (selectedAnnotation !== null) {
            const next = annotations.map((ann, i) => i === selectedAnnotation
              ? { ...ann, class_name: cls, class_id: num - 1 }
              : ann
            )
            setAnnotations(next); saveToHistory(next)
            toast.success(`已指定类别：${cls}`)
          } else {
            toast.info(`当前类别：${cls}`)
          }
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, filteredImages, selectedAnnotation, annotations, polygonPoints, keypointList, selectedDataset, navigateImage, localClasses, sidebarTab])

  // ── Empty state ──────────────────────────────────────────────────────────────
  if (sidebarTab === 'review') return <ReviewWorkspace apiUrl={apiUrl} datasetId={selectedDataset?.id}
    initialJobId={reviewJobId} initialImageKey={reviewImageKey} onChanged={refreshReviewIndex}
    onBack={returnFromReview} />

  if (!selectedDataset) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-muted-foreground">未选择数据集</h3>
          <p className="text-sm text-muted-foreground/70 mt-1">请到「数据集」视图中选择一个数据集</p>
        </div>
      </div>
    )
  }

  if (isLoadingList) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <h3 className="text-lg font-medium">正在加载数据集</h3>
          <p className="text-sm text-muted-foreground mt-1">正在从 {selectedDataset.name} 加载图片…</p>
          <p className="text-xs text-muted-foreground/60 mt-1">数据集较大时可能需要一些时间</p>
        </div>
      </div>
    )
  }

  // ── 未找到图片 after loading ────────────────────────────────────────────
  if (!isLoadingList && filteredImages.length === 0 && allImages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-3">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto" />
          <h3 className="text-lg font-medium text-muted-foreground">未找到图片</h3>
          <p className="text-sm text-muted-foreground/70">
            数据集「{selectedDataset.name}」没有图片，或图片加载失败。
          </p>
          <button
            onClick={() => loadImageList()}
            className="text-sm text-primary underline underline-offset-2"
          >
            重试加载图片
          </button>
          {error && <p className="text-xs text-red-600 dark:text-red-400 mt-1">{error}</p>}
        </div>
      </div>
    )
  }

  // ── Elapsed time helper ───────────────────────────────────────────────────────
  const fmtElapsed = (startedAt: string | number) => {
    const ms = typeof startedAt === 'number' ? startedAt : new Date(startedAt).getTime()
    const s = Math.floor((Date.now() - ms) / 1000)
    if (s < 0 || isNaN(s)) return '—'
    if (s < 60) return `${s}s`
    const m = Math.floor(s / 60)
    return `${m}m ${s % 60}s`
  }

  const datasetJobs = Object.values(batchJobs).filter(j => !selectedDataset || j.dataset_id === selectedDataset.id)
  const runningJobCount = datasetJobs.filter(j => j.status === 'running' || j.status === 'paused').length
  const totalJobCount = datasetJobs.length

  // 云端模型（VLM provider 如通义千问 + VLX-Seek 检测服务）混在几十个本地权重里
  // 根本找不到 —— 在下拉里单独置顶成一组，打开就能看到、点得到。
  const isCloudModel = (m: { type: string }) => m.type === 'vlm' || m.type === 'vlx'
  const vlmModels = models.filter(isCloudModel)
  const localModels = models.filter(m => !isCloudModel(m))
  const selectedModelInfo = models.find(m => m.id === selectedModel)
  const showConfidence = supportsConfidenceThreshold(selectedModelInfo)
  const isTargetModel = selectedModelInfo?.type === 'vlx' || selectedModelInfo?.type === 'vlm'
  const targetClasses = localClasses

  // ── Main UI ──────────────────────────────────────────────────────────────────
  return (
    <div className="h-full flex flex-col">
      {/* Resume-progress banner */}
      {resumeBanner && (
        <div className="flex items-center justify-between gap-3 px-4 py-2 bg-blue-500/10 border-b border-blue-500/20 text-sm">
          <span className="text-blue-700 dark:text-blue-400">
            你上次查看到第 <strong>{resumeBanner.imageIndex + 1}</strong> 张 — <span className="font-mono text-xs">{resumeBanner.imageName}</span>。要从此处继续吗？
          </span>
          <div className="flex gap-2 flex-shrink-0">
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => {
              setResumeBanner(null)
              try { localStorage.removeItem(`cvdm_annotate_${selectedDataset?.id}`) } catch {}
            }}>从头开始</Button>
            <Button size="sm" className="h-7 text-xs" onClick={() => {
              if (!resumeBanner) return
              pendingResumeRef.current = {
                datasetId: selectedDataset?.id ?? '',
                imageId: resumeBanner.imageId,
                imageIndex: resumeBanner.imageIndex,
              }
              setSelectedSplit(resumeBanner.split || 'all')
              setSelectedClass(resumeBanner.classFilter || 'all')
              setResumeRequestVersion(version => version + 1)
              setResumeBanner(null)
            }}>继续</Button>
          </div>
        </div>
      )}
      <div className="flex-1 flex min-h-0">
      {/* Canvas side — always visible */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="flex items-center gap-2 p-2 border-b border-border bg-card flex-wrap">
          {/* Drawing tools */}
          <div className="flex items-center gap-1">
            <Button size="sm" variant={activeTool === 'select' ? 'default' : 'outline'} onClick={() => setActiveTool('select')} title="选择 (V)">
              <MousePointer2 className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'bbox' ? 'default' : 'outline'} onClick={() => setActiveTool('bbox')} title="矩形框 (B)">
              <Square className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'polygon' ? 'default' : 'outline'}
              onClick={() => { setActiveTool('polygon'); setPolygonPoints([]) }} title="多边形 (P)">
              <Pentagon className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'keypoint' ? 'default' : 'outline'}
              onClick={() => { setActiveTool('keypoint'); setKeypointList([]) }} title="关键点 (K)">
              <Crosshair className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'brush' ? 'default' : 'outline'}
              onClick={() => setActiveTool('brush')} title="画笔 / 绘制掩码 (R)">
              <Brush className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'sam' ? 'default' : 'outline'}
              onClick={() => setActiveTool('sam')} title="SAM 点击分割 (Q)">
              {samPendingClicks.some(c => c.status === 'processing') ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
              {samPendingClicks.length > 0 && <span className="ml-1 text-xs">{samPendingClicks.length}</span>}
            </Button>
          </div>

          {/* Brush size (shown when brush active) */}
          {activeTool === 'brush' && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">尺寸：</span>
              <Slider value={[brushSize]} onValueChange={([v]) => setBrushSize(v)} min={5} max={80} step={5} className="w-20" />
              <span className="text-xs w-6">{brushSize}</span>
            </div>
          )}

          <div className="w-px h-6 bg-border" />

          {/* Active annotation class */}
          <Select value={activeClass} onValueChange={cls => {
            setActiveClass(cls)
            // If an annotation is selected, also change its class
            if (selectedAnnotation !== null) {
              const clsIdx = localClasses.indexOf(cls)
              const next = annotations.map((ann, i) =>
                i === selectedAnnotation ? { ...ann, class_name: cls, class_id: clsIdx } : ann
              )
              setAnnotations(next)
              saveToHistory(next)
            }
          }}>
            <SelectTrigger className="w-36 h-8 text-xs">
              <SelectValue placeholder="类别" />
            </SelectTrigger>
            <SelectContent>
              {localClasses.map((cls, i) => (
                <SelectItem key={cls} value={cls}>
                  <span className="flex items-center gap-2">
                    {i < 9 && <kbd className="text-[9px] px-1 py-0.5 bg-muted rounded font-mono">{i+1}</kbd>}
                    {cls}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Add new class button */}
          {addingClass ? (
            <form className="flex items-center gap-1" onSubmit={e => { e.preventDefault(); submitNewClass() }}>
              <Input
                autoFocus
                value={newClassName}
                onChange={e => setNewClassName(e.target.value)}
                placeholder="类别名称"
                className="h-8 w-24 text-xs bg-background border-border"
                onKeyDown={e => { if (e.key === 'Escape') { setAddingClass(false); setNewClassName('') } }}
              />
              <Button type="submit" size="sm" variant="outline" className="h-8 px-2 text-xs" disabled={!newClassName.trim()}>
                添加
              </Button>
              <Button type="button" size="sm" variant="ghost" className="h-8 px-1" onClick={() => { setAddingClass(false); setNewClassName('') }}>
                <X className="w-3 h-3" />
              </Button>
            </form>
          ) : (
            <Button size="sm" variant="outline" className="h-8 px-2" title="新增类别" onClick={() => setAddingClass(true)}>
              <Plus className="w-3.5 h-3.5" />
            </Button>
          )}

          <div className="w-px h-6 bg-border" />

          {/* Split filter */}
          {availableSplits.length > 0 && (
            <Select value={selectedSplit} onValueChange={setSelectedSplit}>
              <SelectTrigger className="w-32 h-8 text-xs">
                <FolderOpen className="w-3 h-3 mr-1" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部划分</SelectItem>
                {availableSplits.map(s => (
                  <SelectItem key={s} value={s}>{s} ({allImages.filter(i => i.split === s).length})</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          {/* Class filter */}
          {availableClasses.length > 0 && (
            <Select value={selectedClass} onValueChange={setSelectedClass}>
              <SelectTrigger className="w-32 h-8 text-xs">
                <Tag className="w-3 h-3 mr-1" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部类别</SelectItem>
                {availableClasses.map(c => (
                  <SelectItem key={c} value={c}>{c}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          <div className="flex-1" />

          {/* Keyboard shortcut hint */}
          <Button size="sm" variant="ghost" onClick={() => setShowShortcutHint(h => !h)} title="键盘快捷键">
            <Keyboard className="w-4 h-4" />
          </Button>

          <span className="text-[10px] text-muted-foreground hidden xl:inline">滚轮：缩放 · 空格 + 拖拽：平移</span>
          {/* Undo + zoom */}
          <Button size="sm" variant="outline" onClick={undo} disabled={historyIndex < 0} title="撤销 (Ctrl+Z)">
            <Undo2 className="w-4 h-4" />
          </Button>
          <div className="w-px h-6 bg-border" />
          <Button size="sm" variant="outline" onClick={() => zoomAt(viewportRef.current.zoom - 0.1)}><ZoomOut className="w-4 h-4" /></Button>
          <span className="text-[10px] font-mono w-12 text-center text-muted-foreground">{Math.round(zoom * 100)}%</span>
          <Button size="sm" variant="outline" onClick={() => zoomAt(viewportRef.current.zoom + 0.1)}><ZoomIn className="w-4 h-4" /></Button>
          <Button size="sm" variant="outline" onClick={resetViewport} title="重置缩放与平移 (Ctrl+0)"><RotateCcw className="w-4 h-4" /></Button>

          <div className="w-px h-6 bg-border" />

          {/* Navigation */}
          <Button size="sm" variant="outline" onClick={() => navigateImage('prev')} disabled={currentIndex === 0}><ChevronLeft className="w-4 h-4" /></Button>
          <span className="text-[10px] font-mono text-muted-foreground w-20 text-center">{currentIndex + 1} / {filteredImages.length}</span>
          <Button size="sm" variant="outline" onClick={() => navigateImage('next')} disabled={currentIndex >= filteredImages.length - 1}><ChevronRight className="w-4 h-4" /></Button>

          {/* Jobs indicator — clicking opens the Jobs sidebar tab */}
          {totalJobCount > 0 && (
            <>
              <div className="w-px h-6 bg-border" />
              <Button
                size="sm"
                variant={sidebarTab === 'jobs' ? 'default' : 'outline'}
                onClick={() => setSidebarTab('jobs')}
                className="gap-1.5 text-xs font-mono"
                title="查看批量任务"
              >
                {runningJobCount > 0 && <span className="live-dot text-primary" />}
                任务
                {runningJobCount > 0 && (
                  <span className="px-1.5 py-0 bg-primary/15 text-primary text-[9px] rounded font-mono leading-4">
                    {runningJobCount}
                  </span>
                )}
              </Button>
            </>
          )}
        </div>

        {/* Keyboard shortcut overlay */}
        {showShortcutHint && (
          <div className="absolute top-16 right-72 z-50 bg-card border border-border rounded-lg shadow-xl p-4 text-xs w-64">
            <div className="font-semibold mb-2 flex items-center justify-between">
              键盘快捷键
              <button onClick={() => setShowShortcutHint(false)} className="text-muted-foreground hover:text-foreground">✕</button>
            </div>
            <div className="space-y-1 text-muted-foreground">
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <span><kbd className="px-1 bg-muted rounded">V</kbd> 选择</span>
                <span><kbd className="px-1 bg-muted rounded">B</kbd> 矩形框</span>
                <span><kbd className="px-1 bg-muted rounded">P</kbd> 多边形</span>
                <span><kbd className="px-1 bg-muted rounded">K</kbd> 关键点</span>
                <span><kbd className="px-1 bg-muted rounded">R</kbd> 画笔</span>
                <span><kbd className="px-1 bg-muted rounded">Q</kbd> SAM</span>
                <span><kbd className="px-1 bg-muted rounded">S</kbd> 保存并下一张</span>
                <span><kbd className="px-1 bg-muted rounded">Del</kbd> 删除</span>
                <span><kbd className="px-1 bg-muted rounded">Enter</kbd> 完成</span>
                <span><kbd className="px-1 bg-muted rounded">Ctrl+Z</kbd> 撤销</span>
                <span><kbd className="px-1 bg-muted rounded">Esc</kbd> 取消</span>
                <span><kbd className="px-1 bg-muted rounded">1-9</kbd> 类别</span>
              </div>
              <div className="mt-2 pt-2 border-t border-border">
                按 <kbd className="px-1 bg-muted rounded">1</kbd>–<kbd className="px-1 bg-muted rounded">9</kbd> 可直接指定前 9 个类别。
              </div>
            </div>
          </div>
        )}

        {/* ── Annotation canvas view ──────────────────────────────────────── */}
        <>

        {/* Viewport translation changes display only; canvas coordinates remain image-relative. */}
        <div ref={containerRef} className="flex-1 min-h-0 overflow-hidden overscroll-contain bg-muted/30 relative"
          aria-label="标注画布视口"
          style={{ cursor: isPanning ? 'grabbing' : spacePressed ? 'grab' : undefined }}
          onPointerDownCapture={handlePanStart} onPointerMoveCapture={handlePanMove}
          onPointerUpCapture={handlePanEnd} onPointerCancel={handlePanEnd} onLostPointerCapture={stopPan}>
          <div className="min-h-full flex items-center justify-center p-4">
          {isImageLoading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-10 h-10 animate-spin text-primary" />
              <p className="text-sm font-medium">正在加载图片…</p>
            </div>
          ) : error && !imageRef.current ? (
            <div className="flex flex-col items-center gap-3 text-destructive">
              <AlertCircle className="w-10 h-10" />
              <p className="text-sm font-medium">{error}</p>
              <p className="text-xs text-muted-foreground">图片文件无法加载 — 请检查数据集路径</p>
            </div>
          ) : currentImage ? (
            <>
              <canvas
                ref={canvasRef}
                style={{ position: 'absolute', left: '50%', top: '50%', maxWidth: 'none',
                  transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px)`,
                  cursor: isPanning ? 'grabbing' : spacePressed ? 'grab' : undefined }}
                className={cn(
                  "shadow-lg rounded",
                  activeTool !== 'select' && "cursor-crosshair",
                  activeTool === 'select' && "cursor-pointer",
                  activeTool === 'brush' && "cursor-cell",
                  activeTool === 'sam' && "cursor-cell"
                )}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseUp}
                onDoubleClick={() => !spaceRef.current && activeTool === 'polygon' && polygonPoints.length >= 3 && completePolygon()}
              />
              {activeTool === 'polygon' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-amber-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  多边形模式 — 点击添加顶点，双击或 Enter 完成
                </div>
              )}
              {activeTool === 'keypoint' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-purple-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  关键点模式 — 点击放置关键点（已放置 {keypointList.length} 个）。按 Enter 提交。
                </div>
              )}
              {activeTool === 'brush' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  画笔模式 — 涂抹掩码。松开鼠标提交多边形。
                </div>
              )}
              {activeTool === 'sam' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-green-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  SAM 模式 — 点击目标进行分割
                  {samPendingClicks.length > 0 && ` · ${samPendingClicks.length} 个待处理`}
                </div>
              )}
              {keypointList.length > 0 && activeTool === 'keypoint' && (
                <div className="absolute bottom-16 right-2 flex gap-2">
                  <Button size="sm" onClick={commitKeypoints} className="bg-purple-600 hover:bg-purple-700 text-white">
                    提交 {keypointList.length} 个关键点
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setKeypointList([])}>清除</Button>
                </div>
              )}
            </>
          ) : (
            <div className="text-center space-y-2">
              <p className="text-muted-foreground text-sm">未加载图片</p>
              {filteredImages.length > 0 && (
                <button
                  onClick={() => loadImageFile(filteredImages[0])}
                  className="text-xs text-primary underline underline-offset-2"
                >
                  加载第一张图片
                </button>
              )}
            </div>
          )}
          </div>{/* end inner centering wrapper */}
        </div>

        {/* Bottom bar */}
        <div className="flex items-center justify-between px-3 py-2 border-t border-border bg-card gap-3 shrink-0">
          <div className="flex items-center gap-2">
            <Button size="sm" variant="destructive" onClick={deleteAnnotation} disabled={selectedAnnotation === null}
              className="h-7 text-xs gap-1">
              <Trash2 className="w-3 h-3" strokeWidth={1.5} /> Delete
            </Button>
            <span className="text-[10px] text-muted-foreground/60 hidden lg:flex items-center gap-1.5">
              <kbd className="px-1 py-0.5 bg-muted rounded font-mono text-[9px]">S</kbd> 保存并下一张
              <kbd className="px-1 py-0.5 bg-muted rounded font-mono text-[9px]">Del</kbd> 删除
              <kbd className="px-1 py-0.5 bg-muted rounded font-mono text-[9px]">1–9</kbd> Class
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground/60 font-mono truncate max-w-[180px]">
            {currentImage?.filename}
          </span>
          <div className="flex items-center gap-1.5">
            <Button size="sm" variant="outline" onClick={() => saveAnnotations(false)} disabled={isSaving}
              className="h-7 text-xs gap-1 border-border">
              {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" strokeWidth={1.5} />}
              保存
            </Button>
            <Button size="sm" onClick={() => saveAnnotations(true)}
              disabled={isSaving || currentIndex >= filteredImages.length - 1}
              className="h-7 text-xs gap-1">
              {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" strokeWidth={1.5} />}
              保存并下一张
            </Button>
          </div>
        </div>
        </>
      </div>

      {/* ── Right panel ─────────────────────────────────────────────────── */}
      <div className="w-64 border-l border-border bg-card flex flex-col shrink-0">

        {/* Tab switcher */}
        <div className="flex border-b border-border shrink-0">
          <button
            onClick={() => setSidebarTab('annotate')}
            className={cn(
              'flex-1 py-2.5 text-[10px] font-mono font-semibold uppercase tracking-wider transition-colors',
              sidebarTab === 'annotate'
                ? 'text-primary border-b-2 border-primary bg-primary/5'
                : 'text-muted-foreground/60 hover:text-foreground'
            )}
          >
            标注
          </button>
          <button
            onClick={() => setSidebarTab('jobs')}
            className={cn(
              'flex-1 py-2.5 text-[10px] font-mono font-semibold uppercase tracking-wider transition-colors flex items-center justify-center gap-1.5',
              sidebarTab === 'jobs'
                ? 'text-primary border-b-2 border-primary bg-primary/5'
                : 'text-muted-foreground/60 hover:text-foreground'
            )}
          >
            {runningJobCount > 0 && <span className="live-dot" />}
            任务
            {totalJobCount > 0 && (
              <span className={cn(
                'px-1.5 py-px rounded font-mono text-[9px] leading-none',
                runningJobCount > 0 ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
              )}>
                {totalJobCount}
              </span>
            )}
          </button>
          <button onClick={() => openReview()} className="flex-1 py-2.5 text-[10px] font-mono font-semibold tracking-wider text-muted-foreground hover:text-primary">
            待复核 {reviewPending > 0 && <span className="text-primary">{reviewPending}</span>}
          </button>
        </div>

        {/* ── Annotate tab ─────────────────────────────────────────────── */}
        {sidebarTab === 'annotate' && <>

        {/* Model + confidence */}
        <div className="p-3 space-y-3 border-b border-border">
          <Select
            value={selectedModel}
            onValueChange={v => {
              const m = models.find(m => m.id === v)
              if (m && !m.downloaded && downloadingModelId !== m.id) downloadModel(m)
              else if (m?.downloaded) {
                setSelectedModel(v)
                if (m.type === 'vlx' || m.type === 'vlm') {
                  setTargetLabel(prev => localClasses.includes(prev)
                    ? prev
                    : (localClasses.includes(activeClass) ? activeClass : (localClasses[0] || '')))
                }
              }
            }}
          >
            <SelectTrigger className="h-8 text-xs bg-background border-border">
              {downloadingModelId
                ? <span className="flex items-center gap-1.5 text-muted-foreground">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    下载中…
                  </span>
                : <SelectValue placeholder="选择模型…" />
              }
            </SelectTrigger>
            <SelectContent>
              {/* Radix 约束：SelectLabel 必须放在 SelectGroup 内。 */}
              {vlmModels.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">
                    大模型（自动标注）
                  </SelectLabel>
                  {vlmModels.map(m => (
                    <SelectItem
                      key={m.id}
                      value={m.id}
                      className="group data-[highlighted]:bg-primary data-[highlighted]:text-primary-foreground data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
                    >
                      <div className="flex items-center gap-2 w-full">
                        <Bot className="w-3 h-3 shrink-0 text-primary group-data-[highlighted]:text-primary-foreground group-data-[state=checked]:text-primary-foreground" />
                        <span className="flex-1 truncate">{m.name}</span>
                        <CheckCircle2 className="w-3 h-3 text-primary group-data-[highlighted]:text-primary-foreground group-data-[state=checked]:text-primary-foreground shrink-0" />
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {vlmModels.length > 0 && <SelectSeparator />}
              <SelectGroup>
                {vlmModels.length > 0 && (
                  <SelectLabel className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    本地模型
                  </SelectLabel>
                )}
                {localModels.map(m => (
                  <SelectItem key={m.id} value={m.id} disabled={!m.downloaded && downloadingModelId !== m.id && downloadingModelId !== null}>
                    <div className="flex items-center gap-2 w-full">
                      <span className="flex-1 truncate">{m.name}</span>
                      {downloadingModelId === m.id
                        ? <span className="text-[9px] text-primary font-mono shrink-0 flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" />下载中</span>
                        : m.downloaded
                          ? <CheckCircle2 className="w-3 h-3 text-primary shrink-0" />
                          : <Download className="w-3 h-3 text-muted-foreground shrink-0" />
                      }
                    </div>
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>

          {onOpenVlmConfig && (
            <button
              type="button"
              onClick={onOpenVlmConfig}
              className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md border border-dashed border-border text-[10px] text-muted-foreground hover:text-primary hover:border-primary/40 transition-colors"
            >
              <Wand2 className="w-3 h-3 shrink-0" strokeWidth={1.5} />
              <span className="truncate">接入大模型（Qwen-VL / GPT-4o / vLLM）做自动标注</span>
              <ChevronRight className="w-3 h-3 shrink-0 ml-auto" />
            </button>
          )}

          {showConfidence && <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">置信度</label>
              <span className="text-xs font-mono text-primary">{Math.round(confidence * 100)}%</span>
            </div>
            <Slider value={[confidence]} onValueChange={([v]) => set置信度(v)} min={0.1} max={1} step={0.05} />
          </div>}

        </div>

        {/* ── 双模型交叉校验（一个标注 + 一个校验）──────────────────────── */}
        <div className="p-3 space-y-2 border-b border-border">
          <AutoAnnotateToggle label="双模型交叉校验" enabled={dualEnabled}
            disabled={isAutoAnnotating} onToggle={() => { setDualEnabled(v => !v); setDualResult(null) }} />

          {dualEnabled && (
            <>
              <Select value={verifyModel} onValueChange={v => { setVerifyModel(v); setDualResult(null) }}>
                <SelectTrigger className="h-8 text-xs bg-background border-border">
                  <SelectValue placeholder="选择校验模型…" />
                </SelectTrigger>
                <SelectContent>
                  {vlmModels.length > 0 && (
                    <SelectGroup>
                      <SelectLabel className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">
                        大模型（需连服务器，默认不选）
                      </SelectLabel>
                      {vlmModels.map(m => (
                        <SelectItem
                          key={m.id}
                          value={m.id}
                          className="group data-[highlighted]:bg-primary data-[highlighted]:text-primary-foreground data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
                        >
                          <div className="flex items-center gap-2 w-full">
                              <Bot className="w-3 h-3 shrink-0 text-primary group-data-[highlighted]:text-primary-foreground group-data-[state=checked]:text-primary-foreground" />
                            <span className="flex-1 truncate">{m.name}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                  {vlmModels.length > 0 && <SelectSeparator />}
                  <SelectGroup>
                    {vlmModels.length > 0 && (
                      <SelectLabel className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        本地模型
                      </SelectLabel>
                    )}
                    {localModels.map(m => (
                      <SelectItem key={m.id} value={m.id}>
                        <div className="flex items-center gap-2 w-full">
                          <span className="flex-1 truncate">{m.name}</span>
                          {m.downloaded
                            ? <CheckCircle2 className="w-3 h-3 text-primary shrink-0" />
                            : <Download className="w-3 h-3 text-muted-foreground shrink-0" />}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>

              {verifyModel && verifyModel === selectedModel && (
                <p className="text-[10px] text-amber-700 dark:text-yellow-300 leading-relaxed">
                  校验模型和标注模型是同一个，等于没校验 —— 请换一个。
                </p>
              )}

              <p className="text-[10px] text-muted-foreground leading-relaxed">
                两个模型各跑一次全图；只有<strong className="text-foreground font-medium">单个模型命中的框</strong>
                会被裁出来交给对方复核。互相印证 → 采纳，单方且被否认 → 待复核。
              </p>

              {dualResult && (
                <div className="flex items-center gap-2 text-[10px] font-mono">
                  <span className="text-emerald-700 dark:text-green-400">采纳 {dualResult.accept}</span>
                  <span className="text-amber-700 dark:text-yellow-300">待复核 {dualResult.review}</span>
                  {dualResult.teachers_down.length > 0 && (
                    <span className="text-red-600 dark:text-red-400 truncate">
                      {dualResult.teachers_down.join('、')} 不可用
                    </span>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        <div className="p-3 space-y-2 border-b border-border">
          <AutoAnnotateToggle label="Tile Pass · 小目标切片二次检测" enabled={tileEnabled}
            disabled={isAutoAnnotating} onToggle={() => setTileEnabled(v => !v)} />
          {tileEnabled && <>
            <div className="flex gap-2 items-center text-xs">
              <label>尺寸</label>
              <select aria-label="切片尺寸" value={tileSize} disabled={isAutoAnnotating} onChange={e => { setTileSize(Number(e.target.value)); setTileOverlap(v => Math.min(v,Number(e.target.value)/2-1)) }} className="bg-background border border-border rounded">
                {[512,640,768,1024].map(v => <option key={v} value={v}>{v}px</option>)}
              </select>
              <label>重叠</label>
              <Input aria-label="切片重叠像素" type="number" value={tileOverlap} min={0} max={tileSize/2-1} step={16} disabled={isAutoAnnotating} onChange={e => setTileOverlap(Math.max(0,Math.min(tileSize/2-1,Number(e.target.value))))} className="h-6 w-20 text-xs" />
            </div>
            <p className="text-[10px] text-muted-foreground">保留全图结果，原分辨率覆盖整图。新增框需校验模型确认；未确认或仅切片边缘的框进入复核。</p>
            {tilePreview && <div className="space-y-2 text-xs">
              <p>{tilePreview.phase === 'verify' ? '候选框校验' : tilePreview.phase === 'full' ? '全图检测' : '切片检测'} · {tilePreview.progress?.completed || 0}/{tilePreview.progress?.total || 0}</p>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => controlTilePreview(tilePreview.status === 'paused' ? 'resume' : 'pause')}>{tilePreview.status === 'paused' ? '继续' : '暂停'}</Button>
                <Button size="sm" variant="outline" onClick={() => controlTilePreview('cancel')}>取消</Button>
              </div>
            </div>}
            {tileSummary && <p className="text-[10px] text-muted-foreground">完成 {tileSummary.completed}/{tileSummary.total} 块 · {tileSummary.coverage_complete ? '覆盖完成' : '覆盖不完整，请重试或检查模型'} · 失败 {tileSummary.failed || 0}</p>}
          </>}
        </div>

        {/* Text prompt section — shown for SAM3, YOLO-World, and GroundingDINO */}
        {(() => {
          const selModel = models.find(m => m.id === selectedModel)
          const isTextModel = selModel && ['sam3', 'yoloworld', 'groundingdino', 'owlvit', 'vlx', 'vlm'].includes(selModel.type)
          if (!isTextModel) return null
          const isTargetModel = selModel.type === 'vlx' || selModel.type === 'vlm'
          // 这个 label 的样式带 uppercase，写 "VLX-Seek" 会被显示成 "VLX-SEEK"，
          // 所以产品名不放在这里（下拉框里已经写明是哪个模型）。
          const sectionLabel = selModel.type === 'sam3'
            ? '文本分割'
            : isTargetModel ? '检测目标' : '文本检测 (Zero-Shot)'
          return (
            <div className="p-3 space-y-2.5 border-b border-border">
              {/* Section label */}
              <div className="flex items-center justify-between">
                <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">
                  {sectionLabel}
                </p>
                {totalJobCount > 0 && (
                  <button onClick={() => setSidebarTab('jobs')}
                    className="text-[9px] font-mono text-muted-foreground hover:text-primary flex items-center gap-1.5 transition-colors">
                    {runningJobCount > 0 && <span className="live-dot text-primary" />}
                    {totalJobCount} 个任务
                  </button>
                )}
              </div>

              {/* ── VLX：类别 / 目标描述 / 高级约束 三者分离 ────────────────── */}
              {isTargetModel ? (
                <div className="space-y-2.5">
                  {/* 类别：来自当前数据集 data.yaml，是唯一会写进 data.yaml 的字段 */}
                  <div>
                    <label className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">类别</label>
                    <div className="flex gap-1 mt-1">
                      <select
                        value={targetLabel}
                        onChange={e => setTargetLabel(e.target.value)}
                        className="flex-1 h-7 text-xs rounded border border-border bg-background text-foreground px-1.5"
                      >
                        <option value="">
                          {targetClasses.length
                            ? '选择类别…'
                            : '（数据集暂无类别）'}
                        </option>
                        {targetClasses.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <Button size="sm" variant="outline"
                        className="h-7 w-7 p-0 shrink-0 border-border"
                        title="新增类别"
                        onClick={() => setAddingTargetClass(v => !v)}>
                        <Plus className="w-3 h-3" />
                      </Button>
                    </div>
                    {addingTargetClass && (
                      <div className="flex gap-1 mt-1">
                        <Input
                          value={newTargetClass}
                          onChange={e => setNewTargetClass(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter' && newTargetClass.trim()) {
                              e.preventDefault()
                              const n = newTargetClass.trim()
                              setNewTargetClass(''); setAddingTargetClass(false); setTargetLabel(n)
                              submitNewClass(n)
                            }
                          }}
                          placeholder="新类别名（如 swimmer）"
                          className="h-7 text-xs flex-1 bg-background border-border placeholder:text-[11px] placeholder:text-muted-foreground/50"
                        />
                        <Button size="sm" variant="outline"
                          className="h-7 px-2 text-[10px] border-border"
                          disabled={!newTargetClass.trim()}
                          onClick={() => {
                            const n = newTargetClass.trim()
                            setNewTargetClass(''); setAddingTargetClass(false); setTargetLabel(n)
                            submitNewClass(n)
                          }}>
                          添加
                        </Button>
                      </div>
                    )}
                  </div>

                  {/* 目标描述：只给模型做语义理解，绝不写进 data.yaml */}
                  <div>
                    <label className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">目标描述</label>
                    <Textarea
                      value={targetDescription}
                      onChange={e => setTargetDescription(e.target.value)}
                      placeholder="寻找水面中正在游泳的人员"
                      className="mt-1 min-h-[96px] text-xs resize-y bg-background border-border placeholder:text-[11px] placeholder:text-muted-foreground/50 leading-relaxed"
                    />
                  </div>

                  {/* 常用描述：实测单字「人」会把船认成人，必须给完整短句 */}
                  <div className="flex flex-wrap gap-1">
                    {PROMPT_PRESETS.map(p => (
                      <button key={p} type="button" onClick={() => setTargetDescription(p)}
                        className="px-1.5 py-0.5 rounded-full border border-border text-[10px] text-muted-foreground hover:text-primary hover:border-primary/40 transition-colors">
                        {p}
                      </button>
                    ))}
                  </div>

                  {/* 高级约束：默认折叠，避免把侧栏撑得太长 */}
                  <div>
                    <button type="button" onClick={() => setShowExtraPrompt(v => !v)}
                      className="text-[10px] text-muted-foreground hover:text-primary flex items-center gap-1 transition-colors">
                      {showExtraPrompt ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                      高级约束
                    </button>
                    {showExtraPrompt && (
                      <Textarea
                        value={targetExtra}
                        onChange={e => setTargetExtra(e.target.value)}
                        placeholder="排除岸边、船上和码头人员"
                        className="mt-1 min-h-[60px] text-xs resize-y bg-background border-border placeholder:text-[11px] placeholder:text-muted-foreground/50 leading-relaxed"
                      />
                    )}
                  </div>

                  <p className="text-[10px] text-muted-foreground leading-relaxed">
                    类别来自当前数据集并限定本次结果；目标描述与高级约束只用于语义理解，不会合并成类别名。
                  </p>
                </div>
              ) : (
                <>
                  {/* Tag chips */}
                  {promptTags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {promptTags.map(tag => (
                        <span key={tag}
                          className="flex items-center gap-0.5 pl-2 pr-1 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded-full text-[10px] font-medium">
                          {tag}
                          <button onClick={() => removePromptTag(tag)}
                            className="hover:text-primary/60 transition-colors ml-0.5">
                            <X className="w-2.5 h-2.5" />
                          </button>
                        </span>
                      ))}
                      <button onClick={() => setPromptTags([])}
                        className="text-[10px] text-muted-foreground/60 hover:text-muted-foreground px-1 transition-colors">
                        clear
                      </button>
                    </div>
                  )}

                  {/* Label input */}
                  <div className="flex gap-1">
                    <Input
                      value={promptInput}
                      onChange={e => setPromptInput(e.target.value)}
                      onKeyDown={e => {
                        if ((e.key === 'Enter' || e.key === ',') && promptInput.trim()) {
                          e.preventDefault(); addPromptTag(promptInput)
                        }
                      }}
                      placeholder={promptTags.length ? '添加标签…' : 'car, person, dog…'}
                      className="h-7 text-xs flex-1 bg-background border-border placeholder:text-[11px] placeholder:text-muted-foreground/50"
                    />
                    <Button size="sm" variant="outline"
                      className="h-7 w-7 p-0 shrink-0 border-border"
                      onClick={() => promptInput.trim() && addPromptTag(promptInput)}
                      disabled={!promptInput.trim()}>
                      <Plus className="w-3 h-3" />
                    </Button>
                  </div>
                  <p className="text-[10px] text-muted-foreground leading-relaxed">
                    按 Enter 或 <kbd className="px-1 bg-muted rounded text-[9px]">,</kbd> 添加标签。
                  </p>
                </>
              )}

              {/* Annotate this image */}
              <Button size="sm" variant="outline"
                className="w-full h-7 text-xs gap-1.5 border-border hover:border-primary/40 hover:text-primary"
                onClick={textAnnotateSingle}
                disabled={(isTargetModel
                  ? !targetLabel.trim() || !targetDescription.trim()
                  : promptTags.length === 0) || isAutoAnnotating}>
                {isAutoAnnotating
                  ? <Loader2 className="w-3 h-3 animate-spin" />
                  : <Wand2 className="w-3 h-3" strokeWidth={1.5} />
                }
                标注当前图片
              </Button>

              {/* VLX、Qwen 与豆包统一走同一个单图 / 批量目标标注流程。 */}
              <Button size="sm"
                className="w-full h-8 text-xs gap-1.5 font-semibold bg-primary text-primary-foreground hover:bg-primary/90"
                onClick={textAnnotateBatch}
                disabled={(isTargetModel
                  ? !targetLabel.trim() || !targetDescription.trim()
                  : promptTags.length === 0) || isAutoAnnotating || isStartingBatch}>
                {isStartingBatch
                  ? <><Loader2 className="w-3 h-3 animate-spin" /> 启动中…</>
                  : <><ChevronsRight className="w-3.5 h-3.5" /> 批量标注数据集</>
                }
              </Button>
            </div>
          )
        })()}

        {/* 目标型云端模型使用上面的统一单张 / 批量入口，不显示无目标配置的通用入口。 */}
        {!isTargetModel && (
          <div className="p-3 border-b border-border">
            <Button
              className="w-full h-8 text-xs gap-1.5 font-medium"
              onClick={autoAnnotate}
              disabled={!selectedModel || isAutoAnnotating || runningJobCount > 0}
            >
              {isModelLoading
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> 正在加载模型…</>
                : isAutoAnnotating
                  ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> 标注中…</>
                  : <><Wand2 className="w-3.5 h-3.5" strokeWidth={1.5} /> 自动标注</>
              }
            </Button>
          </div>
        )}

        {/* Fixed-class batch annotate — YOLO, RT-DETR, RF-DETR (no text prompt needed) */}
        {(() => {
          const selModel = models.find(m => m.id === selectedModel)
          // vlm/vlx 由上面的目标标注面板处理，不算"固定类别"，
          // 否则会冒出一个点进去必然失败的「批量标注数据集」按钮。
          const isFixedClass = selModel && !['sam3', 'yoloworld', 'groundingdino', 'owlvit', 'vlm', 'vlx'].includes(selModel.type)
          if (!isFixedClass || !selModel) return null
          return (
            <div className="p-3 space-y-2.5 border-b border-border">
              <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">
                批量标注
              </p>
              <Button size="sm"
                className="w-full h-8 text-xs gap-1.5 font-semibold bg-primary text-primary-foreground hover:bg-primary/90"
                onClick={textAnnotateBatch}
                disabled={isAutoAnnotating || isStartingBatch}>
                {isStartingBatch
                  ? <><Loader2 className="w-3 h-3 animate-spin" /> 启动中…</>
                  : <><ChevronsRight className="w-3.5 h-3.5" /> 批量标注数据集</>
                }
              </Button>
            </div>
          )
        })()}

        {/* Annotations list */}
        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 flex items-center justify-between border-b border-border shrink-0">
            <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground/60">
              标注列表
            </p>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-primary">{annotations.length}</span>
              {selectedDataset && (
                <button
                  onClick={deleteAllAnnotations}
                  className="text-[9px] text-destructive/60 hover:text-destructive transition-colors"
                  title="删除数据集中的全部标注"
                >
                  全部清除
                </button>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {annotations.length === 0 ? (
              <div className="py-6 text-center">
                <p className="text-xs text-muted-foreground/60">暂无标注。</p>
                <p className="text-[10px] text-muted-foreground/40 mt-0.5">使用上方工具添加标注。</p>
              </div>
            ) : (
              annotations.map((ann, idx) => (
                <div key={idx}
                  onClick={() => setSelectedAnnotation(idx === selectedAnnotation ? null : idx)}
                  className={cn(
                    'px-2.5 py-1.5 rounded-md mb-0.5 cursor-pointer transition-all text-xs',
                    'border',
                    selectedAnnotation === idx
                      ? 'bg-primary/10 border-primary/30 text-primary'
                      : 'border-transparent hover:bg-muted hover:border-border text-foreground'
                  )}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium truncate">{ann.class_name}</span>
                    <span className="text-[10px] text-muted-foreground capitalize shrink-0">{ann.type}</span>
                  </div>
                  {selectedAnnotation === idx && (
                    <div className="mt-1.5 flex items-center gap-1" onClick={e => e.stopPropagation()}>
                      <select
                        value={ann.class_name}
                        onChange={e => {
                          const cls = e.target.value
                          const clsIdx = localClasses.indexOf(cls)
                          const next = annotations.map((a, i) =>
                            i === idx ? { ...a, class_name: cls, class_id: clsIdx } : a
                          )
                          setAnnotations(next); saveToHistory(next); setActiveClass(cls)
                        }}
                        className="flex-1 h-6 text-[10px] rounded border border-border bg-background text-foreground px-1"
                      >
                        {localClasses.map(c => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                      <button
                        onClick={() => {
                          const next = annotations.filter((_, i) => i !== idx)
                          setAnnotations(next); saveToHistory(next); setSelectedAnnotation(null)
                        }}
                        className="p-1 rounded hover:bg-destructive/10 hover:text-destructive transition-colors"
                        title="删除该标注"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {tileReview && tileReview.datasetId === selectedDataset?.id && tileReview.items.length > 0 && (
          <div className="border-t border-border p-3 space-y-2">
            <Button size="sm" className="w-full" onClick={() => openReview(tileReview.jobId)}>开始复核单图候选</Button>
            <p className="text-xs text-muted-foreground">候选已保存，可在「待复核」中继续处理。</p>
          </div>
        )}
        {/* Copy forward */}
        <div className="border-t border-border p-3 space-y-2 shrink-0">
          <button
            onClick={() => setShowBatchPanel(b => !b)}
            className="w-full flex items-center justify-between text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground/50 hover:text-foreground transition-colors"
          >
            <span>向后复制</span>
            <span className="text-[8px]">{showBatchPanel ? '▲' : '▼'}</span>
          </button>
          {showBatchPanel && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Input type="number" value={copyToNext}
                  onChange={e => setCopyToNext(Math.max(1, +e.target.value))}
                  min={1} max={100}
                  className="h-6 w-14 text-xs bg-background border-border font-mono" />
                <span className="text-xs text-muted-foreground">张图片</span>
              </div>
              <Button size="sm" variant="outline"
                className="w-full h-7 text-xs gap-1 border-border"
                onClick={() => copyAnnotationsToNext(copyToNext)}
                disabled={annotations.length === 0}>
                <ChevronsRight className="w-3 h-3" strokeWidth={1.5} />
                复制标注到后续图片
              </Button>
            </div>
          )}
        </div>

        </> /* end sidebarTab === 'annotate' */}

        {/* ── Jobs tab ─────────────────────────────────────────────────── */}
        {sidebarTab === 'jobs' && (
          <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
            {totalJobCount === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 gap-2 text-muted-foreground p-4">
                <p className="text-[10px] font-mono uppercase tracking-widest opacity-30">暂无任务</p>
                <p className="text-xs text-center">在「标注」标签页中启动批量任务</p>
                <button
                  onClick={() => setSidebarTab('annotate')}
                  className="text-xs text-primary underline underline-offset-2 mt-1"
                >
                  前往「标注」
                </button>
              </div>
            ) : (
              <div className="p-2 space-y-2">
                {/* Summary row */}
                <div className="flex items-center gap-2 px-1 py-1.5 text-[9px] font-mono text-muted-foreground/50">
                  <span>{totalJobCount} 个任务</span>
                  {runningJobCount > 0 && (
                    <span className="flex items-center gap-1 text-primary">
                      <span className="live-dot" /> {runningJobCount} 个进行中
                    </span>
                  )}
                </div>
                {datasetJobs.slice().reverse().map(job => {
                  const processed = job.processed ?? 0
                  const isActive = job.status === 'running' || job.status === 'paused'
                  const isRunning = job.status === 'running'
                  return (
                    <div key={job.job_id}
                      className={cn(
                        'rounded-xl bg-background overflow-hidden',
                        isRunning ? 'gradient-border-active' : 'border border-border'
                      )}>

                      {(job.tile_pass || (job.review_summary?.total || 0) > 0) && <div className="px-3 pt-2 space-y-2 text-[10px] text-muted-foreground">
                        <p>{job.tile_pass ? `Tile Pass · ${job.tile_progress?.phase === 'verify' ? '校验' : '切片'} ${job.tile_progress?.completed || 0}/${job.tile_progress?.total || 0}` : '双模型交叉检验'} · 待复核 {job.review_summary?.pending ?? job.dual?.review ?? 0}</p>
                        {(job.review_summary?.total || job.dual?.review || 0) > 0 && <>
                          <Button size="sm" className="w-full h-7 text-xs" onClick={() => openReview(job.job_id)}>
                            {job.status === 'undone' ? '查看已撤销复核记录' : `开始复核 · ${job.review_summary?.pending ?? job.dual?.review ?? 0}`}
                          </Button>
                          {job.review_summary && <p>人工确认 {job.review_summary.accepted} · 已驳回 {job.review_summary.rejected}</p>}
                        </>}
                      </div>}
                      {/* Card header */}
                      <div className="flex items-start gap-2 px-3 pt-3 pb-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex flex-wrap gap-1 mb-1.5">
                            {job.text_prompt.split(',').map(t => t.trim()).filter(Boolean).map(tag => (
                              <span key={tag}
                                className="px-1.5 py-px bg-primary/8 text-primary border border-primary/15 rounded-full text-[9px] font-semibold">
                                {tag}
                              </span>
                            ))}
                          </div>
                          <p className="text-[8px] text-muted-foreground/40 font-mono">
                            {job.job_id} · {fmtElapsed(job.started_at)}
                          </p>
                        </div>
                        <span className={cn(
                          'shrink-0 flex items-center gap-1 text-[8px] px-1.5 py-0.5 rounded-full font-mono uppercase tracking-wider',
                          isRunning                      && 'bg-primary/8 text-primary',
                          job.status === 'paused'        && 'bg-warning/10 text-warning',
                          job.status === 'done'          && 'bg-success/10 text-success',
                          job.status === 'error'         && 'bg-destructive/10 text-destructive',
                          job.status === 'undone'        && 'bg-muted text-muted-foreground',
                          (job.status === 'cancelled' || job.status === 'interrupted') && 'bg-muted text-muted-foreground',
                        )}>
                          {isRunning && <span className="live-dot" />}
                          {job.status}
                        </span>
                      </div>

                      {/* Progress */}
                      <div className="px-3 pb-3 space-y-2">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-[8px] font-mono text-muted-foreground/50">
                            <span>{processed} / {job.total || '?'}</span>
                            <span>{job.progress}%</span>
                          </div>
                          <div className="relative h-[3px] bg-muted/60 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                'absolute inset-y-0 left-0 rounded-full transition-all duration-700',
                                job.status === 'done'        ? 'bg-success' :
                                job.status === 'error'       ? 'bg-destructive' :
                                job.status === 'cancelled'   ? 'bg-muted-foreground' :
                                job.status === 'interrupted' ? 'bg-muted-foreground' :
                                job.status === 'paused'      ? 'bg-warning' : 'bg-primary'
                              )}
                              style={{ width: `${job.progress}%` }}
                            >
                              {isRunning && <span className="shimmer-overlay" />}
                            </div>
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="grid grid-cols-4 gap-0.5 pt-1 border-t border-border/60">
                          {[
                            { label: '已完成',    value: processed,             color: '' },
                            { label: '已标注',   value: job.annotated,         color: 'text-success' },
                            { label: '标注数',  value: job.total_annotations, color: 'text-primary' },
                            { label: '失败',    value: job.failed,            color: job.failed > 0 ? 'text-destructive' : 'text-muted-foreground/30' },
                          ].map(({ label, value, color }) => (
                            <div key={label} className="text-center pt-1.5">
                              <div className={cn('text-xs font-mono font-bold tabular-nums leading-none', color || 'text-foreground')}>
                                {value ?? '–'}
                              </div>
                              <div className="text-[7px] font-mono text-muted-foreground/40 uppercase tracking-widest mt-1">{label}</div>
                            </div>
                          ))}
                        </div>

                        {/* Recent images preview */}
                        {(job.recent_images?.length ?? 0) > 0 && (
                          <div className="pt-1 border-t border-border/60">
                            <div className="flex items-center justify-between mb-1.5">
                              <p className="text-[8px] font-mono text-muted-foreground/40 uppercase tracking-wider">
                                最近 · {job.recent_images!.length} 张已标注
                              </p>
                              {job.status === 'done' && job.dataset_id === selectedDataset?.id && (
                                <button
                                  onClick={() => {
                                    const first = job.recent_images![job.recent_images!.length - 1]
                                    if (first) navigateToAnnotatedImage(first.image_id)
                                  }}
                                  className="text-[8px] font-mono text-primary hover:underline"
                                >
                                  查看标注 →
                                </button>
                              )}
                              {job.status === 'done' && job.dataset_id !== selectedDataset?.id && (
                                <span className="text-[8px] font-mono text-muted-foreground/40">
                                  切换到该数据集后查看
                                </span>
                              )}
                            </div>
                            <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin">
                              {job.recent_images!.slice().reverse().map((img, idx) => (
                                <button
                                  key={img.image_id ?? idx}
                                  className="shrink-0 relative group cursor-pointer"
                                  disabled={job.dataset_id !== selectedDataset?.id}
                                  onClick={() => navigateToAnnotatedImage(img.image_id)}
                                  title={`${img.filename} · ${img.annotation_count} 个标注`}
                                >
                                  <img
                                    src={`${apiUrl}/api/auto-annotate/text-batch/${job.job_id}/annotated/${encodeURIComponent(img.image_id)}`}
                                    alt={img.filename}
                                    className="w-[100px] h-14 object-contain bg-muted/40 rounded border border-border group-hover:border-primary/50 transition-colors"
                                    loading="lazy"
                                  />
                                  <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[7px] text-white font-mono text-center py-0.5 rounded-b opacity-0 group-hover:opacity-100 transition-opacity">
                                    {img.annotation_count} 个标注
                                  </div>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Controls */}
                        {isActive && (
                          <div className="flex items-center gap-1.5 pt-0.5">
                            {isRunning ? (
                              <button onClick={() => pauseJob(job.job_id)}
                                className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-warning/20 bg-warning/5 text-warning hover:bg-warning/10 transition-colors">
                                <Pause className="w-2 h-2" strokeWidth={2} /> 暂停
                              </button>
                            ) : (
                              <button onClick={() => resumeJob(job.job_id)}
                                className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-primary/20 bg-primary/5 text-primary hover:bg-primary/10 transition-colors">
                                <Play className="w-2 h-2" strokeWidth={2} /> 继续
                              </button>
                            )}
                            <button onClick={() => cancelJob(job.job_id)}
                              className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-destructive/20 bg-destructive/5 text-destructive hover:bg-destructive/10 transition-colors">
                              <StopCircle className="w-2 h-2" strokeWidth={2} /> 取消
                            </button>
                          </div>
                        )}

                        {job.status === 'done' && (
                          <div className="flex items-center gap-1.5 pt-0.5 border-t border-border/60">
                            <button
                              onClick={() => undoJob(job.job_id)}
                              disabled={job.dataset_id !== selectedDataset?.id}
                              className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-destructive/20 bg-destructive/5 text-destructive hover:bg-destructive/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                              title={job.dataset_id !== selectedDataset?.id ? '切换到该数据集后撤销' : '移除本次任务创建的全部标注'}
                            >
                              ↩ 撤销批量
                            </button>
                          </div>
                        )}

                        {job.status === 'error' && job.error && (
                          <div className="px-2 py-1.5 rounded bg-destructive/5 border border-destructive/15 text-[9px] font-mono text-destructive">
                            {job.error}
                          </div>
                        )}
                        {job.status === 'interrupted' && (
                          <div className="px-2 py-1.5 rounded bg-muted border border-border text-[9px] font-mono text-muted-foreground">
                            后端已重启 — 任务在 {job.progress}% 处中断
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

      </div>

      {error && (
        <div className="absolute bottom-4 left-4 right-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="w-4 h-4" />{error}
          <button onClick={() => setError(null)} className="ml-auto text-xs underline">忽略</button>
        </div>
      )}
      </div>
    </div>
  )
}
