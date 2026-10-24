'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Loader2, AlertTriangle, X, Database, Menu } from 'lucide-react'
import { Sidebar } from '@/components/sidebar'
import { DatasetsView } from '@/components/datasets-view'
import { DashboardView } from '@/components/dashboard-view'
import { GalleryView } from '@/components/gallery-view'
import { SortingView } from '@/components/sorting-view'
import { AnnotationView } from '@/components/annotation-view'
import { ConvertView } from '@/components/convert-view'
import { MergeView } from '@/components/merge-view'
import { ComputeNodesView } from '@/components/compute-nodes-view'
import { TrainingView } from '@/components/training-view'
import { NativeWorkflowGuide } from '@/components/native-workflow-guide'
import { useAssistantWorkflow } from '@/lib/assistant-workflow'
import { InferenceView } from '@/components/inference-view'
import { EvaluationView } from '@/components/evaluation-view'
import { ModelsView } from '@/components/models-view'
import { SettingsView } from '@/components/settings-view'
import { ClassManagementView } from '@/components/class-management-view'
import { VlmView } from '@/components/vlm-view'
import { LmmView } from '@/components/lmm-view'
import { AugmentationView } from '@/components/augmentation-view'
import { VideoExtractionView } from '@/components/video-extraction-view'
import { SplitView } from '@/components/split-view'
import { HealthView } from '@/components/health-view'
import { CompareView } from '@/components/compare-view'
import { SnapshotView } from '@/components/snapshot-view'
import { YamlWizardView } from '@/components/yaml-wizard-view'
import { DuplicateDetectionView } from '@/components/duplicate-detection-view'
import { BatchJobsView } from '@/components/batch-jobs-view'
import { useSettings } from '@/lib/settings-context'

export type ViewType = 'datasets' | 'dashboard' | 'gallery' | 'sorting' | 'annotate' | 'classes' | 'augmentation' | 'video-extraction' | 'split' | 'convert' | 'merge' | 'compute-nodes' | 'training' | 'inference' | 'evaluate' | 'models' | 'batch-jobs' | 'health' | 'compare' | 'snapshots' | 'yaml-wizard' | 'settings' | 'duplicate-detection' | 'vlm' | 'lmm'

export interface Dataset {
  id: string
  name: string
  path: string
  format: string
  task_type: string
  num_images: number
  num_annotations: number
  classes: string[]
  created_at: string
}

export interface ImageData {
  id: string
  filename: string
  path: string
  width?: number
  height?: number
  split?: string
  class_name?: string
  annotations: Annotation[]
  has_annotations: boolean
}

export interface Annotation {
  id?: string
  type: string
  class_id: number
  class_name: string
  bbox?: number[]
  x_center?: number
  y_center?: number
  width?: number
  height?: number
  points?: number[]
  confidence?: number
  confidence_source?: string
  decision?: string
  decision_reasons?: string[]
  evidence?: any
  normalized?: boolean
}

// Shared image cache so tab-switches don't reload
export type ImageCache = Record<string, ImageData[]>

// All valid view IDs — used to validate URL path segments
const VALID_VIEWS = new Set<ViewType>([
  'datasets', 'dashboard', 'gallery', 'sorting', 'annotate', 'classes',
  'augmentation', 'video-extraction', 'split', 'convert', 'merge',
  'compute-nodes', 'training', 'inference', 'evaluate', 'models', 'batch-jobs', 'health',
  'compare', 'snapshots', 'yaml-wizard', 'settings', 'duplicate-detection', 'vlm', 'lmm',
])

// Views that require a dataset to be selected before they're useful.
// Note: 'video-extraction' is intentionally excluded — it creates a new
// dataset and should be accessible without one already loaded.
const DATASET_REQUIRED_VIEWS = new Set<ViewType>([
  'dashboard', 'gallery', 'sorting', 'annotate', 'classes',
  'augmentation', 'split', 'health',
  'snapshots', 'yaml-wizard', 'duplicate-detection',
])

// 视图 id → 中文名称（移动端顶栏使用）
const VIEW_NAME_ZH: Record<ViewType, string> = {
  datasets: '数据集',
  dashboard: '数据看板',
  gallery: '图库',
  sorting: '筛选排序',
  annotate: '标注',
  classes: '类别',
  augmentation: '数据增强',
  'video-extraction': '视频抽帧',
  split: '训练 / 验证 / 测试',
  convert: '格式转换',
  merge: '合并',
  'compute-nodes': '算力节点',
  training: '模型训练',
  inference: '推理',
  evaluate: '评估',
  models: '模型',
  'batch-jobs': '批量任务',
  health: '健康检查',
  compare: '对比',
  snapshots: '快照',
  'yaml-wizard': 'YAML 配置',
  settings: '设置',
  'duplicate-detection': '重复检测',
  vlm: '大模型配置',
  lmm: '多模态目标检测',
}

// Inline dataset picker shown when a view needs a dataset but none is selected
function DatasetRequiredGuard({
  datasets,
  selectedDataset,
  onSelectDataset,
  children,
}: {
  datasets: Dataset[]
  selectedDataset: Dataset | null
  onSelectDataset: (d: Dataset) => void
  children: React.ReactNode
}) {
  if (selectedDataset) return <>{children}</>

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
      <div className="flex flex-col items-center gap-2 text-center">
        <Database className="w-10 h-10 text-muted-foreground/40 mb-1" />
        <h2 className="text-base font-semibold text-foreground">未选择数据集</h2>
        <p className="text-sm text-muted-foreground max-w-xs">
          {datasets.length === 0
            ? '尚未加载任何数据集。请到「数据集」上传图像、抽取视频帧，或载入已有格式的数据集。'
            : '请在下方选择一个数据集以继续。'}
        </p>
      </div>
      {datasets.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full max-w-2xl">
          {datasets.map((d) => (
            <button
              key={d.id}
              onClick={() => onSelectDataset(d)}
              className="text-left p-3.5 rounded-xl border border-border bg-card hover:bg-accent hover:border-primary/40 transition-all duration-150 group"
            >
              <p className="text-sm font-semibold text-foreground truncate group-hover:text-primary transition-colors">
                {d.name}
              </p>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="text-[10px] px-1.5 py-0.5 bg-muted text-muted-foreground rounded font-mono">
                  {d.format?.toUpperCase()}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {d.num_images?.toLocaleString()} 张图像
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Home() {
  const [activeView, setActiveViewState] = useState<ViewType>('datasets')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [imageCache, setImageCache] = useState<ImageCache>({})
  const [annotateInitialImageId, setAnnotateInitialImageId] = useState<string | null>(null)
  const [gpuStatus, setGpuStatus] = useState<{ state: string; message: string } | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isRunHiveEmbedded, setIsRunHiveEmbedded] = useState(false)
  const annotationBeforeLeave=useRef<()=>Promise<void>>(async()=>{})
  const [annotationAction,setAnnotationAction]=useState<{action:'jobs'|'review'|'annotate';version:number}|null>(null)

  useEffect(() => {
    setIsRunHiveEmbedded(
      window.self !== window.top || new URLSearchParams(window.location.search).get('rh_embed') === '1'
    )
  }, [])

  // Sync activeView from URL after hydration, and on back/forward navigation
  useEffect(() => {
    const syncFromUrl = () => {
      const segment = window.location.pathname.replace(/^\//, '') as ViewType
      if (VALID_VIEWS.has(segment)) {
        setActiveViewState(segment)
      }
    }
    syncFromUrl()
    window.addEventListener('popstate', syncFromUrl)
    return () => window.removeEventListener('popstate', syncFromUrl)
  }, [])

  // Navigate to a view — updates React state AND the browser URL without remounting
  const setActiveView = useCallback((view: ViewType) => {
    setActiveViewState(view)
    const params=new URLSearchParams(location.search)
    window.history.pushState(null, '', `/${view}${params.toString()?'?'+params:''}`)
  }, [])

  // apiUrl now comes from the shared settings context
  const { settings } = useSettings()
  const apiUrl = settings.apiUrl

  // Poll GPU/CUDA install status every 4 s — hide banner once state is "ready"
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>
    const check = async () => {
      try {
        const r = await fetch(`${apiUrl}/api/device-info`)
        if (r.ok) {
          const d = await r.json()
          const s = d.gpu_status
          if (s && s.state !== 'ready' && s.state !== 'no_gpu' && s.state !== 'unknown') {
            setGpuStatus(s)
          } else {
            setGpuStatus(null)
          }
          // Stop polling once ready or no GPU
          if (s?.state === 'ready' || s?.state === 'no_gpu') return
        }
      } catch { /* backend may be restarting */ }
      timer = setTimeout(check, 4000)
    }
    timer = setTimeout(check, 3000)
    return () => clearTimeout(timer)
  }, [apiUrl])

  useEffect(() => {
    let cancelled = false
    const refreshMetadata = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/datasets`)
        if (response.ok) {
          const data = await response.json()
          if (!cancelled) setDatasets(data.datasets || [])
        }
      } catch { /* Keep current metadata if backend is temporarily unavailable. */ }
    }
    if (['merge','split','annotate','training'].includes(activeView)) refreshMetadata()
    window.addEventListener('visos-datasets-changed', refreshMetadata)
    return () => { cancelled = true; window.removeEventListener('visos-datasets-changed', refreshMetadata) }
  }, [apiUrl, activeView])

  const updateImageCache = useCallback((datasetId: string, images: ImageData[]) => {
    setImageCache(prev => ({ ...prev, [datasetId]: images }))
  }, [])

  const invalidateImageCache = useCallback((datasetId: string) => {
    setImageCache(prev => {
      const next = { ...prev }
      delete next[datasetId]
      return next
    })
  }, [])

  const handleClassDatasetUpdate = useCallback((dataset: Dataset) => {
    invalidateImageCache(dataset.id)
    setSelectedDataset(current => current?.id === dataset.id ? dataset : current)
  }, [invalidateImageCache])

  const handleSelectDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset)
  }
  const workflow=useAssistantWorkflow(apiUrl,handleSelectDataset,setActiveView,()=>activeView==='annotate'?annotationBeforeLeave.current():Promise.resolve())

  const renderView = () => {
    if (process.env.NEXT_PUBLIC_RUNHIVE_PREVIEW === '1' && !['datasets', 'dashboard', 'gallery', 'classes', 'health'].includes(activeView)) return <div role="status" className="p-8">该功能未启用。当前预览仅开放合成数据集、概览、图库、类别与数据健康检查；Agent、LLM、Portal、模型训练和远程计算不可用。</div>
    let viewNode: React.ReactNode = null
    switch (activeView) {
      case 'datasets':
        viewNode = (
          <DatasetsView
            datasets={datasets}
            setDatasets={setDatasets}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            onDatasetLoaded={invalidateImageCache}
            onNavigateToVideo={() => setActiveView('video-extraction')}
            apiUrl={apiUrl}
          />
        )
        break
      case 'dashboard':
        viewNode = (
          <DashboardView
            datasets={datasets}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            setActiveView={setActiveView}
            apiUrl={apiUrl}
          />
        )
        break
      case 'gallery':
        viewNode = (
          <GalleryView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
            onOpenAnnotator={(imageId: string) => { setAnnotateInitialImageId(imageId); setActiveView('annotate') }}
          />
        )
        break
      case 'health':
        viewNode = (
          <HealthView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
          />
        )
        break
      case 'compare':
        viewNode = (
          <CompareView
            datasets={datasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'snapshots':
        viewNode = (
          <SnapshotView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'sorting':
        viewNode = (
          <SortingView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
          />
        )
        break
      case 'duplicate-detection':
        viewNode = (
          <DuplicateDetectionView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'annotate':
        viewNode = (
          <AnnotationView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
            initialImageId={annotateInitialImageId}
            onInitialImageConsumed={() => setAnnotateInitialImageId(null)}
            onOpenVlmConfig={() => setActiveView('vlm')}
            assistantDraftId={workflow.flow?.annotation_draft|| (activeView==='annotate'?workflow.draftId:'')}
            onAssistantDraftChange={id=>void workflow.adoptDraft(id)}
            assistantAction={annotationAction}
            registerBeforeLeave={handler=>{annotationBeforeLeave.current=handler}}
          />
        )
        break
      case 'classes':
        viewNode = (
          <ClassManagementView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetUpdated={handleClassDatasetUpdate}
            apiUrl={apiUrl}
          />
        )
        break
      case 'vlm':
        viewNode = (
          <VlmView
            datasets={datasets}
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'lmm':
        viewNode = (
          <LmmView
            apiUrl={apiUrl}
          />
        )
        break
      case 'augmentation':
        viewNode = (
          <AugmentationView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetCreated={setSelectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'video-extraction':
        viewNode = (
          <VideoExtractionView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetCreated={(dataset) => {
              setSelectedDataset(dataset)
              invalidateImageCache(dataset.id)
            }}
            apiUrl={apiUrl}
          />
        )
        break
      case 'split':
        viewNode = (
          <SplitView
            selectedDataset={workflow.flow?(datasets.find(d=>d.id===workflow.flow?.source_dataset_id)||workflow.flow.source_dataset||selectedDataset):selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetCreated={setSelectedDataset}
            apiUrl={apiUrl}
            assistantFlow={workflow.flow}
            onWorkflowChange={()=>void workflow.refresh()}
          />
        )
        break
      case 'convert':
        viewNode = (
          <ConvertView
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'merge':
        viewNode = (
          <MergeView
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'compute-nodes':
        viewNode = <ComputeNodesView apiUrl={apiUrl} />
        break
      case 'training':
        viewNode = (
          <TrainingView
            datasets={datasets}
            apiUrl={apiUrl}
            assistantDraftId={workflow.flow?workflow.flow.training_draft||'':workflow.draftId}
            workflowId={workflow.flow?.id}
            initialDatasetId={workflow.flow?.target_dataset_id}
          />
        )
        break
      case 'inference':
        viewNode = (
          <InferenceView
            apiUrl={apiUrl}
          />
        )
        break
      case 'evaluate':
        viewNode = (
          <EvaluationView
            datasets={datasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'models':
        viewNode = (
          <ModelsView
            apiUrl={apiUrl}
          />
        )
        break
      case 'batch-jobs':
        viewNode = (
          <BatchJobsView
            datasets={datasets}
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'yaml-wizard':
        viewNode = (
          <YamlWizardView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'settings':
        return <SettingsView />
      default:
        return null
    }

    // Wrap views that require a dataset with the inline dataset picker
    if (DATASET_REQUIRED_VIEWS.has(activeView)) {
      return (
        <DatasetRequiredGuard
          datasets={datasets}
          selectedDataset={selectedDataset}
          onSelectDataset={handleSelectDataset}
        >
          {viewNode}
        </DatasetRequiredGuard>
      )
    }
    return viewNode
  }

  return (
    <div className="flex h-screen bg-background flex-col overflow-hidden">
      {/* GPU status banner */}
      {gpuStatus && (
        <div className={`flex items-center gap-2.5 px-4 py-2 text-xs font-medium z-50 shrink-0 ${
          gpuStatus.state === 'failed'
            ? 'bg-destructive/90 text-destructive-foreground'
            : 'bg-warning/90 text-warning-foreground'
        }`}>
          {gpuStatus.state === 'installing' ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
          ) : (
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          )}
          <span className="flex-1 truncate">{gpuStatus.message}</span>
          {gpuStatus.state === 'failed' && (
            <button
              className="shrink-0 opacity-70 hover:opacity-100 transition-opacity"
              onClick={() => setGpuStatus(null)}
              aria-label="关闭提示"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}

      {/* Mobile top bar — hidden on md+ */}
      <div className="md:hidden flex items-center gap-3 px-4 py-3 border-b border-border bg-sidebar shrink-0">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-1.5 rounded-lg text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
          aria-label="打开菜单"
        >
          <Menu className="w-5 h-5" />
        </button>
        <span className="text-sm font-semibold text-foreground">
          {VIEW_NAME_ZH[activeView] ?? activeView}
        </span>
        {selectedDataset && (
          <span className="ml-auto text-[10px] font-mono text-primary truncate max-w-[140px]">
            {selectedDataset.name}
          </span>
        )}
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <Sidebar
          activeView={activeView}
          setActiveView={setActiveView}
          selectedDataset={selectedDataset}
          isOpen={sidebarOpen}
          embedded={isRunHiveEmbedded}
          onClose={() => setSidebarOpen(false)}
        />
        <main className="flex flex-col flex-1 min-h-0 min-w-0 overflow-hidden bg-background">
          {process.env.NEXT_PUBLIC_RUNHIVE_PREVIEW === '1' && <div role="status" className="px-4 py-2 border-b bg-amber-50 text-amber-950 text-sm">独立测试预览 · 仅使用合成数据。Agent、LLM、Portal、模型训练与远程计算未启用。</div>}
          {workflow.resourceGuide&&workflow.resourceGuide.view===activeView?<div className="px-4 py-3 bg-blue-50 border-b border-blue-200 flex items-center gap-3 text-sm"><strong className="text-blue-600">{activeView==='classes'?'添加数据集类别':'导入数据集'}</strong><span className="flex-1">{activeView==='classes'?'在原类别页面添加类别，完成后返回助手继续选择。':'上传图片或数据集，也可在导入目录中输入路径；导入完成后返回助手。'}</span><button className="rounded border bg-white px-3 py-2" onClick={()=>workflow.notify('resource-return')}>已添加，返回助手继续</button><button onClick={()=>workflow.setResourceGuide(null)}>暂停引导</button></div>:<NativeWorkflowGuide controller={workflow} view={activeView} apiUrl={apiUrl} selectedDatasetId={selectedDataset?.id} beforeLeave={()=>activeView==='annotate'?annotationBeforeLeave.current():Promise.resolve()} onAnnotationAction={action=>{if(activeView!=='annotate')void workflow.move('annotate').catch(()=>{});setAnnotationAction({action,version:Date.now()})}} />}
          <div className="flex-1 min-h-0 overflow-y-auto">{renderView()}</div>
        </main>
      </div>
    </div>
  )
}
