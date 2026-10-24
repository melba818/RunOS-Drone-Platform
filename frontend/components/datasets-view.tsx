'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Upload,
  FolderOpen,
  Trash2,
  Download,
  Database,
  Image,
  Tag,
  Calendar,
  CheckCircle,
  AlertCircle,
  Folder,
  ChevronRight,
  HardDrive,
  RefreshCw,
  Loader2,
  Video,
  ImagePlus,
  Plus,
  Pencil,
  X,
  Check
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

interface DatasetsViewProps {
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  selectedDataset: Dataset | null
  onSelectDataset: (dataset: Dataset) => void
  onDatasetLoaded?: (datasetId: string) => void
  onNavigateToVideo?: () => void
  apiUrl: string
}

interface FolderItem {
  name: string
  path: string
  is_directory: boolean
  is_dataset: boolean
  format_hint?: string
}

export function DatasetsView({
  datasets,
  setDatasets,
  selectedDataset,
  onSelectDataset,
  onDatasetLoaded,
  onNavigateToVideo,
  apiUrl
}: DatasetsViewProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState<'uploading' | 'processing' | 'complete' | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [renamingDatasetId, setRenamingDatasetId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  
  const imageInputRef = useRef<HTMLInputElement>(null)

  // Folder browser state
  const [showFolderBrowser, setShowFolderBrowser] = useState(false)
  const [currentPath, setCurrentPath] = useState('')
  const [folderItems, setFolderItems] = useState<FolderItem[]>([])
  const [isLoadingFolder, setIsLoadingFolder] = useState(false)
  const [manualPath, setManualPath] = useState('')

  // Fetch datasets from backend on mount and whenever apiUrl changes.
  // This restores the list after a page refresh or backend restart.
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/datasets`)
        if (response.ok) {
          const data = await response.json()
          if (data.datasets && data.datasets.length > 0) {
            setDatasets(data.datasets)
          }
        }
      } catch {
        // Backend may not be running yet — fail silently
      }
    }
    fetchDatasets()
  }, [apiUrl])

  const browseFolders = async (path: string = '') => {
    setIsLoadingFolder(true)
    try {
      const response = await fetch(`${apiUrl}/api/browse-folders?path=${encodeURIComponent(path || '.')}`, {
        method: 'POST',
      })
      
      if (response.ok) {
        const data = await response.json()
        setCurrentPath(data.current_path || path)
        setFolderItems(data.items || [])
        setManualPath(data.current_path || path)
      }
    } catch (err) {
      console.error('Failed to browse folders:', err)
    } finally {
      setIsLoadingFolder(false)
    }
  }

  const handleOpenFolderBrowser = () => {
    setShowFolderBrowser(true)
    browseFolders('')
  }

  const handleFolderClick = (item: FolderItem) => {
    if (item.is_directory) {
      browseFolders(item.path)
    }
  }

  const handleLoadLocalDataset = async (path: string, formatHint?: string) => {
    setIsLoadingFolder(true)
    setError(null)
    setSuccess(null)
    setSuccess(`正在从 ${path} 加载数据集…`)
    
    try {
      const response = await fetch(`${apiUrl}/api/datasets/load-local`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: path, format_hint: formatHint })
      })
      
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || '加载数据集失败')
      }
      
      const result = await response.json()
      
      if (result.success && result.dataset) {
        setDatasets([...datasets, result.dataset])
        onSelectDataset(result.dataset)
        onDatasetLoaded?.(result.dataset.id)
        setShowFolderBrowser(false)
        setSuccess(`已成功加载 ${result.dataset.name}`)
        setTimeout(() => setSuccess(null), 3000)
      } else if (result.error) {
        throw new Error(result.error)
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err)
      setError(errorMsg === '[object Object]' ? '加载数据集失败 — 请检查文件夹路径' : errorMsg)
    } finally {
      setIsLoadingFolder(false)
    }
  }

  const handleManualPathSubmit = () => {
    if (manualPath.trim()) {
      browseFolders(manualPath.trim())
    }
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      await uploadFiles(files)
    }
  }, [apiUrl, datasets])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      await uploadFiles(files)
    }
  }

  const uploadFiles = async (files: File[]) => {
    setIsUploading(true)
    setUploadProgress(0)
    setUploadStatus('uploading')
    setError(null)

    const formData = new FormData()
    files.forEach(file => formData.append('files', file))
    // Derive a friendly name from the first file (strip .zip extension)
    const firstName = files[0]?.name ?? ''
    const derivedName = firstName.replace(/\.zip$/i, '').replace(/\.[^.]+$/, '') || firstName
    if (derivedName) formData.append('dataset_name', derivedName)

    const xhr = new XMLHttpRequest()

    const uploadPromise = new Promise<{success: boolean, dataset?: Dataset}>((resolve, reject) => {
      // Track upload progress (bytes sent to server)
      xhr.upload.addEventListener('loadstart', () => {
        setUploadProgress(2)
      })

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && e.total > 0) {
          // Upload progress is 0-80% of total; processing takes the rest
          const uploadPercent = Math.round((e.loaded / e.total) * 80)
          setUploadProgress(Math.max(2, uploadPercent))
          if (e.loaded >= e.total) {
            setUploadStatus('processing')
            setUploadProgress(80)
          }
        }
      })

      xhr.upload.addEventListener('load', () => {
        // Upload complete, server is now processing
        setUploadStatus('processing')
        setUploadProgress(85)
      })

      xhr.addEventListener('progress', () => {
        // Server is streaming response (processing)
        if (uploadStatus !== 'processing') {
          setUploadStatus('processing')
        }
      })

      xhr.addEventListener('load', () => {
        setUploadProgress(100)
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText)
            resolve(result)
          } catch {
            reject(new Error('服务器返回了无效的响应'))
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText)
            reject(new Error(err.detail || '上传失败'))
          } catch {
            reject(new Error(`上传失败（${xhr.status}）`))
          }
        }
      })
      
      xhr.addEventListener('error', () => reject(new Error('上传过程中网络出错')))
      xhr.addEventListener('abort', () => reject(new Error('上传已取消')))
      
      xhr.open('POST', `${apiUrl}/api/datasets/load`)
      xhr.send(formData)
    })

    try {
      const result = await uploadPromise
      
      if (result.success && result.dataset) {
        setDatasets([...datasets, result.dataset])
        onSelectDataset(result.dataset)
        onDatasetLoaded?.(result.dataset.id)
        setUploadStatus('complete')
        setSuccess(`已成功加载 ${result.dataset.name}`)
        setTimeout(() => setSuccess(null), 3000)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '上传失败')
    } finally {
      setIsUploading(false)
      setTimeout(() => setUploadStatus(null), 1000)
    }
  }

  const handleDelete = async (datasetId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/datasets/${datasetId}`, { method: 'DELETE' })
      if (response.ok) {
        const newDatasets = datasets.filter(d => d.id !== datasetId)
        setDatasets(newDatasets)
        if (selectedDataset?.id === datasetId) {
          onSelectDataset(newDatasets[0] || null as unknown as Dataset)
        }
        setSuccess('数据集已移除')
        setTimeout(() => setSuccess(null), 3000)
      }
    } catch (err) {
      setError('删除数据集失败')
    }
  }

  const handleExport = async (dataset: Dataset) => {
    try {
      window.open(`${apiUrl}/api/export/${dataset.id}`, '_blank')
    } catch (err) {
      setError('导出数据集失败')
    }
  }

  const startRenaming = (dataset: Dataset, e: React.MouseEvent) => {
    e.stopPropagation()
    setRenamingDatasetId(dataset.id)
    setRenameValue(dataset.name)
  }

  const cancelRenaming = (e?: React.MouseEvent) => {
    e?.stopPropagation()
    setRenamingDatasetId(null)
    setRenameValue('')
  }

  const commitRename = async (datasetId: string, e?: React.MouseEvent) => {
    e?.stopPropagation()
    const trimmed = renameValue.trim()
    if (!trimmed) { cancelRenaming(); return }
    try {
      const response = await fetch(`${apiUrl}/api/datasets/${datasetId}/rename`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: trimmed }),
      })
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || '重命名失败')
      }
      const result = await response.json()
      setDatasets(datasets.map(d => d.id === datasetId ? { ...d, name: trimmed } : d))
      if (selectedDataset?.id === datasetId) {
        onSelectDataset({ ...selectedDataset, name: trimmed })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '重命名失败')
    } finally {
      setRenamingDatasetId(null)
      setRenameValue('')
    }
  }

  const getTaskTypeColor = (taskType: string) => {
    switch (taskType) {
      case 'classification': return 'bg-blue-500/10 text-blue-600 dark:text-blue-400'
      case 'detection': return 'bg-green-500/10 text-green-600 dark:text-green-400'
      case 'segmentation': return 'bg-purple-500/10 text-purple-600 dark:text-purple-400'
      case 'keypoint': return 'bg-orange-500/10 text-orange-600 dark:text-orange-400'
      case 'obb-detection': return 'bg-teal-500/10 text-teal-600 dark:text-teal-400'
      default: return 'bg-muted text-muted-foreground'
    }
  }

  /** 任务类型内部 key 保持英文（用于逻辑判断），仅展示时映射为中文 */
  const TASK_TYPE_LABEL_ZH: Record<string, string> = {
    classification: '分类',
    detection: '检测',
    segmentation: '分割',
    keypoint: '关键点',
    'obb-detection': 'OBB 检测',
  }

  const getFormatLabel = (format: string) => {
    if (!format || format === 'unknown') return '未知格式'
    if (format === 'generic-images') return '图像'
    return format.toUpperCase()
  }

  const getTaskTypeLabel = (taskType: string) => {
    if (!taskType || taskType === 'unknown') return '未知类型'
    return TASK_TYPE_LABEL_ZH[taskType] ?? taskType
  }

  const getProgressLabel = () => {
    if (uploadStatus === 'processing') return '正在服务端处理数据集…'
    if (uploadStatus === 'complete') return '完成！'
    if (uploadProgress < 5) return '开始上传…'
    return `上传中… ${uploadProgress}%`
  }

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground/50 mb-2">数据集</p>
          <h2 className="text-2xl font-display font-bold tracking-tight">我的数据集</h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-lg">
            数据集是一组图像及其标注的集合。在此加载数据集后，即可开始标注、处理或训练。
          </p>
        </div>
        <Button onClick={handleOpenFolderBrowser} variant="outline" className="shrink-0">
          <FolderOpen className="w-4 h-4 mr-2" strokeWidth={1.75} />
          浏览本地文件夹
        </Button>
      </div>

      {/* Alerts */}
      {error && (
        <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2 text-destructive">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-xs underline hover:no-underline">忽略</button>
        </div>
      )}

      {success && (
        <div className={cn(
          "mb-4 p-3 rounded-lg flex items-center gap-2",
          success.includes('正在') || success.includes('…')
            ? "bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400"
            : "bg-green-500/10 border border-green-500/20 text-green-600 dark:text-green-400"
        )}>
          {success.includes('正在') || success.includes('…') ? (
            <Loader2 className="w-4 h-4 flex-shrink-0 animate-spin" />
          ) : (
            <CheckCircle className="w-4 h-4 flex-shrink-0" />
          )}
          <span className="text-sm">{success}</span>
        </div>
      )}

      {/* Create new dataset — three paths */}
      <div className="mb-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Path A: upload a formatted ZIP */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={cn(
            'sm:col-span-1 border-2 border-dashed rounded-xl p-5 transition-all flex flex-col items-center text-center cursor-pointer',
            dragActive
              ? 'border-primary bg-primary/5'
              : 'border-border hover:border-primary/40 hover:bg-primary/[0.02] bg-card/50'
          )}
          onClick={() => { if (!isUploading) document.getElementById('zip-input')?.click() }}
        >
          <div className={cn('w-10 h-10 rounded-full flex items-center justify-center mb-3', dragActive ? 'bg-primary/10' : 'bg-secondary')}>
            {isUploading && uploadStatus === 'processing' ? (
              <Loader2 className="w-5 h-5 text-amber-600 dark:text-amber-400 animate-spin" />
            ) : (
              <Upload className={cn('w-5 h-5', dragActive ? 'text-primary' : 'text-muted-foreground')} />
            )}
          </div>
          <p className="text-sm font-semibold mb-1">
            {isUploading ? (uploadStatus === 'processing' ? '处理中…' : '上传中…') : '上传数据集 ZIP'}
          </p>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            YOLO、COCO、VOC、LabelMe、分类文件夹
          </p>
          {isUploading && (
            <div className="w-full mt-3 space-y-1">
              <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                <div
                  className={cn('h-full rounded-full transition-all duration-300',
                    uploadStatus === 'processing' ? 'bg-amber-500 animate-pulse' :
                    uploadStatus === 'complete' ? 'bg-green-500' : 'bg-primary'
                  )}
                  style={{ width: uploadStatus === 'processing' ? '100%' : `${Math.max(2, uploadProgress)}%` }}
                />
              </div>
              <p className="text-[10px] text-muted-foreground">{getProgressLabel()}</p>
            </div>
          )}
          <input id="zip-input" type="file" accept=".zip" className="hidden" onChange={handleFileSelect} />
        </div>

        {/* Path B: upload raw unannotated images */}
        <div
          className="border-2 border-dashed rounded-xl p-5 transition-all flex flex-col items-center text-center cursor-pointer border-border hover:border-primary/40 hover:bg-primary/[0.02] bg-card/50"
          onClick={() => imageInputRef.current?.click()}
        >
          <div className="w-10 h-10 rounded-full flex items-center justify-center mb-3 bg-secondary">
            <ImagePlus className="w-5 h-5 text-muted-foreground" />
          </div>
          <p className="text-sm font-semibold mb-1">上传原始图像</p>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            拖入未标注的 JPG / PNG 照片 — 在平台上完成标注
          </p>
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>

        {/* Path C: extract frames from video */}
        <div
          className="border-2 border-dashed rounded-xl p-5 transition-all flex flex-col items-center text-center cursor-pointer border-border hover:border-primary/40 hover:bg-primary/[0.02] bg-card/50"
          onClick={onNavigateToVideo}
        >
          <div className="w-10 h-10 rounded-full flex items-center justify-center mb-3 bg-secondary">
            <Video className="w-5 h-5 text-muted-foreground" />
          </div>
          <p className="text-sm font-semibold mb-1">从视频抽取帧</p>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            上传视频并抽取帧，构建新的数据集
          </p>
        </div>
      </div>

      {/* Dataset Grid */}
      <div className="flex-1 overflow-y-auto">
        {datasets.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="w-12 h-12 rounded-xl bg-muted/60 border border-border flex items-center justify-center mb-4">
              <Database className="w-5 h-5 text-muted-foreground/50" strokeWidth={1.5} />
            </div>
            <h3 className="text-base font-semibold">尚未加载数据集</h3>
            <p className="text-xs text-muted-foreground mt-1.5 mb-5 max-w-xs leading-relaxed">
              上传数据集 ZIP、放入原始图像、抽取视频帧，或浏览本地文件夹即可开始。
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              <Button variant="outline" size="sm" onClick={() => imageInputRef.current?.click()}>
                <ImagePlus className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.75} />
                上传图像
              </Button>
              <Button variant="outline" size="sm" onClick={onNavigateToVideo}>
                <Video className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.75} />
                从视频
              </Button>
              <Button variant="outline" size="sm" onClick={handleOpenFolderBrowser}>
                <FolderOpen className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.75} />
                浏览文件夹
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
            {datasets.map((dataset) => (
              <Card
                key={dataset.id}
                className={cn(
                  'cursor-pointer transition-all group hover:border-primary/30 hover:bg-primary/[0.02]',
                  selectedDataset?.id === dataset.id
                    ? 'border-primary/40 bg-primary/[0.03] shadow-sm'
                    : ''
                )}
                onClick={() => onSelectDataset(dataset)}
              >
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      {renamingDatasetId === dataset.id ? (
                        <div className="flex items-center gap-1" onClick={e => e.stopPropagation()}>
                          <Input
                            autoFocus
                            value={renameValue}
                            onChange={e => setRenameValue(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === 'Enter') commitRename(dataset.id)
                              if (e.key === 'Escape') cancelRenaming()
                            }}
                            className="h-7 text-sm font-semibold px-2 py-0"
                          />
                          <button
                            onClick={e => commitRename(dataset.id, e)}
                            className="p-1 rounded hover:bg-green-500/10 text-green-600"
                          >
                            <Check className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={cancelRenaming}
                            className="p-1 rounded hover:bg-muted text-muted-foreground"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ) : (
                        <CardTitle className="text-sm font-semibold break-words">{dataset.name}</CardTitle>
                      )}
                      <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                        <span className="inline-flex items-center px-2 py-0.5 bg-secondary rounded-md text-[10px] font-mono font-medium">
                          {getFormatLabel(dataset.format)}
                        </span>
                        <span className={cn('px-2 py-0.5 rounded-md text-[10px] font-medium', getTaskTypeColor(dataset.task_type))}>
                          {getTaskTypeLabel(dataset.task_type)}
                        </span>
                      </div>
                    </div>
                    {selectedDataset?.id === dataset.id && renamingDatasetId !== dataset.id && (
                      <CheckCircle className="w-4 h-4 text-primary shrink-0 mt-0.5" strokeWidth={2} />
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex items-center gap-4 text-xs text-muted-foreground mb-3">
                    <div className="flex items-center gap-1.5">
                      <Image className="w-3.5 h-3.5" strokeWidth={1.75} />
                      <span>{dataset.num_images.toLocaleString()} 张图像</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Tag className="w-3.5 h-3.5" strokeWidth={1.75} />
                      <span>{dataset.classes?.length || 0} 个类别</span>
                    </div>
                  </div>

                  {dataset.classes && dataset.classes.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-3">
                      {dataset.classes.slice(0, 4).map((cls, idx) => (
                        <span key={idx} className="px-1.5 py-0.5 bg-muted rounded text-[10px] font-medium">{cls}</span>
                      ))}
                      {dataset.classes.length > 4 && (
                        <span className="px-1.5 py-0.5 text-muted-foreground/60 text-[10px]">
                          +{dataset.classes.length - 4} 个
                        </span>
                      )}
                    </div>
                  )}

                  <div className="flex items-center gap-2 pt-3 border-t border-border opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      variant="outline"
                      className="flex-1 h-7 text-xs"
                      onClick={(e) => { e.stopPropagation(); handleExport(dataset) }}
                    >
                      <Download className="w-3 h-3 mr-1" strokeWidth={1.75} />
                      导出
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7"
                      onClick={(e) => startRenaming(dataset, e)}
                    >
                      <Pencil className="w-3 h-3" strokeWidth={1.75} />
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={(e) => { e.stopPropagation(); handleDelete(dataset.id) }}
                    >
                      <Trash2 className="w-3 h-3" strokeWidth={1.75} />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Folder Browser Dialog */}
      <Dialog open={showFolderBrowser} onOpenChange={setShowFolderBrowser}>
        <DialogContent className="max-w-2xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>浏览本地文件夹</DialogTitle>
            <DialogDescription>
              在本机导航到你的数据集文件夹，点击<strong>加载</strong>即可导入 — 无需上传。
              高亮显示的文件夹已被识别为数据集。
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="flex gap-2">
              <Input
                value={manualPath}
                onChange={(e) => setManualPath(e.target.value)}
                placeholder="输入文件夹路径（例如 C:\datasets\my_dataset）"
                onKeyDown={(e) => e.key === 'Enter' && handleManualPathSubmit()}
                className="flex-1"
              />
              <Button onClick={handleManualPathSubmit} variant="secondary">前往</Button>
              <Button onClick={() => browseFolders(currentPath)} variant="ghost" size="icon">
                <RefreshCw className={cn('w-4 h-4', isLoadingFolder && 'animate-spin')} />
              </Button>
            </div>

            {currentPath && (
              <div className="flex items-center gap-1 text-sm text-muted-foreground bg-muted/50 px-3 py-2 rounded-lg overflow-x-auto">
                <HardDrive className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">{currentPath}</span>
              </div>
            )}

            <ScrollArea className="h-[350px] border rounded-lg">
              {isLoadingFolder ? (
                <div className="flex items-center justify-center h-full">
                  <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
                </div>
              ) : folderItems.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <Folder className="w-12 h-12 mb-2 opacity-50" />
                  <p>未找到文件夹</p>
                </div>
              ) : (
                <div className="p-2 space-y-1">
                  {folderItems.map((item, idx) => (
                    <div
                      key={idx}
                      className={cn(
                        'flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors',
                        item.is_dataset 
                          ? 'bg-primary/5 hover:bg-primary/10 border border-primary/20' 
                          : 'hover:bg-muted'
                      )}
                      onClick={() => handleFolderClick(item)}
                    >
                      <Folder className={cn('w-5 h-5 flex-shrink-0', item.is_dataset ? 'text-primary' : 'text-muted-foreground')} />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{item.name}</p>
                        {item.is_dataset && item.format_hint && (
                          <p className="text-xs text-muted-foreground">已识别：{item.format_hint}</p>
                        )}
                      </div>
                      {item.is_dataset ? (
                        <Button size="sm" onClick={(e) => { e.stopPropagation(); handleLoadLocalDataset(item.path, item.format_hint) }}>
                          加载
                        </Button>
                      ) : (
                        <ChevronRight className="w-4 h-4 text-muted-foreground" />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>

            {currentPath && (
              <Button className="w-full" onClick={() => handleLoadLocalDataset(currentPath)} disabled={isLoadingFolder}>
                <FolderOpen className="w-4 h-4 mr-2" />
                将当前文件夹作为数据集加载
              </Button>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
