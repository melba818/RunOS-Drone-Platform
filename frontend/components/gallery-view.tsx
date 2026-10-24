'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { 
  Grid3X3, ZoomIn, ZoomOut, CheckSquare, Trash2, 
  Tag, Search, Filter, PenTool, SlidersHorizontal, X, Loader2
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset, ImageData, ImageCache } from '@/app/page'
import { toast } from 'sonner'

interface GalleryViewProps {
  selectedDataset: Dataset | null
  apiUrl: string
  imageCache: ImageCache
  updateImageCache: (datasetId: string, images: ImageData[]) => void
  onOpenAnnotator?: (imageId: string) => void
}

export function GalleryView({ selectedDataset, apiUrl, imageCache, updateImageCache, onOpenAnnotator }: GalleryViewProps) {
  const [allImages, setAllImages] = useState<ImageData[]>([])
  const [filteredImages, setFilteredImages] = useState<ImageData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [thumbnailSize, setThumbnailSize] = useState(160)
  const [filterSplit, setFilterSplit] = useState('all')
  const [filterClass, setFilterClass] = useState('all')
  const [filterAnnotated, setFilterAnnotated] = useState<'all' | 'annotated' | 'unannotated'>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [availableSplits, setAvailableSplits] = useState<string[]>([])
  const [availableClasses, setAvailableClasses] = useState<string[]>([])
  const [isDeleting, setIsDeleting] = useState(false)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [showSplitDialog, setShowSplitDialog] = useState(false)
  const [targetSplit, setTargetSplit] = useState('train')
  const lastSelectedRef = useRef<string | null>(null)

  useEffect(() => {
    if (!selectedDataset) return
    const cached = imageCache[selectedDataset.id]
    if (cached?.length) {
      initFromImages(cached)
    } else {
      loadImages()
    }
  }, [selectedDataset])

  useEffect(() => {
    applyFilters()
  }, [allImages, filterSplit, filterClass, filterAnnotated, searchQuery])

  const initFromImages = (images: ImageData[]) => {
    setAllImages(images)
    const splits = [...new Set(images.map(i => i.split).filter(Boolean) as string[])]
    const classes = [...new Set(
      images.flatMap(i => [i.class_name, ...(i.annotations || []).map(a => a.class_name)]).filter(Boolean) as string[]
    )]
    setAvailableSplits(splits)
    setAvailableClasses(classes)
  }

  const loadImages = async () => {
    if (!selectedDataset) return
    setIsLoading(true)
    try {
      const res = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images?limit=999999`)
      if (!res.ok) throw new Error('加载失败')
      const data = await res.json()
      const images: ImageData[] = data.images || []
      updateImageCache(selectedDataset.id, images)
      initFromImages(images)
    } catch (e) {
      toast.error('加载图像失败')
    } finally {
      setIsLoading(false)
    }
  }

  const applyFilters = () => {
    let imgs = allImages
    if (filterSplit !== 'all') imgs = imgs.filter(i => i.split === filterSplit)
    if (filterClass !== 'all') {
      imgs = imgs.filter(i =>
        i.class_name === filterClass ||
        (i.annotations || []).some(a => a.class_name === filterClass)
      )
    }
    if (filterAnnotated === 'annotated') imgs = imgs.filter(i => i.has_annotations)
    if (filterAnnotated === 'unannotated') imgs = imgs.filter(i => !i.has_annotations)
    if (searchQuery) imgs = imgs.filter(i => i.filename?.toLowerCase().includes(searchQuery.toLowerCase()))
    setFilteredImages(imgs)
  }

  const toggleSelect = useCallback((id: string, event: React.MouseEvent) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (event.shiftKey && lastSelectedRef.current) {
        // Range select
        const ids = filteredImages.map(i => i.id)
        const a = ids.indexOf(lastSelectedRef.current)
        const b = ids.indexOf(id)
        const [start, end] = a < b ? [a, b] : [b, a]
        for (let i = start; i <= end; i++) next.add(ids[i])
      } else {
        if (next.has(id)) next.delete(id)
        else next.add(id)
      }
      lastSelectedRef.current = id
      return next
    })
  }, [filteredImages])

  const selectAll = () => setSelectedIds(new Set(filteredImages.map(i => i.id)))
  const clearSelection = () => setSelectedIds(new Set())

  const annotationCount = (img: ImageData) => (img.annotations || []).length

  const batchAssignSplit = async () => {
    if (!selectedDataset || selectedIds.size === 0) return
    setIsDeleting(true)
    try {
      const ids = Array.from(selectedIds)
      const res = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images/batch-split`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_ids: ids, split: targetSplit }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as { detail?: string }).detail || '分配划分失败')
      }
      const updated = allImages.map(img =>
        selectedIds.has(img.id) ? { ...img, split: targetSplit } : img
      )
      setAllImages(updated)
      updateImageCache(selectedDataset.id, updated)
      const newSplits = [...new Set(updated.map(i => i.split).filter(Boolean) as string[])]
      setAvailableSplits(newSplits)
      toast.success(`已将 ${ids.length} 张图像分配到 ${targetSplit} 划分`)
      clearSelection()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '分配划分失败')
    } finally {
      setIsDeleting(false)
      setShowSplitDialog(false)
    }
  }

  const batchDelete = async () => {
    if (!selectedDataset || selectedIds.size === 0) return
    if (!confirm(`确定永久删除 ${selectedIds.size} 张图像？此操作无法撤销。`)) return
    setIsDeleting(true)
    try {
      const ids = Array.from(selectedIds)
      const res = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images/batch-delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_ids: ids }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error((err as { detail?: string }).detail || '删除图像失败')
      }
      const remaining = allImages.filter(img => !selectedIds.has(img.id))
      setAllImages(remaining)
      updateImageCache(selectedDataset.id, remaining)
      toast.success(`已删除 ${ids.length} 张图像`)
      clearSelection()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '删除图像失败')
    } finally {
      setIsDeleting(false)
    }
  }

  const getThumbCols = () => {
    if (thumbnailSize <= 100) return 'grid-cols-[repeat(auto-fill,minmax(80px,1fr))]'
    if (thumbnailSize <= 150) return 'grid-cols-[repeat(auto-fill,minmax(120px,1fr))]'
    if (thumbnailSize <= 200) return 'grid-cols-[repeat(auto-fill,minmax(160px,1fr))]'
    if (thumbnailSize <= 280) return 'grid-cols-[repeat(auto-fill,minmax(220px,1fr))]'
    return 'grid-cols-[repeat(auto-fill,minmax(280px,1fr))]'
  }

  if (!selectedDataset) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center">
          <Grid3X3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-lg font-medium">未选择数据集</p>
          <p className="text-sm text-muted-foreground mt-1">请先选择一个数据集，再查看图像库</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-border/50 bg-background/95 backdrop-blur space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">图像库</h2>
            <Badge variant="secondary">{filteredImages.length} / {allImages.length}</Badge>
            {selectedIds.size > 0 && (
              <Badge variant="default" className="bg-primary">已选 {selectedIds.size} 张</Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {selectedIds.size > 0 && (
              <>
                <Button variant="ghost" size="sm" onClick={clearSelection} className="gap-1">
                  <X className="w-3 h-3" /> 清除
                </Button>
                <Button variant="ghost" size="sm" onClick={selectAll} className="gap-1">
                  <CheckSquare className="w-3 h-3" /> 全选
                </Button>
              </>
            )}
            {selectedIds.size === 0 && (
              <Button variant="ghost" size="sm" onClick={selectAll} className="gap-1">
                <CheckSquare className="w-3 h-3" /> 全选
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={loadImages} className="gap-1">
              <Filter className="w-3 h-3" /> 刷新
            </Button>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <div className="relative flex-1 min-w-[180px]">
            <Search className="absolute left-2 top-2 w-3.5 h-3.5 text-muted-foreground" />
            <Input
              placeholder="按文件名搜索…"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-7 h-8 text-sm"
            />
          </div>

          <Select value={filterSplit} onValueChange={setFilterSplit}>
            <SelectTrigger className="h-8 w-28 text-sm">
              <SelectValue placeholder="划分" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部划分</SelectItem>
              {availableSplits.map(s => <SelectItem key={s} value={s}>{s}</SelectItem>)}
            </SelectContent>
          </Select>

          <Select value={filterClass} onValueChange={setFilterClass}>
            <SelectTrigger className="h-8 w-36 text-sm">
              <SelectValue placeholder="类别" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部类别</SelectItem>
              {availableClasses.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
            </SelectContent>
          </Select>

          <Select value={filterAnnotated} onValueChange={v => setFilterAnnotated(v as any)}>
            <SelectTrigger className="h-8 w-36 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部图像</SelectItem>
              <SelectItem value="annotated">已标注</SelectItem>
              <SelectItem value="unannotated">未标注</SelectItem>
            </SelectContent>
          </Select>

          <div className="flex items-center gap-2 ml-auto">
            <ZoomOut className="w-3.5 h-3.5 text-muted-foreground" />
            <Slider
              value={[thumbnailSize]}
              onValueChange={([v]) => setThumbnailSize(v)}
              min={80} max={320} step={20}
              className="w-24"
            />
            <ZoomIn className="w-3.5 h-3.5 text-muted-foreground" />
          </div>
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : filteredImages.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <Grid3X3 className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
              <p className="text-muted-foreground">没有符合当前筛选条件的图像</p>
            </div>
          </div>
        ) : (
          <div className={cn('grid gap-2', getThumbCols())}>
            {filteredImages.map((img) => {
              const isSelected = selectedIds.has(img.id)
              const annCount = annotationCount(img)
              const isHovered = hoveredId === img.id
              return (
                <div
                  key={img.id}
                  className={cn(
                    'relative group rounded-md overflow-hidden cursor-pointer border-2 transition-all',
                    isSelected ? 'border-primary shadow-lg shadow-primary/20' : 'border-transparent hover:border-border',
                  )}
                  style={{ height: thumbnailSize }}
                  onMouseEnter={() => setHoveredId(img.id)}
                  onMouseLeave={() => setHoveredId(null)}
                  onClick={(e) => toggleSelect(img.id, e)}
                  onDoubleClick={() => onOpenAnnotator?.(img.id)}
                >
                  <img
                    src={`${apiUrl}/api/datasets/${selectedDataset.id}/image-file/${img.path}`}
                    alt={img.filename}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />

                  {/* Selection overlay */}
                  <div className={cn(
                    'absolute inset-0 bg-primary/10 transition-opacity',
                    isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-30'
                  )} />

                  {/* Checkbox */}
                  <div className={cn(
                    'absolute top-1.5 left-1.5 transition-opacity',
                    isSelected || isHovered ? 'opacity-100' : 'opacity-0'
                  )}>
                    <div className={cn(
                      'w-5 h-5 rounded border-2 flex items-center justify-center bg-background',
                      isSelected ? 'border-primary bg-primary' : 'border-white/80'
                    )}>
                      {isSelected && <div className="w-2.5 h-2.5 text-white">✓</div>}
                    </div>
                  </div>

                  {/* Annotation badge */}
                  <div className="absolute top-1.5 right-1.5">
                    {annCount > 0 ? (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs bg-primary/90 text-white font-medium">
                        <Tag className="w-2.5 h-2.5" />{annCount}
                      </span>
                    ) : (
                      <span className="inline-flex px-1.5 py-0.5 rounded text-xs bg-black/50 text-white/70">
                        0
                      </span>
                    )}
                  </div>

                  {/* Bottom info - show on hover or larger thumbs */}
                  {(isHovered || thumbnailSize >= 160) && (
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-1.5 pb-1.5 pt-4">
                      <p className="text-white text-[10px] truncate leading-tight">{img.filename}</p>
                      {img.split && (
                        <span className="inline-block text-[9px] px-1 rounded bg-white/20 text-white/80 mt-0.5">{img.split}</span>
                      )}
                    </div>
                  )}

                  {/* Open in annotator button */}
                  {isHovered && onOpenAnnotator && (
                    <button
                      className="absolute bottom-7 right-1.5 p-1 rounded bg-black/60 hover:bg-primary text-white transition-colors"
                      onClick={(e) => { e.stopPropagation(); onOpenAnnotator(img.id) }}
                      title="在标注工具中打开"
                    >
                      <PenTool className="w-3 h-3" />
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Bottom batch actions bar */}
      {selectedIds.size > 0 && (
        <div className="flex-shrink-0 px-4 py-2 bg-primary/5 border-t border-primary/20 flex items-center gap-3">
          <span className="text-sm font-medium">已选 {selectedIds.size} 张图像</span>
          <div className="flex gap-2 ml-auto">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSplitDialog(true)}
              className="gap-1"
            >
              <SlidersHorizontal className="w-3.5 h-3.5" /> 分配划分
            </Button>
            <Button
              variant="destructive"
              size="sm"
              disabled={isDeleting}
              onClick={batchDelete}
              className="gap-1"
            >
              {isDeleting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
              删除所选
            </Button>
          </div>
        </div>
      )}

      {/* Assign split dialog */}
      <Dialog open={showSplitDialog} onOpenChange={setShowSplitDialog}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>分配划分</DialogTitle>
            <DialogDescription>
              将已选的 {selectedIds.size} 张图像移动到某个数据集划分。
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 pt-2">
            <div>
              <label className="text-sm font-medium mb-2 block">目标划分</label>
              <Select value={targetSplit} onValueChange={setTargetSplit}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="train">train</SelectItem>
                  <SelectItem value="val">val</SelectItem>
                  <SelectItem value="test">test</SelectItem>
                  {availableSplits
                    .filter(s => !['train', 'val', 'test'].includes(s))
                    .map(s => <SelectItem key={s} value={s}>{s}</SelectItem>)
                  }
                </SelectContent>
              </Select>
            </div>
            <div className="flex gap-2 justify-end">
              <Button variant="outline" onClick={() => setShowSplitDialog(false)}>取消</Button>
              <Button onClick={batchAssignSplit} disabled={isDeleting} className="gap-2">
                {isDeleting && <Loader2 className="w-4 h-4 animate-spin" />}
                分配
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
