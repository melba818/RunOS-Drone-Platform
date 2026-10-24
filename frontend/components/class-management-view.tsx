'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Slider } from '@/components/ui/slider'
import {
  Layers,
  Trash2,
  GitMerge,
  Copy,
  Plus,
  AlertTriangle,
  CheckCircle,
  ArrowRight,
  Search,
  RefreshCw,
  Pencil,
  ImageMinus
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

const EXPORT_FORMATS = [
  { id: 'yolo',           name: 'YOLO' },
  { id: 'coco',           name: 'COCO JSON' },
  { id: 'pascal-voc',     name: 'Pascal VOC' },
  { id: 'labelme',        name: 'LabelMe' },
  { id: 'tensorflow-csv', name: 'TensorFlow CSV' },
  { id: 'createml',       name: 'CreateML' },
]

/** 内部状态 key 保持英文（用于逻辑判断），仅展示时映射为中文 */
const TAB_LABEL_ZH: Record<string, string> = {
  extract: '提取',
  delete: '删除',
  merge: '合并',
  rename: '重命名',
  add: '新增',
  'remove-unannotated': '移除未标注',
}

interface ClassManagementViewProps {
  selectedDataset: Dataset | null
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  onDatasetUpdated: (dataset: Dataset) => void
  apiUrl: string
}

interface ClassInfo {
  name: string
  count: number
  selected: boolean
}

export function ClassManagementView({ 
  selectedDataset, 
  datasets,
  setDatasets,
  onDatasetUpdated,
  apiUrl 
}: ClassManagementViewProps) {
  const [classes, setClasses] = useState<ClassInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'extract' | 'delete' | 'merge' | 'add' | 'rename' | 'remove-unannotated'>(()=>typeof window!=='undefined'&&new URLSearchParams(location.search).get('assistant_resource')==='classes'?'add':'extract')
  const [newClassName, setNewClassName] = useState('')
  const [mergeTargetName, setMergeTargetName] = useState('')
  const [extractOutputName, setExtractOutputName] = useState('')
  const [extractOutputFormat, setExtractOutputFormat] = useState('')
  const [renameNewName, setRenameNewName] = useState('')
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const [unannotatedCount, setUnannotatedCount] = useState<{ total: number; unannotated: number; annotated: number } | null>(null)
  const [randomRemovalEnabled, setRandomRemovalEnabled] = useState(false)
  const [randomRemovalRatio, setRandomRemovalRatio] = useState(50)
  const plannedRemovalCount = unannotatedCount
    ? randomRemovalEnabled && unannotatedCount.unannotated > 0
      ? Math.min(unannotatedCount.unannotated, Math.max(1, Math.round(unannotatedCount.unannotated * randomRemovalRatio / 100)))
      : randomRemovalEnabled ? 0 : unannotatedCount.unannotated
    : 0

  useEffect(() => {
    if (selectedDataset) {
      loadClasses()
      setExtractOutputName(`${selectedDataset.name}_extracted`)
      setExtractOutputFormat(selectedDataset.format || '')
    }
  }, [selectedDataset])

  useEffect(() => {
    if (activeTab === 'remove-unannotated' && selectedDataset) {
      fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/unannotated-count`)
        .then(r => r.json())
        .then(data => setUnannotatedCount(data))
        .catch(() => setUnannotatedCount(null))
    }
  }, [activeTab, selectedDataset?.id])

  const applyDatasetUpdate = (dataset: Dataset) => {
    setDatasets(datasets.map(d => d.id === dataset.id ? dataset : d))
    onDatasetUpdated(dataset)
  }

  const loadClasses = async () => {
    if (!selectedDataset) return
    setLoading(true)
    setMessage(null)
    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/classes`)
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Server returned ${response.status}`)
      }
      const data = await response.json()
      const classData: Record<string, number> = data.classes || {}

      if (Object.keys(classData).length === 0 && selectedDataset.classes?.length > 0) {
        // Backend returned empty dict but dataset has known classes — use them as fallback
        setClasses(selectedDataset.classes.map(name => ({ name, count: 0, selected: false })))
      } else {
        setClasses(
          Object.entries(classData).map(([name, count]) => ({
            name,
            count: count as number,
            selected: false,
          }))
        )
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : '未知错误'
      setMessage({ type: 'error', text: `加载类别失败：${msg}` })
      // Still show whatever the dataset metadata has so the panel is not blank
      if (selectedDataset.classes?.length > 0) {
        setClasses(selectedDataset.classes.map(name => ({ name, count: 0, selected: false })))
      }
    } finally {
      setLoading(false)
    }
  }

  const toggleClass = (className: string) => {
    setClasses(classes.map(c => 
      c.name === className ? { ...c, selected: !c.selected } : c
    ))
  }

  const selectAll = () => {
    setClasses(classes.map(c => ({ ...c, selected: true })))
  }

  const deselectAll = () => {
    setClasses(classes.map(c => ({ ...c, selected: false })))
  }

  const selectedClasses = classes.filter(c => c.selected)
  const filteredClasses = classes.filter(c => 
    c.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleExtractClasses = async () => {
    if (!selectedDataset || selectedClasses.length === 0) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/extract-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          classes_to_extract: selectedClasses.map(c => c.name),
          output_name: extractOutputName || `${selectedDataset.name}_extracted`,
          output_format: extractOutputFormat || undefined
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.new_dataset) {
          setDatasets([...datasets, data.new_dataset])
        }
        setMessage({
          type: 'success',
          text: `已将 ${selectedClasses.length} 个类别提取到新数据集，共 ${data.extracted_images} 张图片`
        })
        deselectAll()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || '提取类别失败')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '提取类别失败' })
    }
    setLoading(false)
  }

  const handleRenameClass = async () => {
    if (!selectedDataset || selectedClasses.length !== 1 || !renameNewName.trim()) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/rename-class`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          old_name: selectedClasses[0].name,
          new_name: renameNewName.trim()
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          applyDatasetUpdate(data.updated_dataset)
        }
        setMessage({
          type: 'success',
          text: `已将「${selectedClasses[0].name}」重命名为「${renameNewName}」（更新了 ${data.renamed_annotations} 个标注）`
        })
        setRenameNewName('')
        deselectAll()
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || '重命名类别失败')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '重命名类别失败' })
    }
    setLoading(false)
  }

  const handleDeleteClasses = async () => {
    if (!selectedDataset || selectedClasses.length === 0) return
    
    if (!confirm(`确定要删除 ${selectedClasses.length} 个类别吗？此操作无法撤销。`)) {
      return
    }

    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/delete-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          classes_to_delete: selectedClasses.map(c => c.name)
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          applyDatasetUpdate(data.updated_dataset)
        }
        setMessage({
          type: 'success',
          text: `已删除 ${selectedClasses.length} 个类别（移除了 ${data.deleted_annotations} 个标注）`
        })
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || '删除类别失败')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '删除类别失败' })
    }
    setLoading(false)
  }

  const handleMergeClasses = async () => {
    if (!selectedDataset || selectedClasses.length < 2 || !mergeTargetName) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/merge-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          source_classes: selectedClasses.map(c => c.name),
          target_class: mergeTargetName
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          applyDatasetUpdate(data.updated_dataset)
        }
        setMessage({
          type: 'success',
          text: `已将 ${selectedClasses.length} 个类别合并为「${mergeTargetName}」（更新了 ${data.merged_annotations} 个标注）`
        })
        setMergeTargetName('')
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || '合并类别失败')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '合并类别失败' })
    }
    setLoading(false)
  }

  const handleAddClass = async () => {
    if (!selectedDataset || !newClassName.trim()) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/add-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          new_classes: [newClassName.trim()],
          use_model: false
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          applyDatasetUpdate(data.updated_dataset)
        }
        setMessage({ type: 'success', text: `已添加类别「${newClassName}」` })
        setNewClassName('')
        loadClasses()
      } else {
        throw new Error('添加类别失败')
      }
    } catch (err) {
      setMessage({ type: 'error', text: '添加类别失败' })
    }
    setLoading(false)
  }

  const handleRemoveUnannotated = async () => {
    if (!selectedDataset) return
    setLoading(true)
    setMessage(null)
    try {
      const params = new URLSearchParams({
        randomize: String(randomRemovalEnabled),
        ratio: String(randomRemovalRatio),
      })
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/remove-unannotated?${params}`, {
        method: 'POST',
      })
      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          applyDatasetUpdate(data.updated_dataset)
        }
        const remaining = data.remaining ?? data.updated_dataset?.num_images ?? '?'
        const unannotatedRemaining = Number(data.unannotated_remaining
          ?? Math.max(0, (unannotatedCount?.unannotated ?? 0) - Number(data.removed || 0)))
        if (typeof remaining === 'number') {
          setUnannotatedCount({
            total: remaining,
            unannotated: unannotatedRemaining,
            annotated: Math.max(0, remaining - unannotatedRemaining),
          })
        } else {
          setUnannotatedCount(null)
        }
        await loadClasses()
        setMessage({ type: 'success', text: `已移除 ${data.removed} 张未标注图片，剩余 ${remaining} 张，其中未标注 ${unannotatedRemaining} 张。` })
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || '移除未标注图片失败')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '移除未标注图片失败' })
    }
    setLoading(false)
  }

  if (!selectedDataset) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6">
          <Layers className="w-10 h-10 text-muted-foreground" />
        </div>
        <h2 className="text-2xl font-semibold text-foreground mb-2">未选择数据集</h2>
        <p className="text-muted-foreground max-w-md">
          请先选择一个数据集以管理其类别
        </p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold text-foreground">类别管理</h2>
          <p className="text-muted-foreground text-sm mt-1">
            在 {selectedDataset.name} 中提取、删除、合并或新增类别
          </p>
        </div>
      </div>

      {message && (
        <div className={cn(
          'mb-4 p-3 rounded-lg flex items-center gap-2',
          message.type === 'success' ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20' : 'bg-destructive/10 text-destructive border border-destructive/20'
        )}>
          {message.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
          <span className="text-sm">{message.text}</span>
          <button onClick={() => setMessage(null)} className="ml-auto text-xs underline">忽略</button>
        </div>
      )}

      <div className="flex gap-6 flex-1 min-h-0">
        <Card className="w-80 flex-shrink-0 flex flex-col">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">类别（{classes.length}）</CardTitle>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={loadClasses}
                disabled={loading}
                title="刷新类别列表"
              >
                <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
              </Button>
            </div>
            <CardDescription>
              已选择 {selectedClasses.length} 个
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden flex flex-col">
            <div className="relative mb-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input 
                placeholder="搜索类别…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            
            <div className="flex items-center gap-2 mb-3">
              <Button variant="outline" size="sm" onClick={selectAll}>全选</Button>
              <Button variant="outline" size="sm" onClick={deselectAll}>取消全选</Button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-1">
              {filteredClasses.map((cls) => (
                <div 
                  key={cls.name}
                  className={cn(
                    'flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors',
                    cls.selected ? 'bg-primary/10' : 'hover:bg-muted'
                  )}
                  onClick={() => toggleClass(cls.name)}
                >
                  <Checkbox checked={cls.selected} />
                  <span className="flex-1 text-sm truncate">{cls.name}</span>
                  <span className="text-xs text-muted-foreground">{cls.count}</span>
                </div>
              ))}
              {filteredClasses.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">未找到类别</p>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="flex-1 overflow-y-auto">
          <div className="flex flex-wrap gap-2 mb-4">
            {(['extract', 'delete', 'merge', 'rename', 'add', 'remove-unannotated'] as const).map((tab) => (
              <Button
                key={tab}
                variant={activeTab === tab ? 'default' : 'outline'}
                size="sm"
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'extract' && <Copy className="w-4 h-4 mr-2" />}
                {tab === 'delete' && <Trash2 className="w-4 h-4 mr-2" />}
                {tab === 'merge' && <GitMerge className="w-4 h-4 mr-2" />}
                {tab === 'rename' && <Pencil className="w-4 h-4 mr-2" />}
                {tab === 'add' && <Plus className="w-4 h-4 mr-2" />}
                {tab === 'remove-unannotated' && <ImageMinus className="w-4 h-4 mr-2" />}
                {TAB_LABEL_ZH[tab]}
              </Button>
            ))}
          </div>

          {activeTab === 'extract' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Copy className="w-5 h-5" />
                  提取类别到新数据集
                </CardTitle>
                <CardDescription>
                  创建一个仅包含所选类别的新数据集，只有带有这些类别标注的图片才会被收录。
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>输出数据集名称</Label>
                  <Input
                    value={extractOutputName}
                    onChange={(e) => setExtractOutputName(e.target.value)}
                    placeholder="新数据集名称"
                  />
                </div>

                <div className="space-y-2">
                  <Label>输出格式</Label>
                  <select
                    className="w-full border rounded-md px-3 py-2 text-sm bg-background"
                    value={extractOutputFormat}
                    onChange={(e) => setExtractOutputFormat(e.target.value)}
                  >
                    {EXPORT_FORMATS.map(f => (
                      <option key={f.id} value={f.id}>{f.name}</option>
                    ))}
                  </select>
                  <p className="text-xs text-muted-foreground">
                    当前格式：{selectedDataset?.format || '—'}。更改后将在提取时同步转换格式。
                  </p>
                </div>

                {selectedClasses.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">待提取的类别：</p>
                    <div className="flex flex-wrap gap-1">
                      {selectedClasses.map(c => (
                        <span key={c.name} className="px-2 py-0.5 bg-background rounded text-xs">
                          {c.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <Button
                  onClick={handleExtractClasses}
                  disabled={loading || selectedClasses.length === 0}
                  className="w-full"
                >
                  {loading ? '正在提取…' : `提取 ${selectedClasses.length} 个类别`}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'delete' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Trash2 className="w-5 h-5 text-destructive" />
                  删除类别
                </CardTitle>
                <CardDescription>
                  从数据集中移除所选类别及其全部标注
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-destructive">警告</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        此操作将永久删除所选类别的全部标注，
                        图片文件会保留，但这些类别对应的标签将被移除。
                      </p>
                    </div>
                  </div>
                </div>

                {selectedClasses.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">待删除的类别：</p>
                    <div className="flex flex-wrap gap-1">
                      {selectedClasses.map(c => (
                        <span key={c.name} className="px-2 py-0.5 bg-destructive/20 text-destructive rounded text-xs">
                          {c.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <Button 
                  variant="destructive"
                  onClick={handleDeleteClasses} 
                  disabled={loading || selectedClasses.length === 0}
                  className="w-full"
                >
                  {loading ? '正在删除…' : `删除 ${selectedClasses.length} 个类别`}
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'merge' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <GitMerge className="w-5 h-5" />
                  合并类别
                </CardTitle>
                <CardDescription>
                  将多个类别合并为一个类别
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>目标类别名称</Label>
                  <Input 
                    value={mergeTargetName}
                    onChange={(e) => setMergeTargetName(e.target.value)}
                    placeholder="合并后的类别名称"
                  />
                  <p className="text-xs text-muted-foreground">
                    所有选中的类别都会被重命名为该类别
                  </p>
                </div>

                {selectedClasses.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">待合并的类别：</p>
                    <div className="flex flex-wrap gap-1 items-center">
                      {selectedClasses.map((c, idx) => (
                        <span key={c.name}>
                          <span className="px-2 py-0.5 bg-background rounded text-xs">{c.name}</span>
                          {idx < selectedClasses.length - 1 && (
                            <span className="mx-1 text-muted-foreground">+</span>
                          )}
                        </span>
                      ))}
                      {mergeTargetName && (
                        <>
                          <ArrowRight className="w-4 h-4 mx-2 text-muted-foreground" />
                          <span className="px-2 py-0.5 bg-primary text-primary-foreground rounded text-xs">
                            {mergeTargetName}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                )}

                <Button 
                  onClick={handleMergeClasses} 
                  disabled={loading || selectedClasses.length < 2 || !mergeTargetName}
                  className="w-full"
                >
                  {loading ? '正在合并…' : `合并 ${selectedClasses.length} 个类别`}
                </Button>
                
                {selectedClasses.length < 2 && (
                  <p className="text-xs text-muted-foreground text-center">
                    请至少选择 2 个类别再进行合并
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {activeTab === 'rename' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Pencil className="w-5 h-5" />
                  重命名类别
                </CardTitle>
                <CardDescription>
                  在数据集中统一重命名某个类别
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedClasses.length === 1 ? (
                  <>
                    <div className="p-3 bg-muted rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">正在重命名的类别：</p>
                      <p className="font-medium">{selectedClasses[0].name}</p>
                      <p className="text-xs text-muted-foreground mt-1">{selectedClasses[0].count} 个标注</p>
                    </div>
                    <div className="space-y-2">
                      <Label>新的类别名称</Label>
                      <Input
                        value={renameNewName}
                        onChange={(e) => setRenameNewName(e.target.value)}
                        placeholder="请输入新的名称"
                        onKeyDown={(e) => e.key === 'Enter' && handleRenameClass()}
                      />
                    </div>
                    {renameNewName && (
                      <div className="flex items-center gap-2 text-sm p-2 bg-muted rounded-lg">
                        <span className="px-2 py-0.5 bg-background rounded text-xs">{selectedClasses[0].name}</span>
                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        <span className="px-2 py-0.5 bg-primary text-primary-foreground rounded text-xs">{renameNewName}</span>
                      </div>
                    )}
                    <Button
                      onClick={handleRenameClass}
                      disabled={loading || !renameNewName.trim() || renameNewName.trim() === selectedClasses[0].name}
                      className="w-full"
                    >
                      {loading ? '正在重命名…' : '重命名类别'}
                      <Pencil className="w-4 h-4 ml-2" />
                    </Button>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    请只选择 1 个类别进行重命名
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {activeTab === 'add' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Plus className="w-5 h-5" />
                  新增类别
                </CardTitle>
                <CardDescription>
                  为数据集新增一个类别，便于手动标注
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>类别名称</Label>
                  <Input
                    value={newClassName}
                    onChange={(e) => setNewClassName(e.target.value)}
                    placeholder="请输入新的类别名称"
                    onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
                  />
                </div>

                <Button
                  onClick={handleAddClass}
                  disabled={loading || !newClassName.trim()}
                  className="w-full"
                >
                  {loading ? '正在添加…' : '添加类别'}
                  <Plus className="w-4 h-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'remove-unannotated' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <ImageMinus className="w-5 h-5 text-destructive" />
                  移除未标注图片
                </CardTitle>
                <CardDescription>
                  {randomRemovalEnabled
                    ? `从未标注图片中随机删除 ${randomRemovalRatio}%（约 ${plannedRemovalCount} 张）`
                    : '从数据集中永久删除所有没有任何标注的图片'}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {unannotatedCount && (
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="p-3 bg-muted rounded-lg">
                      <p className="text-2xl font-bold">{unannotatedCount.total}</p>
                      <p className="text-xs text-muted-foreground">图片总数</p>
                    </div>
                    <div className="p-3 bg-destructive/10 rounded-lg">
                      <p className="text-2xl font-bold text-destructive">{plannedRemovalCount}</p>
                      <p className="text-xs text-muted-foreground">预计移除</p>
                    </div>
                    <div className="p-3 bg-emerald-500/10 rounded-lg">
                      <p className="text-2xl font-bold text-emerald-700 dark:text-emerald-400">{unannotatedCount.total - plannedRemovalCount}</p>
                      <p className="text-xs text-muted-foreground">预计保留</p>
                    </div>
                  </div>
                )}
                <div className="space-y-3 rounded-lg border border-border/60 bg-muted/20 p-4">
                  <div className="flex items-center gap-3">
                    <Checkbox id="random-remove-unannotated" checked={randomRemovalEnabled}
                      onCheckedChange={checked => setRandomRemovalEnabled(checked === true)} />
                    <Label htmlFor="random-remove-unannotated" className="cursor-pointer font-medium">
                      随机移除未标注图片
                    </Label>
                    <span className="ml-auto text-xs text-muted-foreground">默认关闭</span>
                  </div>
                  <div className="space-y-2 pl-7">
                    <div className="flex items-center justify-between gap-3">
                      <Label htmlFor="random-remove-ratio" className="text-sm text-muted-foreground">随机移除比例</Label>
                      <span className="min-w-12 rounded bg-background px-2 py-1 text-center text-sm font-semibold tabular-nums">
                        {randomRemovalRatio}%
                      </span>
                    </div>
                    <Slider id="random-remove-ratio" value={[randomRemovalRatio]}
                      onValueChange={([value]) => setRandomRemovalRatio(value)}
                      min={5} max={100} step={5} disabled={!randomRemovalEnabled || loading}
                      aria-label="随机移除比例" />
                    <p className="text-xs text-muted-foreground">
                      {randomRemovalEnabled
                        ? `每次操作从 ${unannotatedCount?.unannotated ?? 0} 张未标注图片中随机抽取 ${plannedRemovalCount} 张删除。`
                        : '开启后按比例随机抽取未标注图片；关闭时保持原行为，移除全部未标注图片。'}
                    </p>
                  </div>
                </div>
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-destructive">警告</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {randomRemovalEnabled
                          ? `此操作将随机永久删除 ${plannedRemovalCount} 张未标注图片；其余未标注图片会保留。`
                          : '此操作将永久删除所有标注数为 0 的图片文件，'}
                        对应的标签文件也会一并移除，且无法撤销。
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  可在删除类别后清理数据集，也可以在训练前剔除那些
                  从未标注过的图片。
                </p>
                <Button
                  variant="destructive"
                  onClick={handleRemoveUnannotated}
                  disabled={loading || !unannotatedCount || plannedRemovalCount === 0}
                  className="w-full"
                >
                  {loading
                    ? '正在移除…'
                    : plannedRemovalCount === 0
                      ? '没有可移除的未标注图片'
                      : randomRemovalEnabled
                        ? `随机移除 ${plannedRemovalCount} 张未标注图片`
                        : `移除 ${plannedRemovalCount} 张未标注图片`}
                  <ImageMinus className="w-4 h-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
