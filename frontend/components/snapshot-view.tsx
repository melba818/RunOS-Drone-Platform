'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { 
  GitBranch, Plus, Clock, HardDrive, Download, Trash2, 
  Loader2, CheckCircle2, AlertCircle, RefreshCw 
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'
import { toast } from 'sonner'

interface SnapshotViewProps {
  selectedDataset: Dataset | null
  datasets: Dataset[]
  setDatasets: (d: Dataset[]) => void
  apiUrl: string
}

interface Snapshot {
  id: string
  name: string
  dataset_id: string
  created_at: string
  num_images: number
  num_annotations: number
  description: string
  size_mb?: number
}

export function SnapshotView({ selectedDataset, datasets, setDatasets, apiUrl }: SnapshotViewProps) {
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [isCreating, setIsCreating] = useState(false)
  const [snapshotName, setSnapshotName] = useState('')
  const [snapshotDesc, setSnapshotDesc] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // Load snapshots from backend (with localStorage fallback for locally-created ones)
  const loadSnapshots = async (datasetId: string) => {
    setIsLoading(true)
    try {
      const res = await fetch(`${apiUrl}/api/datasets/${datasetId}/snapshots`)
      if (res.ok) {
        const data = await res.json()
        setSnapshots(data.snapshots || [])
        // Persist to localStorage so they're available offline
        localStorage.setItem(`snapshots_${datasetId}`, JSON.stringify(data.snapshots || []))
        setIsLoading(false)
        return
      }
    } catch {}
    // Fallback: load from localStorage
    const stored = localStorage.getItem(`snapshots_${datasetId}`)
    if (stored) {
      try { setSnapshots(JSON.parse(stored)) } catch {}
    }
    setIsLoading(false)
  }

  useEffect(() => {
    if (!selectedDataset) return
    loadSnapshots(selectedDataset.id)
  }, [selectedDataset?.id, apiUrl])

  const saveSnapshots = (snaps: Snapshot[]) => {
    if (!selectedDataset) return
    setSnapshots(snaps)
    localStorage.setItem(`snapshots_${selectedDataset.id}`, JSON.stringify(snaps))
  }

  const createSnapshot = async () => {
    if (!selectedDataset) return
    if (!snapshotName.trim()) { toast.error('请输入快照名称'); return }
    setIsCreating(true)
    try {
      const res = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/snapshot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: snapshotName.trim(), description: snapshotDesc.trim() })
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: '创建快照失败' }))
        throw new Error(err.detail || '创建快照失败')
      }
      setSnapshotName('')
      setSnapshotDesc('')
      toast.success(`已创建快照「${snapshotName.trim()}」`)
      // Reload from backend to get the accurate list
      await loadSnapshots(selectedDataset.id)
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '创建快照失败')
    } finally {
      setIsCreating(false)
    }
  }

  const deleteSnapshot = async (snap: Snapshot) => {
    if (!confirm('确定删除该快照吗？对应的 .zip 文件将被永久移除。')) return
    try {
      const res = await fetch(`${apiUrl}/api/datasets/${snap.dataset_id}/snapshot/${snap.id}`, {
        method: 'DELETE'
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: '删除失败' }))
        throw new Error(err.detail)
      }
      const updated = snapshots.filter(s => s.id !== snap.id)
      saveSnapshots(updated)
      toast.success('快照已删除')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : '删除快照失败')
    }
  }

  const downloadSnapshot = async (snap: Snapshot) => {
    try {
      // Try to download from backend
      const res = await fetch(`${apiUrl}/api/datasets/${snap.dataset_id}/snapshot/${snap.id}/download`)
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${snap.name}.zip`
        a.click()
        URL.revokeObjectURL(url)
      } else {
        // Fallback: export current dataset
        window.open(`${apiUrl}/api/export/${snap.dataset_id}`, '_blank')
      }
    } catch {
      window.open(`${apiUrl}/api/export/${snap.dataset_id}`, '_blank')
    }
  }

  const restoreSnapshot = async (snap: Snapshot) => {
    if (!confirm(`确定将数据集恢复到快照「${snap.name}」吗？当前的标注将被覆盖。`)) return
    try {
      const res = await fetch(`${apiUrl}/api/datasets/${snap.dataset_id}/snapshot/${snap.id}/restore`, {
        method: 'POST'
      })
      if (res.ok) {
        toast.success(`数据集已恢复到快照「${snap.name}」`)
      } else {
        toast.info('恢复功能需要后端支持。请下载快照 .zip 后手动重新导入。')
      }
    } catch {
      toast.info('恢复功能需要后端支持。请下载快照后手动恢复。')
    }
  }

  if (!selectedDataset) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center">
          <GitBranch className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-lg font-medium">未选择数据集</p>
          <p className="text-sm text-muted-foreground mt-1">选择一个数据集以管理快照</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">数据集快照</h2>
        <p className="text-muted-foreground text-sm mt-1">
          在进行数据增强、划分或重大改动之前，创建冻结的存档点
        </p>
      </div>

      {/* Create snapshot */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Plus className="w-4 h-4" /> 创建快照
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">快照名称 *</label>
            <Input
              placeholder="例如 before-augmentation、v1.0-clean、post-review"
              value={snapshotName}
              onChange={e => setSnapshotName(e.target.value)}
              className="max-w-md"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">描述（可选）</label>
            <Input
              placeholder="记录本次改动内容，或该存档点的重要性"
              value={snapshotDesc}
              onChange={e => setSnapshotDesc(e.target.value)}
              className="max-w-md"
            />
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground pt-1">
            <div className="flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5" />
              <span>{selectedDataset.num_images} 张图片</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{selectedDataset.num_annotations} 个标注</span>
            </div>
          </div>
          <Button
            onClick={createSnapshot}
            disabled={isCreating || !snapshotName.trim()}
            className="gap-2"
          >
            {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitBranch className="w-4 h-4" />}
            创建快照
          </Button>
        </CardContent>
      </Card>

      {/* Snapshot history */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Clock className="w-4 h-4" />
            快照历史（{snapshots.length}）
          </CardTitle>
        </CardHeader>
        <CardContent>
          {snapshots.length === 0 ? (
            <div className="text-center py-10">
              <GitBranch className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
              <p className="text-muted-foreground">尚无快照。</p>
              <p className="text-xs text-muted-foreground mt-1">在进行重大改动前创建快照，以保留当前成果。</p>
            </div>
          ) : (
            <div className="space-y-3">
              {snapshots.map((snap, idx) => (
                <div
                  key={snap.id}
                  className="flex items-start gap-4 p-4 rounded-lg border border-border/50 hover:border-border bg-muted/20"
                >
                  <div className="flex-shrink-0 mt-0.5">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary text-xs font-bold">
                      {idx === 0 ? '●' : `${idx + 1}`}
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="font-medium">{snap.name}</span>
                      {idx === 0 && <Badge variant="default" className="text-xs">最新</Badge>}
                    </div>
                    <p className="text-xs text-muted-foreground">{snap.description}</p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1"><Clock className="w-3 h-3" />{new Date(snap.created_at).toLocaleString()}</span>
                      <span>{snap.num_images} 张图片</span>
                      <span>{snap.num_annotations} 个标注</span>
                      {snap.size_mb && <span>{snap.size_mb.toFixed(1)} MB</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <Button size="sm" variant="outline" className="gap-1 h-7" onClick={() => downloadSnapshot(snap)}>
                      <Download className="w-3.5 h-3.5" />
                      导出
                    </Button>
                    <Button size="sm" variant="outline" className="gap-1 h-7" onClick={() => restoreSnapshot(snap)}>
                      <RefreshCw className="w-3.5 h-3.5" />
                      恢复
                    </Button>
                    <Button size="sm" variant="ghost" className="h-7 text-destructive hover:text-destructive" onClick={() => deleteSnapshot(snap)}>
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Info card */}
      <Card className="border-blue-500/20 bg-blue-500/5">
        <CardContent className="flex items-start gap-3 pt-4">
          <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-medium text-blue-600 dark:text-blue-400">关于快照</p>
            <p className="text-muted-foreground mt-1">
              快照是数据集的完整 .zip 压缩包，保存在后端的 <code className="text-xs bg-muted px-1 rounded">workspace/snapshots/</code> 目录中，
              重启后端或更换浏览器后依然保留。点击<strong>导出</strong>可下载快照压缩包，点击<strong>恢复</strong>可将数据集回退到该存档状态。
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
