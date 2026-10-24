'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import {
  BarChart3,
  Image as ImageIcon,
  Tag,
  Layers,
  TrendingUp,
  Database,
  RefreshCw,
  Download,
  ChevronRight,
  PenTool,
  ArrowUpDown,
  ArrowLeftRight,
  Cpu,
  PieChart,
  AlertCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset, ViewType } from '@/app/page'

interface DashboardViewProps {
  datasets: Dataset[]
  selectedDataset: Dataset | null
  onSelectDataset: (dataset: Dataset) => void
  setActiveView: (view: ViewType) => void
  apiUrl: string
}

interface DatasetStats {
  dataset_id: string
  name: string
  format: string
  task_type: string
  total_images: number
  total_annotations: number
  classes: Record<string, any>
  class_distribution: Record<string, number>
  splits: Record<string, number>
  image_sizes: Record<string, any>
  avg_annotations_per_image: number
  created_at: string
}

const CLASS_COLORS = [
  'bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500',
  'bg-violet-500', 'bg-cyan-500', 'bg-orange-500', 'bg-pink-500',
  'bg-teal-500', 'bg-indigo-500', 'bg-lime-500', 'bg-fuchsia-500',
]

/** 任务类型内部 key 保持英文（用于逻辑判断），仅展示时映射为中文 */
const TASK_TYPE_LABEL_ZH: Record<string, string> = {
  classification: '分类',
  detection: '检测',
  segmentation: '分割',
  keypoint: '关键点',
  'obb-detection': 'OBB 检测',
}

const SPLIT_COLORS: Record<string, string> = {
  train: 'bg-primary',
  val: 'bg-emerald-500',
  valid: 'bg-emerald-500',
  validation: 'bg-emerald-500',
  test: 'bg-amber-500',
}

export function DashboardView({
  datasets,
  selectedDataset,
  onSelectDataset,
  setActiveView,
  apiUrl,
}: DashboardViewProps) {
  const [stats, setStats] = useState<DatasetStats | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [statsError, setStatsError] = useState<string | null>(null)

  useEffect(() => {
    if (selectedDataset) {
      setStats(null)
      setStatsError(null)
      loadStats()
    }
  }, [selectedDataset])

  const loadStats = async (forceRefresh = false) => {
    if (!selectedDataset) return
    setIsLoading(true)
    setStatsError(null)
    try {
      const url = `${apiUrl}/api/datasets/${selectedDataset.id}/stats${forceRefresh ? '?force_refresh=true' : ''}`
      const response = await fetch(url)
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      } else {
        setStatsError('加载统计信息失败')
      }
    } catch {
      setStatsError('无法连接后端服务')
    } finally {
      setIsLoading(false)
    }
  }

  const exportReport = () => {
    if (!stats || !selectedDataset) return
    const report = {
      dataset: {
        id: selectedDataset.id,
        name: selectedDataset.name,
        format: selectedDataset.format,
        task_type: selectedDataset.task_type,
        created_at: selectedDataset.created_at,
      },
      summary: {
        total_images: stats.total_images,
        total_annotations: stats.total_annotations,
        num_classes: Object.keys(stats.class_distribution || {}).length,
        avg_annotations_per_image: stats.avg_annotations_per_image,
      },
      class_distribution: stats.class_distribution,
      splits: stats.splits,
      classes: stats.classes,
      generated_at: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selectedDataset.name}_report_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    const csv = ['class,count,percentage']
    const total = Object.values(stats.class_distribution || {}).reduce((a: number, b: unknown) => a + (b as number), 0) as number
    Object.entries(stats.class_distribution || {}).forEach(([cls, count]) => {
      csv.push(`${cls},${count},${total > 0 ? (((count as number) / total) * 100).toFixed(2) : 0}%`)
    })
    const csvBlob = new Blob([csv.join('\n')], { type: 'text/csv' })
    const csvUrl = URL.createObjectURL(csvBlob)
    const csvA = document.createElement('a')
    csvA.href = csvUrl
    csvA.download = `${selectedDataset.name}_classes_${new Date().toISOString().split('T')[0]}.csv`
    setTimeout(() => { csvA.click(); URL.revokeObjectURL(csvUrl) }, 300)
  }

  const totalAnnotations = stats?.class_distribution
    ? Object.values(stats.class_distribution).reduce((a, b) => a + b, 0)
    : 0

  const sortedClasses = stats?.class_distribution
    ? Object.entries(stats.class_distribution).sort(([, a], [, b]) => b - a)
    : []

  /* ── No dataset selected ───────────────────────────────────────────── */
  if (!selectedDataset) {
    return (
      <div className="h-full flex flex-col p-6 overflow-y-auto">
        {/* Header */}
        <div className="mb-6">
          <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground/50 mb-2">数据概览</p>
          <h2 className="text-2xl font-display font-bold tracking-tight">数据集概览</h2>
          <p className="text-sm text-muted-foreground mt-1">
            请选择一个数据集，查看其统计与分析数据。
          </p>
        </div>

        {datasets.length === 0 ? (
          /* No datasets at all — onboarding */
          <div className="flex-1 flex items-center justify-center">
            <div className="max-w-sm w-full text-center space-y-6">
              <div className="w-16 h-16 rounded-2xl bg-primary/8 border border-primary/15 flex items-center justify-center mx-auto">
                <Database className="w-7 h-7 text-primary" strokeWidth={1.5} />
              </div>

              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold">尚未加载任何数据集</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  加载一个图像数据集，即可开始标注、处理与训练 AI 模型。
                </p>
              </div>

              {/* Quick-start steps */}
              <div className="text-left bg-muted/40 rounded-xl border border-border p-4 space-y-3">
                <p className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground/60">使用流程</p>
                {[
                  { n: '1', label: '加载数据集', desc: '上传 ZIP 或指定本地文件夹' },
                  { n: '2', label: '标注图像', desc: '在图像上绘制边界框与标签' },
                  { n: '3', label: '训练 AI 模型', desc: '实时监控损失与准确率' },
                ].map(({ n, label, desc }) => (
                  <div key={n} className="flex items-start gap-3">
                    <span className="w-5 h-5 rounded-full bg-primary/15 border border-primary/25 text-primary text-[9px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {n}
                    </span>
                    <div>
                      <p className="text-xs font-medium">{label}</p>
                      <p className="text-[11px] text-muted-foreground">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>

              <Button onClick={() => setActiveView('datasets')} className="w-full" size="lg">
                <Database className="w-4 h-4 mr-2" strokeWidth={1.75} />
                加载你的第一个数据集
              </Button>
            </div>
          </div>
        ) : (
          /* Datasets exist — show picker */
          <div>
            <p className="text-xs text-muted-foreground mb-4">
              点击一个数据集，查看其统计信息
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {datasets.map((dataset) => (
                <button
                  key={dataset.id}
                  className="p-4 rounded-xl border border-border bg-card hover:border-primary/30 hover:bg-primary/5 transition-all text-left group"
                  onClick={() => onSelectDataset(dataset)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="w-8 h-8 rounded-lg bg-primary/8 border border-primary/15 flex items-center justify-center">
                      <Database className="w-3.5 h-3.5 text-primary" strokeWidth={1.75} />
                    </div>
                    <ChevronRight className="w-4 h-4 text-muted-foreground/30 group-hover:text-primary transition-colors" />
                  </div>
                  <p className="font-semibold text-sm truncate mb-1">{dataset.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(!dataset.format || dataset.format === 'unknown' || dataset.format === 'generic-images') ? (dataset.format === 'generic-images' ? '图像' : '未知') : dataset.format.toUpperCase()} · {dataset.num_images.toLocaleString()} 张图像 · {dataset.classes?.length || 0} 个类别
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  /* ── Dataset selected ──────────────────────────────────────────────── */
  return (
    <div className="p-6 space-y-5 overflow-y-auto h-full">

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-muted-foreground/50 mb-1">数据概览</p>
          <h2 className="text-2xl font-display font-bold tracking-tight">{selectedDataset.name}</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            {(!selectedDataset.format || selectedDataset.format === 'unknown' ? '未知格式' : selectedDataset.format === 'generic-images' ? '仅图像' : selectedDataset.format.toUpperCase())} · {(TASK_TYPE_LABEL_ZH[selectedDataset.task_type] ?? (selectedDataset.task_type || '未知类型'))} · 创建于 {new Date(selectedDataset.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="outline" size="sm" onClick={() => loadStats(true)} disabled={isLoading}>
            <RefreshCw className={cn('w-3.5 h-3.5 mr-1.5', isLoading && 'animate-spin')} strokeWidth={1.75} />
            刷新
          </Button>
          <Button variant="outline" size="sm" onClick={exportReport} disabled={!stats}>
            <Download className="w-3.5 h-3.5 mr-1.5" strokeWidth={1.75} />
            导出报告
          </Button>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          {
            label: '图像总数',
            value: (stats?.total_images ?? selectedDataset.num_images).toLocaleString(),
            icon: ImageIcon,
            iconColor: 'text-blue-600 dark:text-blue-400',
            iconBg: 'bg-blue-500/10',
            desc: '该数据集中的图像数量',
          },
          {
            label: '标注总数',
            value: (stats?.total_annotations ?? selectedDataset.num_annotations).toLocaleString(),
            icon: Tag,
            iconColor: 'text-emerald-600 dark:text-emerald-400',
            iconBg: 'bg-emerald-500/10',
            desc: '图像上已绘制的标签数',
          },
          {
            label: '类别数',
            value: Object.keys(stats?.class_distribution || {}).length || selectedDataset.classes.length,
            icon: Layers,
            iconColor: 'text-amber-600 dark:text-amber-400',
            iconBg: 'bg-amber-500/10',
            desc: '不同的目标类别数',
          },
          {
            label: '每图像均值',
            value: (stats?.avg_annotations_per_image ?? (selectedDataset.num_images > 0 ? selectedDataset.num_annotations / selectedDataset.num_images : 0)).toFixed(1),
            icon: TrendingUp,
            iconColor: 'text-violet-500',
            iconBg: 'bg-violet-500/10',
            desc: '每张图像的标注数',
          },
        ].map(({ label, value, icon: Icon, iconColor, iconBg, desc }) => (
          <div key={label} className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-start justify-between mb-3">
              <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/70 leading-tight">{label}</p>
              <div className={cn('w-7 h-7 rounded-lg flex items-center justify-center shrink-0', iconBg)}>
                <Icon className={cn('w-3.5 h-3.5', iconColor)} strokeWidth={1.75} />
              </div>
            </div>
            <p className="text-3xl font-display font-bold tracking-tight">{value}</p>
            <p className="text-[10px] text-muted-foreground/60 mt-1">{desc}</p>
          </div>
        ))}
      </div>

      {/* Class distribution + Splits */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Class distribution */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <BarChart3 className="w-3.5 h-3.5 text-muted-foreground" strokeWidth={1.75} />
                  类别分布
                </CardTitle>
                <CardDescription className="text-[11px] mt-0.5">
                  标注在各目标类别之间的分布情况
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-[340px] overflow-y-auto pr-1">
              {isLoading && !stats ? (
                /* Loading skeleton */
                Array.from({ length: 4 }).map((_, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1.5">
                      <Skeleton className="h-3 w-24 rounded" />
                      <Skeleton className="h-3 w-16 rounded" />
                    </div>
                    <Skeleton className="h-1.5 w-full rounded-full" />
                  </div>
                ))
              ) : statsError ? (
                <div className="flex flex-col items-center gap-2 py-6 text-center">
                  <AlertCircle className="w-4 h-4 text-destructive" />
                  <p className="text-xs text-muted-foreground">{statsError}</p>
                </div>
              ) : sortedClasses.length > 0 ? (
                sortedClasses.map(([className, count], idx) => {
                  const percentage = totalAnnotations > 0 ? (count / totalAnnotations) * 100 : 0
                  return (
                    <div key={className}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium truncate flex-1 mr-2">{className}</span>
                        <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                          {count.toLocaleString()} <span className="text-muted-foreground/50">({percentage.toFixed(1)}%)</span>
                        </span>
                      </div>
                      <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={cn('h-full rounded-full transition-all', CLASS_COLORS[idx % CLASS_COLORS.length])}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  )
                })
              ) : (
                <p className="text-xs text-muted-foreground text-center py-6">
                  该数据集暂无类别数据
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Dataset splits */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <PieChart className="w-3.5 h-3.5 text-muted-foreground" strokeWidth={1.75} />
              数据集划分
            </CardTitle>
            <CardDescription className="text-[11px]">
              图像在 train / val / test 之间的划分情况
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading && !stats ? (
              /* Loading skeleton */
              <div className="space-y-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between mb-1.5">
                      <Skeleton className="h-3 w-12 rounded" />
                      <Skeleton className="h-3 w-20 rounded" />
                    </div>
                    <Skeleton className="h-2 w-full rounded-full" />
                  </div>
                ))}
              </div>
            ) : statsError ? (
              <div className="flex flex-col items-center gap-2 py-6 text-center">
                <AlertCircle className="w-4 h-4 text-destructive" />
                <p className="text-xs text-muted-foreground">{statsError}</p>
              </div>
            ) : stats?.splits && Object.keys(stats.splits).length > 0 ? (
              <div className="space-y-4">
                {Object.entries(stats.splits).map(([split, count]) => {
                  const totalImgs = stats?.total_images || selectedDataset.num_images || 1
                  const percentage = (count / totalImgs) * 100
                  const colorClass = SPLIT_COLORS[split] || 'bg-muted-foreground'
                  return (
                    <div key={split}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium capitalize">{split}</span>
                        <span className="text-xs text-muted-foreground tabular-nums">
                          {count.toLocaleString()} <span className="text-muted-foreground/50">({percentage.toFixed(1)}%)</span>
                        </span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={cn('h-full rounded-full transition-all', colorClass)}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-8 space-y-3">
                <p className="text-xs text-muted-foreground">
                  尚未配置划分。请对数据集进行划分，以区分训练数据与测试数据。
                </p>
                <Button variant="outline" size="sm" onClick={() => setActiveView('split')}>
                  创建划分
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Dataset info */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">数据集信息</CardTitle>
          <CardDescription className="text-[11px]">该数据集的技术细节</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
            {[
              { label: '格式', value: (!selectedDataset.format || selectedDataset.format === 'unknown' ? '未知' : selectedDataset.format === 'generic-images' ? '仅图像（无标注）' : selectedDataset.format.toUpperCase()) },
              { label: '任务类型', value: (TASK_TYPE_LABEL_ZH[stats?.task_type || selectedDataset.task_type] ?? (stats?.task_type || selectedDataset.task_type || '未知')) },
              { label: '创建时间', value: new Date(selectedDataset.created_at).toLocaleDateString() },
              { label: '数据集 ID', value: `${selectedDataset.id.slice(0, 12)}…`, mono: true },
            ].map(({ label, value, mono }) => (
              <div key={label}>
                <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/60 mb-1">{label}</p>
                <p className={cn('text-sm font-medium capitalize', mono && 'font-mono text-xs')}>{value}</p>
              </div>
            ))}
          </div>

          {selectedDataset.classes.length > 0 && (
            <div>
              <p className="text-[10px] font-mono uppercase tracking-[0.12em] text-muted-foreground/60 mb-2">类别</p>
              <div className="flex flex-wrap gap-1.5">
                {selectedDataset.classes.map((cls, idx) => (
                  <span
                    key={cls}
                    className={cn(
                      'px-2 py-0.5 rounded-md text-[11px] font-medium text-white',
                      CLASS_COLORS[idx % CLASS_COLORS.length]
                    )}
                  >
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick actions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold">快捷操作</CardTitle>
          <CardDescription className="text-[11px]">针对该数据集的常用下一步</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { view: 'annotate' as ViewType, icon: PenTool, label: '标注', desc: '为图像打标签' },
              { view: 'sorting' as ViewType, icon: ArrowUpDown, label: '筛选排序', desc: '整理数据' },
              { view: 'convert' as ViewType, icon: ArrowLeftRight, label: '格式转换', desc: '切换数据格式' },
              { view: 'training' as ViewType, icon: Cpu, label: '训练模型', desc: '开始训练' },
            ].map(({ view, icon: Icon, label, desc }) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className="h-auto py-4 flex flex-col items-center gap-1.5 rounded-xl border border-border bg-card hover:border-primary/30 hover:bg-primary/5 transition-all"
              >
                <Icon className="w-5 h-5 text-muted-foreground" strokeWidth={1.75} />
                <span className="text-xs font-semibold">{label}</span>
                <span className="text-[10px] text-muted-foreground/70">{desc}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
