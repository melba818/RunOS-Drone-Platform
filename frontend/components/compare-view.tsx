'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { GitCompare, ArrowRight, TrendingUp, TrendingDown, Minus, RefreshCw, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'
import { toast } from 'sonner'

interface CompareViewProps {
  datasets: Dataset[]
  apiUrl: string
}

interface DatasetStats {
  total_images: number
  total_annotations: number
  class_distribution: Record<string, number>
  splits: Record<string, number>
  avg_annotations_per_image: number
  classes: Record<string, any>
}

export function CompareView({ datasets, apiUrl }: CompareViewProps) {
  const [datasetA, setDatasetA] = useState('')
  const [datasetB, setDatasetB] = useState('')
  const [statsA, setStatsA] = useState<DatasetStats | null>(null)
  const [statsB, setStatsB] = useState<DatasetStats | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const fetchStats = async (id: string): Promise<DatasetStats | null> => {
    const res = await fetch(`${apiUrl}/api/datasets/${id}/stats`)
    if (!res.ok) return null
    return res.json()
  }

  const runComparison = async () => {
    if (!datasetA || !datasetB) {
      toast.error('请先选择两个数据集进行对比')
      return
    }
    if (datasetA === datasetB) {
      toast.error('请选择两个不同的数据集')
      return
    }
    setIsLoading(true)
    try {
      const [a, b] = await Promise.all([fetchStats(datasetA), fetchStats(datasetB)])
      setStatsA(a)
      setStatsB(b)
    } catch {
      toast.error('获取统计信息失败')
    } finally {
      setIsLoading(false)
    }
  }

  const delta = (a: number, b: number) => {
    const diff = b - a
    const pct = a !== 0 ? ((diff / a) * 100).toFixed(1) : '∞'
    return { diff, pct, positive: diff > 0 }
  }

  const nameOf = (id: string) => datasets.find(d => d.id === id)?.name || id

  // All classes across both datasets
  const allClasses = [...new Set([
    ...Object.keys(statsA?.class_distribution || {}),
    ...Object.keys(statsB?.class_distribution || {})
  ])].sort()

  const DeltaBadge = ({ a, b, lowerBetter = false }: { a: number; b: number; lowerBetter?: boolean }) => {
    const d = delta(a, b)
    if (d.diff === 0) return <Badge variant="secondary" className="text-xs gap-1"><Minus className="w-3 h-3" />持平</Badge>
    const good = lowerBetter ? !d.positive : d.positive
    return (
      <Badge variant={good ? 'default' : 'destructive'} className="text-xs gap-1">
        {d.positive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
        {d.positive ? '+' : ''}{d.diff.toLocaleString()} ({d.pct}%)
      </Badge>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">数据集对比</h2>
        <p className="text-muted-foreground text-sm mt-1">并排比较两个数据集的统计信息</p>
      </div>

      {/* Dataset selectors */}
      <Card>
        <CardContent className="pt-5">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <label className="text-xs text-muted-foreground mb-1 block">数据集 A</label>
              <Select value={datasetA} onValueChange={setDatasetA}>
                <SelectTrigger><SelectValue placeholder="选择数据集 A…" /></SelectTrigger>
                <SelectContent>
                  {datasets.map(d => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <ArrowRight className="w-5 h-5 text-muted-foreground mt-5 flex-shrink-0" />
            <div className="flex-1 min-w-[200px]">
              <label className="text-xs text-muted-foreground mb-1 block">数据集 B</label>
              <Select value={datasetB} onValueChange={setDatasetB}>
                <SelectTrigger><SelectValue placeholder="选择数据集 B…" /></SelectTrigger>
                <SelectContent>
                  {datasets.map(d => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={runComparison} disabled={isLoading || !datasetA || !datasetB} className="mt-5 gap-2">
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitCompare className="w-4 h-4" />}
              对比
            </Button>
          </div>
        </CardContent>
      </Card>

      {!statsA && !statsB && !isLoading && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <GitCompare className="w-12 h-12 text-muted-foreground mb-4" />
          <p className="text-lg font-medium">选择两个数据集进行对比</p>
          <p className="text-sm text-muted-foreground mt-1">查看规模、类别、划分与标注密度的差异</p>
        </div>
      )}

      {statsA && statsB && (
        <>
          {/* Summary comparison */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">总体对比</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 text-muted-foreground font-medium">指标</th>
                      <th className="text-right py-2 text-blue-700 dark:text-blue-400">{nameOf(datasetA)}</th>
                      <th className="text-right py-2 text-emerald-700 dark:text-emerald-400">{nameOf(datasetB)}</th>
                      <th className="text-right py-2 text-muted-foreground">变化量</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {[
                      { label: '图片总数', a: statsA.total_images, b: statsB.total_images },
                      { label: '标注总数', a: statsA.total_annotations, b: statsB.total_annotations },
                      { label: '类别数', a: Object.keys(statsA.class_distribution || {}).length, b: Object.keys(statsB.class_distribution || {}).length },
                      { label: '平均标注数 / 图', a: Math.round(statsA.avg_annotations_per_image * 100) / 100, b: Math.round(statsB.avg_annotations_per_image * 100) / 100 },
                    ].map(row => (
                      <tr key={row.label}>
                        <td className="py-2 font-medium">{row.label}</td>
                        <td className="text-right py-2">{row.a.toLocaleString()}</td>
                        <td className="text-right py-2">{row.b.toLocaleString()}</td>
                        <td className="text-right py-2"><DeltaBadge a={row.a} b={row.b} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Class distribution comparison */}
          {allClasses.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">类别分布</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto max-h-96">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border sticky top-0 bg-card">
                        <th className="text-left py-2 text-muted-foreground font-medium">类别</th>
                        <th className="text-right py-2 text-blue-700 dark:text-blue-400">{nameOf(datasetA)}</th>
                        <th className="text-right py-2 text-emerald-700 dark:text-emerald-400">{nameOf(datasetB)}</th>
                        <th className="text-right py-2 text-muted-foreground">变化</th>
                        <th className="py-2">存在情况</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/40">
                      {allClasses.map(cls => {
                        const a = (statsA.class_distribution || {})[cls] || 0
                        const b = (statsB.class_distribution || {})[cls] || 0
                        const onlyA = a > 0 && b === 0
                        const onlyB = a === 0 && b > 0
                        return (
                          <tr key={cls} className={cn(onlyA && 'bg-blue-500/5', onlyB && 'bg-emerald-500/5')}>
                            <td className="py-1.5 font-medium">{cls}</td>
                            <td className={cn('text-right py-1.5', a === 0 && 'text-muted-foreground')}>{a.toLocaleString()}</td>
                            <td className={cn('text-right py-1.5', b === 0 && 'text-muted-foreground')}>{b.toLocaleString()}</td>
                            <td className="text-right py-1.5">
                              {a > 0 && b > 0 ? <DeltaBadge a={a} b={b} /> : null}
                            </td>
                            <td className="py-1.5 text-center">
                              {onlyA && <Badge variant="secondary" className="text-xs bg-blue-500/20 text-blue-600">仅 A</Badge>}
                              {onlyB && <Badge variant="secondary" className="text-xs bg-emerald-500/20 text-emerald-600">仅 B</Badge>}
                              {!onlyA && !onlyB && <span className="text-xs text-muted-foreground">两者都有</span>}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Splits comparison */}
          {(statsA.splits || statsB.splits) && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">划分分布</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-6">
                  {[
                    { label: nameOf(datasetA), splits: statsA.splits, total: statsA.total_images, color: 'bg-blue-500' },
                    { label: nameOf(datasetB), splits: statsB.splits, total: statsB.total_images, color: 'bg-emerald-500' },
                  ].map(({ label, splits, total, color }) => (
                    <div key={label}>
                      <p className="text-sm font-medium mb-3">{label}</p>
                      {splits && Object.keys(splits).length > 0 ? (
                        <div className="space-y-2">
                          {Object.entries(splits).map(([split, count]) => (
                            <div key={split}>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="capitalize">{split}</span>
                                <span className="text-muted-foreground">{count} ({total > 0 ? ((count / total) * 100).toFixed(1) : 0}%)</span>
                              </div>
                              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                                <div className={cn('h-full rounded-full', color)} style={{ width: `${total > 0 ? (count / total) * 100 : 0}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">未定义划分</p>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
