'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  ShieldCheck, AlertTriangle, XCircle, CheckCircle2, 
  BarChart3, ImageOff, FileX, TrendingUp, RefreshCw, 
  Loader2, Eye, Scale
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset, ImageData, ImageCache } from '@/app/page'
import { toast } from 'sonner'

interface HealthViewProps {
  selectedDataset: Dataset | null
  apiUrl: string
  imageCache: ImageCache
  updateImageCache: (datasetId: string, images: ImageData[]) => void
}

interface HealthReport {
  totalImages: number
  annotatedImages: number
  unannotatedImages: number
  coveragePercent: number
  classImbalance: { class: string; count: number; ratio: number }[]
  maxImbalanceRatio: number
  smallImages: { id: string; filename: string; width: number; height: number }[]
  unannotatedList: { id: string; filename: string }[]
  classDistribution: Record<string, number>
  splitDistribution: Record<string, number>
  warnings: string[]
  score: number
}

export function HealthView({ selectedDataset, apiUrl, imageCache, updateImageCache }: HealthViewProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [report, setReport] = useState<HealthReport | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'imbalance' | 'coverage' | 'quality'>('overview')

  useEffect(() => {
    if (selectedDataset) setReport(null)
  }, [selectedDataset?.id])

  const runAnalysis = async () => {
    if (!selectedDataset) return
    setIsAnalyzing(true)
    try {
      // Fetch images (use cache if available)
      let images: ImageData[] = imageCache[selectedDataset.id] || []
      if (!images.length) {
        const res = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images?limit=999999`)
        const data = await res.json()
        images = data.images || []
        updateImageCache(selectedDataset.id, images)
      }

      // Fetch stats
      const statsRes = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/stats`)
      const stats = statsRes.ok ? await statsRes.json() : {}

      // Compute health metrics
      const annotated = images.filter(i => i.has_annotations || (i.annotations?.length ?? 0) > 0)
      const unannotated = images.filter(i => !i.has_annotations && !(i.annotations?.length))
      const coverage = images.length > 0 ? (annotated.length / images.length) * 100 : 0

      // Class distribution
      const classCounts: Record<string, number> = {}
      images.forEach(img => {
        (img.annotations || []).forEach(ann => {
          if (ann.class_name) classCounts[ann.class_name] = (classCounts[ann.class_name] || 0) + 1
        })
      })

      // Imbalance detection
      const classEntries = Object.entries(classCounts).sort(([, a], [, b]) => b - a)
      const maxCount = classEntries[0]?.[1] ?? 1
      const minCount = classEntries[classEntries.length - 1]?.[1] ?? 1
      const imbalanceRatio = minCount > 0 ? maxCount / minCount : maxCount
      const classImbalance = classEntries.map(([cls, count]) => ({
        class: cls, count, ratio: maxCount > 0 ? count / maxCount : 1
      }))

      // Small images (< 100px in any dimension based on metadata)
      const smallImages = images
        .filter(i => (i.width && i.width < 100) || (i.height && i.height < 100))
        .slice(0, 20)
        .map(i => ({ id: i.id, filename: i.filename, width: i.width || 0, height: i.height || 0 }))

      // Warnings
      const warnings: string[] = []
      if (coverage < 50) warnings.push(`标注覆盖率偏低：仅 ${coverage.toFixed(0)}% 的图片已标注`)
      if (imbalanceRatio > 10) warnings.push(`类别严重失衡：最大类别的样本数是最小类别的 ${imbalanceRatio.toFixed(0)} 倍`)
      else if (imbalanceRatio > 3) warnings.push(`类别中度失衡：最大类别与最小类别之比为 ${imbalanceRatio.toFixed(1)}×`)
      if (smallImages.length > 0) warnings.push(`有 ${smallImages.length} 张图片尺寸过小（< 100px），可能影响模型质量`)
      if (unannotated.length > images.length * 0.3) warnings.push(`有 ${unannotated.length} 张图片没有任何标注 — 建议自动标注或将其移除`)

      // Health score (0-100)
      let score = 100
      if (coverage < 100) score -= (100 - coverage) * 0.3
      if (imbalanceRatio > 3) score -= Math.min(25, (imbalanceRatio - 3) * 2)
      if (smallImages.length > 0) score -= Math.min(10, smallImages.length * 0.5)
      score = Math.max(0, Math.min(100, score))

      setReport({
        totalImages: images.length,
        annotatedImages: annotated.length,
        unannotatedImages: unannotated.length,
        coveragePercent: coverage,
        classImbalance,
        maxImbalanceRatio: imbalanceRatio,
        smallImages,
        unannotatedList: unannotated.slice(0, 50).map(i => ({ id: i.id, filename: i.filename })),
        classDistribution: classCounts,
        splitDistribution: stats.splits || {},
        warnings,
        score
      })
    } catch (e) {
      toast.error('分析失败：' + (e instanceof Error ? e.message : '未知错误'))
    } finally {
      setIsAnalyzing(false)
    }
  }

  const scoreColor = (s: number) =>
    s >= 80 ? 'text-emerald-700 dark:text-green-400' : s >= 60 ? 'text-amber-700 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'
  const scoreLabel = (s: number) =>
    s >= 80 ? '良好' : s >= 60 ? '一般' : '较差'

  if (!selectedDataset) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center">
          <ShieldCheck className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-lg font-medium">未选择数据集</p>
          <p className="text-sm text-muted-foreground mt-1">选择一个数据集以运行健康检查</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">数据集健康检查</h2>
          <p className="text-muted-foreground text-sm mt-1">{selectedDataset.name}</p>
        </div>
        <Button onClick={runAnalysis} disabled={isAnalyzing} className="gap-2">
          {isAnalyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          {report ? '重新分析' : '开始分析'}
        </Button>
      </div>

      {!report && !isAnalyzing && (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <ShieldCheck className="w-16 h-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">运行健康检查</h3>
            <p className="text-muted-foreground text-sm text-center max-w-md">
              扫描数据集的标注覆盖缺口、类别失衡、图像质量等问题。
            </p>
            <Button onClick={runAnalysis} className="mt-6 gap-2">
              <ShieldCheck className="w-4 h-4" /> 分析数据集
            </Button>
          </CardContent>
        </Card>
      )}

      {isAnalyzing && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Loader2 className="w-10 h-10 animate-spin text-primary mb-4" />
            <p className="text-muted-foreground">正在分析数据集质量…</p>
          </CardContent>
        </Card>
      )}

      {report && (
        <>
          {/* Score + Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="md:col-span-1 flex flex-col items-center justify-center py-6">
              <div className={cn('text-5xl font-bold', scoreColor(report.score))}>
                {report.score.toFixed(0)}
              </div>
              <div className={cn('text-sm font-medium mt-1', scoreColor(report.score))}>
                {scoreLabel(report.score)}
              </div>
              <div className="text-xs text-muted-foreground mt-1">健康评分</div>
            </Card>

            <Card>
              <CardContent className="pt-5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                    <Eye className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">标注覆盖率</p>
                    <p className="text-xl font-bold">{report.coveragePercent.toFixed(1)}%</p>
                    <p className="text-xs text-muted-foreground">{report.annotatedImages}/{report.totalImages} 已标注</p>
                  </div>
                </div>
                <Progress value={report.coveragePercent} className="h-1.5 mt-3" />
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
                    <Scale className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">类别失衡度</p>
                    <p className="text-xl font-bold">{report.maxImbalanceRatio.toFixed(1)}×</p>
                    <p className="text-xs text-muted-foreground">最大/最小比</p>
                  </div>
                </div>
                <div className="mt-3 h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className={cn('h-full rounded-full', report.maxImbalanceRatio > 10 ? 'bg-red-500' : report.maxImbalanceRatio > 3 ? 'bg-amber-500' : 'bg-green-500')}
                    style={{ width: `${Math.min(100, (1 / report.maxImbalanceRatio) * 100)}%` }}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-rose-500/10 flex items-center justify-center">
                    <AlertTriangle className="w-5 h-5 text-rose-500" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">发现问题</p>
                    <p className="text-xl font-bold">{report.warnings.length}</p>
                    <p className="text-xs text-muted-foreground">条待处理告警</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Warnings */}
          {report.warnings.length > 0 && (
            <Card className="border-amber-500/30 bg-amber-500/5">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2 text-amber-600 dark:text-amber-400">
                  <AlertTriangle className="w-4 h-4" /> 告警
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {report.warnings.map((w, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <XCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                    <span>{w}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Class imbalance chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> 类别分布
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-72 overflow-y-auto">
                {report.classImbalance.map(({ class: cls, count, ratio }) => (
                  <div key={cls}>
                    <div className="flex items-center justify-between mb-1 text-sm">
                      <span className="font-medium truncate flex-1 mr-2">{cls}</span>
                      <span className={cn('text-xs', ratio < 0.1 ? 'text-red-600 dark:text-red-400 font-medium' : 'text-muted-foreground')}>
                        {count.toLocaleString()}
                        {ratio < 0.1 && <Badge variant="destructive" className="ml-2 text-xs py-0">样本不足</Badge>}
                      </span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={cn('h-full rounded-full', ratio < 0.1 ? 'bg-red-500' : ratio < 0.3 ? 'bg-amber-500' : 'bg-primary')}
                        style={{ width: `${ratio * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Unannotated images */}
          {report.unannotatedImages > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <FileX className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                  未标注图片（{report.unannotatedImages}）
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-muted-foreground mb-3">
                  这些图片没有任何标注，建议使用自动标注或手动补标。
                </div>
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {report.unannotatedList.map(img => (
                    <div key={img.id} className="flex items-center gap-2 py-1 border-b border-border/30 last:border-0">
                      <ImageOff className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
                      <span className="text-sm font-mono">{img.filename}</span>
                    </div>
                  ))}
                  {report.unannotatedImages > 50 && (
                    <p className="text-xs text-muted-foreground pt-2">
                      …以及另外 {report.unannotatedImages - 50} 张
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Small images */}
          {report.smallImages.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <ImageOff className="w-4 h-4 text-rose-500" />
                  低分辨率图片（{report.smallImages.length}）
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  小于 100px 的图片可能会降低模型表现。
                </p>
                <div className="max-h-48 overflow-y-auto space-y-1">
                  {report.smallImages.map(img => (
                    <div key={img.id} className="flex items-center gap-3 py-1 border-b border-border/30 last:border-0">
                      <span className="text-sm font-mono flex-1">{img.filename}</span>
                      <Badge variant="secondary" className="text-xs">{img.width}×{img.height}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {report.warnings.length === 0 && (
            <Card className="border-green-500/30 bg-green-500/5">
              <CardContent className="flex items-center gap-3 py-4">
                <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-green-400" />
                <div>
                  <p className="font-medium text-green-700 dark:text-green-400">数据集状态良好！</p>
                  <p className="text-sm text-muted-foreground">未发现明显问题。</p>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
