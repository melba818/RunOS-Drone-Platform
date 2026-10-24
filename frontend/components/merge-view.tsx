"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { 
  Layers, 
  ArrowRight, 
  Play,
  CheckCircle2,
  AlertCircle,
  FileType,
  Images,
  RefreshCw
} from "lucide-react"
import type { Dataset } from "@/app/page"

interface Format {
  id: string
  name: string
}

interface MergeViewProps {
  datasets?: Dataset[]
  setDatasets?: (datasets: Dataset[]) => void
  apiUrl?: string
}

export function MergeView({ datasets: propDatasets = [], setDatasets: propSetDatasets, apiUrl = "http://localhost:8000" }: MergeViewProps) {
  const [formats, setFormats] = useState<Format[]>([])
  const [outputFormat, setOutputFormat] = useState("yolo")
  const [outputName, setOutputName] = useState("merged_dataset")
  const [isMerging, setIsMerging] = useState(false)
  const [mergeStatus, setMergeStatus] = useState<"idle" | "merging" | "success" | "error">("idle")
  const [mergeError, setMergeError] = useState<string | null>(null)
  const [mergedDatasetName, setMergedDatasetName] = useState<string | null>(null)
  const [remapClasses, setRemapClasses] = useState(true)
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string>>(new Set())

  // Fetch available formats from the API
  useEffect(() => {
    fetch(`${apiUrl}/api/formats`)
      .then(r => r.json())
      .then(data => { if (data.formats) setFormats(data.formats) })
      .catch(() => {})
  }, [apiUrl])

  // Auto-select all datasets when the list changes
  useEffect(() => {
    if (propDatasets.length > 0) {
      setSelectedDatasets(new Set(propDatasets.map(d => d.id)))
    }
  }, [propDatasets])

  const toggleDatasetSelection = (id: string) => {
    setSelectedDatasets(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleMerge = async () => {
    if (selectedDatasets.size < 2) return
    if (!outputName.trim()) return

    setIsMerging(true)
    setMergeStatus("merging")
    setMergeError(null)
    setMergedDatasetName(null)

    try {
      const res = await fetch(`${apiUrl}/api/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_ids: Array.from(selectedDatasets),
          output_name: outputName.trim(),
          output_format: outputFormat,
          class_mapping: remapClasses ? null : undefined,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }

      const data = await res.json()

      if (data.success && data.merged_dataset) {
        setMergeStatus("success")
        setMergedDatasetName(data.merged_dataset.name)
        if (propSetDatasets) {
          const refreshed = await fetch(`${apiUrl}/api/datasets`)
          if (refreshed.ok) propSetDatasets((await refreshed.json()).datasets || [])
          else propSetDatasets([...propDatasets, data.merged_dataset])
        }
      } else {
        throw new Error("合并返回了意外的响应")
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setMergeStatus("error")
      setMergeError(msg)
    } finally {
      setIsMerging(false)
    }
  }

  const getAllClasses = () => {
    const classSet = new Set<string>()
    propDatasets
      .filter(d => selectedDatasets.has(d.id))
      .forEach(d => d.classes.forEach(c => classSet.add(c)))
    return Array.from(classSet)
  }

  const getTotalImages = () =>
    propDatasets
      .filter(d => selectedDatasets.has(d.id))
      .reduce((sum, d) => sum + d.num_images, 0)

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">合并数据集</h1>
        <p className="text-muted-foreground mt-1">
          将多个数据集合并为一个统一的数据集
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Datasets List */}
        <div className="lg:col-span-2">
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Layers className="w-5 h-5 text-primary" />
                源数据集
              </CardTitle>
              <CardDescription>
                选择要合并的已加载数据集。不同格式会自动转换。
              </CardDescription>
            </CardHeader>
            <CardContent>
              {propDatasets.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4">
                    <Layers className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <p className="text-muted-foreground mb-2">尚未加载任何数据集</p>
                  <p className="text-sm text-muted-foreground/70">
                    请先到「数据集」标签页加载至少两个数据集。
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {propDatasets.map((dataset, index) => (
                    <div
                      key={dataset.id}
                      className={`flex items-center gap-4 p-4 rounded-lg border transition-all ${
                        selectedDatasets.has(dataset.id)
                          ? "border-primary/50 bg-primary/5"
                          : "border-border/50 bg-muted/20"
                      }`}
                    >
                      <Checkbox
                        checked={selectedDatasets.has(dataset.id)}
                        onCheckedChange={() => toggleDatasetSelection(dataset.id)}
                      />

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium text-foreground">{dataset.name}</span>
                          <Badge variant="secondary" className="text-xs">
                            <FileType className="w-3 h-3 mr-1" />
                            {dataset.format.toUpperCase()}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground truncate">{dataset.path}</p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Images className="w-3 h-3" />
                            {dataset.num_images.toLocaleString()} 张图像
                          </span>
                          <span>{dataset.num_annotations.toLocaleString()} 个标注</span>
                          <span>{dataset.classes.length} 个类别</span>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-1 max-w-[200px]">
                        {dataset.classes.slice(0, 3).map(cls => (
                          <Badge key={cls} variant="outline" className="text-xs">{cls}</Badge>
                        ))}
                        {dataset.classes.length > 3 && (
                          <Badge variant="outline" className="text-xs">+{dataset.classes.length - 3}</Badge>
                        )}
                      </div>

                      {index < propDatasets.length - 1 && selectedDatasets.has(dataset.id) && (
                        <ArrowRight className="w-4 h-4 text-primary shrink-0" />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Merge Settings */}
        <div className="space-y-4">
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg">输出设置</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">数据集名称</label>
                <Input
                  value={outputName}
                  onChange={(e) => setOutputName(e.target.value)}
                  placeholder="merged_dataset"
                  disabled={isMerging}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">输出格式</label>
                <Select value={outputFormat} onValueChange={setOutputFormat} disabled={isMerging}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {formats.map(f => (
                      <SelectItem key={f.id} value={f.id}>{f.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2 pt-2">
                <Checkbox
                  id="remap"
                  checked={remapClasses}
                  onCheckedChange={(checked) => setRemapClasses(checked as boolean)}
                  disabled={isMerging}
                />
                <label htmlFor="remap" className="text-sm text-muted-foreground cursor-pointer">
                  自动重映射类别 ID
                </label>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg">合并概览</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">已选数据集</p>
                  <p className="text-2xl font-semibold text-foreground">{selectedDatasets.size}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">图像总数</p>
                  <p className="text-2xl font-semibold text-foreground">{getTotalImages().toLocaleString()}</p>
                </div>
              </div>

              {getAllClasses().length > 0 && (
                <div>
                  <p className="text-sm text-muted-foreground mb-2">
                    合并后的类别（{getAllClasses().length}）
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {getAllClasses().map(cls => (
                      <Badge key={cls} variant="secondary" className="text-xs">{cls}</Badge>
                    ))}
                  </div>
                </div>
              )}

              {mergeStatus === "merging" && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  正在后端合并…
                </div>
              )}

              {mergeStatus === "success" && (
                <div className="flex items-center gap-2 text-sm text-emerald-700 dark:text-green-400">
                  <CheckCircle2 className="w-4 h-4" />
                  已合并为「{mergedDatasetName}」！
                </div>
              )}

              {mergeStatus === "error" && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-sm text-destructive">
                    <AlertCircle className="w-4 h-4" />
                    合并失败
                  </div>
                  {mergeError && (
                    <p className="text-xs text-destructive/80 break-all">{mergeError}</p>
                  )}
                </div>
              )}

              <Button
                className="w-full gap-2"
                disabled={selectedDatasets.size < 2 || isMerging || !outputName.trim()}
                onClick={handleMerge}
              >
                {isMerging ? (
                  <><RefreshCw className="w-4 h-4 animate-spin" />合并中…</>
                ) : (
                  <><Play className="w-4 h-4" />合并数据集</>
                )}
              </Button>
              {selectedDatasets.size < 2 && (
                <p className="text-xs text-muted-foreground text-center">
                  请至少选择 2 个数据集进行合并
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
