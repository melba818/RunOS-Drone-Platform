'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import { 
  Shuffle,
  CheckCircle,
  AlertCircle,
  Loader2,
  FolderOpen,
  Image
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'
import {assistantCall,type Workflow} from '@/lib/assistant-workflow'
import {splitCounts} from '@/lib/native-workflow.mjs'

interface SplitViewProps {
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  selectedDataset: Dataset | null
  onDatasetCreated?: (dataset: Dataset) => void
  apiUrl: string
  assistantFlow?:Workflow|null
  onWorkflowChange?:()=>void
}

interface SplitConfig {
  train: number
  val: number
  test: number
}

interface SplitResult {
  success: boolean
  message: string
  train_count: number
  val_count: number
  test_count: number
}

export function SplitView({ datasets, setDatasets, selectedDataset, onDatasetCreated, apiUrl,assistantFlow,onWorkflowChange }: SplitViewProps) {
  const [splitConfig, setSplitConfig] = useState<SplitConfig>({ train: 70, val: 20, test: 10 })
  const [outputName, setOutputName] = useState('')
  const [isSplitting, setIsSplitting] = useState(false)
  const [result, setResult] = useState<SplitResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [seed, setSeed] = useState(42)
  const [preview,setPreview]=useState<any>(null)
  const [currentSplits, setCurrentSplits] = useState<Record<string, number> | null>(null)

  useEffect(() => {
    if (!selectedDataset) { setCurrentSplits(null); return }
    fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/stats`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.splits && Object.keys(data.splits).length > 0) {
          setCurrentSplits(data.splits)
        } else {
          setCurrentSplits(null)
        }
      })
      .catch(() => setCurrentSplits(null))
  }, [selectedDataset?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // Calculate counts based on percentages
  const totalImages = selectedDataset?.num_images || 0
  const counts=splitCounts(totalImages,splitConfig.train/100,splitConfig.val/100,splitConfig.test/100)
  const trainCount=preview?.counts?.train??counts.train,valCount=preview?.counts?.val??counts.val,testCount=preview?.counts?.test??counts.test
  const plan=()=>({train_ratio:splitConfig.train/100,val_ratio:splitConfig.val/100,test_ratio:splitConfig.test/100,output_name:outputName||`${assistantFlow?.source_dataset?.name||selectedDataset?.name}_split`,seed,shuffle:true,stratify:false})
  useEffect(()=>{
    if(assistantFlow?.split){const p=assistantFlow.split.data;setSplitConfig({train:p.train_ratio*100,val:p.val_ratio*100,test:p.test_ratio*100});setOutputName(p.output_name);setSeed(p.seed);setResult({success:true,message:'本流程已完成划分',train_count:assistantFlow.split.counts.train,val_count:assistantFlow.split.counts.val,test_count:assistantFlow.split.counts.test})}
  },[assistantFlow?.id,assistantFlow?.split?.created_at])
  useEffect(()=>{
    if(assistantFlow?.stage!=='split_ready'){setPreview(null);return}
    setPreview(null)
    let live=true;const timer=setTimeout(()=>void assistantCall(apiUrl,`flows/${assistantFlow.id}/split-preview`,'POST',plan()).then(v=>{if(live){setPreview(v);setError(null)}}).catch(err=>{if(live){setPreview(null);setError(err.message)}}),300)
    return()=>{live=false;clearTimeout(timer)}
  },[assistantFlow?.id,assistantFlow?.revision,splitConfig,seed,outputName,apiUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleTrainChange = (value: number[]) => {
    const train = value[0]
    const remaining = 100 - train
    const valRatio = splitConfig.val / (splitConfig.val + splitConfig.test) || 0.67
    const val = Math.round(remaining * valRatio)
    const test = remaining - val
    setSplitConfig({ train, val, test })
  }

  const handleValChange = (value: number[]) => {
    const val = value[0]
    const maxVal = 100 - splitConfig.train
    const clampedVal = Math.min(val, maxVal)
    const test = maxVal - clampedVal
    setSplitConfig({ ...splitConfig, val: clampedVal, test })
  }

  const handleSplit = async () => {
    if (!selectedDataset) return

    setIsSplitting(true)
    setError(null)
    setResult(null)

    try {
      let data:any
      if(assistantFlow){
        const latest=await assistantCall(apiUrl,`flows/${assistantFlow.id}`)
        const flow=await assistantCall(apiUrl,`flows/${assistantFlow.id}/split`,'POST',{revision:latest.revision,data:plan()})
        data={success:true,new_dataset:flow.target_dataset,train_count:flow.split.counts.train,val_count:flow.split.counts.val,test_count:flow.split.counts.test};onWorkflowChange?.()
      }else{
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/split`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          train_ratio: splitConfig.train / 100,
          val_ratio: splitConfig.val / 100,
          test_ratio: splitConfig.test / 100,
          output_name: outputName || `${selectedDataset.name}_split`,
          seed,
          stratify:false
        })
      })

      if (!response.ok) {
        const err = await response.json()
        const detail = err.detail
        const msg = Array.isArray(detail)
          ? detail.map((e: { msg?: string }) => e.msg || JSON.stringify(e)).join('; ')
          : (typeof detail === 'string' ? detail : detail?.message||'划分未完成，请核对参数。')
        throw new Error(msg)
      }

      data = await response.json()
      }

      if (data.success) {
        setResult({
          success: true,
          message: '数据集划分成功！',
          train_count: data.train_count ?? data.splits?.train ?? trainCount,
          val_count: data.val_count ?? data.splits?.val ?? valCount,
          test_count: data.test_count ?? data.splits?.test ?? testCount
        })

        // Refresh datasets list
        const datasetsResponse = await fetch(`${apiUrl}/api/datasets`)
        if (datasetsResponse.ok) {
          const datasetsData = await datasetsResponse.json()
          setDatasets(datasetsData.datasets || [])
        }

        if (data.new_dataset && onDatasetCreated) {
          onDatasetCreated(data.new_dataset)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '划分失败')
    } finally {
      setIsSplitting(false)
    }
  }

  if (!selectedDataset) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-muted-foreground">未选择数据集</h3>
          <p className="text-sm text-muted-foreground/70 mt-1">
            请先在「数据集」视图中选择一个数据集，再创建 train/val/test 划分
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-foreground">训练集 / 验证集 / 测试集划分</h2>
        <p className="text-muted-foreground text-sm mt-1">
          将数据集划分为训练集、验证集与测试集
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-6">
        {/* Source Dataset Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">源数据集</CardTitle>
            <CardDescription>待划分的数据集</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="p-4 bg-muted/50 rounded-lg space-y-3">
              <div className="flex items-center gap-3">
                <FolderOpen className="w-5 h-5 text-primary" />
                <div>
                  <p className="font-medium">{selectedDataset.name}</p>
                  <p className="text-sm text-muted-foreground">{selectedDataset.format.toUpperCase()}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Image className="w-4 h-4 text-muted-foreground" />
                <span>{totalImages.toLocaleString()} 张图像</span>
              </div>
              {selectedDataset.classes && selectedDataset.classes.length > 0 && (
                <div className="pt-2 border-t">
                  <p className="text-sm text-muted-foreground mb-2">类别（{selectedDataset.classes.length}）</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedDataset.classes.slice(0, 6).map((cls, idx) => (
                      <span key={idx} className="px-2 py-0.5 bg-muted rounded text-xs">{cls}</span>
                    ))}
                    {selectedDataset.classes.length > 6 && (
                      <span className="px-2 py-0.5 text-muted-foreground text-xs">
                        另有 {selectedDataset.classes.length - 6} 个
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Current Split Breakdown */}
        {currentSplits && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">当前划分</CardTitle>
              <CardDescription>该数据集现有的 train/val/test 分布</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(currentSplits).map(([split, count]) => {
                  const total = Object.values(currentSplits).reduce((a, b) => a + b, 0)
                  const pct = total > 0 ? Math.round((count / total) * 100) : 0
                  const color = split === 'train' ? 'bg-primary' : split === 'val' || split === 'valid' ? 'bg-amber-500' : 'bg-blue-500'
                  const textColor = split === 'train' ? 'text-primary' : split === 'val' || split === 'valid' ? 'text-amber-700 dark:text-amber-500' : 'text-blue-700 dark:text-blue-400'
                  return (
                    <div key={split} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize font-medium">{split}</span>
                        <span className={cn('font-medium', textColor)}>{count} 张图像（{pct}%）</span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <div className={cn('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Split Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">划分配置</CardTitle>
            <CardDescription>调整各划分的占比</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Train Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">训练集</label>
                <span className="text-sm text-primary font-medium">{splitConfig.train}%（{trainCount} 张图像）</span>
              </div>
              <Slider
                value={[splitConfig.train]}
                onValueChange={handleTrainChange}
                min={10}
                max={90}
                step={5}
                className="[&>span:first-child]:bg-primary"
              />
            </div>

            {/* Val Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">验证集</label>
                <span className="text-sm text-amber-700 dark:text-amber-500 font-medium">{splitConfig.val}%（{valCount} 张图像）</span>
              </div>
              <Slider
                value={[splitConfig.val]}
                onValueChange={handleValChange}
                min={0}
                max={100 - splitConfig.train}
                step={5}
                className="[&>span:first-child]:bg-amber-500"
              />
            </div>

            {/* Test (calculated) */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">测试集</label>
                <span className="text-sm text-blue-700 dark:text-blue-400 font-medium">{splitConfig.test}%（{testCount} 张图像）</span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: `${splitConfig.test}%` }} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Visual Split Preview */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">划分预览</CardTitle>
          <CardDescription>划分结果的可视化展示</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-8 rounded-lg overflow-hidden flex">
            <div 
              className="bg-primary flex items-center justify-center text-white text-xs font-medium transition-all"
              style={{ width: `${splitConfig.train}%` }}
            >
              {splitConfig.train >= 20 && `Train ${splitConfig.train}%`}
            </div>
            <div 
              className="bg-amber-500 flex items-center justify-center text-white text-xs font-medium transition-all"
              style={{ width: `${splitConfig.val}%` }}
            >
              {splitConfig.val >= 15 && `Val ${splitConfig.val}%`}
            </div>
            <div 
              className="bg-blue-500 flex items-center justify-center text-white text-xs font-medium transition-all"
              style={{ width: `${splitConfig.test}%` }}
            >
              {splitConfig.test >= 10 && `Test ${splitConfig.test}%`}
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-4 text-center">
            <div className="p-3 bg-primary/10 rounded-lg">
              <p className="text-2xl font-bold text-primary">{trainCount}</p>
              <p className="text-xs text-muted-foreground">训练图像</p>
            </div>
            <div className="p-3 bg-amber-500/10 rounded-lg">
              <p className="text-2xl font-bold text-amber-700 dark:text-amber-500">{valCount}</p>
              <p className="text-xs text-muted-foreground">验证图像</p>
            </div>
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <p className="text-2xl font-bold text-blue-700 dark:text-blue-400">{testCount}</p>
              <p className="text-xs text-muted-foreground">测试图像</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Options */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">选项</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">输出名称</label>
              <Input
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
                placeholder={`${selectedDataset.name}_split`}
              />
            </div>
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">随机种子</label>
              <Input
                type="number"
                min={0}
                max={2147483647}
                value={seed}
                onChange={(e) => setSeed(Number.isInteger(Number(e.target.value))?Number(e.target.value):42)}
                placeholder="42"
              />
            </div>
          </div>
          
          <p className="text-xs text-muted-foreground">按随机种子打乱后划分；创建新的数据集，保留源图片和正式标签。</p>
        </CardContent>
      </Card>

      {/* Action */}
      <div className="flex items-center justify-center gap-4 mb-6">
        <Button
          size="lg"
          onClick={handleSplit}
          disabled={isSplitting || totalImages === 0 || (assistantFlow&& (assistantFlow.stage!=='split_ready'||!preview||!!error))}
        >
          {isSplitting ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Shuffle className="w-4 h-4 mr-2" />
          )}
          {isSplitting ? '划分中…' : '创建划分'}
        </Button>
      </div>

      {/* Result */}
      {result && (
        <Card className="border-primary/50 bg-primary/5 mb-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium">{result.message}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Train: {result.train_count} | Val: {result.val_count} | Test: {result.test_count}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <Card className="border-destructive/50 bg-destructive/5 mb-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-destructive" />
              <p className="text-destructive">{error}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Common Presets */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">快速预设</CardTitle>
          <CardDescription>机器学习中常用的划分比例</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { name: '70/20/10', train: 70, val: 20, test: 10 },
              { name: '80/10/10', train: 80, val: 10, test: 10 },
              { name: '60/20/20', train: 60, val: 20, test: 20 },
              { name: '90/5/5', train: 90, val: 5, test: 5 },
            ].map((preset) => (
              <Button
                key={preset.name}
                variant="outline"
                className={cn(
                  'h-auto py-3 flex-col',
                  splitConfig.train === preset.train && 
                  splitConfig.val === preset.val && 
                  'border-primary bg-primary/5'
                )}
                onClick={() => setSplitConfig({ train: preset.train, val: preset.val, test: preset.test })}
              >
                <span className="font-medium">{preset.name}</span>
                <span className="text-xs text-muted-foreground">Train/Val/Test</span>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
