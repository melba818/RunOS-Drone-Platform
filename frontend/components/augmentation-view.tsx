'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import {
  Wand2,
  RefreshCcw,
  ImagePlus,
  Contrast,
  Maximize2,
  RotateCw,
  FlipHorizontal,
  FlipVertical,
  Crop,
  Droplets,
  Palette,
  Play,
  Eye,
  SunDim,
  Shuffle,
  Target,
  Users
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

interface AugmentationViewProps {
  selectedDataset: Dataset | null
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  onDatasetCreated?: (dataset: Dataset) => void
  apiUrl: string
}

interface AugmentationConfig {
  horizontalFlip: boolean
  horizontalFlipProb: number
  verticalFlip: boolean
  verticalFlipProb: number
  rotate90: boolean
  rotate90Prob: number
  randomRotate: boolean
  rotateLimit: number
  randomCrop: boolean
  cropScale: [number, number]
  resize: boolean
  resizeWidth: number
  resizeHeight: number
  brightness: boolean
  brightnessLimit: number
  contrast: boolean
  contrastLimit: number
  saturation: boolean
  saturationLimit: number
  hue: boolean
  hueLimit: number
  blur: boolean
  blurLimit: number
  noise: boolean
  noiseVar: number
  mosaic: boolean
  mosaicProb: number
  mixup: boolean
  mixupAlpha: number
  cutout: boolean
  cutoutSize: number
}

interface PreviewItem {
  data_url: string | null
  augmentations: string[]
  error?: string
}

const defaultConfig: AugmentationConfig = {
  horizontalFlip: true,
  horizontalFlipProb: 0.5,
  verticalFlip: false,
  verticalFlipProb: 0.5,
  rotate90: false,
  rotate90Prob: 0.5,
  randomRotate: true,
  rotateLimit: 15,
  randomCrop: false,
  cropScale: [0.8, 1.0],
  resize: true,
  resizeWidth: 640,
  resizeHeight: 640,
  brightness: true,
  brightnessLimit: 0.2,
  contrast: true,
  contrastLimit: 0.2,
  saturation: true,
  saturationLimit: 0.2,
  hue: false,
  hueLimit: 0.1,
  blur: true,
  blurLimit: 3,
  noise: false,
  noiseVar: 0.1,
  mosaic: false,
  mosaicProb: 0.5,
  mixup: false,
  mixupAlpha: 0.5,
  cutout: false,
  cutoutSize: 32
}

const presetConfigs = {
  light: {
    ...defaultConfig,
    horizontalFlip: true,
    randomRotate: true,
    rotateLimit: 10,
    brightness: true,
    brightnessLimit: 0.15,
    contrast: true,
    contrastLimit: 0.15,
    saturation: false,
    blur: false,
    noise: false
  },
  medium: {
    ...defaultConfig,
    horizontalFlip: true,
    verticalFlip: true,
    randomRotate: true,
    rotateLimit: 20,
    brightness: true,
    brightnessLimit: 0.25,
    contrast: true,
    contrastLimit: 0.25,
    saturation: true,
    blur: true
  },
  heavy: {
    ...defaultConfig,
    horizontalFlip: true,
    verticalFlip: true,
    rotate90: true,
    randomRotate: true,
    rotateLimit: 30,
    randomCrop: true,
    brightness: true,
    brightnessLimit: 0.3,
    contrast: true,
    contrastLimit: 0.3,
    saturation: true,
    hue: true,
    blur: true,
    noise: true,
    mosaic: true,
    cutout: true
  }
}

// Preset button with accent hover style
function PresetButton({ label, onClick }: { label: string; onClick: () => void }) {
  const [hovered, setHovered] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="px-3 py-1 text-sm rounded-md border transition-colors"
      style={{
        borderColor: hovered ? 'var(--primary)' : undefined,
        color: hovered ? 'var(--primary)' : undefined,
        backgroundColor: 'transparent'
      }}
    >
      {label}
    </button>
  )
}

export function AugmentationView({
  selectedDataset,
  datasets,
  setDatasets,
  onDatasetCreated,
  apiUrl
}: AugmentationViewProps) {
  const [config, setConfig] = useState<AugmentationConfig>(defaultConfig)
  // Size mode: 'factor' or 'absolute'
  const [sizeMode, setSizeMode] = useState<'factor' | 'absolute'>('factor')
  const [augmentFactor, setAugmentFactor] = useState(2)
  const [factorInput, setFactorInput] = useState('2')
  const [targetSize, setTargetSize] = useState(0)
  const [outputName, setOutputName] = useState('')
  const [previews, setPreviews] = useState<PreviewItem[]>([])
  const [previewLoading, setPreviewLoading] = useState(false)
  const [isAugmenting, setIsAugmenting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  // Class targets: class_name -> target count
  const [classTargets, setClassTargets] = useState<Record<string, number>>({})
  const [classAugEnabled, setClassAugEnabled] = useState(false)

  useEffect(() => {
    if (selectedDataset) {
      setOutputName(`${selectedDataset.name}_augmented`)
      setTargetSize(Math.ceil(selectedDataset.num_images * augmentFactor))
    }
  }, [selectedDataset])

  const updateConfig = <K extends keyof AugmentationConfig>(
    key: K,
    value: AugmentationConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handlePreview = async () => {
    if (!selectedDataset) return
    setPreviewLoading(true)
    setPreviews([])
    try {
      const response = await fetch(`${apiUrl}/api/augment/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          config,
          num_previews: 6
        })
      })
      if (response.ok) {
        const data = await response.json()
        setPreviews(data.previews || [])
      } else {
        console.error('Preview failed:', await response.text())
      }
    } catch (err) {
      console.error('Preview failed:', err)
    }
    setPreviewLoading(false)
  }

  const computedTargetSize = sizeMode === 'factor'
    ? Math.ceil((selectedDataset?.num_images ?? 0) * augmentFactor)
    : targetSize

  const handleAugment = async () => {
    if (!selectedDataset) return
    setIsAugmenting(true)
    setProgress(0)
    setMessage(null)

    try {
      const body: Record<string, unknown> = {
        dataset_id: selectedDataset.id,
        config,
        output_name: outputName || `${selectedDataset.name}_augmented`,
      }
      if (sizeMode === 'factor') {
        body.augment_factor = augmentFactor
      } else {
        body.target_size = targetSize
      }
      if (classAugEnabled && Object.keys(classTargets).length > 0) {
        body.class_targets = classTargets
      }

      const startResp = await fetch(`${apiUrl}/api/augment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!startResp.ok) {
        const err = await startResp.json().catch(() => ({}))
        throw new Error(err.detail || '数据增强失败')
      }

      const { job_id } = await startResp.json()

      await new Promise<void>((resolve) => {
        const poll = setInterval(async () => {
          try {
            const statusResp = await fetch(`${apiUrl}/api/augment/${job_id}/status`)
            if (!statusResp.ok) return
            const job = await statusResp.json()
            setProgress(job.progress ?? 0)

            if (job.status === 'done') {
              clearInterval(poll)
              setProgress(100)
              if (job.new_dataset) {
                setDatasets([...datasets, job.new_dataset])
                onDatasetCreated?.(job.new_dataset)
              }
              setMessage({ type: 'success', text: `已生成增强数据集，共 ${job.total_images} 张图像` })
              resolve()
            } else if (job.status === 'error') {
              clearInterval(poll)
              setMessage({ type: 'error', text: job.error || '数据增强失败' })
              setProgress(0)
              resolve()
            }
          } catch (_) {}
        }, 500)
      })
    } catch (err: unknown) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : '数据增强失败' })
    }
    setIsAugmenting(false)
  }

  const datasetClasses: string[] = (selectedDataset as unknown as Record<string, unknown>)?.classes as string[] ?? []

  if (!selectedDataset) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6">
          <Wand2 className="w-10 h-10 text-muted-foreground" />
        </div>
        <h2 className="text-2xl font-semibold text-foreground mb-2">未选择数据集</h2>
        <p className="text-muted-foreground max-w-md">
          选择一个数据集以配置并应用数据增强
        </p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold text-foreground">数据增强</h2>
          <p className="text-muted-foreground text-sm mt-1">
            为 {selectedDataset.name} 配置并应用数据增强
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={handlePreview}
            disabled={previewLoading}
          >
            <Eye className="w-4 h-4 mr-2" />
            {previewLoading ? '加载中…' : '预览'}
          </Button>
          <Button
            onClick={handleAugment}
            disabled={isAugmenting}
          >
            {isAugmenting ? (
              <>
                <RefreshCcw className="w-4 h-4 mr-2 animate-spin" />
                正在增强…
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                开始增强
              </>
            )}
          </Button>
        </div>
      </div>

      {isAugmenting && (
        <Card className="mb-4">
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <Progress value={progress} className="flex-1" />
              <span className="text-sm text-muted-foreground w-12">{progress}%</span>
            </div>
          </CardContent>
        </Card>
      )}

      {message && (
        <div className={cn(
          'mb-4 p-3 rounded-lg text-sm',
          message.type === 'success' ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20' : 'bg-destructive/10 text-destructive border border-destructive/20'
        )}>
          {message.text}
        </div>
      )}

      <div className="flex gap-6 flex-1 min-h-0">
        <div className="flex-1 overflow-y-auto pr-2">
          {/* Quick Presets */}
          <div className="flex items-center gap-2 mb-4">
            <span className="text-sm text-muted-foreground">快速预设：</span>
            <PresetButton label="轻度" onClick={() => setConfig(presetConfigs.light)} />
            <PresetButton label="中度" onClick={() => setConfig(presetConfigs.medium)} />
            <PresetButton label="重度" onClick={() => setConfig(presetConfigs.heavy)} />
            <PresetButton label="↺ 重置" onClick={() => setConfig(defaultConfig)} />
          </div>

          <Tabs defaultValue="geometric" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="geometric">几何变换</TabsTrigger>
              <TabsTrigger value="color">颜色与噪声</TabsTrigger>
              <TabsTrigger value="advanced">高级</TabsTrigger>
              <TabsTrigger value="output">输出</TabsTrigger>
            </TabsList>

            <TabsContent value="geometric" className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <FlipHorizontal className="w-4 h-4" />
                    翻转变换
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FlipHorizontal className="w-4 h-4 text-muted-foreground" />
                      <Label>水平翻转</Label>
                    </div>
                    <Switch
                      checked={config.horizontalFlip}
                      onCheckedChange={(v) => updateConfig('horizontalFlip', v)}
                    />
                  </div>
                  {config.horizontalFlip && (
                    <div className="pl-6 space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>概率</span>
                        <span className="text-muted-foreground">{config.horizontalFlipProb}</span>
                      </div>
                      <Slider
                        value={[config.horizontalFlipProb]}
                        onValueChange={([v]) => updateConfig('horizontalFlipProb', v)}
                        min={0} max={1} step={0.1}
                      />
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FlipVertical className="w-4 h-4 text-muted-foreground" />
                      <Label>垂直翻转</Label>
                    </div>
                    <Switch
                      checked={config.verticalFlip}
                      onCheckedChange={(v) => updateConfig('verticalFlip', v)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <RotateCw className="w-4 h-4" />
                    旋转
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>90° 旋转</Label>
                    <Switch
                      checked={config.rotate90}
                      onCheckedChange={(v) => updateConfig('rotate90', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>随机旋转</Label>
                    <Switch
                      checked={config.randomRotate}
                      onCheckedChange={(v) => updateConfig('randomRotate', v)}
                    />
                  </div>
                  {config.randomRotate && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>最大角度</span>
                        <span className="text-muted-foreground">±{config.rotateLimit}°</span>
                      </div>
                      <Slider
                        value={[config.rotateLimit]}
                        onValueChange={([v]) => updateConfig('rotateLimit', v)}
                        min={0} max={45} step={5}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Maximize2 className="w-4 h-4" />
                    缩放与裁剪
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Crop className="w-4 h-4 text-muted-foreground" />
                      <Label>随机裁剪</Label>
                    </div>
                    <Switch
                      checked={config.randomCrop}
                      onCheckedChange={(v) => updateConfig('randomCrop', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>缩放</Label>
                    <Switch
                      checked={config.resize}
                      onCheckedChange={(v) => updateConfig('resize', v)}
                    />
                  </div>
                  {config.resize && (
                    <div className="grid grid-cols-2 gap-4 pl-6">
                      <div className="space-y-2">
                        <Label className="text-xs">宽度</Label>
                        <Input
                          type="number"
                          value={config.resizeWidth}
                          onChange={(e) => updateConfig('resizeWidth', parseInt(e.target.value) || 640)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">高度</Label>
                        <Input
                          type="number"
                          value={config.resizeHeight}
                          onChange={(e) => updateConfig('resizeHeight', parseInt(e.target.value) || 640)}
                        />
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="color" className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <SunDim className="w-4 h-4" />
                    亮度与对比度
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>亮度</Label>
                    <Switch
                      checked={config.brightness}
                      onCheckedChange={(v) => updateConfig('brightness', v)}
                    />
                  </div>
                  {config.brightness && (
                    <div className="space-y-2 pl-6">
                      <div className="flex items-center justify-between text-sm">
                        <span>幅度</span>
                        <span className="text-muted-foreground">±{(config.brightnessLimit * 100).toFixed(0)}%</span>
                      </div>
                      <Slider
                        value={[config.brightnessLimit]}
                        onValueChange={([v]) => updateConfig('brightnessLimit', v)}
                        min={0} max={0.5} step={0.05}
                      />
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Contrast className="w-4 h-4 text-muted-foreground" />
                      <Label>对比度</Label>
                    </div>
                    <Switch
                      checked={config.contrast}
                      onCheckedChange={(v) => updateConfig('contrast', v)}
                    />
                  </div>
                  {config.contrast && (
                    <div className="space-y-2 pl-6">
                      <div className="flex items-center justify-between text-sm">
                        <span>幅度</span>
                        <span className="text-muted-foreground">±{(config.contrastLimit * 100).toFixed(0)}%</span>
                      </div>
                      <Slider
                        value={[config.contrastLimit]}
                        onValueChange={([v]) => updateConfig('contrastLimit', v)}
                        min={0} max={0.5} step={0.05}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Palette className="w-4 h-4" />
                    色彩调整
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>饱和度</Label>
                    <Switch
                      checked={config.saturation}
                      onCheckedChange={(v) => updateConfig('saturation', v)}
                    />
                  </div>
                  {config.saturation && (
                    <div className="space-y-2 pl-6">
                      <Slider
                        value={[config.saturationLimit]}
                        onValueChange={([v]) => updateConfig('saturationLimit', v)}
                        min={0} max={0.5} step={0.05}
                      />
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <Label>色相偏移</Label>
                    <Switch
                      checked={config.hue}
                      onCheckedChange={(v) => updateConfig('hue', v)}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Droplets className="w-4 h-4" />
                    模糊与噪声
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>高斯模糊</Label>
                    <Switch
                      checked={config.blur}
                      onCheckedChange={(v) => updateConfig('blur', v)}
                    />
                  </div>
                  {config.blur && (
                    <div className="space-y-2 pl-6">
                      <div className="flex items-center justify-between text-sm">
                        <span>核大小</span>
                        <span className="text-muted-foreground">{config.blurLimit}px</span>
                      </div>
                      <Slider
                        value={[config.blurLimit]}
                        onValueChange={([v]) => updateConfig('blurLimit', v)}
                        min={1} max={7} step={2}
                      />
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <Label>高斯噪声</Label>
                    <Switch
                      checked={config.noise}
                      onCheckedChange={(v) => updateConfig('noise', v)}
                    />
                  </div>
                  {config.noise && (
                    <div className="space-y-2 pl-6">
                      <div className="flex items-center justify-between text-sm">
                        <span>方差</span>
                        <span className="text-muted-foreground">{config.noiseVar.toFixed(2)}</span>
                      </div>
                      <Slider
                        value={[config.noiseVar]}
                        onValueChange={([v]) => updateConfig('noiseVar', v)}
                        min={0.01} max={0.5} step={0.01}
                      />
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="advanced" className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Shuffle className="w-4 h-4" />
                    Mosaic 与 Mixup
                  </CardTitle>
                  <CardDescription>
                    将多张图像组合在一起的高级增强方式
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Mosaic</Label>
                      <p className="text-xs text-muted-foreground">将 4 张图像拼接进一张</p>
                    </div>
                    <Switch
                      checked={config.mosaic}
                      onCheckedChange={(v) => updateConfig('mosaic', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Mixup</Label>
                      <p className="text-xs text-muted-foreground">将两张图像按比例混合</p>
                    </div>
                    <Switch
                      checked={config.mixup}
                      onCheckedChange={(v) => updateConfig('mixup', v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Cutout</Label>
                      <p className="text-xs text-muted-foreground">随机矩形遮挡</p>
                    </div>
                    <Switch
                      checked={config.cutout}
                      onCheckedChange={(v) => updateConfig('cutout', v)}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="output" className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <ImagePlus className="w-4 h-4" />
                    输出设置
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>输出数据集名称</Label>
                    <Input
                      value={outputName}
                      onChange={(e) => setOutputName(e.target.value)}
                      placeholder="增强后的数据集名称"
                    />
                  </div>

                  {/* Size mode toggle */}
                  <div className="space-y-3">
                    <Label>目标规模</Label>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSizeMode('factor')}
                        className="flex-1 px-3 py-1.5 text-sm rounded-md border transition-colors"
                        style={{
                          borderColor: sizeMode === 'factor' ? 'var(--primary)' : undefined,
                          color: sizeMode === 'factor' ? 'var(--primary)' : undefined,
                          backgroundColor: sizeMode === 'factor' ? 'rgba(64,158,255,0.08)' : 'transparent'
                        }}
                      >
                        倍数
                      </button>
                      <button
                        onClick={() => setSizeMode('absolute')}
                        className="flex-1 px-3 py-1.5 text-sm rounded-md border transition-colors"
                        style={{
                          borderColor: sizeMode === 'absolute' ? 'var(--primary)' : undefined,
                          color: sizeMode === 'absolute' ? 'var(--primary)' : undefined,
                          backgroundColor: sizeMode === 'absolute' ? 'rgba(64,158,255,0.08)' : 'transparent'
                        }}
                      >
                        绝对数量
                      </button>
                    </div>

                    {sizeMode === 'factor' ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            value={factorInput}
                            min={1.1}
                            max={20}
                            step={0.1}
                            className="w-24"
                            onChange={(e) => {
                              setFactorInput(e.target.value)
                              const v = parseFloat(e.target.value)
                              if (!isNaN(v) && v >= 1.1) setAugmentFactor(v)
                            }}
                          />
                          <span className="text-sm text-muted-foreground">× 倍数</span>
                        </div>
                        <Slider
                          value={[augmentFactor]}
                          onValueChange={([v]) => { setAugmentFactor(v); setFactorInput(String(v)) }}
                          min={1.1}
                          max={10}
                          step={0.1}
                        />
                        <p className="text-xs text-muted-foreground">
                          {selectedDataset.num_images} → 约 {computedTargetSize} 张图像（{augmentFactor.toFixed(1)}×）
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            value={targetSize || ''}
                            min={selectedDataset.num_images}
                            placeholder={String(selectedDataset.num_images * 2)}
                            onChange={(e) => setTargetSize(parseInt(e.target.value) || 0)}
                          />
                          <span className="text-sm text-muted-foreground">张</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          当前：{selectedDataset.num_images} 张图像
                          {targetSize > 0 && ` → ${targetSize} 张图像（${(targetSize / selectedDataset.num_images).toFixed(2)}×）`}
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Class-based augmentation */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Users className="w-4 h-4" />
                    类别均衡
                  </CardTitle>
                  <CardDescription>
                    为每个类别单独设置目标数量，以缓解类别不平衡
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>启用按类别目标</Label>
                    <Switch
                      checked={classAugEnabled}
                      onCheckedChange={setClassAugEnabled}
                    />
                  </div>
                  {classAugEnabled && (
                    <div className="space-y-2">
                      {datasetClasses.length === 0 ? (
                        <p className="text-xs text-muted-foreground">该数据集没有可用的类别信息。</p>
                      ) : (
                        datasetClasses.map((cls) => (
                          <div key={cls} className="flex items-center gap-3">
                            <span className="text-sm flex-1 truncate">{cls}</span>
                            <Input
                              type="number"
                              className="w-24"
                              placeholder="目标数量"
                              value={classTargets[cls] ?? ''}
                              onChange={(e) => {
                                const v = parseInt(e.target.value)
                                setClassTargets(prev => ({
                                  ...prev,
                                  [cls]: isNaN(v) ? 0 : v
                                }))
                              }}
                            />
                            <span className="text-xs text-muted-foreground w-14">张</span>
                          </div>
                        ))
                      )}
                      <p className="text-xs text-muted-foreground pt-1">
                        留空则对该类别使用全局目标规模。
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Preview panel */}
        <Card className="w-80 flex-shrink-0">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Target className="w-4 h-4" />
              预览
            </CardTitle>
            <CardDescription>增强后的样本 — 每张采用随机组合</CardDescription>
          </CardHeader>
          <CardContent>
            {previewLoading ? (
              <div className="flex items-center justify-center h-48">
                <RefreshCcw className="w-8 h-8 text-muted-foreground animate-spin" />
              </div>
            ) : previews.length > 0 ? (
              <div className="grid grid-cols-2 gap-2">
                {previews.map((p, i) => (
                  <div key={i} className="space-y-1">
                    <div className="aspect-square bg-muted rounded overflow-hidden">
                      {p.data_url ? (
                        <img src={p.data_url} alt={`预览 ${i + 1}`} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Eye className="w-5 h-5 text-muted-foreground" />
                        </div>
                      )}
                    </div>
                    {p.augmentations.length > 0 && (
                      <p className="text-[10px] text-muted-foreground leading-tight truncate" title={p.augmentations.join(', ')}>
                        {p.augmentations.join(', ')}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-48 text-center">
                <Eye className="w-8 h-8 text-muted-foreground mb-2" />
                <p className="text-sm text-muted-foreground">
                  点击「生成预览」查看增强后的样本
                </p>
              </div>
            )}
            <Button
              variant="outline"
              className="w-full mt-4"
              onClick={handlePreview}
              disabled={previewLoading}
            >
              <RefreshCcw className={cn("w-4 h-4 mr-2", previewLoading && "animate-spin")} />
              生成预览
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
