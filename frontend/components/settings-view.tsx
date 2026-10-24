"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Server,
  FolderOpen,
  Cpu,
  Palette,
  Save,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Shield,
  RotateCcw,
  Square,
  FileText,
  Folder,
  ChevronRight,
  HardDrive,
  Sun,
} from "lucide-react"
import { useSettings } from "@/lib/settings-context"

interface FolderItem {
  name: string
  path: string
  is_directory: boolean
}

function FolderPickerButton({
  value,
  onChange,
  apiUrl,
}: {
  value: string
  onChange: (path: string) => void
  apiUrl: string
}) {
  const [open, setOpen] = useState(false)
  const [currentPath, setCurrentPath] = useState("")
  const [items, setItems] = useState<FolderItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [manualPath, setManualPath] = useState("")

  const browse = async (path: string = "") => {
    setIsLoading(true)
    try {
      const res = await fetch(`${apiUrl}/api/browse-folders?path=${encodeURIComponent(path || ".")}`, { method: "POST" })
      if (res.ok) {
        const data = await res.json()
        setCurrentPath(data.current_path || path)
        setManualPath(data.current_path || path)
        setItems((data.items || []).filter((i: FolderItem) => i.is_directory))
      }
    } catch {}
    finally { setIsLoading(false) }
  }

  const handleOpen = () => {
    setOpen(true)
    browse(value || "")
  }

  return (
    <>
      <Button variant="outline" size="icon" onClick={handleOpen} title="浏览文件夹">
        <FolderOpen className="w-4 h-4" />
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-lg flex flex-col" style={{ maxHeight: "70vh" }}>
          <DialogHeader>
            <DialogTitle>选择文件夹</DialogTitle>
            <DialogDescription>浏览并选择目录路径。</DialogDescription>
          </DialogHeader>
          <div className="space-y-3 flex-1 flex flex-col min-h-0">
            <div className="flex gap-2">
              <Input
                value={manualPath}
                onChange={e => setManualPath(e.target.value)}
                onKeyDown={e => e.key === "Enter" && browse(manualPath)}
                placeholder="输入路径后按回车"
                className="flex-1"
              />
              <Button variant="secondary" size="sm" onClick={() => browse(manualPath)}>前往</Button>
              <Button variant="ghost" size="icon" onClick={() => browse(currentPath)}>
                <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
              </Button>
            </div>
            {currentPath && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground bg-muted/50 px-3 py-1.5 rounded">
                <HardDrive className="w-3 h-3 flex-shrink-0" />
                <span className="truncate">{currentPath}</span>
              </div>
            )}
            <ScrollArea className="border rounded-lg" style={{ minHeight: 200, maxHeight: 300 }}>
              {isLoading ? (
                <div className="flex items-center justify-center h-32">
                  <RefreshCw className="w-5 h-5 animate-spin text-muted-foreground" />
                </div>
              ) : items.length === 0 ? (
                <div className="flex items-center justify-center h-32 text-muted-foreground">
                  <Folder className="w-8 h-8 opacity-30 mr-2" />
                  <span className="text-sm">无子目录</span>
                </div>
              ) : (
                <div className="p-2 space-y-0.5">
                  {items.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-center gap-2 p-2 rounded hover:bg-muted cursor-pointer"
                      onClick={() => browse(item.path)}
                    >
                      <Folder className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                      <span className="flex-1 text-sm truncate">{item.name}</span>
                      <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
            <div className="flex gap-2 justify-end pt-1">
              <Button variant="outline" onClick={() => setOpen(false)}>取消</Button>
              <Button onClick={() => { onChange(currentPath); setOpen(false) }} disabled={!currentPath}>
                选择此文件夹
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

interface HardwareInfo {
  platform: string
  python_version: string
  cuda_available: boolean
  cuda_version: string | null
  gpu_count: number
  gpus: { index: number; name: string; memory_total_mb: number; memory_free_mb: number | null }[]
  cpu_cores: number | null
  cpu_model: string | null
  ram_total_gb: number | null
  ram_available_gb: number | null
}

export function SettingsView() {
  const { settings, saveSettings, isSaving, lastSaved } = useSettings()

  // Local draft state — only committed on "Save Changes"
  const [draft, setDraft] = useState({ ...settings })
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "disconnected" | "checking">("disconnected")
  const [saved, setSaved] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [restarting, setRestarting] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [hwInfo, setHwInfo] = useState<HardwareInfo | null>(null)
  const [hwLoading, setHwLoading] = useState(false)

  const fetchHardware = async () => {
    setHwLoading(true)
    try {
      const res = await fetch(`${draft.apiUrl}/api/hardware`)
      if (res.ok) setHwInfo(await res.json())
    } catch {}
    finally { setHwLoading(false) }
  }

  // Auto-fetch on mount
  useEffect(() => { fetchHardware() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Keep draft in sync if settings change externally (e.g. first hydration from localStorage)
  useEffect(() => {
    setDraft({ ...settings })
  }, [settings.apiUrl]) // re-sync when apiUrl changes (covers initial load)

  const patch = (key: keyof typeof draft, value: unknown) =>
    setDraft(prev => ({ ...prev, [key]: value }))

  const checkConnection = async () => {
    setConnectionStatus("checking")
    try {
      const response = await fetch(`${draft.apiUrl}/api/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      })
      setConnectionStatus(response.ok ? "connected" : "disconnected")
    } catch {
      setConnectionStatus("disconnected")
    }
  }

  const handleSave = async () => {
    // Push draft into context (localStorage + backend)
    const synced = await saveSettings(draft)
    setSaved(synced)
    setSaveError(synced ? null : "设置已保存在本地，但未能同步到服务端。请检查连接后重试。")
    if (synced) setTimeout(() => setSaved(false), 2500)
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">设置</h1>
          <p className="text-muted-foreground mt-1">
            配置应用偏好与后端连接
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastSaved && (
            <span className="text-xs text-muted-foreground">
              上次保存 {lastSaved.toLocaleTimeString()}
            </span>
          )}
          <Button onClick={handleSave} disabled={isSaving} className="gap-2">
            {isSaving ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                保存中…
              </>
            ) : saved ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                已保存
              </>
            ) : (
              <>
                <Save className="w-4 h-4" />
                保存修改
              </>
            )}
          </Button>
        </div>
      </div>

      {saveError && <p role="alert" className="text-sm text-destructive">{saveError}</p>}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Backend Connection ────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Server className="w-5 h-5 text-primary" />
              后端连接
            </CardTitle>
            <CardDescription>
              配置 Python 后端服务连接。保存后立即生效。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">API URL</label>
              <div className="flex gap-2">
                <Input
                  value={draft.apiUrl}
                  onChange={e => patch("apiUrl", e.target.value)}
                  placeholder="http://localhost:8000"
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  onClick={checkConnection}
                  disabled={connectionStatus === "checking"}
                >
                  {connectionStatus === "checking" ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    "测试"
                  )}
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {connectionStatus === "connected" && (
                <Badge className="bg-green-500/10 text-green-700 dark:text-green-400">
                  <CheckCircle2 className="w-3 h-3 mr-1" />
                  已连接
                </Badge>
              )}
              {connectionStatus === "disconnected" && (
                <Badge variant="destructive" className="bg-destructive/10">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  未连接
                </Badge>
              )}
              {connectionStatus === "checking" && (
                <Badge variant="secondary">
                  <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                  检测中…
                </Badge>
              )}
            </div>

            <p className="text-xs text-muted-foreground">
              启动后端命令：{" "}
              <code className="bg-muted px-1 py-0.5 rounded">
                uvicorn main:app --reload --port 8000
              </code>
            </p>

            <div className="pt-2 border-t border-border space-y-2">
              <p className="text-sm font-medium mb-3">服务控制</p>

              {/* Restart backend — zero-downtime via uvicorn --reload */}
              <Button
                variant="outline"
                size="sm"
                className="w-full gap-2"
                disabled={restarting}
                onClick={async () => {
                  setRestarting(true)
                  try {
                    const r = await fetch(`${settings.apiUrl}/api/restart`, { method: "POST" })
                    if (r.ok) {
                      // Poll until backend comes back
                      let attempts = 0
                      const poll = setInterval(async () => {
                        attempts++
                        try {
                          const h = await fetch(`${settings.apiUrl}/api/health`, { signal: AbortSignal.timeout(1000) })
                          if (h.ok) { clearInterval(poll); setRestarting(false); setConnectionStatus("connected") }
                        } catch { /* still reloading */ }
                        if (attempts > 20) { clearInterval(poll); setRestarting(false) }
                      }, 500)
                    } else {
                      setRestarting(false)
                    }
                  } catch {
                    setRestarting(false)
                  }
                }}
              >
                {restarting
                  ? <><RefreshCw className="w-4 h-4 animate-spin" /> 重新加载中…</>
                  : <><RotateCcw className="w-4 h-4" /> 重载后端</>
                }
              </Button>
              <p className="text-xs text-muted-foreground">
                触发 <code className="bg-muted px-1 py-0.5 rounded">uvicorn --reload</code> — 约 1 秒内生效，无需停机即可加载 Python 改动。
              </p>

              {/* View logs */}
              <Button
                variant="ghost"
                size="sm"
                className="w-full gap-2 text-muted-foreground"
                onClick={() => window.open(`${settings.apiUrl}/docs`, "_blank")}
              >
                <FileText className="w-4 h-4" />
                打开 API 文档
              </Button>

              {/* Hard stop */}
              <div className="pt-1 border-t border-border">
                <p className="text-sm font-medium mb-2 text-destructive">危险操作</p>
                <Button
                  variant="destructive"
                  size="sm"
                  className="w-full gap-2"
                  disabled={stopping}
                  onClick={async () => {
                    if (!confirm("确定停止后端服务吗？在重新执行  python run.py start  之前，应用将无法使用。")) return
                    setStopping(true)
                    try {
                      await fetch(`${settings.apiUrl}/api/shutdown`, { method: "POST" })
                    } catch { /* expected — server is shutting down */ }
                    setConnectionStatus("disconnected")
                    setStopping(false)
                  }}
                >
                  {stopping
                    ? <><RefreshCw className="w-4 h-4 animate-spin" /> 停止中…</>
                    : <><Square className="w-4 h-4" /> 停止后端</>
                  }
                </Button>
                <p className="text-xs text-muted-foreground mt-1">
                  完全终止 Python 进程。可通过 <code className="bg-muted px-1 py-0.5 rounded">python run.py restart-back</code> 重新启动。
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ── Storage Paths ─────────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-primary" />
              存储路径
            </CardTitle>
            <CardDescription>
              配置后端使用的默认目录。保存时路径会同步到服务端。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">模型目录</label>
              <div className="flex gap-2">
                <Input
                  value={draft.modelsPath}
                  onChange={e => patch("modelsPath", e.target.value)}
                  placeholder="/path/to/models"
                  className="flex-1"
                />
                <FolderPickerButton
                  value={draft.modelsPath}
                  onChange={path => patch("modelsPath", path)}
                  apiUrl={draft.apiUrl}
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">数据集目录</label>
              <div className="flex gap-2">
                <Input
                  value={draft.datasetsPath}
                  onChange={e => patch("datasetsPath", e.target.value)}
                  placeholder="/path/to/datasets"
                  className="flex-1"
                />
                <FolderPickerButton
                  value={draft.datasetsPath}
                  onChange={path => patch("datasetsPath", path)}
                  apiUrl={draft.apiUrl}
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">输出目录</label>
              <div className="flex gap-2">
                <Input
                  value={draft.outputPath}
                  onChange={e => patch("outputPath", e.target.value)}
                  placeholder="/path/to/output"
                  className="flex-1"
                />
                <FolderPickerButton
                  value={draft.outputPath}
                  onChange={path => patch("outputPath", path)}
                  apiUrl={draft.apiUrl}
                />
              </div>
            </div>

            <p className="text-xs text-muted-foreground pt-1">
              路径修改会在保存时发送到后端，并在多次会话之间保留。
            </p>
          </CardContent>
        </Card>

        {/* ── Hardware ─────────────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" />
              硬件
            </CardTitle>
            <CardDescription>
              配置 GPU 与计算设置，将应用于训练与自动标注任务。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">使用 GPU</p>
                <p className="text-xs text-muted-foreground">启用 CUDA / MPS 加速</p>
              </div>
              <Switch
                checked={draft.useGpu}
                onCheckedChange={val => patch("useGpu", val)}
              />
            </div>

            {draft.useGpu && (
              <div className="space-y-2">
                <label className="text-sm font-medium">GPU Device ID（设备编号）</label>
                <Input
                  value={draft.gpuDevice}
                  onChange={e => patch("gpuDevice", e.target.value)}
                  placeholder="0"
                  className="w-24"
                />
                <p className="text-xs text-muted-foreground">
                  多卡时使用逗号分隔的编号（例如 <code>0,1</code>）
                </p>
              </div>
            )}

            <div className="pt-2 space-y-2">
              <div className="flex items-center justify-between mb-1">
                <p className="text-sm font-medium">系统信息</p>
                <Button variant="ghost" size="sm" className="h-7 gap-1 text-xs" onClick={fetchHardware} disabled={hwLoading}>
                  <RefreshCw className={`w-3 h-3 ${hwLoading ? "animate-spin" : ""}`} />
                  刷新
                </Button>
              </div>

              {hwLoading && !hwInfo && (
                <p className="text-xs text-muted-foreground">加载中…</p>
              )}

              {hwInfo && (
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">CUDA</span>
                    {hwInfo.cuda_available
                      ? <Badge className="bg-green-500/10 text-green-700 dark:text-green-400">可用 {hwInfo.cuda_version ? `(${hwInfo.cuda_version})` : ""}</Badge>
                      : <Badge variant="secondary">不可用</Badge>}
                  </div>

                  {hwInfo.gpus.length > 0 ? hwInfo.gpus.map(gpu => (
                    <div key={gpu.index} className="space-y-0.5">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">GPU {hwInfo.gpus.length > 1 ? gpu.index : ""}</span>
                        <span className="font-medium text-xs text-right max-w-[55%] truncate">{gpu.name}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">VRAM</span>
                        <span className="text-xs">
                          {gpu.memory_free_mb != null
                            ? `${gpu.memory_free_mb} MB 可用 / ${gpu.memory_total_mb} MB`
                            : `${gpu.memory_total_mb} MB`}
                        </span>
                      </div>
                    </div>
                  )) : (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">GPU</span>
                      <Badge variant="secondary">未检测到</Badge>
                    </div>
                  )}

                  {hwInfo.cpu_model && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">CPU</span>
                      <span className="font-medium text-xs text-right max-w-[60%] truncate">{hwInfo.cpu_model}</span>
                    </div>
                  )}

                  {hwInfo.cpu_cores && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">CPU 核心数</span>
                      <span className="font-medium text-xs">{hwInfo.cpu_cores} logical</span>
                    </div>
                  )}

                  {hwInfo.ram_total_gb && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">RAM</span>
                      <span className="font-medium text-xs">
                        {hwInfo.ram_available_gb} GB 可用 / {hwInfo.ram_total_gb} GB
                      </span>
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">平台</span>
                    <span className="font-medium text-xs">{hwInfo.platform} · Python {hwInfo.python_version}</span>
                  </div>
                </div>
              )}
            </div>

            <p className="text-xs text-muted-foreground">
              GPU 设置在保存后应用于新的训练任务与自动标注任务。
            </p>
          </CardContent>
        </Card>

        {/* ── Preferences ──────────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Palette className="w-5 h-5 text-primary" />
              偏好设置
            </CardTitle>
            <CardDescription>
              外观与 RunHive 统一为蓝白主题，其余设置在保存后生效。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">外观模式</p>
                <p className="text-xs text-muted-foreground">
                  与 RunHive 使用相同的蓝白界面
                </p>
              </div>
              <Badge variant="secondary" className="gap-1.5 bg-primary/10 text-primary border border-primary/15">
                <Sun className="w-3.5 h-3.5" />
                RunHive 蓝白主题
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">自动保存标注</p>
                <p className="text-xs text-muted-foreground">
                  在图像之间切换时自动保存
                </p>
              </div>
              <Switch
                checked={draft.autoSave}
                onCheckedChange={val => patch("autoSave", val)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">通知</p>
                <p className="text-xs text-muted-foreground">
                  任务完成时显示提示通知
                </p>
              </div>
              <Switch
                checked={draft.notifications}
                onCheckedChange={val => patch("notifications", val)}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── About ────────────────────────────────────────────────── */}
      <Card className="border-border/50 bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            关于
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">版本</p>
              <p className="font-medium">3.0.0</p>
            </div>
            <div>
              <p className="text-muted-foreground">许可证</p>
              <p className="font-medium">MIT</p>
            </div>
            <div>
              <p className="text-muted-foreground">后端</p>
              <p className="font-medium">FastAPI + Python</p>
            </div>
            <div>
              <p className="text-muted-foreground">前端</p>
              <p className="font-medium">Next.js + React</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
