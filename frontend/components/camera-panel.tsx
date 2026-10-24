"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Camera, RefreshCw, Play, Pause, Square } from "lucide-react"

interface Options {
  compute_node_id?: string | null
  model_path: string; task: string; confidence: number; iou: number
  imgsz: number; tracker: string; classes: number[]
  // Tracking / ReID (only used when task === "track")
  with_reid?: boolean; reid_model?: string | null
  track_high_thresh?: number; new_track_thresh?: number
  match_thresh?: number; track_buffer?: number
}
interface Props { apiUrl: string; options: Options; enabled: boolean }
interface Device { index: number; name: string; openable: boolean; resolutions: number[][]; error?: string }
interface LiveState {
  id: string; status: string; error?: string; index: number; resolution?: number[]
  model_path: string; camera_fps?: number; inference_fps?: number; latency_ms?: number
  object_count?: number; sequence?: number; annotated_b64?: string
  device?: string
}
const labels: Record<string,string> = {
  starting:"启动中",running:"运行中",paused:"已暂停",stopping:"停止中",stopped:"已停止",error:"出错",
}

export function CameraPanel({ apiUrl, options, enabled }: Props) {
  const [computeDevice,setComputeDevice] = useState("cuda:0")
  const [devices,setDevices] = useState<Device[]>([])
  const [device,setDevice] = useState("0")
  const [resolution,setResolution] = useState("1280x720")
  const [scanning,setScanning] = useState(false)
  const [busy,setBusy] = useState(false)
  const [live,setLive] = useState<LiveState | null>(null)
  const [preview,setPreview] = useState<string | null>(null)
  const [error,setError] = useState<string | null>(null)
  const mounted = useRef(false)
  const session = useRef<string | null>(null)
  const action = useRef(false)
  const sequence = useRef(0)
  const settings = useRef({...options, device:computeDevice})
  const applied = useRef("")
  const scanningRef = useRef(false)
  const ready = useRef(enabled)
  settings.current = {...options, device:computeDevice}
  ready.current = enabled
  const active = !!live && !["stopped","error"].includes(live.status)

  async function request(path: string, init?: RequestInit) {
    const response = await fetch(`${apiUrl}/api/camera${path}`,{...init,signal:AbortSignal.timeout(15000)})
    const data = await response.json()
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
    return data
  }

  useEffect(() => {
    mounted.current = true
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const release = () => {
      const id = session.current
      session.current = null
      if (mounted.current) { setLive(previous=>previous ? {...previous,status:"stopped"} : null);setPreview(null) }
      if (id) {
        const url = `${apiUrl}/api/camera/${id}/stop`
        if (!navigator.sendBeacon?.(url)) void fetch(url,{method:"POST",keepalive:true}).catch(()=>{})
      }
    }
    const tick = async () => {
      try {
        if (scanningRef.current) {
          const scan = await request("/devices")
          if (!disposed && scan.status !== "scanning") {
            scanningRef.current = false; setScanning(false); setDevices(scan.devices)
            if (scan.error) setError(scan.error)
            else if (!scan.devices.length) setError("未检测到摄像头。请连接 USB 摄像头后重新扫描。")
            const first = scan.devices.find((d: Device)=>d.openable)
            if (first) setDevice(String(first.index))
          }
        }
        const id = session.current
        if (id) {
          const encoded = JSON.stringify(settings.current)
          if (ready.current && encoded !== applied.current) {
            try {
              await request(`/${id}/options`,{method:"PATCH",headers:{"Content-Type":"application/json"},body:encoded})
              applied.current = encoded
            } catch (e) { if (!disposed) setError(String(e)) }
          }
          const data: LiveState = await request(`/${id}/frame?after=${sequence.current}`)
          if (!disposed && session.current === id) {
            setLive(data)
            if (data.annotated_b64) {
              setPreview(`data:image/jpeg;base64,${data.annotated_b64}`)
              sequence.current = data.sequence || 0
            }
            if (data.error) setError(data.error)
            if (["stopped","error"].includes(data.status)) session.current = null
          }
        }
      } catch (e) { if (!disposed) setError(String(e)) }
      finally { if (!disposed) timer = setTimeout(tick,100) }
    }
    void tick()
    window.addEventListener("pagehide",release)
    return () => {
      disposed = true; mounted.current = false; clearTimeout(timer)
      window.removeEventListener("pagehide",release); release()
    }
    // Poll requests are sequential and read settings through refs; parameter changes never restart capture.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[apiUrl])

  async function scan() {
    if (action.current || session.current || scanningRef.current) return
    setError(null); setScanning(true)
    try { await request("/scan",{method:"POST"}); scanningRef.current=true }
    catch (e) { scanningRef.current=false;setScanning(false);setError(String(e)) }
  }
  async function run(command: "start" | "pause" | "stop") {
    if (action.current) return
    action.current = true; setBusy(true); setError(null)
    try {
      if (command === "start") {
        const [width,height] = resolution.split("x").map(Number)
        const data: LiveState = await request("/start",{method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({...settings.current,index:Number(device),width,height})})
        if (!mounted.current) {
          void fetch(`${apiUrl}/api/camera/${data.id}/stop`,{method:"POST",keepalive:true}).catch(()=>{})
          return
        }
        session.current=data.id;sequence.current=0;applied.current=JSON.stringify(settings.current)
        setPreview(null);setLive(data)
      } else if (session.current) {
        const id = session.current
        if (command === "pause") {
          await request(`/${id}/pause?paused=${live?.status !== "paused"}`,{method:"POST"})
        } else {
          const data = await request(`/${id}/stop`,{method:"POST"})
          if (data.status !== "stopping") session.current=null
          if (mounted.current) { setLive(data);setPreview(null) }
        }
      }
    } catch (e) { if (mounted.current) setError(String(e)) }
    finally { action.current=false;if (mounted.current) setBusy(false) }
  }
  const selected = devices.find(d=>String(d.index)===device)
  const resolutions = [...new Set(["640x480","1280x720","1920x1080",...(selected?.resolutions||[]).map(r=>r.join("x"))])]
  return <Card className="border-border/50 bg-card/50">
    <CardHeader className="pb-3"><CardTitle className="flex items-center gap-2 text-sm"><Camera className="w-4 h-4"/>实时摄像头</CardTitle></CardHeader>
    <CardContent className="space-y-4">
      <p className="text-xs text-muted-foreground">使用左侧的 YOLO 模型、任务类型、置信度、IoU、图像尺寸与类别设置，修改会自动生效。实时视频不会保存到磁盘。</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1"><label className="text-xs">摄像头设备</label>
          <Select value={device} onValueChange={setDevice} disabled={active || busy || scanning}>
            <SelectTrigger aria-label="摄像头设备"><SelectValue/></SelectTrigger><SelectContent>
              {!devices.length && <SelectItem value="0">摄像头 0</SelectItem>}
              {devices.map(d=><SelectItem key={d.index} value={String(d.index)}>摄像头 {d.index} - {d.name}{!d.openable ? "（不可用）" : ""}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1"><label className="text-xs">请求分辨率</label>
          <Select value={resolution} onValueChange={setResolution} disabled={active || busy}>
            <SelectTrigger aria-label="请求分辨率"><SelectValue/></SelectTrigger><SelectContent>
              {resolutions.map(r=><SelectItem key={r} value={r}>{r.replace("x"," × ")}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="space-y-1">
        <label className="text-xs">推理设备</label>
        <Select value={computeDevice} onValueChange={setComputeDevice} disabled={active || busy}>
          <SelectTrigger aria-label="推理设备"><SelectValue/></SelectTrigger>
          <SelectContent>
            <SelectItem value="cpu">CPU</SelectItem>
            <SelectItem value="cuda:0">NVIDIA GPU</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" size="sm" onClick={scan} disabled={active || busy || scanning}><RefreshCw className={`w-3.5 h-3.5 mr-1 ${scanning ? "animate-spin" : ""}`}/>{scanning ? "扫描中…" : "扫描设备"}</Button>
        <Button size="sm" onClick={()=>run("start")} disabled={active || busy || scanning || !enabled}><Play className="w-3.5 h-3.5 mr-1"/>启动</Button>
        <Button variant="outline" size="sm" onClick={()=>run("pause")} disabled={busy || !live || !["running","paused"].includes(live.status)}><Pause className="w-3.5 h-3.5 mr-1"/>{live?.status === "paused" ? "继续" : "暂停"}</Button>
        <Button variant="outline" size="sm" onClick={()=>run("stop")} disabled={busy || !active}><Square className="w-3.5 h-3.5 mr-1"/>停止</Button>
      </div>
      {!enabled && <p className="text-xs text-muted-foreground">请在左侧选择 YOLO 模型，并等待其类别加载完成。</p>}
      {(error || selected?.error) && <p role="alert" className="text-xs text-destructive break-words">{error || selected?.error}</p>}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs rounded-lg bg-muted/30 p-3" aria-live="polite">
        <span>设备：摄像头 {live?.index ?? device}</span><span>实际分辨率：{live?.resolution?.join(" × ") || "—"}</span>
        <span>状态：{labels[live?.status || "stopped"]}</span>
        <span>摄像头 FPS：{live?.camera_fps?.toFixed(1) ?? "—"}</span>
        <span>推理 FPS：{live?.status === "paused" ? "0.0" : live?.inference_fps?.toFixed(1) ?? "—"}</span>
        <span>推理延迟：{live?.latency_ms ?? "—"} ms</span><span>目标数：{live?.object_count ?? 0}</span>
        <span>计算设备：{live?.device || "—"}</span>
        <span className="col-span-2 break-all">模型：{(live?.model_path || options.model_path).split(/[\\/]/).pop() || "—"}</span>
      </div>
      {live?.resolution && live.resolution.join("x") !== resolution && <p className="text-xs text-muted-foreground">摄像头正在使用其支持的分辨率：{live.resolution.join(" × ")}。</p>}
      <div className="min-h-64 bg-black/90 rounded-lg flex items-center justify-center overflow-hidden">
        {preview ? <img src={preview} alt="实时摄像头检测结果" className="w-full h-auto max-h-[70vh] object-contain"/> : <p className="text-sm text-white/70 p-8">请选择摄像头与模型，然后开始实时推理</p>}
      </div>
    </CardContent>
  </Card>
}
