"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { FolderOpen, Play, Square, Download } from "lucide-react"
import { hasDetectorConfidence } from "@/lib/model-capabilities"

interface Prediction { annotated_b64: string; detections?: { class_name: string; confidence: number; confidence_source?: string }[] }
interface Props { enabled: boolean; infer: (file: File, signal: AbortSignal) => Promise<Prediction> }
interface Item {
  file: File; name: string; status: "Pending" | "Running" | "Done" | "Failed"
  result?: string; error?: string; detections?: Prediction["detections"]
}

const STATUS_LABELS: Record<string, string> = {
  Pending: "待处理",
  Running: "处理中",
  Done: "已完成",
  Failed: "失败",
}

export function FolderInferencePanel({ enabled, infer }: Props) {
  const [items,setItems] = useState<Item[]>([])
  const [selected,setSelected] = useState(0)
  const [running,setRunning] = useState(false)
  const [stopping,setStopping] = useState(false)
  const [ignored,setIgnored] = useState(0)
  const [notice,setNotice] = useState("")
  const [inputUrl,setInputUrl] = useState("")
  const picker = useRef<HTMLInputElement>(null)
  const mounted = useRef(false)
  const busy = useRef(false)
  const stopRequested = useRef(false)
  const controller = useRef<AbortController | null>(null)
  const resultUrls = useRef(new Set<string>())

  function clearResults() {
    resultUrls.current.forEach(url=>URL.revokeObjectURL(url))
    resultUrls.current.clear()
  }
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false; stopRequested.current = true
      controller.current?.abort(); clearResults()
    }
  },[])
  const current = items[selected]
  useEffect(() => {
    if (!current?.file) { setInputUrl(""); return }
    const url = URL.createObjectURL(current.file)
    setInputUrl(url)
    return () => URL.revokeObjectURL(url)
  },[current?.file])

  function choose(files: FileList | null) {
    if (!files || busy.current) return
    const all = Array.from(files)
    const images = all.filter(file=>/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name))
      .sort((a,b)=>(a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name,undefined,{numeric:true}))
    clearResults(); setSelected(0); setIgnored(all.length-images.length)
    setItems(images.map(file=>({file,name:file.webkitRelativePath || file.name,status:"Pending"})))
    setNotice(images.length ? "" : "该文件夹中未找到受支持的图像。")
  }

  async function run() {
    if (busy.current || !enabled || !items.length) return
    busy.current = true; stopRequested.current = false;setRunning(true);setStopping(false);setNotice("")
    // Freeze this run's model and parameters through the callback captured at Start.
    const predict = infer
    const work = items.map(item=>({...item}))
    try {
      for (let i=0;i<work.length;i++) {
        if (stopRequested.current || !mounted.current) break
        if (work[i].status === "Done") continue
        const request = new AbortController();controller.current = request
        work[i] = {...work[i],status:"Running",error:undefined}
        setItems([...work]);setSelected(i)
        try {
          const prediction = await predict(work[i].file,request.signal)
          if (!mounted.current) break
          if (!prediction.annotated_b64) throw new Error("服务器未返回结果图像")
          const bytes = Uint8Array.from(atob(prediction.annotated_b64),c=>c.charCodeAt(0))
          const url = URL.createObjectURL(new Blob([bytes],{type:"image/jpeg"}))
          resultUrls.current.add(url)
          work[i] = {...work[i],status:"Done",result:url,detections:prediction.detections || []}
        } catch (e) {
          if (!mounted.current) break
          work[i] = {...work[i],status:"Failed",error:e instanceof Error ? e.message : String(e)}
        }
        if (mounted.current) setItems([...work])
      }
      if (mounted.current) setNotice(stopRequested.current ? "已在当前图像处理后停止。再次运行可继续处理剩余或失败的图像。" : "批处理完成。选择图像以查看或保存其结果。")
    } finally {
      controller.current=null;busy.current=false
      if (mounted.current) {setRunning(false);setStopping(false)}
    }
  }
  const done = items.filter(item=>item.status === "Done").length
  const failed = items.filter(item=>item.status === "Failed").length
  return <div className="space-y-4">
    <input ref={picker} type="file" multiple {...{webkitdirectory:"",directory:""}}
      className="hidden" aria-label="上传图像文件夹" disabled={running}
      onChange={event=>{choose(event.target.files);event.target.value=""}} />
    <div className="rounded-lg border-2 border-dashed border-border/50 p-6 text-center space-y-2">
      <Button variant="outline" onClick={()=>picker.current?.click()} disabled={running} className="gap-2">
        <FolderOpen className="w-4 h-4"/> 上传文件夹
      </Button>
      <p className="text-xs text-muted-foreground">支持 JPG、PNG、BMP、WebP 与 TIFF，包含子文件夹。图像将逐张处理。</p>
      {!!ignored && <p className="text-xs text-muted-foreground">已忽略 {ignored} 个不支持的文件。</p>}
    </div>
    <p className="text-xs text-muted-foreground">批量处理开始时沿用左侧的 YOLO 模型、任务类型与参数。请保持本标签页打开，并在离开或选择其它文件夹前保存结果。</p>
    {!enabled && <p className="text-xs text-muted-foreground">请选择 YOLO 模型并等待其类别加载完成。大模型仍可在「单张图像」模式下使用。</p>}
    {!!items.length && <>
      <div className="flex flex-wrap gap-2 items-center">
        <Button onClick={run} disabled={!enabled || running || done===items.length} className="gap-2"><Play className="w-4 h-4"/>{running ? "运行中…" : "运行文件夹推理"}</Button>
        {running && <Button variant="outline" disabled={stopping} onClick={()=>{stopRequested.current=true;setStopping(true)}} className="gap-2"><Square className="w-4 h-4"/>{stopping ? "停止中…" : "处理完当前图像后停止"}</Button>}
        <span className="text-xs text-muted-foreground">{items.length} 张图像 · 已完成 {done} · 失败 {failed}</span>
      </div>
      <Progress value={(done+failed)/items.length*100}/>
      <select aria-label="文件夹图像" className="w-full rounded-md border border-border bg-background p-2 text-xs" value={selected} onChange={event=>setSelected(Number(event.target.value))}>
        {items.map((item,index)=><option key={item.name} value={index}>{index+1}. {item.name} — {STATUS_LABELS[item.status] ?? item.status}</option>)}
      </select>
      <div className="flex gap-2 items-center">
        <Button variant="outline" size="sm" disabled={selected===0} onClick={()=>setSelected(i=>i-1)}>上一张</Button>
        <span className="text-xs">{selected+1} / {items.length}</span>
        <Button variant="outline" size="sm" disabled={selected===items.length-1} onClick={()=>setSelected(i=>i+1)}>下一张</Button>
        {current?.result && <Button variant="outline" size="sm" className="gap-1 ml-auto" onClick={()=>{
          const link=document.createElement("a");link.href=current.result!;
          link.download=current.name.replace(/[\\/]/g,"__").replace(/\.[^.]+$/,"")+"_result.jpg";link.click()
        }}><Download className="w-4 h-4"/>保存结果</Button>}
      </div>
      {current?.error && <p role="alert" className="text-xs text-destructive break-words">{current.name}: {current.error}</p>}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {inputUrl && <div><p className="text-xs text-muted-foreground mb-1">输入</p><img src={inputUrl} alt={current?.name || "输入"} className="w-full rounded border border-border/40"/></div>}
        {current?.result && <div><p className="text-xs text-muted-foreground mb-1">结果 · {current.detections?.length || 0} 个检测结果</p><img src={current.result} alt="文件夹推理结果" className="w-full rounded border border-border/40"/></div>}
      </div>
      {!!current?.detections?.length && <div className="max-h-40 overflow-auto text-xs space-y-1">{current.detections.map((d,i)=><p key={i}>{d.class_name}{hasDetectorConfidence(d) && <> · {(d.confidence*100).toFixed(1)}%</>}</p>)}</div>}
    </>}
    {notice && <p className="text-xs text-muted-foreground" role="status">{notice}</p>}
  </div>
}
