"use client"
import { useCallback, useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { CheckCircle2, ChevronLeft, ChevronRight, Loader2, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'

type Box = [number, number, number, number]
type Counts = { pending: number; accepted: number; rejected: number; total: number }
type Candidate = { candidate_id: string; image_key: string; image: string; bbox: Box; class_name: string;
  review_status: 'pending' | 'accepted' | 'rejected'; decision_reasons: string[]; evidence?: any }
type Job = { job_id: string; dataset_id: string; dataset_name: string; kind: string; status: string; readonly: boolean;
  readonly_reason: string; summary: Counts; revision: number }
type Queue = Job & { candidates: Candidate[]; classes: string[]; format: string }
type Detail = { width: number; height: number; image_url: string; existing_annotations: { class_name: string; bbox: Box }[] }
type Props = { apiUrl: string; datasetId?: string; initialJobId?: string | null; initialImageKey?: string | null;
  onBack: (tab: 'annotate' | 'jobs') => void; onChanged: () => void }
const STATUS = { pending: '待复核', accepted: '已确认', rejected: '已驳回' }
const modelName = (id: string) => id === 'vlx-seek' ? 'VLX-Seek' : id.startsWith('qwen') ? 'Qwen' : id
function reasons(items: string[]) {
  return [...new Set(items.map(r => r.includes('class_conflict') ? '模型对类别判断不一致'
    : r.includes('tile_overlaps_full') ? '切片检测框与全图检测框存在重叠，目标一致性尚待复核'
    : r.includes('tile_edge_only') ? '只在切片边缘发现，目标可能不完整'
    : r.includes('teacher_down') || r.includes('verify_unknown') ? '校验模型未能完成检测'
    : r.includes('cross_denied') ? '另一模型未提供符合匹配条件的检测框'
    : r.includes('tile_only_requires') || r.includes('no_verify') ? '切片新增候选尚未通过独立校验'
    : '该检测结果尚待复核'))].join('；')
}
async function readJson(response: Response) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败 (${response.status})`)
  return data
}

function Crop({ url, box, width, height, scale }: { url: string; box: Box; width: number; height: number; scale: number }) {
  const w = Math.min(width, 280 / scale), h = Math.min(height, 180 / scale)
  const x = Math.max(0, Math.min(width - w, box[0] + box[2] / 2 - w / 2))
  const y = Math.max(0, Math.min(height - h, box[1] + box[3] / 2 - h / 2))
  return <svg aria-label="候选目标局部放大图" viewBox={`${x} ${y} ${w} ${h}`} style={{ width: '100%', height: 180, background: '#111827' }}>
    <image href={url} width={width} height={height} />
    <rect x={box[0]} y={box[1]} width={Math.max(0,box[2])} height={Math.max(0,box[3])} fill="none" stroke="#f59e0b" strokeWidth="2" vectorEffect="non-scaling-stroke" strokeDasharray="6 4" />
  </svg>
}

function ReviewCanvas({ url, detail, box, editing, onBox, onReady }: {
  url: string; detail: Detail; box: Box; editing: boolean; onBox: (box: Box) => void; onReady: (ready: boolean) => void
}) {
  const ref = useRef<SVGSVGElement>(null)
  const [zoom, setZoom] = useState(1)
  const [view, setView] = useState({ x: detail.width / 2, y: detail.height / 2, width: detail.width, height: detail.height })
  const gesture = useRef<any>(null)
  useEffect(() => {
    setZoom(1); setView({ x: detail.width / 2, y: detail.height / 2, width: detail.width, height: detail.height })
  }, [url, detail.width, detail.height])
  const vw = view.width / zoom, vh = view.height / zoom
  const point = (event: React.PointerEvent<SVGSVGElement>) => {
    const svg = ref.current!, p = svg.createSVGPoint(); p.x = event.clientX; p.y = event.clientY
    return p.matrixTransform(svg.getScreenCTM()!.inverse())
  }
  const down = (event: React.PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return
    const handle = (event.target as SVGElement).getAttribute('data-handle')
    const p = point(event), matrix = ref.current!.getScreenCTM()!
    gesture.current = { mode: editing && handle ? handle : 'pan', point: p, box: [...box], view: { ...view },
      cx: event.clientX, cy: event.clientY, sx: matrix.a, sy: matrix.d }
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const move = (event: React.PointerEvent<SVGSVGElement>) => {
    const g = gesture.current
    if (!g) return
    if (g.mode === 'pan') {
      setView({ ...g.view, x: g.view.x - (event.clientX - g.cx) / g.sx, y: g.view.y - (event.clientY - g.cy) / g.sy }); return
    }
    const p = point(event), dx = p.x - g.point.x, dy = p.y - g.point.y
    let [x, y, w, h] = g.box as Box
    if (g.mode === 'move') {
      x = Math.max(0, Math.min(detail.width - w, x + dx)); y = Math.max(0, Math.min(detail.height - h, y + dy))
    } else {
      let right = x + w, bottom = y + h
      if (g.mode.includes('w')) x = Math.max(0, Math.min(right - 1, x + dx))
      if (g.mode.includes('e')) right = Math.min(detail.width, Math.max(x + 1, right + dx))
      if (g.mode.includes('n')) y = Math.max(0, Math.min(bottom - 1, y + dy))
      if (g.mode.includes('s')) bottom = Math.min(detail.height, Math.max(y + 1, bottom + dy))
      w = right - x; h = bottom - y
    }
    onBox([x, y, w, h].map(v => Number(v.toFixed(2))) as Box)
  }
  const reset = () => { setZoom(1);setView({ x: detail.width / 2, y: detail.height / 2, width: detail.width, height: detail.height }) }
  const focus = () => { setZoom(1);setView({ x: box[0]+box[2]/2,y:box[1]+box[3]/2,width:Math.max(160,box[2]*6),height:Math.max(120,box[3]*6) }) }
  const handleSize = vw / 80
  return <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 360 }}>
    <div className="flex items-center gap-2 p-2 border-b border-border flex-wrap">
      <Button size="sm" variant="outline" onClick={reset}>全图</Button>
      <Button size="sm" variant="outline" onClick={focus}>定位目标</Button>
      <Button size="sm" variant="outline" aria-label="缩小复核图片" onClick={() => setZoom(v => Math.max(.5,v/1.5))}><ZoomOut className="w-4 h-4" /></Button>
      <Button size="sm" variant="outline" aria-label="放大复核图片" onClick={() => setZoom(v => Math.min(8,v*1.5))}><ZoomIn className="w-4 h-4" /></Button>
      <span className="text-xs text-muted-foreground">拖动空白处平移 · {editing ? '拖动橙色框或四角修框' : '橙色：候选；绿色：正式标签'}</span>
    </div>
    <svg ref={ref} aria-label="人工复核图片画布" viewBox={`${view.x-vw/2} ${view.y-vh/2} ${vw} ${vh}`}
      style={{ width:'100%',flex:1,minHeight:300,background:'#111827',touchAction:'none',cursor:gesture.current ? 'grabbing' : 'grab' }}
      onPointerDown={down} onPointerMove={move} onPointerUp={() => {gesture.current=null}} onPointerCancel={() => {gesture.current=null}}
      onLostPointerCapture={() => {gesture.current=null}}>
      <image href={url} width={detail.width} height={detail.height} onLoad={() => onReady(true)} onError={() => onReady(false)} />
      {detail.existing_annotations.map((a,i) => <rect key={i} x={a.bbox[0]} y={a.bbox[1]} width={Math.max(0,a.bbox[2])} height={Math.max(0,a.bbox[3])} fill="none" stroke="#22c55e" strokeWidth="1.5" vectorEffect="non-scaling-stroke" pointerEvents="none" />)}
      <rect data-handle="move" x={box[0]} y={box[1]} width={Math.max(0,box[2])} height={Math.max(0,box[3])}
        fill="transparent" stroke="#f59e0b" strokeWidth="2.5" strokeDasharray="6 4" vectorEffect="non-scaling-stroke"
        style={{ cursor: editing ? 'move' : 'grab' }} />
      {editing && [['nw',box[0],box[1]],['ne',box[0]+box[2],box[1]],['sw',box[0],box[1]+box[3]],['se',box[0]+box[2],box[1]+box[3]]].map(([key,x,y]) =>
        <rect key={key} data-handle={key} x={Number(x)-handleSize/2} y={Number(y)-handleSize/2} width={handleSize} height={handleSize} fill="#f59e0b" stroke="#fff" vectorEffect="non-scaling-stroke" style={{cursor:`${key}-resize`}} />)}
    </svg>
  </div>
}

export function ReviewWorkspace({ apiUrl, datasetId, initialJobId, initialImageKey, onBack, onChanged }: Props) {
  const [jobs,setJobs] = useState<Job[]>([]), [jobId,setJobId] = useState('')
  const [queue,setQueue] = useState<Queue|null>(null), [selectedId,setSelectedId] = useState('')
  const [filter,setFilter] = useState<'pending'|'accepted'|'rejected'|'all'>('pending')
  const [archived,setArchived] = useState(false), [loading,setLoading] = useState(true), [saving,setSaving] = useState(false)
  const [detail,setDetail] = useState<Detail|null>(null), [error,setError] = useState(''), [editing,setEditing] = useState(false)
  const [box,setBox] = useState<Box>([0,0,1,1]), [label,setLabel] = useState(''), [cropScale,setCropScale] = useState(4)
  const [ready,setReady] = useState(false)
  const candidate = queue?.candidates.find(c => c.candidate_id===selectedId)
  const visible = queue?.candidates.filter(c => filter==='all' || c.review_status===filter) || []
  const position = visible.findIndex(c => c.candidate_id===selectedId)
  const sequence = useRef(0)
  const refreshIndex = useCallback(async () => {
    const data = await readJson(await fetch(`${apiUrl}/api/reviews/jobs`));setJobs(data.jobs || []);return data.jobs as Job[]
  },[apiUrl])
  useEffect(() => {
    let stopped=false
    refreshIndex().then(list => {
      if (stopped) return
      let saved='';try {saved=localStorage.getItem(`runos_review_job_${datasetId||'all'}`)||''}catch{}
      const wanted=list.find(j => j.job_id===(initialJobId||saved))
        || list.find(j => (!datasetId||j.dataset_id===datasetId) && !j.readonly && j.summary.pending>0)
        || list.find(j => !datasetId||j.dataset_id===datasetId)
      if (wanted) {setArchived(wanted.readonly);setJobId(wanted.job_id)} else setLoading(false)
    }).catch(e => {if(!stopped){setError(e.message);setLoading(false)}})
    return () => {stopped=true;sequence.current++}
  },[apiUrl,datasetId,initialJobId,refreshIndex])
  const loadQueue = useCallback(async (id: string, preferred?: string) => {
    const seq=++sequence.current;setLoading(true);setError('')
    try {
      const data: Queue=await readJson(await fetch(`${apiUrl}/api/reviews/jobs/${id}`))
      if (seq!==sequence.current) return
      setQueue(data)
      let saved='';try{saved=localStorage.getItem(`runos_review_cursor_${id}`)||''}catch{}
      const chosen=data.candidates.find(c => c.candidate_id===preferred)
        || data.candidates.find(c => c.image_key===initialImageKey && c.review_status==='pending')
        || data.candidates.find(c => c.candidate_id===saved && c.review_status==='pending')
        || data.candidates.find(c => c.review_status==='pending')
      setSelectedId(chosen?.candidate_id||'');setFilter('pending')
    } catch(e) {if(seq===sequence.current){setQueue(null);setError(e instanceof Error?e.message:String(e))}}
    finally {if(seq===sequence.current)setLoading(false)}
  },[apiUrl,initialImageKey])
  useEffect(() => {
    if(!jobId)return
    setQueue(null);setDetail(null);loadQueue(jobId)
    try{localStorage.setItem(`runos_review_job_${datasetId||'all'}`,jobId)}catch{}
  },[jobId,loadQueue,datasetId])
  useEffect(() => {
    if(!candidate||!queue){setDetail(null);return}
    const controller=new AbortController();setDetail(null);setReady(false);setEditing(false);setError('')
    setBox([...candidate.bbox] as Box);setLabel(candidate.class_name)
    try{localStorage.setItem(`runos_review_cursor_${queue.job_id}`,candidate.candidate_id)}catch{}
    fetch(`${apiUrl}/api/reviews/jobs/${queue.job_id}/candidates/${candidate.candidate_id}`,{signal:controller.signal})
      .then(readJson).then(setDetail).catch(e => {if(e.name!=='AbortError')setError(e.message)})
    return () => controller.abort()
  },[candidate?.candidate_id,queue?.job_id,apiUrl])
  const navigate = (step:number) => {
    if(saving||!visible.length)return
    const index=position<0?0:(position+step+visible.length)%visible.length;setSelectedId(visible[index].candidate_id)
  }
  const changeFilter = (value:typeof filter) => {
    if(saving)return
    setFilter(value);const items=queue?.candidates.filter(c=>value==='all'||c.review_status===value)||[]
    setSelectedId(items.some(c=>c.candidate_id===selectedId)?selectedId:items[0]?.candidate_id||'')
  }
  const decide = async (action:'accept'|'reject') => {
    if(!candidate||!queue||queue.readonly||saving)return
    setSaving(true);setError('')
    const next=visible.find((c,i)=>i>position&&c.candidate_id!==candidate.candidate_id)
      ||visible.find(c=>c.candidate_id!==candidate.candidate_id)
    try {
      const result=await readJson(await fetch(`${apiUrl}/api/reviews/jobs/${queue.job_id}/candidates/${candidate.candidate_id}/decision`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,revision:queue.revision,...(action==='accept'?{bbox:box,class_name:label}:{})})}))
      const fresh:Queue=await readJson(await fetch(`${apiUrl}/api/reviews/jobs/${queue.job_id}`))
      setQueue(fresh);setSelectedId(next?.candidate_id||'');setEditing(false)
      await refreshIndex();onChanged()
    } catch(e) {setError(e instanceof Error?e.message:String(e))}
    finally {setSaving(false)}
  }
  const readonly=queue?.readonly||candidate?.review_status!=='pending'
  const selectedJobs=jobs.filter(j=>archived||!j.readonly||j.job_id===jobId)
  const pending=jobs.filter(j=>!j.readonly&&(!datasetId||j.dataset_id===datasetId)).reduce((s,j)=>s+j.summary.pending,0)
  const imageUrl=detail?`${apiUrl}${detail.image_url}`:''
  return <div style={{height:'100%',display:'flex',flexDirection:'column',minHeight:0}}>
    <style>{`@media(max-width:1000px){.runos-review-layout{grid-template-columns:180px minmax(0,1fr)!important;overflow:auto!important}.runos-review-detail{grid-column:1/-1;max-height:380px;overflow:auto}}`}</style>
    <div className="flex items-center gap-2 p-2 border-b border-border bg-card flex-wrap">
      <Button size="sm" variant="outline" disabled={saving} onClick={()=>onBack('annotate')}>标注</Button>
      <Button size="sm" variant="outline" disabled={saving} onClick={()=>onBack('jobs')}>任务</Button>
      <Button size="sm" aria-current="page">待复核 {pending}</Button>
      <span className="text-xs font-medium">{queue?.dataset_name||'自动标注复核'}</span>
      <Button size="sm" variant="outline" style={{marginLeft:'auto'}} disabled={saving||loading} onClick={()=>{refreshIndex().catch(e=>setError(e.message));if(jobId)loadQueue(jobId,selectedId)}}><RotateCcw className="w-3 h-3 mr-1"/>刷新进度</Button>
    </div>
    <div className="runos-review-layout" style={{display:'grid',gridTemplateColumns:'210px minmax(0,1fr) 310px',flex:1,minHeight:0,overflow:'hidden'}}>
      <aside className="border-r border-border bg-card" style={{display:'flex',flexDirection:'column',minHeight:0}}>
        <div className="p-3 space-y-2 border-b border-border">
          <label className="text-xs font-medium" htmlFor="review-job">复核任务</label>
          <select id="review-job" value={jobId} disabled={saving} onChange={e=>setJobId(e.target.value)} className="bg-background border border-border rounded text-xs" style={{width:'100%',padding:6}}>
            <option value="" disabled>选择任务</option>
            {selectedJobs.map(j=><option key={j.job_id} value={j.job_id}>{j.dataset_name} · {j.kind==='preview'?'单图':'批量'} {j.job_id.slice(0,8)} · {j.readonly?'只读':`待复核 ${j.summary.pending}`}</option>)}
          </select>
          <label className="flex items-center gap-2 text-xs text-muted-foreground"><input type="checkbox" checked={archived} disabled={saving} onChange={e=>setArchived(e.target.checked)}/>显示已撤销／只读任务</label>
          <div className="flex gap-1 flex-wrap">{(['pending','accepted','rejected','all'] as const).map(s=><button key={s} disabled={saving} onClick={()=>changeFilter(s)} className={cn('rounded border px-2 py-1 text-xs',filter===s?'text-primary border-primary/30 bg-primary/15':'border-border text-muted-foreground')}>{s==='all'?'全部':STATUS[s]}</button>)}</div>
        </div>
        <div style={{flex:1,overflowY:'auto'}} className="p-2 space-y-1">
          {visible.map((c,i)=><button key={c.candidate_id} disabled={saving} onClick={()=>setSelectedId(c.candidate_id)} className={cn('rounded border p-2 text-xs text-left',c.candidate_id===selectedId?'border-primary/30 bg-primary/15':'border-border hover:bg-muted/30')} style={{display:'block',width:'100%'}}>
            <span className="block truncate">{i+1}. {c.image.split('/').pop()}</span><span className="text-muted-foreground">{c.class_name} · {STATUS[c.review_status]}</span>
          </button>)}
          {!loading&&!visible.length&&<p className="text-xs text-muted-foreground p-2">{queue?'此分类没有候选框':'暂无复核任务'}</p>}
        </div>
      </aside>
      <main style={{display:'flex',flexDirection:'column',minHeight:0,minWidth:0}}>
        {loading?<div className="flex-1 flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-primary"/></div>
          :candidate&&detail?<ReviewCanvas key={`${jobId}-${candidate.candidate_id}`} url={imageUrl} detail={detail} box={box} editing={editing&&!readonly} onBox={setBox} onReady={setReady}/>
          :candidate?<div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">{error||'正在加载复核图片…'}</div>
          :<div className="flex-1 flex flex-col items-center justify-center gap-3 text-muted-foreground"><CheckCircle2 className="w-10 h-10"/><p>{queue?.summary.pending===0?'本任务的候选已全部处理':'请选择复核任务或候选框'}</p><p className="text-xs">确认后才写入正式标签；进度保存在服务器</p></div>}
        <div className="flex items-center gap-2 p-2 border-t border-border">
          <Button size="sm" variant="outline" disabled={saving||visible.length<2} onClick={()=>navigate(-1)}><ChevronLeft className="w-4 h-4"/>上一候选</Button>
          <span className="text-xs text-muted-foreground">{position<0?0:position+1} / {visible.length}</span>
          <Button size="sm" variant="outline" disabled={saving||visible.length<2} onClick={()=>navigate(1)}>下一候选<ChevronRight className="w-4 h-4"/></Button>
        </div>
      </main>
      <aside className="runos-review-detail border-l border-border bg-card p-3 space-y-3" style={{overflowY:'auto'}}>
        <div className="flex flex-wrap gap-3 text-xs"><span className="text-primary">待复核 {queue?.summary.pending||0}</span><span>已确认 {queue?.summary.accepted||0}</span><span>已驳回 {queue?.summary.rejected||0}</span></div>
        {queue?.readonly&&<p role="status" className="border border-border rounded p-2 text-xs text-muted-foreground">{queue.readonly_reason}</p>}
        {error&&<div role="alert" className="border border-border rounded p-2 text-xs text-destructive">{error}</div>}
        {candidate&&<>
          <p className="text-sm font-medium">{candidate.class_name} · {STATUS[candidate.review_status]}</p>
          <p className="text-xs text-muted-foreground">{reasons(candidate.decision_reasons||[])}</p>
          <p className="text-xs text-muted-foreground">来源：{(candidate.evidence?.teachers||[]).map(modelName).join('、')||'自动标注模型'}</p>
          {detail&&<><div className="flex items-center gap-2 text-xs"><span>局部放大</span>{[2,4,8].map(v=><button key={v} onClick={()=>setCropScale(v)} className={cn('rounded border px-2 py-1',cropScale===v?'text-primary border-primary/30':'border-border')}>{v}×</button>)}</div><Crop url={imageUrl} box={box} width={detail.width} height={detail.height} scale={cropScale}/></>}
          <div className="flex gap-2">
            <Button size="sm" variant="outline" disabled={!!readonly||saving||!detail} onClick={()=>setEditing(v=>!v)}>{editing?'结束修框':'调整框／类别'}</Button>
            {editing&&<Button size="sm" variant="outline" disabled={saving} onClick={()=>{setBox([...candidate.bbox] as Box);setLabel(candidate.class_name)}}>重置修改</Button>}
          </div>
          {editing&&<div className="space-y-2">
            <label className="text-xs">类别<select value={label} onChange={e=>setLabel(e.target.value)} disabled={saving} className="bg-background border border-border rounded" style={{width:'100%',padding:6}}>{queue?.classes.map(c=><option key={c} value={c}>{c}</option>)}</select></label>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>{['X','Y','宽','高'].map((name,i)=><label key={name} className="text-xs">{name}<Input aria-label={`复核框${name}`} type="number" step="0.1" min={0} value={box[i]} disabled={saving} className="h-8 text-xs" onChange={e=>setBox(prev=>prev.map((v,j)=>j===i?Number(e.target.value):v) as Box)}/></label>)}</div>
            <p className="text-xs text-muted-foreground">坐标以原图像素为单位，也可拖动橙色框和四角调整。</p>
          </div>}
          <div className="space-y-2">
            <Button style={{width:'100%'}} disabled={!!readonly||saving||!detail||!ready} onClick={()=>decide('accept')}>{saving?<Loader2 className="w-4 h-4 animate-spin mr-1"/>:<CheckCircle2 className="w-4 h-4 mr-1"/>}{editing?'调整后确认并保存':'确认目标并保存'}</Button>
            <div className="flex gap-2"><Button variant="outline" style={{flex:1}} disabled={!!readonly||saving||!detail} onClick={()=>decide('reject')}>驳回误检</Button><Button variant="outline" style={{flex:1}} disabled={saving||visible.length<2} onClick={()=>navigate(1)}>暂时跳过</Button></div>
            {!readonly&&<p className="text-xs text-muted-foreground">确认：合并到正式标签；驳回：不写标签；跳过：保持待复核。</p>}
          </div>
        </>}
      </aside>
    </div>
  </div>
}
