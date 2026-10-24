'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

type Draft = { id:string; kind:'annotation'|'training'; revision:number; status:string; data:Record<string,any>; job?:Record<string,any>; offer_training?:boolean }

export function AssistantDraftReview({apiUrl}:{apiUrl:string}) {
  const [id,setId]=useState(''),[draft,setDraft]=useState<Draft|null>(null),[text,setText]=useState('')
  const [notice,setNotice]=useState(''),[busy,setBusy]=useState(false),[catalog,setCatalog]=useState<any>(null)
  const [architecture,setArchitecture]=useState('yolov8n'),[weight,setWeight]=useState('builtin')
  useEffect(()=>{
    const value=new URLSearchParams(location.search).get('assistant_draft')||''
    if (/^[a-f0-9]{32}$/.test(value))setId(value)
  },[])
  async function call(path:string,method='GET',body?:unknown){
    const r=await fetch(`${apiUrl}/api/assistant/${path}`,{method,headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),cache:'no-store'})
    const data=await r.json()
    if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:data.detail?.message||'操作未完成，请检查参数。')
    return data
  }
  useEffect(()=>{
    if(!id)return
    let live=true
    const load=async()=>{try{const v=await call(`drafts/${id}`);if(live){setDraft(v);if(v.status==='prepared')setText(JSON.stringify(v.data,null,2))}}catch(e){if(live)setNotice((e as Error).message)}}
    void load();void call('catalog').then(v=>{if(live)setCatalog(v)}).catch(()=>{})
    const timer=setInterval(()=>{if(!document.hidden)void call(`drafts/${id}`).then(v=>{if(live)setDraft(v)}).catch(()=>{})},5000)
    return()=>{live=false;clearInterval(timer)}
  },[id,apiUrl]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(()=>{
    const ds=catalog?.datasets?.find((d:any)=>d.id===draft?.data.dataset_id)
    const domain=ds?.task_type==='classification'?'classification':ds?.task_type==='segmentation'?'segmentation':'detection'
    const choice=catalog?.training_architectures?.find((a:any)=>a.taskType===domain)
    if(choice)setArchitecture(choice.value)
  },[catalog,draft?.data.dataset_id])
  if(!id||!draft||draft.kind==='training')return null
  const ds=catalog?.datasets?.find((d:any)=>d.id===draft.data.dataset_id)
  async function execute(){
    if(!draft)return;setBusy(true);setNotice('')
    try{
      const data=JSON.parse(text)
      const updated=await call(`drafts/${id}`,'PUT',{kind:'annotation',data,revision:draft.revision})
      setDraft(updated)
      setDraft(await call(`drafts/${id}/execute`,'POST',{revision:updated.revision}))
      setNotice('标注任务已创建。已有正式标签按原规则保留。')
    }catch(e){setNotice((e as Error).message)}finally{setBusy(false)}
  }
  async function prepareTraining(){
    setBusy(true);setNotice('')
    try{
      const arch=catalog.training_architectures.find((a:any)=>a.value===architecture)
      const type=architecture.startsWith('rtdetrv2')?'rtdetrv2':architecture.startsWith('rtdetr')?'rtdetr':architecture.startsWith('rfdetr')?'rfdetr':arch.taskType==='classification'?'classification':arch.taskType==='segmentation'?'segmentation':'yolo'
      const result=await call('drafts','POST',{kind:'training',data:{dataset_id:draft!.data.dataset_id,name:`${ds?.name||'dataset'}_${new Date().toISOString().slice(0,10)}`,model_type:type,model_arch:architecture,initial_weights:weight==='builtin'?'builtin':'custom',custom_weights:weight==='builtin'?null:weight,epochs:80,batch_size:1,image_size:640,lr0:.001,optimizer:'AdamW',amp:false,cos_lr:true,save_period:10}})
      location.assign(`/training?assistant_draft=${result.id}`)
    }catch(e){setNotice((e as Error).message)}finally{setBusy(false)}
  }
  const setField=(key:string,value:unknown)=>{try{setText(JSON.stringify({...JSON.parse(text),[key]:value},null,2))}catch{setNotice('请先检查高级参数中的 JSON 格式。')}}
  const job=draft.job
  const domain=ds?.task_type==='classification'?'classification':ds?.task_type==='segmentation'?'segmentation':'detection'
  return <Card className="mx-4 my-4 border-primary/30">
    <CardHeader><CardTitle className="text-base">平台助手 · 标注参数核对</CardTitle></CardHeader>
    <CardContent className="space-y-3 text-sm">
      <p>数据集：{ds?.name||draft.data.dataset_id}　·　状态：{job?.status||draft.status}</p>
      {draft.status==='prepared'&&<>
        <div className="grid gap-3 md:grid-cols-3">
          <label>标注模型<select className="mt-1 w-full rounded border bg-background p-2" value={(()=>{try{return JSON.parse(text).model_id}catch{return draft.data.model_id}})()} onChange={e=>setField('model_id',e.target.value)}>{catalog?.annotation_models?.map((m:any)=><option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
          <label>目标类别<select className="mt-1 w-full rounded border bg-background p-2" value={(()=>{try{return JSON.parse(text).label}catch{return draft.data.label}})()} onChange={e=>setField('label',e.target.value)}>{ds?.classes?.map((c:string)=><option key={c}>{c}</option>)}</select></label>
          <label>复核模型<select className="mt-1 w-full rounded border bg-background p-2" value={(()=>{try{return JSON.parse(text).verify_model_id||''}catch{return ''}})()} onChange={e=>setField('verify_model_id',e.target.value||null)}><option value="">按当前方案，不增加复核模型</option>{catalog?.annotation_models?.map((m:any)=><option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
        </div>
        <details><summary className="cursor-pointer">全部参数（可编辑）</summary><textarea aria-label="标注参数" value={text} onChange={e=>setText(e.target.value)} className="mt-2 h-64 w-full rounded border bg-background p-3 font-mono text-xs"/></details>
        <p className="text-muted-foreground">已有正式标签按原规则保留。待复核候选不会自动作为正式训练标签。</p>
        <Button disabled={busy} onClick={execute}>{busy?'正在提交…':'确认并执行标注'}</Button>
      </>}
      {job&&<p>进度：{job.processed||0}/{job.total||0}　标注：{job.annotated||0}　失败：{job.failed||0}　待复核：{job.review_summary?.pending||0}</p>}
      {job&&['running','paused'].includes(job.status)&&<div className="flex gap-2">{(['pause','resume','cancel'] as const).map(a=><Button key={a} variant="outline" disabled={busy||(a==='resume'?job.status!=='paused':a==='pause'&&job.status!=='running')} onClick={async()=>{setBusy(true);try{setDraft(await call(`drafts/${id}/control/${a}`,'POST',{}))}catch(e){setNotice((e as Error).message)}finally{setBusy(false)}}}>{({pause:'暂停',resume:'恢复',cancel:'停止'})[a]}</Button>)}</div>}
      {draft.offer_training&&catalog&&<div className="space-y-3 rounded border p-3">
        <p className="font-medium">标注已完成，是否准备训练？</p>
        <p className="text-muted-foreground">下方方案用于准备训练表单，不会立即启动训练。待复核候选仍需在标注页复核。</p>
        <div className="grid gap-3 md:grid-cols-2">
          <label>训练架构<select className="mt-1 w-full rounded border bg-background p-2" value={architecture} onChange={e=>setArchitecture(e.target.value)}>{catalog.training_architectures.filter((a:any)=>a.taskType===domain).map((a:any)=><option key={a.value} value={a.value}>{a.label}</option>)}</select></label>
          <label>初始权重<select className="mt-1 w-full rounded border bg-background p-2" value={weight} onChange={e=>setWeight(e.target.value)}><option value="builtin">内置预训练权重</option>{catalog.initial_weights.map((w:any)=><option key={w.name} value={w.name}>{w.name}</option>)}</select></label>
        </div>
        <div className="flex gap-2"><Button disabled={busy} onClick={prepareTraining}>准备训练并打开表单</Button><Button variant="outline" onClick={async()=>setDraft(await call(`drafts/${id}/control/skip_training`,'POST',{}))}>暂不训练</Button></div>
      </div>}
      {notice&&<p role="status">{notice}</p>}
    </CardContent>
  </Card>
}
