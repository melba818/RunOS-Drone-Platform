'use client'
import {useCallback,useEffect,useRef,useState} from 'react'
import {validNavigation} from './native-workflow.mjs'
import type {Dataset,ViewType} from '@/app/page'
export type Workflow={id:string;revision:number;stage:string;source_dataset_id:string;target_dataset_id?:string;annotation_draft:string;training_draft?:string;source_dataset?:Dataset;target_dataset?:Dataset;annotation_job?:any;training_status?:string;training_id?:string;training_job?:any;split?:any;review?:any}
export async function assistantCall(apiUrl:string,path:string,method='GET',data?:unknown){
  const response=await fetch(`${apiUrl}/api/assistant/${path}`,{method,headers:{'Content-Type':'application/json'},body:data===undefined?undefined:JSON.stringify(data),cache:'no-store'})
  const value=await response.json()
  if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:value.detail?.message||value.message||'操作未完成，请检查当前步骤的参数。')
  return value
}
export function useAssistantWorkflow(apiUrl:string,select:(ds:Dataset)=>void,navigate:(view:ViewType)=>void,beforeNavigate:()=>Promise<void>){
  const [flow,setFlow]=useState<Workflow|null>(null),[draftId,setDraftId]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false)
  const [resourceGuide,setResourceGuide]=useState<{view:'datasets'|'classes';datasetId?:string}|null>(null)
  const flowRef=useRef<Workflow|null>(null),identity=useRef(0),channel=useRef(''),parentOrigin=useRef(''),callbacks=useRef({select,navigate,beforeNavigate})
  callbacks.current={select,navigate,beforeNavigate};flowRef.current=flow
  const storage=(kind:string)=>`runos-assistant:${identity.current}:${kind}`
  const notify=useCallback((type:string,data:Record<string,unknown>={})=>{if(window.parent!==window&&parentOrigin.current&&channel.current)window.parent.postMessage({source:'runos-native-assistant',type,channel:channel.current,admin_id:identity.current,...data},parentOrigin.current)},[])
  const refresh=useCallback(async()=>{const id=flowRef.current?.id;if(!id)return;const v=await assistantCall(apiUrl,`flows/${id}`);if(flowRef.current?.id!==id)return;flowRef.current=v;setFlow(v);notify('workflow-state',{flow:v,view:location.pathname.slice(1)});return v},[apiUrl,notify])
  const activate=useCallback(async(id:string,view:ViewType='annotate',requestedDraft?:string)=>{
    const v:Workflow=await assistantCall(apiUrl,`flows/${id}`)
    const dataset=view==='training'&&v.target_dataset_id?v.target_dataset:v.source_dataset
    if(!dataset?.name)throw Error('流程数据集不可用，请重新选择。')
    const draft=view==='training'?(requestedDraft||v.training_draft||''):v.annotation_draft
    await callbacks.current.beforeNavigate()
    const params=new URLSearchParams(location.search);params.set('assistant_flow',id);if(draft)params.set('assistant_draft',draft);else params.delete('assistant_draft');history.replaceState(null,'',`/${view}?${params}`)
    const catalog=await fetch(`${apiUrl}/api/datasets`,{cache:'no-store'}).then(r=>r.ok?r.json():null)
    callbacks.current.select(catalog?.datasets?.find((d:Dataset)=>d.id===dataset.id)||dataset);flowRef.current=v;setFlow(v);setDraftId(draft);localStorage.setItem(storage('flow'),id);localStorage.setItem(storage('view'),view);callbacks.current.navigate(view)
    notify('workflow-state',{flow:v,view});return v
  },[apiUrl,notify])
  const standaloneTraining=useCallback(async(id:string)=>{
    await callbacks.current.beforeNavigate()
    flowRef.current=null;setFlow(null);setDraftId(id);localStorage.removeItem(storage('flow'));localStorage.removeItem(storage('view'))
    const params=new URLSearchParams(location.search);params.delete('assistant_flow');params.set('assistant_draft',id);history.replaceState(null,'',`/training?${params}`);callbacks.current.navigate('training');notify('workflow-closed')
  },[notify])
  const adoptDraft=useCallback(async(id:string)=>{await activate(id,'annotate',id)},[activate])
  useEffect(()=>{
    let live=true
    const listen=async(e:MessageEvent)=>{
      if(e.source!==window.parent||e.origin!==parentOrigin.current)return
      if(e.data?.source==='runhive-platform-assistant'&&e.data.type==='native-ping'){notify('native-ready');return}
      if(!validNavigation(e.data,channel.current,identity.current))return
      const d=e.data
      try{
        if(['datasets','classes'].includes(d.view)){
          await callbacks.current.beforeNavigate()
          if(d.dataset_id){const r=await fetch(`${apiUrl}/api/datasets`,{cache:'no-store'});if(!r.ok)throw Error('数据集列表暂不可用。');const all=await r.json(),dataset=all.datasets?.find((ds:Dataset)=>ds.id===d.dataset_id);if(!dataset)throw Error('数据集已不可用，请重新选择。');callbacks.current.select(dataset)}
          setResourceGuide({view:d.view,datasetId:d.dataset_id});const params=new URLSearchParams();params.set('assistant_resource',d.view);history.replaceState(null,'',`/${d.view}?${params}`);callbacks.current.navigate(d.view)
        }else if(d.flow_id){setResourceGuide(null);await activate(d.flow_id,d.view,d.draft_id)}
        else if(d.draft_id){const draft=await assistantCall(apiUrl,`drafts/${d.draft_id}`);if(draft.kind==='annotation')await activate(d.draft_id,'annotate',d.draft_id);else if(draft.kind==='training')await standaloneTraining(d.draft_id);else throw Error('任务类型无效。')}
        notify('native-navigated',{request_id:d.request_id,view:d.view,flow:flowRef.current})
      }catch(err){setError((err as Error).message);notify('native-navigation-error',{request_id:d.request_id,message:(err as Error).message})}
    }
    void fetch('/api/runos-auth/identity',{cache:'no-store'}).then(async r=>{
      if(!r.ok)throw Error('应用登录已失效。');const me=await r.json();if(!live)return
      identity.current=Number(me.admin_id);channel.current=crypto.randomUUID().replaceAll('-','');parentOrigin.current=document.referrer?new URL(document.referrer).origin:location.origin
      window.addEventListener('message',listen);notify('native-ready')
      const params=new URLSearchParams(location.search),id=params.get('assistant_flow')||params.get('assistant_draft')||localStorage.getItem(storage('flow'))||''
      if(/^[a-f0-9]{32}$/.test(id)){
        const candidate=params.get('assistant_flow')?id:params.get('assistant_draft')?id:localStorage.getItem(storage('flow'))||''
        const savedView=localStorage.getItem(storage('view')) as ViewType
        const view=params.has('assistant_flow')||params.has('assistant_draft')?(location.pathname==='/training'?'training':location.pathname==='/split'?'split':'annotate'):['annotate','split','training'].includes(savedView)?savedView:'annotate'
        try{if(params.has('assistant_draft')&&!params.has('assistant_flow')){const draft=await assistantCall(apiUrl,`drafts/${id}`);if(draft.kind==='training')await standaloneTraining(id);else await activate(id,view,id)}else await activate(candidate,view,params.get('assistant_draft')||undefined)}catch(err){if(live)setError((err as Error).message)}
      }
    }).catch(err=>{if(live)setError(err.message)})
    return()=>{live=false;window.removeEventListener('message',listen)}
  },[apiUrl,activate,notify,standaloneTraining])
  useEffect(()=>{if(!flow)return;const timer=setInterval(()=>{if(!document.hidden)void refresh().catch(()=>setError('暂时无法更新任务状态，请重新连接后继续。'))},4000);return()=>clearInterval(timer)},[flow?.id,refresh])
  const move=async(view:ViewType)=>{
    const v=flowRef.current;if(!v)return
    try{setError('');await activate(v.id,view)}catch(err){setError((err as Error).message);throw err}
  }
  const dismiss=()=>{flowRef.current=null;setFlow(null);setDraftId('');setError('');localStorage.removeItem(storage('flow'));localStorage.removeItem(storage('view'));const params=new URLSearchParams(location.search);params.delete('assistant_flow');params.delete('assistant_draft');history.replaceState(null,'',location.pathname+(params.toString()?'?'+params:''));notify('workflow-closed')}
  return {flow,setFlow,draftId,error,setError,busy,setBusy,refresh,activate,adoptDraft,move,notify,dismiss,resourceGuide,setResourceGuide}
}
export type WorkflowController=ReturnType<typeof useAssistantWorkflow>
