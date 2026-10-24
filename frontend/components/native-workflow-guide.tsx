'use client'
import {useState} from 'react'
import {Button} from '@/components/ui/button'
import {assistantCall,type WorkflowController} from '@/lib/assistant-workflow'
import type {ViewType} from '@/app/page'
export function NativeWorkflowGuide({controller,view,apiUrl,beforeLeave,onAnnotationAction,selectedDatasetId}:{controller:WorkflowController;view:ViewType;apiUrl:string;beforeLeave:()=>Promise<void>;onAnnotationAction:(action:'jobs'|'review'|'annotate')=>void;selectedDatasetId?:string}){
  const {flow,busy,setBusy,setError,error,move,refresh,notify}=controller
  const [excludePending,setExcludePending]=useState(false),[acceptPartial,setAcceptPartial]=useState(false)
  if(!flow)return null
  const job=flow.annotation_job||{},pending=job.review_summary?.pending||0
  const review=['review_required','split_ready'].includes(flow.stage)&&view==='annotate'
  const step=flow.stage.startsWith('annotation')?0:flow.stage==='review_required'?1:flow.stage.startsWith('split')?2:3
  const titles=['批量标注','检查效果','数据划分','模型训练']
  async function confirmReview(){
    setBusy(true);setError('')
    try{await beforeLeave();const latest=await refresh();const value=await assistantCall(apiUrl,`flows/${flow!.id}/review`,'POST',{revision:latest!.revision,confirmed:true,exclude_pending:excludePending,accept_partial:acceptPartial});controller.setFlow(value);await controller.activate(value.id,'split');setExcludePending(false);setAcceptPartial(false)}catch(err){setError((err as Error).message)}finally{setBusy(false)}
  }
  async function reopen(){
    setBusy(true);setError('')
    try{await beforeLeave();const latest=await refresh();const value=await assistantCall(apiUrl,`flows/${flow!.id}/reopen-review`,'POST',{revision:latest!.revision});controller.setFlow(value);await controller.activate(value.id,'annotate')}catch(err){setError((err as Error).message)}finally{setBusy(false)}
  }
  async function go(view:ViewType){try{await move(view)}catch{}}
  async function chooseTraining(){setBusy(true);try{await move('training');notify('workflow-ask-training',{flow_id:flow!.id,dataset_id:flow!.target_dataset_id})}catch{}finally{setBusy(false)}}
  if(view==='annotate'&&selectedDatasetId!==flow.source_dataset_id)return <section aria-label="助手流程引导" className="border-b bg-primary/5 px-4 py-3 text-sm"><span>当前选择的数据集与流程不一致。</span><Button className="ml-2" size="sm" onClick={()=>void go('annotate')}>返回流程数据集</Button><Button className="ml-2" size="sm" variant="ghost" onClick={controller.dismiss}>暂停引导</Button>{error&&<p role="alert" className="text-destructive">{error}</p>}</section>
  return <section aria-label="助手流程引导" className="border-b border-primary/20 bg-primary/5 px-4 py-3 text-sm">
    <div className="flex flex-wrap items-center justify-between gap-2"><div className="flex flex-wrap items-center gap-3">{titles.map((title,i)=><span key={title} className={i===step?'font-semibold text-primary':'text-muted-foreground'}>{i+1}. {title}</span>)}</div><div className="flex items-center gap-2"><span className="text-xs text-muted-foreground">{flow.target_dataset?.name||flow.source_dataset?.name}</span><Button variant="ghost" size="sm" onClick={controller.dismiss}>暂停引导</Button></div></div>
    <div className="mt-2 flex flex-wrap items-center gap-2">
      {flow.stage==='annotation_ready'&&<><span className="text-muted-foreground">模型、类别和阈值已填入。核对右侧参数后点击“批量标注数据集”。</span><Button size="sm" variant="outline" onClick={()=>onAnnotationAction('annotate')}>查看标注参数</Button></>}
      {flow.stage==='annotation_running'&&<><span>标注进度 {job.processed||0}/{job.total||0}，完成后留在当前页检查效果。</span><Button size="sm" variant="outline" onClick={()=>onAnnotationAction('jobs')}>查看批量任务</Button></>}
      {flow.stage==='annotation_problem'&&<><span>标注尚未完成，请在任务面板检查或调整参数重试。</span><Button size="sm" variant="outline" onClick={()=>onAnnotationAction('jobs')}>查看任务</Button><Button size="sm" variant="outline" onClick={()=>onAnnotationAction('annotate')}>调整标注参数</Button></>}
      {review&&<><span>请检查图片上的框和类别。成功 {job.annotated||0} 张，失败 {job.failed||0} 张，待复核 {pending} 条。</span>{pending>0&&<><Button size="sm" variant="outline" onClick={()=>onAnnotationAction('review')}>处理待复核</Button><label className="flex items-center gap-1 text-xs"><input type="checkbox" checked={excludePending} onChange={e=>setExcludePending(e.target.checked)}/>仅使用正式标签，排除待复核候选</label></>}{job.failed>0&&<label className="flex items-center gap-1 text-xs"><input type="checkbox" checked={acceptPartial} onChange={e=>setAcceptPartial(e.target.checked)}/>接受当前部分结果</label>}<Button size="sm" disabled={busy} onClick={confirmReview}>{busy?'检查中…':flow.stage==='split_ready'?'重新确认效果，进入划分':'效果已确认，设置划分'}</Button></>}
      {flow.stage==='review_required'&&view!=='annotate'&&<Button size="sm" onClick={()=>void go('annotate')}>返回标注页检查效果</Button>}
      {flow.stage==='split_ready'&&view!=='annotate'&&<><span className="text-muted-foreground">划分会创建新数据集，保留源图片和标注。核对各组数量后点击“创建划分”。</span><Button size="sm" variant="outline" onClick={()=>void go('annotate')}>返回检查标注</Button>{view!=='split'&&<Button size="sm" onClick={()=>void go('split')}>打开数据划分</Button>}</>}
      {['split_done','training_ready'].includes(flow.stage)&&<><span>划分已完成：训练 {flow.split?.counts?.train||0} / 验证 {flow.split?.counts?.val||0} / 测试 {flow.split?.counts?.test||0}。</span><Button size="sm" variant="outline" onClick={()=>void go('split')}>查看划分结果</Button><Button size="sm" disabled={busy} onClick={()=>void chooseTraining()}>{flow.stage==='training_ready'?'重新选择模型与参数':'选择训练模型与参数'}</Button><Button size="sm" variant="outline" disabled={busy} onClick={()=>void reopen()}>重新检查与划分（保留旧结果）</Button></>}
      {flow.stage==='training_running'&&<><span>训练{flow.training_job?.status==='paused'?'已暂停':'进行中'}，第 {flow.training_job?.current_epoch||0}/{flow.training_job?.total_epochs||0} 轮。</span>{view!=='training'&&<Button size="sm" onClick={()=>void go('training')}>查看训练任务</Button>}<span className="text-muted-foreground">暂停、恢复及停止使用训练页原有控件。</span></>}
      {['training_done','training_problem'].includes(flow.stage)&&<><span>{flow.stage==='training_done'?'训练已完成，请检查指标及训练结果。':'训练未完成，请在训练页检查任务状态与日志。'}</span><Button size="sm" variant="outline" onClick={()=>void go('training')}>查看训练结果</Button>{flow.training_job?.status!=='unavailable'&&flow.training_status!=='uncertain'&&<Button size="sm" variant="outline" disabled={busy} onClick={()=>void reopen()}>重新检查与划分（保留旧结果）</Button>}</>}
      {flow.stage==='split_pending'&&<span>正在创建划分，请勿重复提交。</span>}
      {flow.stage==='split_uncertain'&&<span>划分结果需要核查，请先检查数据集列表。</span>}
    </div>
    {error&&<p role="alert" className="mt-2 text-destructive">{error}</p>}
  </section>
}
