export function splitCounts(total,train,val,test){
  const ratios=[train,val,test],sum=ratios.reduce((a,b)=>a+b,0)
  if(!Number.isInteger(total)||total<0||ratios.some(r=>!Number.isFinite(r)||r<0||r>1)||Math.abs(sum-1)>1e-6)throw Error('划分比例合计必须为 100%。')
  const raw=ratios.map(r=>total*r/sum),counts=raw.map(Math.floor)
  const order=[0,1,2].sort((a,b)=>(raw[b]-counts[b])-(raw[a]-counts[a])||a-b)
  for(const i of order.slice(0,total-counts.reduce((a,b)=>a+b,0)))counts[i]++
  return {train:counts[0],val:counts[1],test:counts[2]}
}
export function validNavigation(data,channel,adminId){
  if(data?.source!=='runhive-platform-assistant'||data?.type!=='native-nav'||data.channel!==channel||data.admin_id!==adminId||!/^[a-f0-9]{32}$/.test(data.request_id||'')||!['annotate','split','training','datasets','classes'].includes(data.view))return false
  if(['datasets','classes'].includes(data.view))return !data.draft_id&&!data.flow_id&&(!data.dataset_id||typeof data.dataset_id==='string'&&data.dataset_id.length<=240&&!/[\\/\u0000-\u001f]/u.test(data.dataset_id))&&(data.view!=='classes'||!!data.dataset_id)
  return (!data.draft_id||/^[a-f0-9]{32}$/.test(data.draft_id))&&(!data.flow_id||/^[a-f0-9]{32}$/.test(data.flow_id))
}
