const $=selector=>document.querySelector(selector),params=new URLSearchParams(location.search);
const queuePath=params.get("queue"),resultPath=params.get("result"),video=$("#video");
const state={queue:null,rallies:[],pendingStart:null};
const clock=value=>{const m=Math.floor(value/60),s=value-m*60;return `${String(m).padStart(2,"0")}:${s.toFixed(2).padStart(5,"0")}`};
function render(){
  state.rallies.sort((a,b)=>a.start_s-b.start_s);
  $("#count").textContent=`${state.rallies.length} розыгрышей`;
  const pending=state.pendingStart;
  $("#pending").classList.toggle("active",Boolean(pending));
  $("#pending").textContent=pending?`Начало: ${clock(pending.time_s)} (${pending.confidence==="exact"?"видно":"оценка"}). Теперь отметь конец.`:"Жду начало следующего розыгрыша";
  $("#start").disabled=Boolean(pending);$("#startInferred").disabled=Boolean(pending);
  $("#end").disabled=!pending;$("#endInferred").disabled=!pending;
  $("#rallies").innerHTML=state.rallies.map((row,index)=>`<li><b>${clock(row.start_s)}–${clock(row.end_s)}</b> <small>${(row.end_s-row.start_s).toFixed(1)} с · ${row.start_confidence==="exact"?"начало видно":"начало оценено"} · ${row.end_confidence==="exact"?"конец видно":"конец оценён"}</small><br><button data-index="${index}" data-side="start">к началу</button><button data-index="${index}" data-side="end">к концу</button></li>`).join("");
  document.querySelectorAll("#rallies button").forEach(button=>button.onclick=()=>{const row=state.rallies[Number(button.dataset.index)];video.pause();video.currentTime=row[`${button.dataset.side}_s`]});
}
async function save(){
  const payload={auditPath:resultPath,schema_version:1,kind:"volleytag.rally-interval-timeline-results",queue:queuePath,split:state.queue.split,frozen_tests_used:Boolean(state.queue.frozen_tests_used),rallies:state.rallies,pending_start:state.pendingStart,updated_at:new Date().toISOString()};
  $("#saved").textContent="сохраняю…";
  try{const response=await fetch("/api/save-dataset-audit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!response.ok)throw new Error(await response.text());$("#saved").textContent="✓ сохранено"}catch(error){$("#saved").textContent=`ошибка: ${error.message}`}
}
function current(){return Number(video.currentTime.toFixed(3))}
function markStart(confidence){const time=current();if(time<state.queue.start_s||time>state.queue.end_s)return;const previous=state.rallies.at(-1);if(previous&&time<=previous.end_s){$("#saved").textContent="начало должно быть после предыдущего конца";return}state.pendingStart={time_s:time,confidence};render();save()}
function markEnd(confidence){if(!state.pendingStart)return;const time=current();if(time<=state.pendingStart.time_s){$("#saved").textContent="конец должен быть после начала";return}state.rallies.push({id:`rally-${String(state.rallies.length+1).padStart(3,"0")}`,start_s:state.pendingStart.time_s,end_s:time,start_confidence:state.pendingStart.confidence,end_confidence:confidence});state.pendingStart=null;render();save()}
function seek(delta){video.pause();video.currentTime=Math.max(state.queue.start_s,Math.min(state.queue.end_s,video.currentTime+delta))}
video.ontimeupdate=()=>{$("#clock").textContent=clock(video.currentTime);$("#timeline").value=video.currentTime;if(state.queue&&video.currentTime>=state.queue.end_s){video.pause();video.currentTime=state.queue.end_s}};
$("#timeline").oninput=event=>{video.pause();video.currentTime=Number(event.target.value)};
$("#start").onclick=()=>markStart("exact");$("#startInferred").onclick=()=>markStart("inferred");$("#end").onclick=()=>markEnd("exact");$("#endInferred").onclick=()=>markEnd("inferred");
$("#back").onclick=()=>seek(-2);$("#forward").onclick=()=>seek(2);$("#backFine").onclick=()=>seek(-.1);$("#forwardFine").onclick=()=>seek(.1);
$("#undo").onclick=()=>{if(state.pendingStart)state.pendingStart=null;else state.rallies.pop();render();save()};
document.addEventListener("keydown",event=>{if(event.code==="Space"){event.preventDefault();video.paused?video.play():video.pause();return}const key=event.key.toLowerCase();if(key==="s")markStart("exact");else if(key==="i")markStart("inferred");else if(key==="e")markEnd("exact");else if(key==="u")markEnd("inferred");else if(event.key==="ArrowLeft"){event.preventDefault();seek(event.shiftKey ? -.1 : -2)}else if(event.key==="ArrowRight"){event.preventDefault();seek(event.shiftKey ? .1 : 2)}});
Promise.all([fetch(`/${queuePath}?v=${Date.now()}`).then(response=>response.json()),fetch(`/${resultPath}?v=${Date.now()}`).then(response=>response.ok?response.json():null).catch(()=>null)]).then(([queue,result])=>{state.queue=queue;state.rallies=result?.rallies||[];state.pendingStart=result?.pending_start||null;video.src=new URL(`/${queue.video}`,location.origin).href;$("#timeline").min=queue.start_s;$("#timeline").max=queue.end_s;video.addEventListener("loadedmetadata",()=>{video.currentTime=state.pendingStart?.time_s||state.rallies.at(-1)?.end_s||queue.start_s;render();$("#saved").textContent="✓ сохранено"},{once:true})}).catch(error=>$("#saved").textContent=`Не удалось открыть задание: ${error.message}`);
