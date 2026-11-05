const $=s=>document.querySelector(s),video=$("#video");
const repoUrl=path=>/^(?:[a-z]+:|\/)/i.test(path)?path:`/${path}`;
const params=new URLSearchParams(location.search);
const source={
  id:params.get("sourceId")||"training-keaz-set1",
  name:params.get("sourceName")||"КЭАЗ — X5 Tech · партия 1",
  video:params.get("video")||"vision/runs/training-keaz/source.mp4",
  exclude:params.get("exclude")||"vision/runs/training-keaz/non-match-intervals.json",
  start_s:Number(params.get("start")||2.048),
  end_s:Number(params.get("end")||852.048),
};
const resultPath=params.get("result")||"vision/runs/training-keaz/rotation-state-v1.json";
const order=[3,2,1,6,5,4];
const state={events:[],started:false,complete:false,rotation:3,owner:"theirs",variant:"front",excluded:[]};
const clock=value=>{const s=Math.max(0,Number(value)||0),m=Math.floor(s/60);return `${String(m).padStart(2,"0")}:${(s-m*60).toFixed(1).padStart(4,"0")}`};
const ownerName=value=>value==="ours"?"подаём мы":"подаёт соперник";
function nextRotation(value){return order[(order.indexOf(Number(value))+1)%order.length]}
function rule(rotation,owner,variant){
  const r=Number(rotation);
  if(r===1)return owner==="ours"?{start:"подача из-за поля",base:[6]}:{start:[1],base:[6]};
  if(r===2){if(owner==="ours")return{start:[2],base:[4]};return variant==="front"?{start:[1],base:[2]}:{start:[5],base:[4]}}
  if(r===3)return owner==="ours"?{start:[3],base:[4]}:{start:[5],base:[4]};
  if(r===4)return owner==="ours"?{start:[4],base:[4]}:{start:[5],base:[4]};
  if(r===5)return{start:[5],base:[6]};
  return{start:[6],base:[6]};
}
function ruleText(rotation=state.rotation,owner=state.owner,variant=state.variant){const value=rule(rotation,owner,variant),start=Array.isArray(value.start)?`зона ${value.start.join("/")}`:value.start,base=`зона ${value.base.join("/")}`;return start===base?`Весь розыгрыш преимущественно ${base}.`:`До подачи/на приёме: ${start}; затем игровая база: ${base}.`}
function intervals(){const completion=state.events.findLast?.(event=>event.event==="complete"),events=state.events.filter(event=>event.event!=="complete");return events.map((event,index)=>{const end=events[index+1]?.time_s??completion?.time_s??null,value=rule(event.rotation,event.serve_owner,state.variant);return{start_s:event.time_s,end_s:end,rotation:event.rotation,serve_owner:event.serve_owner,pre_serve:value.start,playing_base:value.base,hard_filter:false}})}
function flash(id){const el=$(id);el.classList.add("show");clearTimeout(el._t);el._t=setTimeout(()=>el.classList.remove("show"),220)}
function render(){
  $("#eventCount").textContent=`${Math.max(0,state.events.length-1)} смен подачи`;
  $("#stateTitle").textContent=state.started?`позиция ${state.rotation} · ${ownerName(state.owner)}`:"ещё не начато";
  $("#stateRule").textContent=state.started?ruleText():"Выбери стартовые параметры.";
  $("#setup").querySelectorAll("select").forEach(el=>el.disabled=state.started&&!state.complete);
  $("#start").disabled=state.started&&!state.complete;$("#toOurs").disabled=!state.started||state.complete||state.owner==="ours";$("#toTheirs").disabled=!state.started||state.complete||state.owner==="theirs";$("#complete").disabled=!state.started||state.complete;
  const rows=state.events.slice().reverse();$("#history").innerHTML=rows.length?rows.map(e=>`<li><b>${clock(e.time_s)}</b> · позиция ${e.rotation} · ${ownerName(e.serve_owner)}</li>`).join(""):'<li class="empty">Пока событий нет</li>';
}
async function save(){
  $("#saveState").textContent="Сохраняю…";
  const payload={auditPath:resultPath,schema_version:1,kind:"volleytag.5-1-rotation-state",source,outside_variant:state.variant,rotation_order:order,events:state.events,intervals:intervals(),complete:state.complete,position_prior_policy:{hard_filter:false,expected_zone:"small bonus",adjacent_zone:"smaller bonus",other_zone:"neutral; candidate is never rejected"},rules:{r1:{ours:"serve outside -> 6",theirs:"1 -> 6"},r2:{ours:"2 -> 4",theirs_front:"1 -> 2",theirs_back:"5 -> 4"},r3:{ours:"3 -> 4",theirs:"5 -> 4"},r4:{ours:"4 -> 4",theirs:"5 -> 4"},r5:{ours:"5 -> 6",theirs:"5 -> 6"},r6:{ours:"6 -> 6",theirs:"6 -> 6"}},updated_at:new Date().toISOString()};
  try{const response=await fetch("/api/save-dataset-audit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!response.ok)throw new Error(await response.text());$("#saveState").textContent="Сохранено ✓"}catch(error){$("#saveState").textContent=`Ошибка: ${error.message}`}
}
function start(){state.events=[];state.rotation=Number($("#initialRotation").value);state.owner=$("#initialOwner").value;state.variant=$("#outsideVariant").value;state.started=true;state.complete=false;state.events.push({time_s:Number(video.currentTime.toFixed(3)),rotation:state.rotation,serve_owner:state.owner,event:"start"});render();flash("#flash");void save()}
function changeOwner(owner){if(!state.started||state.complete||owner===state.owner)return;if(state.owner==="theirs"&&owner==="ours")state.rotation=nextRotation(state.rotation);state.owner=owner;state.events.push({time_s:Number(video.currentTime.toFixed(3)),rotation:state.rotation,serve_owner:state.owner,event:"service_change"});render();flash("#flash");void save()}
function undo(){if(!state.events.length)return;state.events.pop();if(!state.events.length){state.started=false;state.complete=false}else{const last=state.events.at(-1);state.rotation=last.rotation;state.owner=last.serve_owner;state.complete=false}render();void save()}
function complete(){if(!state.started)return;state.complete=true;state.events.push({time_s:Number(video.currentTime.toFixed(3)),rotation:state.rotation,serve_owner:state.owner,event:"complete"});render();void save()}
function seek(delta){video.currentTime=Math.max(source.start_s,Math.min(source.end_s,video.currentTime+delta))}function toggle(){video.paused?video.play():video.pause()}
$("#start").onclick=start;$("#toOurs").onclick=()=>changeOwner("ours");$("#toTheirs").onclick=()=>changeOwner("theirs");$("#undo").onclick=undo;$("#complete").onclick=complete;$("#back").onclick=()=>seek(-5);$("#forward").onclick=()=>seek(5);$("#play").onclick=toggle;$("#speed").onchange=()=>video.playbackRate=Number($("#speed").value);$("#timeline").oninput=e=>video.currentTime=Number(e.target.value);
video.onplay=()=>$("#play").textContent="⏸ Пауза";video.onpause=()=>$("#play").textContent="▶ Смотреть";
video.ontimeupdate=()=>{const t=video.currentTime;$("#timecode").textContent=clock(t);$("#timeline").value=t;localStorage.setItem(`rotation-review-${source.id}-time`,String(t));const interval=state.excluded.find(x=>t>=Number(x.start_s??x.start)&&t<Number(x.end_s??x.end));if(interval){video.currentTime=Number(interval.end_s??interval.end)+.05;flash("#skip")}};
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName))return;if(event.code==="Space"){event.preventDefault();toggle()}else if(event.code==="Digit1")changeOwner("ours");else if(event.code==="Digit2")changeOwner("theirs");else if(event.code==="Backspace"){event.preventDefault();undo()}else if(event.code==="ArrowLeft")seek(-3);else if(event.code==="ArrowRight")seek(3)});
async function init(){
  try{const response=await fetch(repoUrl(`${resultPath}?v=${Date.now()}`));if(response.ok){const old=await response.json();state.events=old.events||[];state.variant=old.outside_variant||"front";state.started=state.events.length>0;state.complete=Boolean(old.complete);if(state.events.length){const last=state.events.at(-1);state.rotation=last.rotation;state.owner=last.serve_owner}}}catch(_){}
  try{const response=await fetch(repoUrl(`${source.exclude}?v=${Date.now()}`));if(response.ok)state.excluded=(await response.json()).excluded_intervals||[]}catch(_){}
  video.src=new URL(`/${source.video}`,location.origin).href;video.load();video.onloadedmetadata=()=>{video.playbackRate=1.5;$("#timeline").min=source.start_s;$("#timeline").max=source.end_s;const resume=Number(localStorage.getItem(`rotation-review-${source.id}-time`)||source.start_s);video.currentTime=Math.max(source.start_s,Math.min(source.end_s-.2,resume));$("#saveState").textContent="Видео готово"};render();
}
init();
