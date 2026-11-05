const $=s=>document.querySelector(s),video=$("#video");
const repoUrl=path=>/^(?:[a-z]+:|\/)/i.test(path)?path:`/${path}`;
const params=new URLSearchParams(location.search);
const source={id:"training-keaz-set1",name:"КЭАЗ — X5 Tech · партия 1",video:"vision/runs/training-keaz/source.mp4",exclude:"vision/runs/training-keaz/non-match-intervals.json",start_s:2.048,end_s:852.048};
const resultPath=params.get("result")||"vision/runs/training-keaz/position-cycle-v2.json";
const rotationOrder=[3,2,1,6,5,4];
const momentNames={serve_start:"старт до подачи",serve_receive:"приёмная расстановка",playing_base:"игровая база",attack:"разбег / атака",defense:"защита",serve:"моя подача"};
const state={markers:[],rotationIndex:0,started:false,complete:false,cycleStart:null,cycleEnd:null,excluded:[]};
const clock=value=>{const s=Math.max(0,Number(value)||0),m=Math.floor(s/60);return `${String(m).padStart(2,"0")}:${(s-m*60).toFixed(1).padStart(4,"0")}`};
function flash(id){const el=$(id);el.classList.add("show");clearTimeout(el._t);el._t=setTimeout(()=>el.classList.remove("show"),220)}
function currentRotation(){return rotationOrder[state.rotationIndex]}
function update(){
  $("#counter").textContent=`${state.markers.length} отметок`;
  $("#rotation").textContent=state.started?`позиция ${currentRotation()}`:"ещё не начато";
  $("#rotationArrow").textContent=`${currentRotation()} → ${rotationOrder[(state.rotationIndex+1)%6]}`;
  $("#nextRotation").disabled=!state.started||state.complete;
  $("#finishCycle").disabled=!state.started||state.complete;
  $("#startCycle").disabled=state.started&&!state.complete;
  const recent=state.markers.slice(-12).reverse();
  $("#markers").innerHTML=recent.length?recent.map(x=>`<li><b>${clock(x.time_s)}</b> · ротация ${x.rotation_zone} · ${momentNames[x.moment]||x.event}</li>`).join(""):'<li class="empty">Пока отметок нет</li>';
}
async function save(){
  $("#saveState").textContent="Сохраняю…";
  const payload={auditPath:resultPath,schema_version:2,kind:"volleytag.5-1-rotation-landmarks",source,cycle_start_s:state.cycleStart,cycle_end_s:state.cycleEnd,complete:state.complete,rotation_order:rotationOrder,coordinate_source:"personal detector/track at marked timestamp",markers:state.markers,updated_at:new Date().toISOString()};
  try{const response=await fetch("/api/save-dataset-audit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!response.ok)throw new Error(await response.text());$("#saveState").textContent="Сохранено ✓"}catch(e){$("#saveState").textContent=`Ошибка: ${e.message}`}
}
function addMarker(event,moment=null){
  const row={id:`${Date.now().toString(36)}-${Math.random().toString(36).slice(2,7)}`,time_s:Number(video.currentTime.toFixed(3)),event,rotation_zone:currentRotation(),moment,created_at:new Date().toISOString()};
  state.markers.push(row);update();flash("#flash");void save();
}
function startCycle(){state.markers=[];state.started=true;state.complete=false;state.rotationIndex=0;state.cycleStart=Number(video.currentTime.toFixed(3));state.cycleEnd=null;addMarker("cycle_start","serve_start")}
function nextRotation(){if(!state.started||state.complete)return;state.rotationIndex=(state.rotationIndex+1)%6;addMarker("rotation_change")}
function finishCycle(){if(!state.started||state.complete)return;state.cycleEnd=Number(video.currentTime.toFixed(3));state.complete=true;addMarker("cycle_end")}
function markMoment(moment){if(!state.started){$("#saveState").textContent="Сначала зафиксируй начало цикла";return}addMarker("landmark",moment)}
function undo(){const removed=state.markers.pop();if(!removed)return;if(removed.event==="cycle_end"){state.complete=false;state.cycleEnd=null}else if(removed.event==="rotation_change"){state.rotationIndex=(state.rotationIndex+5)%6}else if(removed.event==="cycle_start"){state.started=false;state.cycleStart=null}update();void save()}
function seek(delta){video.currentTime=Math.max(source.start_s,Math.min(source.end_s,video.currentTime+delta))}
function toggle(){video.paused?video.play():video.pause()}
document.querySelectorAll("[data-moment]").forEach(button=>button.onclick=()=>markMoment(button.dataset.moment));
$("#startCycle").onclick=startCycle;$("#nextRotation").onclick=nextRotation;$("#finishCycle").onclick=finishCycle;$("#undo").onclick=undo;$("#back").onclick=()=>seek(-5);$("#forward").onclick=()=>seek(5);$("#play").onclick=toggle;
$("#speed").onchange=()=>video.playbackRate=Number($("#speed").value);$("#timeline").oninput=e=>video.currentTime=Number(e.target.value);
video.onplay=()=>$("#play").textContent="⏸ Пауза";video.onpause=()=>$("#play").textContent="▶ Смотреть";
video.ontimeupdate=()=>{const t=video.currentTime;$("#timecode").textContent=clock(t);$("#timeline").value=t;localStorage.setItem("position-cycle-keaz-v2-time",String(t));const interval=state.excluded.find(x=>t>=Number(x.start_s??x.start)&&t<Number(x.end_s??x.end));if(interval){video.currentTime=Number(interval.end_s??interval.end)+.05;flash("#skip")}};
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName))return;if(event.code==="Space"){event.preventDefault();toggle()}else if(/^Digit[1-6]$/.test(event.code)){event.preventDefault();document.querySelectorAll("[data-moment]")[Number(event.code.at(-1))-1]?.click()}else if(event.key.toLowerCase()==="n")nextRotation();else if(event.code==="Backspace"){event.preventDefault();undo()}else if(event.code==="ArrowLeft")seek(-3);else if(event.code==="ArrowRight")seek(3)});
async function init(){
  try{const response=await fetch(repoUrl(`${resultPath}?v=${Date.now()}`));if(response.ok){const old=await response.json();state.markers=old.markers||[];state.cycleStart=old.cycle_start_s??null;state.cycleEnd=old.cycle_end_s??null;state.started=state.cycleStart!==null;state.complete=Boolean(old.complete);const last=state.markers.at(-1);if(last)state.rotationIndex=Math.max(0,rotationOrder.indexOf(Number(last.rotation_zone)))}}catch(_){}
  try{const response=await fetch(repoUrl(`${source.exclude}?v=${Date.now()}`));if(response.ok)state.excluded=(await response.json()).excluded_intervals||[]}catch(_){}
  video.src=new URL(`/${source.video}`,location.origin).href;video.load();video.onloadedmetadata=()=>{video.playbackRate=1.5;$("#timeline").min=source.start_s;$("#timeline").max=source.end_s;const resume=Number(localStorage.getItem("position-cycle-keaz-v2-time")||source.start_s);video.currentTime=Math.max(source.start_s,Math.min(source.end_s-.2,resume));$("#saveState").textContent="Видео готово"};update();
}
init();
