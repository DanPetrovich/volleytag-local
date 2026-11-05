const $ = (selector) => document.querySelector(selector);
const params = new URLSearchParams(location.search);
const queuePath = params.get("queue") || "vision/runs/touch-temporal-v1/exact-contact-queue.json";
const resultPath = params.get("result") || "vision/runs/touch-temporal-v1/exact-contact-results.json";
const video = $("#video");
const state = { queue:[], answers:{}, index:0, saveTimer:null };
const frameStep = 1 / 30;

function clock(seconds){const m=Math.floor(seconds/60),s=seconds-m*60;return `${String(m).padStart(2,"0")}:${s.toFixed(3).padStart(6,"0")}`}
function current(){return state.queue[state.index]}
function answered(){return Object.keys(state.answers).length}
function updateProgress(){ $("#progress").textContent=`${state.index+1} / ${state.queue.length} · проверено ${answered()}` }

async function save(){
  clearTimeout(state.saveTimer); $("#saved").textContent="Сохраняю…";
  const payload={auditPath:resultPath,schema_version:1,kind:"volleytag.exact-contact-results",queue:queuePath,answers:state.answers,updated_at:new Date().toISOString()};
  try{const response=await fetch("/api/save-dataset-audit",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!response.ok)throw new Error(await response.text());$("#saved").textContent="Сохранено ✓"}catch(error){$("#saved").textContent=`Ошибка: ${error.message}`}
}

function loadItem(index, autoplay=true){
  state.index=Math.max(0,Math.min(state.queue.length-1,index));
  const item=current(); video.pause(); video.src=new URL(`/${item.video}`,location.origin).href; video.load();
  video.addEventListener("loadedmetadata",()=>{
    $("#timeline").min=item.window_start_s; $("#timeline").max=Math.min(item.window_end_s,video.duration);
    video.currentTime=item.window_start_s; if(autoplay) video.play();
    $("#meta").textContent=`Источник ${item.source+1} · ${item.action||"контроль"} · приблизительная метка ${clock(item.approx_time_s)}`;
    updateProgress();
  },{once:true});
}

function flash(){const node=$("#flash");node.classList.add("show");setTimeout(()=>node.classList.remove("show"),160)}
function answer(kind){
  const item=current(); const exact=["contact","nongame_contact"].includes(kind)?Number(video.currentTime.toFixed(3)):null;
  state.answers[item.id]={item_id:item.id,decision:kind,exact_contact_time_s:exact,offset_from_approx_s:exact===null?null:Number((exact-item.approx_time_s).toFixed(3)),answered_at:new Date().toISOString()};
  flash(); save(); updateProgress(); setTimeout(()=>{if(state.index<state.queue.length-1)loadItem(state.index+1);else{$("#saved").textContent="Готово — вся очередь проверена ✓";video.pause()}},180);
}
function replay(){const item=current();video.currentTime=item.window_start_s;video.play()}
function step(delta){video.pause();const item=current();video.currentTime=Math.max(item.window_start_s,Math.min(item.window_end_s,video.currentTime+delta))}

video.addEventListener("timeupdate",()=>{const item=current();if(!item)return;$("#time").textContent=clock(video.currentTime);$("#timeline").value=video.currentTime;if(video.currentTime>=item.window_end_s){video.pause();video.currentTime=item.window_end_s}});
$("#timeline").addEventListener("input",e=>{video.pause();video.currentTime=Number(e.target.value)});
$("#play").addEventListener("click",()=>video.paused?video.play():video.pause());
$("#replay").addEventListener("click",replay);$("#prevFrame").addEventListener("click",()=>step(-frameStep));$("#nextFrame").addEventListener("click",()=>step(frameStep));
$("#contact").addEventListener("click",()=>answer("contact"));$("#nongame").addEventListener("click",()=>answer("nongame_contact"));$("#none").addEventListener("click",()=>answer("no_contact"));$("#offscreen").addEventListener("click",()=>answer("offscreen_contact"));$("#uncertain").addEventListener("click",()=>answer("uncertain"));
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName))return;if(event.code==="Space"){event.preventDefault();video.paused?video.play():video.pause()}else if(event.code==="ArrowLeft"){event.preventDefault();step(-frameStep)}else if(event.code==="ArrowRight"){event.preventDefault();step(frameStep)}else if(event.key.toLowerCase()==="r")replay();else if(event.key.toLowerCase()==="c")answer("contact");else if(event.key.toLowerCase()==="g")answer("nongame_contact");else if(event.key.toLowerCase()==="n")answer("no_contact");else if(event.key.toLowerCase()==="o")answer("offscreen_contact");else if(event.key.toLowerCase()==="u")answer("uncertain");else if(event.code==="Backspace"){event.preventDefault();loadItem(state.index-1,false)}});

Promise.all([fetch(`${queuePath}?v=${Date.now()}`).then(r=>r.json()),fetch(`${resultPath}?v=${Date.now()}`).then(r=>r.ok?r.json():null).catch(()=>null)]).then(([queue,results])=>{state.queue=queue.items;state.answers=results?.answers||{};const first=state.queue.findIndex(item=>!state.answers[item.id]);loadItem(first<0?state.queue.length-1:first)}).catch(error=>{$("#saved").textContent=`Не удалось открыть очередь: ${error.message}`});
