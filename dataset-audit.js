const params=new URLSearchParams(location.search);
const queuePath=params.get('queue')||'vision/runs/danya-detector-v5/audit/manual-sample.json';
const resultPath=params.get('result')||'vision/runs/danya-detector-v5/audit/manual-sample-results.json';
const editBoxes=params.get('editBoxes')==='1';
const ballMode=params.get('ball')==='1';
const selectPerson=params.get('selectPerson')==='1'||ballMode;
const benchmark=params.get('benchmark')==='1';
const excludePath=params.get('exclude');
const els={frame:document.querySelector('#frame'),overlay:document.querySelector('#overlay'),meta:document.querySelector('#meta'),reasons:document.querySelector('#reasons'),progress:document.querySelector('#progress'),saved:document.querySelector('#saved'),jump:document.querySelector('#jump')};
let queue,index=0,results={schema_version:1,queue:queuePath,started_at:new Date().toISOString(),decisions:{}},drawing=false,saving=false,startPoint=null,draftBox=null,imageGeometry=null,candidateRects=[];
const storageKey=`volleytag-audit:${resultPath}`;

function compactResults(value){
  const decisions={};
  for(const [id,decision] of Object.entries(value.decisions||{})){
    const {item,...compact}=decision;
    decisions[id]=compact;
  }
  return {...value,decisions};
}
function persistLocal(){
  // Benchmark decisions used to include the full frame item (often dozens of
  // detector candidates) for every click and quickly exhausted localStorage.
  // Disk is authoritative; localStorage is only a compact recovery copy.
  const payload=benchmark?compactResults(results):results;
  try{localStorage.setItem(storageKey,JSON.stringify(payload));}
  catch(error){
    try{localStorage.removeItem(storageKey);localStorage.setItem(storageKey,JSON.stringify(compactResults(results)));}
    catch(_ignored){}
  }
}
function rememberIndex(){try{localStorage.setItem(`${storageKey}:index`,String(index))}catch(_ignored){}}

async function load(){
  queue=await fetch(queuePath).then(r=>{if(!r.ok)throw new Error(`Queue ${r.status}`);return r.json()});
  if(excludePath){
    const exclusionData=await fetch(excludePath,{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`Exclude ${r.status}`);return r.json()});
    const intervals=Array.isArray(exclusionData)?exclusionData:(exclusionData.intervals||[]);
    queue={...queue,items:queue.items.filter(item=>!intervals.some(interval=>item.time_s>=Number(interval.start_s??interval.start)&&item.time_s<=Number(interval.end_s??interval.end)))};
    queue.total=queue.items.length;
  }
  const disk=await fetch(resultPath,{cache:'no-store'}).then(r=>r.ok?r.json():null).catch(()=>null);
  if(disk)results=disk;
  const cached=localStorage.getItem(storageKey);
  if(cached){
    const local=JSON.parse(cached);
    results={...results,...local,decisions:{...(results.decisions||{}),...(local.decisions||{})}};
  }
  if(benchmark)results=compactResults(results);
  const firstUnreviewed=queue.items.findIndex(item=>!results.decisions?.[item.id]);
  index=benchmark&&firstUnreviewed>=0?firstUnreviewed:Math.min(Number(localStorage.getItem(`${storageKey}:index`)||0),queue.items.length-1);
  render();
}
function draw(){
  const item=queue.items[index],img=els.frame,canvas=els.overlay,stage=img.parentElement;
  const rect=img.getBoundingClientRect(),host=stage.getBoundingClientRect();
  canvas.width=rect.width;canvas.height=rect.height;canvas.style.left=`${rect.left-host.left}px`;canvas.style.top=`${rect.top-host.top}px`;canvas.style.width=`${rect.width}px`;canvas.style.height=`${rect.height}px`;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=4;ctx.strokeStyle='#b7ff36';ctx.fillStyle='#b7ff36';ctx.font='bold 16px system-ui';
  // `object-fit: contain` may letterbox the bitmap inside the <img> element.
  // Draw labels against the actual rendered bitmap, not the element box.
  const scale=Math.min(canvas.width/img.naturalWidth,canvas.height/img.naturalHeight);
  const imageWidth=img.naturalWidth*scale,imageHeight=img.naturalHeight*scale;
  const offsetX=(canvas.width-imageWidth)/2,offsetY=(canvas.height-imageHeight)/2;
  imageGeometry={offsetX,offsetY,imageWidth,imageHeight};
  candidateRects=[];
  if(selectPerson){for(const candidate of item.candidates||[]){const [,cx,cy,w,h]=candidate.label,x=offsetX+(cx-w/2)*imageWidth,y=offsetY+(cy-h/2)*imageHeight,rect={x,y,w:w*imageWidth,h:h*imageHeight,label:candidate.label,confidence:candidate.confidence};candidateRects.push(rect);ctx.strokeStyle='#ffad33';ctx.lineWidth=2;ctx.strokeRect(rect.x,rect.y,rect.w,rect.h);}}
  const corrected=results.decisions[item.id]?.correctedLabel;
  for(const label of corrected?[corrected]:(item.labels||[])){const [,cx,cy,w,h]=label,x=offsetX+(cx-w/2)*imageWidth,y=offsetY+(cy-h/2)*imageHeight;ctx.strokeStyle=corrected?'#55d8ff':'#b7ff36';ctx.fillStyle=ctx.strokeStyle;ctx.strokeRect(x,y,w*imageWidth,h*imageHeight);ctx.fillText(corrected?'ИСПРАВЛЕНО':(ballMode?'МЯЧ':'ДАНЯ'),x+3,Math.max(18,y-5));}
  if(draftBox){ctx.strokeStyle='#55d8ff';ctx.setLineDash([9,6]);ctx.strokeRect(draftBox.x,draftBox.y,draftBox.w,draftBox.h);ctx.setLineDash([]);}
}
function render(){
  const item=queue.items[index],done=Object.keys(results.decisions).length;
  drawing=false;draftBox=null;startPoint=null;els.overlay.classList.remove('drawing');
  els.progress.textContent=`${index+1} / ${queue.items.length} · проверено ${done}`;els.jump.max=queue.items.length;els.jump.value=index+1;
  // Clear stale hit regions immediately. Local JPEGs can resolve from cache so
  // quickly that relying only on `onload` occasionally leaves the previous
  // frame's overlay active over the new bitmap.
  candidateRects=[];imageGeometry=null;els.overlay.width=1;els.overlay.height=1;
  const requestedImage=item.image;
  const drawCurrent=()=>{if(queue.items[index].image===requestedImage)draw()};
  els.frame.onload=drawCurrent;els.frame.src=requestedImage;
  if(els.frame.complete)els.frame.decode().then(drawCurrent).catch(()=>{});
  els.meta.textContent=ballMode?`Источник ${item.source+1} · ${fmt(item.time_s)} · ${item.action||'касание'} · смещение ${(Number(item.sample_offset_s)||0).toFixed(2)} с`:`Источник ${item.source+1} · ${fmt(item.time_s)} · ${item.domain} · ${item.kind} · ${item.split}`;
  const kindHelp={hard_negative:'Это отрицательный кроп. Если тебя в нём нет — нажми 2. Если ты здесь есть, но зелёной рамки нет — нажми 3.',background_negative:'Это отрицательный полный кадр без рамки. Если тебя здесь нет — нажми 2. Если ты есть — нажми 3.',positive:'Это положительный кадр: зелёная рамка должна охватывать именно тебя.',manual_positive:'Это кадр с рамкой, которую ты нарисовал вручную.'};
  const risk=item.risk_reasons?.length?`\nПочему поднят на проверку: ${item.risk_reasons.join(', ')}`:'';
  els.reasons.textContent=(ballMode?'Оранжевые рамки — черновые кандидаты обычного детектора. Выбирай их только если внутри действительно волейбольный мяч.':(kindHelp[item.kind]||''))+risk;
  const savedVerdict=results.decisions[item.id]?.verdict;
  const normalized=savedVerdict==='wrong_player'?'not_in_box':savedVerdict==='correct'?(item.labels?.length?'in_box':'not_in_box'):savedVerdict;
  document.querySelectorAll('[data-verdict]').forEach(button=>button.classList.toggle('selected',normalized===button.dataset.verdict));
  document.querySelector('#instructions').textContent=ballMode?'Кликни по правильной оранжевой рамке. Если рамки нет — R и нарисуй мяч. Если мяч не виден — 5.':benchmark?'Кликни по своей оранжевой рамке — кадр сразу сменится. Кнопок 1–4 здесь нет, потому что они не указывают, какую из многих рамок выбрать. Плохая или отсутствующая рамка — R и нарисуй.':selectPerson?'Кликни прямо по своей оранжевой рамке. Если тебя нет — 2. Если рамка пропущена — R и нарисуй её.':'Проверяй только то, что реально лежит в обучающем архиве. Зелёная рамка — текущая YOLO-разметка.';
  const selectionVerdicts=ballMode?['not_visible','unusable','uncertain']:benchmark?['not_visible','occluded','unusable','uncertain']:['not_in_box','unusable'];
  document.querySelectorAll('[data-verdict]').forEach(button=>button.hidden=selectPerson&&!selectionVerdicts.includes(button.dataset.verdict));
  if(ballMode){document.querySelector('[data-verdict="not_visible"]').innerHTML='<kbd>5</kbd> Мяч не виден';document.querySelector('[data-verdict="unusable"]').innerHTML='<kbd>7</kbd> Кадр непригоден';document.querySelector('[data-verdict="uncertain"]').innerHTML='<kbd>8</kbd> Не уверен';document.querySelector('#redraw').innerHTML='<kbd>R</kbd> Нарисовать рамку мяча';}
  document.querySelector('#redraw').hidden=!editBoxes;
  els.overlay.classList.toggle('selection',selectPerson&&!drawing);
  rememberIndex();
}
function fmt(seconds){const m=Math.floor(seconds/60),s=(seconds%60).toFixed(1).padStart(4,'0');return `${String(m).padStart(2,'0')}:${s}`}
async function decide(verdict){
  const item=queue.items[index],previous=results.decisions[item.id]||{};results.decisions[item.id]={...previous,verdict,reviewed_at:new Date().toISOString(),...(benchmark?{}:{item})};results.updated_at=new Date().toISOString();
  persistLocal();els.saved.textContent='Сохраняю…';
  try{const response=await fetch('/api/save-dataset-audit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...results,auditPath:resultPath})});if(!response.ok)throw new Error(await response.text());els.saved.textContent='Сохранено на диск.'}catch(error){els.saved.textContent=`Локально сохранено; сервер: ${error.message}`}
  if(index<queue.items.length-1)index++;render();
}
function canvasPoint(event){const r=els.overlay.getBoundingClientRect();return{x:(event.clientX-r.left)*els.overlay.width/r.width,y:(event.clientY-r.top)*els.overlay.height/r.height}}
async function saveCorrectedBox(box){
  if(saving)return;
  const g=imageGeometry,x1=Math.max(g.offsetX,Math.min(g.offsetX+g.imageWidth,box.x)),y1=Math.max(g.offsetY,Math.min(g.offsetY+g.imageHeight,box.y)),x2=Math.max(g.offsetX,Math.min(g.offsetX+g.imageWidth,box.x+box.w)),y2=Math.max(g.offsetY,Math.min(g.offsetY+g.imageHeight,box.y+box.h));
  if(x2-x1<4||y2-y1<8)return;
  saving=true;
  const label=[0,((x1+x2)/2-g.offsetX)/g.imageWidth,((y1+y2)/2-g.offsetY)/g.imageHeight,(x2-x1)/g.imageWidth,(y2-y1)/g.imageHeight],item=queue.items[index];
  results.decisions[item.id]={verdict:'corrected_box',correctedLabel:label,reviewed_at:new Date().toISOString(),...(benchmark?{}:{item})};results.updated_at=new Date().toISOString();persistLocal();els.saved.textContent='Исправленная рамка — сохраняю…';draw();
  try{
    const [response]=await Promise.all([fetch('/api/save-dataset-audit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...results,auditPath:resultPath})}),new Promise(resolve=>setTimeout(resolve,250))]);
    if(!response.ok)throw new Error(await response.text());
    els.saved.textContent='Исправленная рамка сохранена ✓';
    if(index<queue.items.length-1)index++;
    render();
  }catch(error){els.saved.textContent=`Локально сохранено; сервер: ${error.message}`}
  finally{saving=false}
}
async function saveCandidate(label){
  if(saving)return;saving=true;
  const item=queue.items[index];results.decisions[item.id]={verdict:'selected_candidate',correctedLabel:label,reviewed_at:new Date().toISOString(),...(benchmark?{}:{item})};results.updated_at=new Date().toISOString();persistLocal();els.saved.textContent='Сохраняю выбор…';
  // Show the chosen box on the current frame before advancing. Consecutive
  // benchmark frames look very similar, so an instantaneous switch is easy to
  // mistake for no reaction at all.
  try{
    // Rebuilding the canvas during pointerdown can cancel the rest of the
    // gesture in some Chromium builds. Paint the confirmation on the existing
    // canvas instead; saving and advancing must not depend on a redraw.
    const rect=candidateRects.find(r=>r.label===label||JSON.stringify(r.label)===JSON.stringify(label));
    if(rect){
      const ctx=els.overlay.getContext('2d');
      ctx.save();ctx.strokeStyle='#55d8ff';ctx.lineWidth=7;ctx.strokeRect(rect.x,rect.y,rect.w,rect.h);ctx.restore();
    }
    const [response]=await Promise.all([
      fetch('/api/save-dataset-audit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...results,auditPath:resultPath})}),
      new Promise(resolve=>setTimeout(resolve,250)),
    ]);
    if(!response.ok)throw new Error(await response.text());
    els.saved.textContent='Выбрано и сохранено ✓';
    if(index<queue.items.length-1)index++;
    render();
  }catch(error){
    els.saved.textContent=`Локально сохранено; сервер: ${error.message}`;
  }finally{saving=false}
}
function selectCandidateAt(point){
  if(saving)return false;
  // Exact hit-testing is intentional: nearby player boxes frequently overlap,
  // and invisible padding can select the wrong neighbour.
  const hits=candidateRects.filter(r=>point.x>=r.x&&point.x<=r.x+r.w&&point.y>=r.y&&point.y<=r.y+r.h).sort((a,b)=>a.w*a.h-b.w*b.h);
  if(hits.length){void saveCandidate(hits[0].label);return true}
  els.saved.textContent='Клик не попал в рамку. Нажми R и нарисуй её либо кликни ближе к центру.';
  return false;
}
document.querySelector('#redraw').onclick=()=>{drawing=!drawing;draftBox=null;startPoint=null;els.overlay.classList.toggle('drawing',drawing);els.saved.textContent=drawing?(ballMode?'Проведи маленькую рамку вокруг мяча. R ещё раз — отмена.':'Проведи мышью новую рамку вокруг себя. R ещё раз — отмена.'):'Рисование отменено.'};
els.overlay.onpointerdown=e=>{const p=canvasPoint(e);if(selectPerson&&!drawing){selectCandidateAt(p);return}if(!drawing)return;startPoint=p;els.overlay.setPointerCapture(e.pointerId)};
els.overlay.onpointermove=e=>{if(!drawing||!startPoint)return;const p=canvasPoint(e);draftBox={x:Math.min(startPoint.x,p.x),y:Math.min(startPoint.y,p.y),w:Math.abs(p.x-startPoint.x),h:Math.abs(p.y-startPoint.y)};draw()};
els.overlay.onpointerup=e=>{if(!drawing)return;const point=canvasPoint(e);if(!draftBox||draftBox.w<4||draftBox.h<8){drawing=false;startPoint=null;draftBox=null;els.overlay.classList.remove('drawing');if(selectPerson)selectCandidateAt(point);return}const box=draftBox;drawing=false;startPoint=null;draftBox=null;els.overlay.classList.remove('drawing');saveCorrectedBox(box)};
document.querySelectorAll('[data-verdict]').forEach(button=>button.onclick=()=>decide(button.dataset.verdict));
document.querySelector('#previous').onclick=()=>{index=Math.max(0,index-1);render()};document.querySelector('#next').onclick=()=>{index=Math.min(queue.items.length-1,index+1);render()};
els.jump.onchange=()=>{index=Math.max(0,Math.min(queue.items.length-1,Number(els.jump.value)-1));render()};
addEventListener('resize',draw);addEventListener('keydown',event=>{if(['INPUT','TEXTAREA'].includes(event.target.tagName))return;if(editBoxes&&(event.code==='KeyR'||event.key.toLowerCase()==='r'||event.key.toLowerCase()==='к')){event.preventDefault();document.querySelector('#redraw').click()}if(/^[1-8]$/.test(event.key))document.querySelector(`[data-verdict="${['in_box','not_in_box','missing_box','bad_box','not_visible','occluded','unusable','uncertain'][Number(event.key)-1]}"]`).click();if(event.key==='ArrowLeft')document.querySelector('#previous').click();if(event.key==='ArrowRight')document.querySelector('#next').click()});
addEventListener('error',event=>{els.saved.textContent=`Ошибка интерфейса: ${event.message}`});
addEventListener('unhandledrejection',event=>{els.saved.textContent=`Ошибка интерфейса: ${event.reason?.message||event.reason}`});
load().catch(error=>{els.progress.textContent=`Ошибка: ${error.message}`});
