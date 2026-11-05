const $ = (selector) => document.querySelector(selector);
const repoUrl = (path) => /^(?:[a-z]+:|\/)/i.test(path) ? path : `/${path}`;
const video = $("#video");
const playlist = [
  { id:"training-set1", name:"Cloud · партия 1", video:"vision/runs/training-set1/source.mp4", training:"vision/runs/training-set1/player-51-training-data.json", split:"train", completed:true },
  { id:"training-set2", name:"Cloud · партия 2", video:"vision/runs/training-set2/source.mp4", training:"vision/runs/training-set2/player-51-training-data.json", split:"train", completed:true },
  { id:"training-old-straight-set1", name:"Матч 2 · партия 1 · прямая камера", video:"vision/runs/training-old-straight/set1.mp4", training:"vision/runs/training-old-straight/player-51-training-data.json", split:"train", metadata_offset_s:0, completed:true },
  { id:"training-straight-far2", name:"Матч 3 · ГАЛС · партия 2 · дальняя", video:"vision/runs/training-straight-far2/far-source.mp4", training:"vision/runs/training-straight-far2/player-51-training-data.json", split:"train", completed:true },
  { id:"training-straight-far3", name:"Матч 4 · партия 2 · дальняя", video:"vision/runs/training-straight-far3/far-source.mp4", training:"vision/runs/training-straight-far3/player-51-training-data.json", split:"train", completed:true },
  { id:"set3-unseen", name:"Матч 1 · партия 3 · уже размечена (10) · FROZEN", video:"vision/runs/set3-unseen/source.mp4", split:"frozen_eval", completed:true },
  // Same GALS set as source 3, only with a 100-second preamble. Keep the
  // stable index for old saved data but never offer this duplicate for review.
  { id:"training-straight-far2-duplicate", name:"ДУБЛИКАТ · ГАЛС · партия 2", video:"vision/runs/training-straight-far2/source.mp4", exclude:"vision/runs/training-straight-far2/non-match-intervals.json", split:"excluded_duplicate", hidden:true },
  { id:"training-straight-far3-near", name:"Матч 4 · партия 1 · ближняя", video:"vision/runs/training-straight-far3/original.mp4", start_s:110, end_s:1000, split:"train" },
  // Danya did not play this set. It is excluded in the source annotations as
  // player_benched and must not appear in action or identity training.
  { id:"training-old-straight-set2", name:"НЕ УЧАСТВУЕТ · матч 2 · партия 2", video:"vision/runs/training-old-straight/set2.mp4", training:"vision/runs/training-old-straight/player-51-training-data.json", split:"excluded_player_benched", metadata_offset_s:847.098, hidden:true },
  { id:"training-old-straight-set3", name:"Матч 2 · партия 3 · прямая камера", video:"vision/runs/training-old-straight/set3.mp4", training:"vision/runs/training-old-straight/player-51-training-data.json", split:"train", metadata_offset_s:1796.185 },
  { id:"training-keaz-set1", name:"КЭАЗ — X5 Tech · партия 1 · ближняя", video:"vision/runs/training-keaz/source.mp4", exclude:"vision/runs/training-keaz/non-match-intervals.json", start_s:2.048, end_s:852.048, split:"train" },
  { id:"training-keaz-set2", name:"КЭАЗ — X5 Tech · партия 2 · дальняя", video:"vision/runs/training-keaz/source.mp4", exclude:"vision/runs/training-keaz/non-match-intervals.json", start_s:912.048, end_s:2112.048, split:"train" },
];
const actionNames = { attack:"атака", serve:"подача", reception:"приём подачи", block:"блок", defense:"защита", set:"передача / пас", other:"другое движение" };
const params = new URLSearchParams(location.search);
const defaultResultPath = "vision/runs/danya-detector-v8-action/action-events.json";
const resultPath = params.get("result") || defaultResultPath;
const allowedSources = params.get("sources")
  ? new Set(params.get("sources").split(",").map(Number).filter(Number.isFinite)) : null;
const state = { source:0, events:[], excluded:[], saving:false, progress:{} };

function resumeKey(item) { return `action-review:${resultPath}:${item.id}`; }

function clock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60), rest = value - minutes * 60;
  return `${String(minutes).padStart(2,"0")}:${rest.toFixed(1).padStart(4,"0")}`;
}

function eventId(source, time) { return `${playlist[source].id}-${time.toFixed(3)}-${Date.now().toString(36)}`; }

async function save() {
  state.saving = true; $("#saveState").textContent = "Сохраняю…";
  const payload = {
    auditPath: resultPath,
    schema_version:1,
    kind:"volleytag.action-events",
    window:{ before_s:1.5, after_s:3.0 },
    playlist:playlist.map(({id,name,video}) => ({id,name,video})),
    events:state.events,
    review_progress:state.progress,
    updated_at:new Date().toISOString(),
  };
  try {
    const response = await fetch("/api/save-dataset-audit", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
    if (!response.ok) throw new Error((await response.json()).error || "save failed");
    $("#saveState").textContent = "Сохранено ✓";
  } catch (error) {
    $("#saveState").textContent = `Ошибка сохранения: ${error.message}`;
  } finally { state.saving = false; }
}

function renderEvents() {
  $("#eventCount").textContent = `${state.events.length} ${state.events.length === 1 ? "событие" : "событий"}`;
  const recent = state.events.slice(-10).reverse();
  $("#events").innerHTML = recent.length ? recent.map((event) =>
    `<li><b>${playlist[event.source]?.name || event.video}</b><br>${clock(event.time_s)} · ${actionNames[event.action] || "действие"} · окно ${clock(event.start_s)}–${clock(event.end_s)}</li>`
  ).join("") : '<li class="empty">Пока меток нет</li>';
  refreshSourceOptions();
}

function sourceEventCount(index) { return state.events.filter((event) => Number(event.source) === Number(index)).length; }

function refreshSourceOptions() {
  const selected = String(state.source);
  $("#source").innerHTML = "";
  playlist.forEach((item,index) => {
    if (item.hidden || (allowedSources && !allowedSources.has(index))) return;
    const progress = state.progress[item.id] || {};
    const done = Boolean(progress.completed ?? (resultPath === defaultResultPath && item.completed));
    const count = sourceEventCount(index);
    const status = done ? "✓ готово" : count ? `• в процессе · ${count}` : "○ не начато";
    $("#source").insertAdjacentHTML("beforeend", `<option value="${index}">${status} · ${item.name}</option>`);
  });
  $("#source").value = selected;
  const item = playlist[state.source], progress = state.progress[item?.id] || {};
  const done = Boolean(progress.completed ?? (resultPath === defaultResultPath && item?.completed));
  const count = sourceEventCount(state.source);
  $("#sourceProgress").textContent = `Метки: ${count}${progress.max_time_s ? ` · просмотрено до ${clock(progress.max_time_s)}` : ""}`;
  $("#completeSource").classList.toggle("done", done);
  $("#completeSource").textContent = done ? "✓ Партия размечена · отменить статус" : "✓ Отметить партию размеченной";
}

async function loadExisting() {
  try {
    const response = await fetch(repoUrl(`${resultPath}?v=${Date.now()}`));
    if (response.ok) {
      const payload = await response.json();
      state.events = payload.events || [];
      state.progress = payload.review_progress || {};
    }
  } catch (_) {}
  renderEvents();
}

async function selectSource(index) {
  state.source = Number(index);
  const item = playlist[state.source];
  video.pause();
  video.removeAttribute("src");
  video.load();
  state.excluded = [];
  try {
    const metadata = item.training || item.exclude;
    const response = metadata ? await fetch(repoUrl(`${metadata}?v=${Date.now()}`)) : null;
    if (response?.ok) {
      const offset = Number(item.metadata_offset_s || 0);
      state.excluded = ((await response.json()).excluded_intervals || []).map((interval) => ({
        ...interval,
        start_s:Number(interval.start_s ?? interval.start) - offset,
        end_s:Number(interval.end_s ?? interval.end) - offset,
      })).filter((interval) => interval.end_s > 0);
    }
  } catch (_) {}
  // Use an origin-absolute URL and explicitly reload the media element. Yandex
  // Browser can otherwise retain MEDIA_ERR_SRC_NOT_SUPPORTED after switching
  // a large range-served MP4 in the same element.
  video.src = new URL(`/${item.video}`, location.origin).href;
  video.load();
  $("#source").value = String(state.source);
  refreshSourceOptions();
  const legacyResume = resultPath === defaultResultPath ? localStorage.getItem(`action-review:${item.id}`) : null;
  const resume = Number(localStorage.getItem(resumeKey(item)) || legacyResume || item.start_s || 0);
  video.addEventListener("loadedmetadata", () => {
    video.playbackRate = Number($("#speed").value);
    const start = Number(item.start_s || 0), end = Math.min(Number(item.end_s || video.duration), video.duration);
    video.currentTime = Math.max(start, Math.min(resume, Math.max(start, end - .2)));
    $("#timeline").min = start; $("#timeline").max = end;
    $("#saveState").textContent = "Видео готово";
  }, {once:true});
}

function flash(selector) {
  const node = $(selector); node.classList.add("show");
  clearTimeout(node._timer); node._timer = setTimeout(() => node.classList.remove("show"), 220);
}

function mark(action = "other") {
  if (!Number.isFinite(video.currentTime)) return;
  const time = video.currentTime;
  const rangeEnd = Number(playlist[state.source].end_s || video.duration || Infinity);
  const event = { id:eventId(state.source,time), source:state.source, video:playlist[state.source].video,
    time_s:Number(time.toFixed(3)), start_s:Number(Math.max(0,time-1.5).toFixed(3)),
    end_s:Number(Math.min(video.duration || Infinity,rangeEnd,time+3).toFixed(3)), label:"action_candidate", action,
    result:null, split:playlist[state.source].split || "train",
    created_at:new Date().toISOString() };
  if (state.events.some((other) => other.source === state.source && Math.abs(other.time_s-time) < .75)) {
    $("#saveState").textContent = "Рядом уже есть метка"; return;
  }
  state.events.push(event);
  const key = playlist[state.source].id;
  state.progress[key] = { ...(state.progress[key] || {}), max_time_s:Math.max(Number(state.progress[key]?.max_time_s || 0), time), updated_at:new Date().toISOString() };
  renderEvents(); flash("#flash"); save();
}

function undo() {
  for (let index=state.events.length-1; index>=0; index-=1) {
    if (state.events[index].source === state.source) { state.events.splice(index,1); renderEvents(); save(); return; }
  }
}

function seek(delta) {
  const item = playlist[state.source], start = Number(item.start_s || 0);
  const end = Math.min(Number(item.end_s || video.duration || Infinity), video.duration || Infinity);
  video.currentTime = Math.max(start, Math.min(end, video.currentTime + delta));
}
function togglePlay() { video.paused ? video.play() : video.pause(); }

$("#source").addEventListener("change", (event) => selectSource(event.target.value));
$("#completeSource").addEventListener("click", () => {
  const item = playlist[state.source], current = state.progress[item.id] || {};
  const done = Boolean(current.completed ?? (resultPath === defaultResultPath && item.completed));
  state.progress[item.id] = { ...current, completed:!done, completed_at:!done ? new Date().toISOString() : null };
  refreshSourceOptions(); save();
});
$("#speed").addEventListener("change", () => { video.playbackRate = Number($("#speed").value); });
$("#play").addEventListener("click", togglePlay); $("#mark").addEventListener("click", () => mark("other"));
document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => mark(button.dataset.action)));
$("#undo").addEventListener("click", undo); $("#back").addEventListener("click", () => seek(-5));
$("#forward").addEventListener("click", () => seek(5));
$("#timeline").addEventListener("input", (event) => { video.currentTime = Number(event.target.value); });
video.addEventListener("play", () => $("#play").textContent = "⏸ Пауза");
video.addEventListener("pause", () => $("#play").textContent = "▶ Смотреть");
video.addEventListener("error", () => {
  const messages = { 1:"загрузка отменена", 2:"ошибка сети", 3:"ошибка декодирования", 4:"формат не поддержан" };
  $("#saveState").textContent = `Ошибка видео: ${messages[video.error?.code] || "неизвестная"}`;
});
video.addEventListener("ratechange", () => { if (Number($("#speed").value) !== video.playbackRate) video.playbackRate = Number($("#speed").value); });
video.addEventListener("timeupdate", () => {
  $("#timecode").textContent = clock(video.currentTime);
  const item = playlist[state.source], end = Math.min(Number(item.end_s || video.duration || 1), video.duration || Infinity);
  $("#timeline").max = end; $("#timeline").value = video.currentTime;
  localStorage.setItem(resumeKey(playlist[state.source]), String(video.currentTime));
  const key = item.id, previous = Number(state.progress[key]?.max_time_s || 0);
  if (video.currentTime > previous) state.progress[key] = { ...(state.progress[key] || {}), max_time_s:Number(video.currentTime.toFixed(2)) };
  if (Number.isFinite(end) && video.currentTime >= end) {
    state.progress[key] = { ...(state.progress[key] || {}), completed:true, completed_at:new Date().toISOString() };
    video.pause(); video.currentTime = Math.max(Number(item.start_s || 0), end - .05); refreshSourceOptions(); save();
  }
  const interval = state.excluded.find((item) => video.currentTime >= Number(item.start_s ?? item.start) && video.currentTime < Number(item.end_s ?? item.end));
  if (interval) { video.currentTime = Math.min(video.duration, Number(interval.end_s ?? interval.end) + .05); flash("#skip"); }
});
document.addEventListener("keydown", (event) => {
  if (["INPUT","SELECT","TEXTAREA"].includes(event.target.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); togglePlay(); }
  else if (event.code === "Digit0") { event.preventDefault(); mark("other"); }
  else if (/^Digit[1-6]$/.test(event.code)) {
    event.preventDefault(); mark(["attack","serve","reception","block","defense","set"][Number(event.code.at(-1)) - 1]);
  }
  else if (event.code === "Backspace") { event.preventDefault(); undo(); }
  else if (event.code === "ArrowLeft") { event.preventDefault(); seek(-3); }
  else if (event.code === "ArrowRight") { event.preventDefault(); seek(3); }
  else if (event.key.toLowerCase() === "f") $("#videoWrap").requestFullscreen?.();
  else if (event.key.toLowerCase() === "p") togglePlay();
});

loadExisting().then(() => {
  const requested = Number(params.get("source"));
  const fallback = playlist.findIndex((item, index) => !item.hidden && (!allowedSources || allowedSources.has(index)));
  selectSource(Number.isFinite(requested) && (!allowedSources || allowedSources.has(requested)) ? requested : fallback);
});
