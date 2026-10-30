const $ = (selector) => document.querySelector(selector);
const video = $("#video");
const canvas = $("#overlay");
const context = canvas.getContext("2d");

const state = {
  videoName: "",
  tracksName: "",
  observations: [],
  chunks: [],
  chunkByKey: new Map(),
  scores: new Map(),
  anchors: [],
  route: [],
  pausedForGap: false,
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function buildChunks(rows) {
  const grouped = new Map();
  rows.forEach((row) => {
    const item = {
      time: Number(row.time_s), frame: Number(row.frame), id: Number(row.track_id),
      confidence: Number(row.confidence),
      x1: Number(row.x1), y1: Number(row.y1), x2: Number(row.x2), y2: Number(row.y2),
    };
    if (!grouped.has(item.id)) grouped.set(item.id, []);
    grouped.get(item.id).push(item);
  });
  const chunks = [];
  grouped.forEach((items, id) => {
    items.sort((a, b) => a.time - b.time);
    let part = [];
    let chunkIndex = 0;
    const save = () => {
      if (!part.length) return;
      const key = `${id}:${chunkIndex++}`;
      part.forEach((item) => { item.key = key; });
      chunks.push({ key, id, items: part, start: part[0].time, end: part.at(-1).time });
    };
    items.forEach((item) => {
      if (part.length && item.time - part.at(-1).time > .25) {
        save();
        part = [];
      }
      part.push(item);
    });
    save();
  });
  chunks.sort((a, b) => a.start - b.start);
  state.observations = chunks.flatMap((chunk) => chunk.items).sort((a, b) => a.time - b.time);
  state.chunks = chunks;
  state.chunkByKey = new Map(chunks.map((chunk) => [chunk.key, chunk]));
}

function center(item) { return [(item.x1 + item.x2) / 2, (item.y1 + item.y2) / 2]; }
function height(item) { return item.y2 - item.y1; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

function rebuildRoute() {
  const anchors = [...state.anchors].sort((a, b) => a.time - b.time);
  const selected = new Map();
  anchors.forEach((anchor) => {
    const chunk = state.chunkByKey.get(anchor.key);
    if (chunk) selected.set(chunk.key, { chunk, source: "CONFIRMED" });
  });

  anchors.forEach((anchor, anchorIndex) => {
    let current = state.chunkByKey.get(anchor.key);
    const stopAt = anchors[anchorIndex + 1]?.time ?? Infinity;
    const visited = new Set();
    while (current && current.end < stopAt && !visited.has(current.key)) {
      visited.add(current.key);
      const end = current.items.at(-1);
      const candidates = state.chunks.filter((candidate) => {
        if (selected.has(candidate.key) || candidate.start >= stopAt) return false;
        const gap = candidate.start - current.end;
        if (gap < -.05 || gap > .8) return false;
        const margin = state.scores.get(candidate.id);
        if (state.scores.size && (margin === undefined || margin < 0)) return false;
        const first = candidate.items[0];
        return distance(center(end), center(first)) / Math.max(height(end), height(first), 1) <= 1.25;
      });
      candidates.sort((left, right) => {
        const score = (candidate) => {
          const first = candidate.items[0];
          const gap = Math.max(0, candidate.start - current.end);
          const geometry = distance(center(end), center(first)) / Math.max(height(end), height(first), 1);
          return geometry + gap * .35 - Math.max(-.1, Math.min(.1, state.scores.get(candidate.id) || 0)) * 1.5;
        };
        return score(left) - score(right);
      });
      current = candidates[0];
      if (current) selected.set(current.key, { chunk: current, source: "TRACK" });
    }
  });
  state.route = [...selected.values()].sort((a, b) => a.chunk.start - b.chunk.start);
}

function nearest(items, time, maxDelta = .09) {
  let best = null;
  items.forEach((item) => {
    const delta = Math.abs(item.time - time);
    if (delta <= maxDelta && (!best || delta < best.delta)) best = { item, delta };
  });
  return best?.item || null;
}

function selectionAt(time) {
  for (const entry of state.route) {
    if (time >= entry.chunk.start - .06 && time <= entry.chunk.end + .06) {
      const item = nearest(entry.chunk.items, time);
      if (item) return { item, source: entry.source };
    }
  }
  for (let index = 0; index < state.route.length - 1; index += 1) {
    const left = state.route[index].chunk;
    const right = state.route[index + 1].chunk;
    const gap = right.start - left.end;
    if (gap <= 0 || gap > .8 || time <= left.end || time >= right.start) continue;
    const a = left.items.at(-1), b = right.items[0];
    if (distance(center(a), center(b)) / Math.max(height(a), height(b), 1) > 1.25) continue;
    const ratio = (time - left.end) / gap;
    const item = { time, key: right.key, id: right.id };
    ["x1", "y1", "x2", "y2"].forEach((key) => { item[key] = a[key] + (b[key] - a[key]) * ratio; });
    return { item, source: "BRIDGE" };
  }
  return null;
}

function observationsAt(time) {
  return state.observations.filter((item) => Math.abs(item.time - time) <= .055);
}

function geometry() {
  const rect = video.getBoundingClientRect();
  const vw = video.videoWidth || 1920, vh = video.videoHeight || 1080;
  const scale = Math.min(rect.width / vw, rect.height / vh);
  return { rect, scale, left: (rect.width - vw * scale) / 2, top: (rect.height - vh * scale) / 2, vw, vh };
}

function drawBox(item, color, width, label) {
  const g = geometry();
  const rx = Number($("#roiX").value) || 0, ry = Number($("#roiY").value) || 0;
  const x = g.left + (item.x1 + rx) * g.scale, y = g.top + (item.y1 + ry) * g.scale;
  const w = (item.x2 - item.x1) * g.scale, h = (item.y2 - item.y1) * g.scale;
  context.strokeStyle = color; context.lineWidth = width; context.strokeRect(x, y, w, h);
  if (label) {
    context.font = "700 13px system-ui";
    const textWidth = context.measureText(label).width + 12;
    context.fillStyle = color; context.fillRect(x, Math.max(0, y - 23), textWidth, 23);
    context.fillStyle = "#0d0f17"; context.fillText(label, x + 6, Math.max(16, y - 7));
  }
}

function render() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * ratio) || canvas.height !== Math.round(rect.height * ratio)) {
    canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, rect.width, rect.height);
  observationsAt(video.currentTime).forEach((item) => drawBox(item, "rgba(200,210,235,.32)", 1, ""));
  const selected = selectionAt(video.currentTime);
  if (selected) {
    const color = selected.source === "BRIDGE" ? "#ffd84a" : "#b7ff36";
    drawBox(selected.item, color, 4, `Даня #51 · ${selected.source}`);
    setStatus(selected.source === "BRIDGE" ? "bridge" : "ok", selected.source === "BRIDGE" ? "Короткий разрыв" : "Игрок сопровождается", `${selected.source} · track ${selected.item.id}`);
    state.pausedForGap = false;
  } else if (state.observations.length) {
    setStatus("lost", "Игрок потерян", "Кликни по своей рамке, чтобы продолжить.");
    if (!video.paused && $("#pauseOnLost").checked && !state.pausedForGap) {
      state.pausedForGap = true; video.pause();
    }
  }
  $("#timecode").textContent = formatTime(video.currentTime);
  requestAnimationFrame(render);
}

function setStatus(kind, title, detail) {
  const status = $("#status"); status.className = `status ${kind}`;
  status.querySelector("b").textContent = title; status.querySelector("span").textContent = detail;
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60), rest = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
}

function storageKey() { return `volleytag-tracking:${state.videoName}:${state.tracksName}`; }
function saveAnchors() {
  if (state.videoName && state.tracksName) localStorage.setItem(storageKey(), JSON.stringify(state.anchors));
  renderAnchors();
}
function loadAnchors() {
  try { state.anchors = JSON.parse(localStorage.getItem(storageKey()) || "[]"); } catch { state.anchors = []; }
  state.anchors = state.anchors.filter((anchor) => state.chunkByKey.has(anchor.key));
  rebuildRoute(); renderAnchors();
}
function renderAnchors() {
  $("#saveState").textContent = `Якорей: ${state.anchors.length}`;
  const list = $("#anchorList");
  if (!state.anchors.length) { list.innerHTML = "<p>Пока пусто.</p>"; return; }
  list.innerHTML = [...state.anchors].sort((a, b) => a.time - b.time).map((anchor, index) =>
    `<div class="anchor"><time data-time="${anchor.time}">${formatTime(anchor.time)}</time><div><b>CONFIRMED</b><small>${anchor.key}</small></div><button data-delete="${index}">×</button></div>`
  ).join("");
  list.querySelectorAll("time").forEach((element) => element.addEventListener("click", () => { video.currentTime = Number(element.dataset.time); }));
  list.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    state.anchors.splice(Number(button.dataset.delete), 1); rebuildRoute(); saveAnchors();
  }));
}

function toast(text) { const node = $("#toast"); node.textContent = text; node.classList.add("show"); setTimeout(() => node.classList.remove("show"), 1800); }

canvas.addEventListener("click", (event) => {
  if (!state.observations.length || !video.videoWidth) return;
  const g = geometry(), rx = Number($("#roiX").value) || 0, ry = Number($("#roiY").value) || 0;
  const x = (event.offsetX - g.left) / g.scale - rx, y = (event.offsetY - g.top) / g.scale - ry;
  const candidates = observationsAt(video.currentTime).filter((item) => x >= item.x1 && x <= item.x2 && y >= item.y1 && y <= item.y2);
  candidates.sort((a, b) => (a.x2 - a.x1) * (a.y2 - a.y1) - (b.x2 - b.x1) * (b.y2 - b.y1));
  if (!candidates.length) { toast("Здесь нет найденной рамки"); return; }
  const item = candidates[0];
  state.anchors = state.anchors.filter((anchor) => Math.abs(anchor.time - video.currentTime) > .35);
  state.anchors.push({ time: video.currentTime, key: item.key, trackId: item.id });
  rebuildRoute(); saveAnchors(); toast(`Подтверждено: ${item.key}`);
});

$("#videoFile").addEventListener("change", (event) => {
  const file = event.target.files[0]; if (!file) return;
  state.videoName = `${file.name}:${file.size}:${file.lastModified}`;
  video.src = URL.createObjectURL(file); $("#empty").hidden = true;
  if (state.tracksName) loadAnchors();
});
$("#tracksFile").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  state.tracksName = `${file.name}:${file.size}:${file.lastModified}`;
  buildChunks(parseCsv(await file.text())); loadAnchors();
  setStatus("lost", "Рамки загружены", "Запусти видео и кликни по себе.");
});
$("#scoresFile").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  state.scores = new Map(parseCsv(await file.text()).map((row) => [Number(row.track_id), Number(row.margin)]));
  rebuildRoute(); toast("ReID-оценки загружены");
});
$("#play").addEventListener("click", () => { if (video.paused) video.play(); else video.pause(); });
document.querySelectorAll("[data-seek]").forEach((button) => button.addEventListener("click", () => { video.currentTime = Math.max(0, video.currentTime + Number(button.dataset.seek)); }));
$("#clear").addEventListener("click", () => { state.anchors = []; rebuildRoute(); saveAnchors(); });
$("#export").addEventListener("click", () => {
  const payload = { version: 1, video: state.videoName, tracks: state.tracksName, roi: [Number($("#roiX").value), Number($("#roiY").value)], anchors: state.anchors };
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })); link.download = "player-51-anchors.json"; link.click(); URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.target.matches("input")) { event.preventDefault(); if (video.paused) video.play(); else video.pause(); }
  if (event.code === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - 5);
  if (event.code === "ArrowRight") video.currentTime += 5;
});
video.addEventListener("play", () => { $("#play").textContent = "❚❚ Пауза"; });
video.addEventListener("pause", () => { $("#play").textContent = "▶ Смотреть"; });
window.addEventListener("resize", () => { canvas.width = 0; });

async function loadQueryFiles() {
  const params = new URLSearchParams(location.search);
  const videoUrl = params.get("video"), tracksUrl = params.get("tracks"), scoresUrl = params.get("scores");
  if (videoUrl) {
    state.videoName = videoUrl;
    video.src = videoUrl;
    $("#empty").hidden = true;
  }
  if (tracksUrl) {
    state.tracksName = tracksUrl;
    const response = await fetch(tracksUrl);
    if (!response.ok) throw new Error(`Не удалось загрузить ${tracksUrl}`);
    buildChunks(parseCsv(await response.text()));
  }
  if (scoresUrl) {
    const response = await fetch(scoresUrl);
    if (!response.ok) throw new Error(`Не удалось загрузить ${scoresUrl}`);
    state.scores = new Map(parseCsv(await response.text()).map((row) => [Number(row.track_id), Number(row.margin)]));
  }
  if (tracksUrl) {
    loadAnchors();
    setStatus("lost", "Рамки загружены", "Запусти видео и кликни по себе.");
  }
}

loadQueryFiles().catch((error) => {
  setStatus("lost", "Ошибка загрузки", error.message);
});
requestAnimationFrame(render);
