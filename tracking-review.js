const $ = (selector) => document.querySelector(selector);
const video = $("#video");
const canvas = $("#overlay");
const context = canvas.getContext("2d");

const state = {
  videoName: "",
  tracksName: "",
  observations: [],
  frames: [],
  chunks: [],
  chunkByKey: new Map(),
  scores: new Map(),
  anchors: [],
  route: [],
  pausedForGap: false,
  preview: null,
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
  const byFrame = new Map();
  state.observations.forEach((item) => {
    if (!byFrame.has(item.frame)) byFrame.set(item.frame, []);
    byFrame.get(item.frame).push(item);
  });
  state.frames = [...byFrame.values()].map((items) => ({
    time: items[0].time,
    items: deduplicate(items),
  })).sort((a, b) => a.time - b.time);
  state.chunks = chunks;
  state.chunkByKey = new Map(chunks.map((chunk) => [chunk.key, chunk]));
}

function center(item) { return [(item.x1 + item.x2) / 2, (item.y1 + item.y2) / 2]; }
function height(item) { return item.y2 - item.y1; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

function boxIou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return intersection / Math.max(1, areaA + areaB - intersection);
}

function deduplicate(items) {
  const kept = [];
  [...items].sort((a, b) => b.confidence - a.confidence).forEach((item) => {
    const duplicate = kept.some((other) => {
      const heightRatio = height(item) / Math.max(1, height(other));
      return boxIou(item, other) > .55 && heightRatio > .68 && heightRatio < 1.47
        && distance(center(item), center(other)) < Math.min(height(item), height(other)) * .42;
    });
    if (!duplicate) kept.push(item);
  });
  return kept;
}

function rebuildRoute() {
  const anchors = [...state.anchors].sort((a, b) => a.time - b.time);
  const confirmedIds = new Set(anchors.map((anchor) => anchor.trackId));
  const points = new Map();
  const putPoint = (item, source, anchorDistance) => {
    const current = points.get(item.frame);
    if (!current || anchorDistance <= current.anchorDistance) points.set(item.frame, { item, source, anchorDistance });
  };
  anchors.forEach((anchor, anchorIndex) => {
    const chunk = state.chunkByKey.get(anchor.key);
    if (!chunk) return;
    const seed = nearest(chunk.items, anchor.time, .2);
    if (!seed) return;
    const stopAt = anchors[anchorIndex + 1]?.time ?? Infinity;
    const previousAnchorTime = anchors[anchorIndex - 1]?.time ?? -Infinity;
    let last = seed, previous = null, lastSeenTime = anchor.time;
    putPoint(seed, "CONFIRMED", 0);
    {
      const earlierFrames = state.frames.filter((frame) => frame.time < anchor.time - .02 && frame.time > previousAnchorTime + .02).reverse();
      for (const frame of earlierFrames) {
        const gap = lastSeenTime - frame.time;
        if (gap > .85) break;
        const lastCenter = center(last);
        let vx = 0, vy = 0;
        if (previous && previous.time > last.time) {
          const previousCenter = center(previous), dt = previous.time - last.time;
          vx = (lastCenter[0] - previousCenter[0]) / dt;
          vy = (lastCenter[1] - previousCenter[1]) / dt;
          const speed = Math.hypot(vx, vy), maxSpeed = height(last) * 4;
          if (speed > maxSpeed) { vx *= maxSpeed / speed; vy *= maxSpeed / speed; }
        }
        const predicted = [lastCenter[0] + vx * gap, lastCenter[1] + vy * gap];
        const maxDistance = .38 + gap * 1.35;
        const ranked = frame.items.map((candidate) => {
          const normalizedDistance = distance(predicted, center(candidate)) / Math.max(height(last), height(candidate), 1);
          const sizePenalty = Math.abs(Math.log(Math.max(1, height(candidate)) / Math.max(1, height(last)))) * .28;
          const sameTrackBonus = candidate.id === last.id ? .32 : 0;
          const confirmedBonus = confirmedIds.has(candidate.id) ? .2 : 0;
          const appearanceBonus = Math.max(-.12, Math.min(.12, state.scores.get(candidate.id) || 0)) * .75;
          return { candidate, normalizedDistance, score: normalizedDistance + sizePenalty - sameTrackBonus - confirmedBonus - appearanceBonus };
        }).filter((entry) => entry.normalizedDistance <= maxDistance
          && height(entry.candidate) / Math.max(1, height(last)) > .52
          && height(entry.candidate) / Math.max(1, height(last)) < 1.9)
          .sort((a, b) => a.score - b.score);
        const chosen = ranked[0]?.candidate;
        if (chosen) {
          previous = last; last = chosen; lastSeenTime = frame.time;
          putPoint(chosen, "TRACK", anchor.time - frame.time);
        } else if (gap <= .35) {
          const bridge = { ...last, time: frame.time, frame: frame.items[0]?.frame ?? last.frame };
          const dx = vx * gap, dy = vy * gap;
          bridge.x1 += dx; bridge.x2 += dx; bridge.y1 += dy; bridge.y2 += dy;
          putPoint(bridge, "BRIDGE", anchor.time - frame.time);
        }
      }
      last = seed; previous = null; lastSeenTime = anchor.time;
    }
    const frames = state.frames.filter((frame) => frame.time > anchor.time + .02 && frame.time < stopAt - .02);
    for (const frame of frames) {
      const gap = frame.time - lastSeenTime;
      if (gap > .85) break;
      const lastCenter = center(last);
      let vx = 0, vy = 0;
      if (previous && last.time > previous.time) {
        const previousCenter = center(previous), dt = last.time - previous.time;
        vx = (lastCenter[0] - previousCenter[0]) / dt;
        vy = (lastCenter[1] - previousCenter[1]) / dt;
        const speed = Math.hypot(vx, vy), maxSpeed = height(last) * 4;
        if (speed > maxSpeed) { vx *= maxSpeed / speed; vy *= maxSpeed / speed; }
      }
      const predicted = [lastCenter[0] + vx * gap, lastCenter[1] + vy * gap];
      const maxDistance = .38 + gap * 1.35;
      const ranked = frame.items.map((candidate) => {
        const normalizedDistance = distance(predicted, center(candidate)) / Math.max(height(last), height(candidate), 1);
        const sizePenalty = Math.abs(Math.log(Math.max(1, height(candidate)) / Math.max(1, height(last)))) * .28;
        const sameTrackBonus = candidate.id === last.id ? .32 : 0;
        const confirmedBonus = confirmedIds.has(candidate.id) ? .2 : 0;
        const appearanceBonus = Math.max(-.12, Math.min(.12, state.scores.get(candidate.id) || 0)) * .75;
        return { candidate, normalizedDistance, score: normalizedDistance + sizePenalty - sameTrackBonus - confirmedBonus - appearanceBonus };
      }).filter((entry) => entry.normalizedDistance <= maxDistance
        && height(entry.candidate) / Math.max(1, height(last)) > .52
        && height(entry.candidate) / Math.max(1, height(last)) < 1.9)
        .sort((a, b) => a.score - b.score);
      const chosen = ranked[0]?.candidate;
      if (chosen) {
        previous = last; last = chosen; lastSeenTime = frame.time;
        putPoint(chosen, "TRACK", frame.time - anchor.time);
      } else if (gap <= .35) {
        const bridge = { ...last, time: frame.time, frame: frame.items[0]?.frame ?? last.frame };
        const dx = vx * gap, dy = vy * gap;
        bridge.x1 += dx; bridge.x2 += dx; bridge.y1 += dy; bridge.y2 += dy;
        putPoint(bridge, "BRIDGE", frame.time - anchor.time);
      }
    }
  });
  state.route = [...points.values()].sort((a, b) => a.item.time - b.item.time);
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
  let best = null;
  state.route.forEach((entry) => {
    const delta = Math.abs(entry.item.time - time);
    if (delta <= .07 && (!best || delta < best.delta)) best = { ...entry, delta };
  });
  return best ? { item: best.item, source: best.source } : null;
}

function observationsAt(time) {
  let best = null;
  state.frames.forEach((frame) => {
    const delta = Math.abs(frame.time - time);
    if (delta <= .055 && (!best || delta < best.delta)) best = { items: frame.items, delta };
  });
  return best?.items || [];
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
  if (state.preview && Math.abs(state.preview.time - video.currentTime) <= .15) {
    drawBox(state.preview.item, "#b584ff", 5,
      `Кандидат ${state.preview.index + 1}/${state.preview.candidates.length} · кликни ещё для перебора`);
    setStatus("bridge", "Рамка ещё не подтверждена", "Повторный клик перебирает рамки; затем нажми кнопку подтверждения.");
  }
  $("#timecode").textContent = formatTime(video.currentTime);
  $("#timeline").value = video.currentTime;
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

function storageKey() { return `volleytag-tracking-v3:${state.videoName}:${state.tracksName}`; }
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
  const ordered = state.anchors.map((anchor, originalIndex) => ({ ...anchor, originalIndex })).sort((a, b) => a.time - b.time);
  list.innerHTML = ordered.map((anchor) =>
    `<div class="anchor"><time data-time="${anchor.time}">${formatTime(anchor.time)}</time><div><b>CONFIRMED</b><small>${anchor.key}</small></div><button data-delete="${anchor.originalIndex}">×</button></div>`
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
  candidates.sort((a, b) => distance(center(a), [x, y]) - distance(center(b), [x, y]));
  if (!candidates.length) { toast("Здесь нет найденной рамки"); return; }
  const sameClick = state.preview && Math.abs(state.preview.time - video.currentTime) <= .15
    && Math.hypot(state.preview.x - x, state.preview.y - y) < 40;
  const index = sameClick ? (state.preview.index + 1) % candidates.length : 0;
  state.preview = { time: video.currentTime, x, y, candidates, index, item: candidates[index] };
  $("#confirmBox").disabled = false;
  $("#stageConfirm").hidden = false;
  video.pause();
  toast(candidates.length > 1 ? `Здесь ${candidates.length} рамки — повторный клик перебирает` : "Проверь фиолетовую рамку и подтверди");
});

function confirmPreview() {
  if (!state.preview) return;
  const { item, time } = state.preview;
  state.anchors = state.anchors.filter((anchor) => Math.abs(anchor.time - time) > .35);
  state.anchors.push({ time, key: item.key, trackId: item.id });
  state.preview = null; $("#confirmBox").disabled = true; $("#stageConfirm").hidden = true;
  rebuildRoute(); saveAnchors(); toast(`Подтверждено: ${item.key}`);
}

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
$("#timeline").addEventListener("input", (event) => { video.currentTime = Number(event.target.value); state.preview = null; $("#confirmBox").disabled = true; $("#stageConfirm").hidden = true; });
$("#confirmBox").addEventListener("click", confirmPreview);
$("#stageConfirm").addEventListener("click", confirmPreview);
$("#fullscreen").addEventListener("click", () => { $("#stage").requestFullscreen?.(); });
$("#undoAnchor").addEventListener("click", () => { state.anchors.pop(); rebuildRoute(); saveAnchors(); });
$("#clear").addEventListener("click", () => { state.anchors = []; rebuildRoute(); saveAnchors(); });
$("#export").addEventListener("click", () => {
  const payload = { version: 1, video: state.videoName, tracks: state.tracksName, roi: [Number($("#roiX").value), Number($("#roiY").value)], anchors: state.anchors };
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })); link.download = "player-51-anchors.json"; link.click(); URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.target.matches("input")) { event.preventDefault(); if (video.paused) video.play(); else video.pause(); }
  if (event.code === "ArrowLeft") video.currentTime = Math.max(0, video.currentTime - 5);
  if (event.code === "ArrowRight") video.currentTime += 5;
  if (event.code === "Enter" && state.preview) confirmPreview();
});
video.addEventListener("play", () => { $("#play").textContent = "❚❚ Пауза"; });
video.addEventListener("pause", () => { $("#play").textContent = "▶ Смотреть"; });
video.addEventListener("loadedmetadata", () => {
  $("#timeline").max = video.duration || 1;
  const start = Number(new URLSearchParams(location.search).get("start") || 0);
  if (start > 0 && start < video.duration) video.currentTime = start;
});
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
