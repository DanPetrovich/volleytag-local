const $ = (selector) => document.querySelector(selector);
const video = $("#video");
const videoFrame = $("#videoFrame");
const frameContext = videoFrame.getContext("2d", { alpha: false });
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
  chunkScores: new Map(),
  anchors: [],
  negatives: [],
  segments: [],
  lostEvents: [],
  activeSegmentIndex: null,
  route: [],
  pausedForGap: false,
  lostSince: null,
  preview: null,
  renderedVideoTime: -1,
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
function appearanceScore(item) { return state.chunkScores.get(item.key) ?? state.scores.get(item.id) ?? 0; }

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
  // A tracker ID/chunk can contain the right player before or after an ID switch.
  // Human "not me" feedback therefore excludes only a short temporal neighbourhood,
  // rather than blacklisting that key for the entire match.
  const isNegativeHere = (candidate) => state.negatives.some((negative) =>
    negative.key === candidate.key && Math.abs(negative.time - candidate.time) <= 1.25);
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
        const ranked = frame.items.filter((candidate) => !isNegativeHere(candidate)).map((candidate) => {
          const normalizedDistance = distance(predicted, center(candidate)) / Math.max(height(last), height(candidate), 1);
          const sizePenalty = Math.abs(Math.log(Math.max(1, height(candidate)) / Math.max(1, height(last)))) * .28;
          const sameTrackBonus = candidate.id === last.id ? .32 : 0;
          const confirmedBonus = confirmedIds.has(candidate.id) ? .2 : 0;
          const appearanceBonus = Math.max(-.12, Math.min(.12, appearanceScore(candidate))) * .75;
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
      const ranked = frame.items.filter((candidate) => !isNegativeHere(candidate)).map((candidate) => {
        const normalizedDistance = distance(predicted, center(candidate)) / Math.max(height(last), height(candidate), 1);
        const sizePenalty = Math.abs(Math.log(Math.max(1, height(candidate)) / Math.max(1, height(last)))) * .28;
        const sameTrackBonus = candidate.id === last.id ? .32 : 0;
        const confirmedBonus = confirmedIds.has(candidate.id) ? .2 : 0;
        const appearanceBonus = Math.max(-.12, Math.min(.12, appearanceScore(candidate))) * .75;
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
  if (videoFrame.width !== Math.round(rect.width * ratio) || videoFrame.height !== Math.round(rect.height * ratio)) {
    videoFrame.width = Math.round(rect.width * ratio); videoFrame.height = Math.round(rect.height * ratio);
    state.renderedVideoTime = -1;
  }
  if (video.readyState >= 2 && (state.renderedVideoTime !== video.currentTime || !video.paused)) {
    const vw = video.videoWidth || 1920, vh = video.videoHeight || 1080;
    const scale = Math.min(videoFrame.width / vw, videoFrame.height / vh);
    const width = vw * scale, height = vh * scale;
    frameContext.fillStyle = "#050608"; frameContext.fillRect(0, 0, videoFrame.width, videoFrame.height);
    frameContext.drawImage(video, (videoFrame.width - width) / 2, (videoFrame.height - height) / 2, width, height);
    state.renderedVideoTime = video.currentTime;
  }
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
    state.lostSince = null;
  } else if (state.observations.length) {
    if (state.lostSince === null) state.lostSince = video.currentTime;
    const lostFor = Math.max(0, video.currentTime - state.lostSince);
    setStatus(lostFor < 1.0 ? "bridge" : "lost",
      lostFor < 1.0 ? "Короткий разрыв" : "Игрок потерян",
      lostFor < 1.0 ? "Жду следующую рамку, видео не останавливаю." : "Кликни по своей рамке, чтобы продолжить.");
    if (!video.paused && $("#pauseOnLost").checked && !state.pausedForGap && lostFor >= 1.0) {
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

function domainFor(item) {
  const manual = $("#domainMode").value;
  if (manual !== "auto") return manual;
  return item.y2 < 470 ? "far_front" : "near_back";
}

function domainLabel(domain) {
  return domain === "near_back" ? "ближняя · спина" : "дальняя · лицо";
}

function closeActiveSegment(end, reason) {
  const index = state.activeSegmentIndex;
  if (index === null || !state.segments[index]) return;
  state.segments[index].end = Math.max(state.segments[index].start, end);
  state.segments[index].endReason = reason;
  state.activeSegmentIndex = null;
}

function startPositiveSegment(item, time) {
  closeActiveSegment(time, "new-positive");
  const domain = domainFor(item);
  state.segments.push({
    label: "positive", start: time, end: null, key: item.key,
    trackId: item.id, domain, source: "human-confirmed",
  });
  state.activeSegmentIndex = state.segments.length - 1;
  return domain;
}

function seekTo(time) {
  closeActiveSegment(video.currentTime, "seek");
  video.currentTime = Math.max(0, Math.min(video.duration || Infinity, time));
  state.lostSince = null; state.pausedForGap = false;
  state.preview = null; $("#confirmBox").disabled = true; $("#stageConfirm").hidden = true;
  saveAnchors();
}

function storageKey() { return `volleytag-tracking-v3:${state.videoName}:${state.tracksName}`; }
function saveAnchors() {
  if (state.videoName && state.tracksName) localStorage.setItem(storageKey(), JSON.stringify({
    version: 3, anchors: state.anchors, negatives: state.negatives,
    segments: state.segments, lostEvents: state.lostEvents,
    activeSegmentIndex: state.activeSegmentIndex,
  }));
  renderAnchors();
}
function loadAnchors() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey()) || "[]");
    state.anchors = Array.isArray(saved) ? saved : (saved.anchors || []);
    state.negatives = Array.isArray(saved) ? [] : (saved.negatives || []);
    state.segments = Array.isArray(saved) ? [] : (saved.segments || []);
    state.lostEvents = Array.isArray(saved) ? [] : (saved.lostEvents || []);
    state.activeSegmentIndex = Array.isArray(saved) ? null : (saved.activeSegmentIndex ?? null);
  } catch { state.anchors = []; state.negatives = []; state.segments = []; state.lostEvents = []; state.activeSegmentIndex = null; }
  state.anchors = state.anchors.filter((anchor) => state.chunkByKey.has(anchor.key));
  state.negatives = state.negatives.filter((negative) => state.chunkByKey.has(negative.key));
  rebuildRoute(); renderAnchors();
}
function renderAnchors() {
  $("#saveState").textContent = `Я: ${state.segments.length} · не я: ${state.negatives.length}`;
  $("#datasetStats").textContent = `Интервалов «я»: ${state.segments.length} · «не я»: ${state.negatives.length} · потерян: ${state.lostEvents.length}`;
  const active = state.activeSegmentIndex === null ? null : state.segments[state.activeSegmentIndex];
  const activeNode = $("#activeSegment");
  activeNode.className = `active-segment${active ? " live" : ""}`;
  activeNode.textContent = active
    ? `● ${formatTime(active.start)} → сейчас · ${domainLabel(active.domain)} · ${active.key}`
    : "Сейчас нет активного интервала";
  const list = $("#anchorList");
  if (!state.anchors.length && !state.negatives.length) { list.innerHTML = "<p>Пока пусто.</p>"; return; }
  const ordered = state.anchors.map((anchor, originalIndex) => ({ ...anchor, originalIndex })).sort((a, b) => a.time - b.time);
  list.innerHTML = ordered.map((anchor) =>
    `<div class="anchor"><time data-time="${anchor.time}">${formatTime(anchor.time)}</time><div><b>CONFIRMED</b><small>${anchor.key}</small></div><button data-delete="${anchor.originalIndex}">×</button></div>`
  ).join("") + state.negatives.map((negative) =>
    `<div class="anchor"><time data-time="${negative.time}">${formatTime(negative.time)}</time><div><b style="color:#ff98a6">НЕ Я</b><small>${negative.key}</small></div></div>`
  ).join("");
  list.querySelectorAll("time").forEach((element) => element.addEventListener("click", () => { video.currentTime = Number(element.dataset.time); }));
  list.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    const removed = state.anchors.splice(Number(button.dataset.delete), 1)[0];
    const segmentIndex = state.segments.findIndex((segment) => removed
      && Math.abs(segment.start - removed.time) < .35 && segment.key === removed.key);
    if (segmentIndex >= 0) {
      state.segments.splice(segmentIndex, 1);
      if (state.activeSegmentIndex === segmentIndex) state.activeSegmentIndex = null;
      else if (state.activeSegmentIndex > segmentIndex) state.activeSegmentIndex -= 1;
    }
    rebuildRoute(); saveAnchors();
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
  const selected = state.preview?.item || selectionAt(video.currentTime)?.item;
  if (!selected) { toast("Сначала кликни по своей рамке"); return; }
  const item = selected, time = video.currentTime;
  const previous = selectionAt(time);
  if (previous && previous.item.key !== item.key && !state.negatives.some((negative) =>
    negative.key === previous.item.key && Math.abs(negative.time - time) < .5)) {
    state.negatives.push({ time, key: previous.item.key, trackId: previous.item.id,
      reason: "corrected", domain: domainFor(previous.item) });
  }
  state.anchors = state.anchors.filter((anchor) => Math.abs(anchor.time - time) > .35);
  const domain = startPositiveSegment(item, time);
  state.anchors.push({ time, key: item.key, trackId: item.id, domain });
  state.preview = null; $("#confirmBox").disabled = true; $("#stageConfirm").hidden = true;
  rebuildRoute(); saveAnchors(); toast(`Начат интервал: ${domainLabel(domain)}`);
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
  const rows = parseCsv(await file.text());
  state.scores = new Map(rows.map((row) => [Number(row.track_id), Number(row.margin)]));
  state.chunkScores = new Map(rows.filter((row) => row.key).map((row) => [row.key, Number(row.margin)]));
  rebuildRoute(); toast("ReID-оценки загружены");
});
$("#play").addEventListener("click", () => { if (video.paused) video.play(); else video.pause(); });
document.querySelectorAll("[data-seek]").forEach((button) => button.addEventListener("click", () => seekTo(video.currentTime + Number(button.dataset.seek))));
$("#timeline").addEventListener("input", (event) => seekTo(Number(event.target.value)));
$("#confirmBox").addEventListener("click", confirmPreview);
$("#stageConfirm").addEventListener("click", confirmPreview);
$("#rejectCurrent").addEventListener("click", () => {
  const selected = state.preview ? { item: state.preview.item } : selectionAt(video.currentTime);
  if (!selected) { toast("Сейчас нет зелёной рамки"); return; }
  closeActiveSegment(video.currentTime, "negative");
  if (!state.negatives.some((negative) => negative.key === selected.item.key
    && Math.abs(negative.time - video.currentTime) < .5)) {
    state.negatives.push({ time: video.currentTime, key: selected.item.key, trackId: selected.item.id,
      reason: "manual", domain: domainFor(selected.item) });
  }
  video.pause(); rebuildRoute(); saveAnchors();
  setStatus("lost", "Отмечено: это не ты", "Теперь кликни по своей рамке и подтверди её.");
});
$("#markLost").addEventListener("click", () => {
  closeActiveSegment(video.currentTime, "lost");
  state.lostEvents.push({ time: video.currentTime, reason: "not-visible" });
  state.preview = null; video.pause(); saveAnchors();
  setStatus("lost", "Отмечено: тебя не видно", "Продолжай видео и нажми Q, когда снова найдёшь себя.");
});
$("#fullscreen").addEventListener("click", () => { $("#stage").requestFullscreen?.(); });
$("#undoAnchor").addEventListener("click", () => {
  state.anchors.pop();
  if (state.segments.length) state.segments.pop();
  state.activeSegmentIndex = null; rebuildRoute(); saveAnchors();
});
$("#clear").addEventListener("click", () => { state.anchors = []; state.negatives = []; state.segments = []; state.lostEvents = []; state.activeSegmentIndex = null; rebuildRoute(); saveAnchors(); });
$("#export").addEventListener("click", () => {
  const segments = state.segments.map((segment, index) => {
    const end = segment.end ?? (index === state.activeSegmentIndex ? video.currentTime : segment.start);
    const routeKeys = [...new Set(state.route.filter((entry) =>
      entry.item.time >= segment.start && entry.item.time <= end).map((entry) => entry.item.key))];
    return { ...segment, end, routeKeys };
  });
  const payload = { version: 3, video: state.videoName, tracks: state.tracksName,
    roi: [Number($("#roiX").value), Number($("#roiY").value)],
    domains: { near_back: "ближняя сторона, спиной", far_front: "дальняя сторона, лицом" },
    segments, anchors: state.anchors, negatives: state.negatives, lostEvents: state.lostEvents };
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })); link.download = "player-51-training-data.json"; link.click(); URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.target.matches("input")) { event.preventDefault(); if (video.paused) video.play(); else video.pause(); }
  if (event.code === "ArrowLeft") seekTo(video.currentTime - 5);
  if (event.code === "ArrowRight") seekTo(video.currentTime + 5);
  if (event.code === "Enter" && state.preview) confirmPreview();
  if (event.code === "KeyQ" && !event.target.matches("input,select")) { event.preventDefault(); confirmPreview(); }
  if (event.code === "KeyW" && !event.target.matches("input,select")) { event.preventDefault(); $("#rejectCurrent").click(); }
  if (event.code === "KeyE" && !event.target.matches("input,select")) { event.preventDefault(); $("#markLost").click(); }
});
video.addEventListener("play", () => { $("#play").textContent = "❚❚ Пауза"; });
video.addEventListener("pause", () => { $("#play").textContent = "▶ Смотреть"; });
video.addEventListener("loadedmetadata", () => {
  $("#timeline").max = video.duration || 1;
  const start = Number(new URLSearchParams(location.search).get("start") || 0);
  if (start > 0 && start < video.duration) video.currentTime = start;
});
video.addEventListener("loadeddata", () => { state.renderedVideoTime = -1; });
video.addEventListener("seeked", () => { state.renderedVideoTime = -1; });
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
    const rows = parseCsv(await response.text());
    state.scores = new Map(rows.map((row) => [Number(row.track_id), Number(row.margin)]));
    state.chunkScores = new Map(rows.filter((row) => row.key).map((row) => [row.key, Number(row.margin)]));
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
