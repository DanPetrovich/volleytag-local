const $ = (selector) => document.querySelector(selector);
const video = $("#video");
const videoFrame = $("#videoFrame");
const frameContext = videoFrame.getContext("2d", { alpha: false });
const canvas = $("#overlay");
const context = canvas.getContext("2d");

const state = {
  mode: "training",
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
  manualBoxes: [],
  activeSegmentIndex: null,
  route: [],
  pausedForGap: false,
  lostSince: null,
  preview: null,
  renderedVideoTime: -1,
  manualMode: false,
  manualStart: null,
  manualDraft: null,
  suppressClick: false,
  evaluation: VolleyTagEvaluation.emptyEvaluation(),
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
  state.manualBoxes.filter((box) => Math.abs(box.time - video.currentTime) <= .15)
    .forEach((box) => drawBox(box, "#66e6f2", 4, "Даня #51 · MANUAL"));
  if (state.manualDraft) drawBox(state.manualDraft, "#66e6f2", 3, "Ручная рамка");
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

function evaluationDomain() { return $("#evaluationDomain").value; }
function evaluationStorageKey() { return VolleyTagEvaluation.storageKey(state.videoName, state.tracksName); }
function nextEvaluationId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function closeEvaluationInterval(collection, activeKey, time) {
  const id = state.evaluation[activeKey];
  const interval = state.evaluation[collection].find((entry) => entry.id === id);
  if (interval) interval.end_s = Math.max(interval.start_s + .001, time);
  if (collection === "presence_intervals" && interval?.state === "visible") {
    const trajectory = state.evaluation.trajectory_intervals.find((entry) => entry.presence_id === interval.id);
    if (trajectory) trajectory.end_s = interval.end_s;
  }
  state.evaluation[activeKey] = null;
}

function closeEvaluationActive(time) {
  closeEvaluationInterval("presence_intervals", "activePresenceId", time);
  closeEvaluationInterval("reviewed_ranges", "activeReviewedId", time);
  closeEvaluationInterval("validated_gaps", "activeGapId", time);
}

function saveEvaluation() {
  state.evaluation.review ||= { complete: false, annotator: "", notes: [] };
  state.evaluation.review.annotator = $("#evalAnnotator")?.value.trim() || "";
  state.evaluation.inputs = {};
  [["video_sha256", "#evalVideoHash"], ["tracks_sha256", "#evalTracksHash"],
    ["route_or_model_sha256", "#evalRouteHash"], ["code_commit", "#evalCommit"]].forEach(([key, selector]) => {
    const value = $(selector)?.value.trim(); if (value) state.evaluation.inputs[key] = value;
  });
  if (state.videoName || state.tracksName) {
    localStorage.setItem(evaluationStorageKey(), JSON.stringify(state.evaluation));
  }
  renderEvaluation();
}

function loadEvaluation() {
  try {
    const saved = JSON.parse(localStorage.getItem(evaluationStorageKey()) || "null");
    state.evaluation = saved && typeof saved === "object" ? saved : VolleyTagEvaluation.emptyEvaluation();
  } catch { state.evaluation = VolleyTagEvaluation.emptyEvaluation(); }
  ["reviewed_ranges", "presence_intervals", "trajectory_intervals", "validated_gaps", "corrections"].forEach((key) => {
    if (!Array.isArray(state.evaluation[key])) state.evaluation[key] = [];
  });
  state.evaluation.review ||= { complete: false, annotator: "", notes: [] };
  state.evaluation.inputs ||= {};
  $("#reviewComplete").checked = state.evaluation.review.complete === true;
  $("#evalAnnotator").value = state.evaluation.review.annotator || "";
  $("#evalVideoHash").value = state.evaluation.inputs.video_sha256 || "";
  $("#evalTracksHash").value = state.evaluation.inputs.tracks_sha256 || "";
  $("#evalRouteHash").value = state.evaluation.inputs.route_or_model_sha256 || "";
  $("#evalCommit").value = state.evaluation.inputs.code_commit || "";
  renderEvaluation();
}

function startPresence(presenceState) {
  const time = video.currentTime;
  closeEvaluationInterval("presence_intervals", "activePresenceId", time);
  const id = nextEvaluationId("presence"), domain = evaluationDomain();
  state.evaluation.presence_intervals.push({ id, start_s: time, end_s: time + .001, state: presenceState, domain });
  state.evaluation.activePresenceId = id;
  if (presenceState === "visible") {
    state.evaluation.trajectory_intervals.push({
      id: nextEvaluationId("trajectory"), presence_id: id, start_s: time, end_s: time + .001,
      domain, source: "human_ground_truth", keyframes: [],
    });
  }
  saveEvaluation();
  toast(`Начат интервал ${presenceState}: ${domainLabel(domain)}`);
}

function toggleSimpleEvaluationInterval(collection, activeKey, prefix) {
  const active = state.evaluation[activeKey];
  if (active) {
    closeEvaluationInterval(collection, activeKey, video.currentTime);
  } else {
    const interval = { id: nextEvaluationId(prefix), start_s: video.currentTime, end_s: video.currentTime + .001 };
    if (collection === "validated_gaps") Object.assign(interval, { domain: evaluationDomain(), validated: true, reason: "human-validated" });
    state.evaluation[collection].push(interval);
    state.evaluation[activeKey] = interval.id;
  }
  saveEvaluation();
}

function activeVisibleTrajectory() {
  const presence = state.evaluation.presence_intervals.find((entry) => entry.id === state.evaluation.activePresenceId);
  if (!presence || presence.state !== "visible") return null;
  return state.evaluation.trajectory_intervals.find((entry) => entry.presence_id === presence.id) || null;
}

function addGroundTruthKeyframe(item) {
  let trajectory = activeVisibleTrajectory();
  if (!trajectory) {
    startPresence("visible");
    trajectory = activeVisibleTrajectory();
  }
  const keyframe = {
    time_s: video.currentTime, box: [item.x1, item.y1, item.x2, item.y2],
    identity: { key: item.key, track_id: item.id }, source: item.id === -1 ? "human-drawn" : "human-confirmed",
  };
  trajectory.keyframes = trajectory.keyframes.filter((entry) => Math.abs(entry.time_s - keyframe.time_s) > .001);
  trajectory.keyframes.push(keyframe);
  trajectory.keyframes.sort((left, right) => left.time_s - right.time_s);
  trajectory.end_s = Math.max(trajectory.end_s, video.currentTime + .001);
  saveEvaluation();
}

function evaluationPayload() {
  const inputs = {};
  const mappings = [["video_sha256", "#evalVideoHash"], ["tracks_sha256", "#evalTracksHash"],
    ["route_or_model_sha256", "#evalRouteHash"], ["code_commit", "#evalCommit"]];
  mappings.forEach(([key, selector]) => { const value = $(selector).value.trim(); if (value) inputs[key] = value; });
  return {
    schema: VolleyTagEvaluation.SCHEMA,
    frozen_test: true,
    created_at: new Date().toISOString(),
    video: {
      name: state.videoName, duration_s: video.duration, width: video.videoWidth,
      height: video.videoHeight, roi: [Number($("#roiX").value), Number($("#roiY").value)],
    },
    tracks: { name: state.tracksName }, inputs,
    review: { ...state.evaluation.review, annotator: $("#evalAnnotator").value.trim() },
    reviewed_ranges: state.evaluation.reviewed_ranges,
    presence_intervals: state.evaluation.presence_intervals,
    trajectory_intervals: state.evaluation.trajectory_intervals,
    validated_gaps: state.evaluation.validated_gaps,
    corrections: state.evaluation.corrections,
  };
}

function renderEvaluation() {
  if (!$("#evaluationSummary")) return;
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const coverage = VolleyTagEvaluation.reviewedCoverage(state.evaluation.reviewed_ranges, duration);
  $("#evaluationSummary").textContent = `Reviewed: ${coverage.seconds.toFixed(1)} / ${duration.toFixed(1)} с (${(coverage.ratio * 100).toFixed(1)}%) · presence: ${state.evaluation.presence_intervals.length} · corrections: ${state.evaluation.corrections.length}`;
  document.querySelectorAll("[data-presence]").forEach((button) => {
    const active = state.evaluation.presence_intervals.find((entry) => entry.id === state.evaluation.activePresenceId);
    button.classList.toggle("active", active?.state === button.dataset.presence);
  });
  $("#toggleReviewed").classList.toggle("active", Boolean(state.evaluation.activeReviewedId));
  $("#toggleGap").classList.toggle("active", Boolean(state.evaluation.activeGapId));
  const rows = [
    ...state.evaluation.reviewed_ranges.map((entry) => ({ collection: "reviewed_ranges", label: "reviewed", entry })),
    ...state.evaluation.presence_intervals.map((entry) => ({ collection: "presence_intervals", label: `${entry.state} · ${entry.domain}`, entry })),
    ...state.evaluation.validated_gaps.map((entry) => ({ collection: "validated_gaps", label: `gap · ${entry.domain}`, entry })),
  ].sort((left, right) => left.entry.start_s - right.entry.start_s);
  const list = $("#evaluationList");
  list.innerHTML = rows.map(({ collection, label, entry }) => `<div class="evaluation-row"><span>${label}</span><input type="number" step="0.001" value="${entry.start_s}" data-field="start_s" data-collection="${collection}" data-id="${entry.id}"><input type="number" step="0.001" value="${entry.end_s}" data-field="end_s" data-collection="${collection}" data-id="${entry.id}"><button data-eval-delete="${entry.id}" data-collection="${collection}">×</button></div>`).join("");
  list.querySelectorAll("input").forEach((input) => input.addEventListener("change", () => {
    const entry = state.evaluation[input.dataset.collection].find((candidate) => candidate.id === input.dataset.id);
    if (entry) entry[input.dataset.field] = Number(input.value);
    saveEvaluation();
  }));
  list.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
    state.evaluation[button.dataset.collection] = state.evaluation[button.dataset.collection].filter((entry) => entry.id !== button.dataset.evalDelete);
    if (button.dataset.collection === "presence_intervals") {
      state.evaluation.trajectory_intervals = state.evaluation.trajectory_intervals.filter((entry) => entry.presence_id !== button.dataset.evalDelete);
    }
    saveEvaluation();
  }));
}

function setAnnotationMode(mode) {
  state.mode = mode;
  $("#annotationMode").value = mode;
  const evaluation = mode === "evaluation";
  $("#evaluationCard").hidden = !evaluation;
  $(".dataset-card").hidden = evaluation;
  $(".help").hidden = evaluation;
  $(".anchors").hidden = evaluation;
  $("#rejectCurrent").hidden = evaluation;
  $("#markLost").hidden = evaluation;
  $("#confirmBox").textContent = evaluation ? "Q · ✓ Добавить human GT keyframe" : "Q · ✓ Это я — начать интервал";
  $("#manualBox").textContent = evaluation ? "R · ▱ Нарисовать human GT рамку" : "R · ▱ Нарисовать рамку вручную";
  if (evaluation) loadEvaluation();
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
  if (state.mode === "evaluation") closeEvaluationActive(video.currentTime);
  video.currentTime = Math.max(0, Math.min(video.duration || Infinity, time));
  state.lostSince = null; state.pausedForGap = false;
  state.preview = null; $("#confirmBox").disabled = true; $("#stageConfirm").hidden = true;
  if (state.mode === "evaluation") saveEvaluation(); else saveAnchors();
}

function storageKey() { return `volleytag-tracking-v3:${state.videoName}:${state.tracksName}`; }
function saveAnchors() {
  if (state.videoName && state.tracksName) localStorage.setItem(storageKey(), JSON.stringify({
    version: 3, anchors: state.anchors, negatives: state.negatives,
    segments: state.segments, lostEvents: state.lostEvents, manualBoxes: state.manualBoxes,
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
    state.manualBoxes = Array.isArray(saved) ? [] : (saved.manualBoxes || []);
    state.activeSegmentIndex = Array.isArray(saved) ? null : (saved.activeSegmentIndex ?? null);
  } catch { state.anchors = []; state.negatives = []; state.segments = []; state.lostEvents = []; state.manualBoxes = []; state.activeSegmentIndex = null; }
  state.anchors = state.anchors.filter((anchor) => state.chunkByKey.has(anchor.key));
  state.negatives = state.negatives.filter((negative) => state.chunkByKey.has(negative.key));
  rebuildRoute(); renderAnchors();
}
function renderAnchors() {
  $("#saveState").textContent = `Я: ${state.segments.length} · не я: ${state.negatives.length}`;
  $("#datasetStats").textContent = `Интервалов «я»: ${state.segments.length} · ручных рамок: ${state.manualBoxes.length} · «не я»: ${state.negatives.length} · потерян: ${state.lostEvents.length}`;
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
  if (state.manualMode || state.suppressClick) return;
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

function canvasPoint(event) {
  const g = geometry(), rx = Number($("#roiX").value) || 0, ry = Number($("#roiY").value) || 0;
  return [(event.offsetX - g.left) / g.scale - rx, (event.offsetY - g.top) / g.scale - ry];
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.manualMode) return;
  video.pause(); canvas.setPointerCapture?.(event.pointerId);
  state.manualStart = canvasPoint(event);
  state.manualDraft = null;
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.manualMode || !state.manualStart) return;
  const [x, y] = canvasPoint(event), [sx, sy] = state.manualStart;
  state.manualDraft = { x1: Math.min(sx, x), y1: Math.min(sy, y), x2: Math.max(sx, x), y2: Math.max(sy, y) };
});
canvas.addEventListener("pointerup", (event) => {
  if (!state.manualMode || !state.manualStart) return;
  const [x, y] = canvasPoint(event), [sx, sy] = state.manualStart;
  const box = { time: video.currentTime, frame: Math.round(video.currentTime * 50), id: -1,
    key: `manual:${video.currentTime.toFixed(3)}:${state.manualBoxes.length}`,
    x1: Math.min(sx, x), y1: Math.min(sy, y), x2: Math.max(sx, x), y2: Math.max(sy, y), confidence: 1 };
  state.manualStart = null; state.manualDraft = null; state.manualMode = false; state.suppressClick = true;
  $("#stage").classList.remove("manual-draw"); $("#manualBox").classList.remove("active");
  setTimeout(() => { state.suppressClick = false; }, 0);
  if (box.x2 - box.x1 < 12 || box.y2 - box.y1 < 20) { toast("Рамка слишком маленькая — попробуй ещё раз"); return; }
  if (state.mode === "evaluation") {
    box.domain = evaluationDomain(); addGroundTruthKeyframe(box);
    toast(`Human GT рамка сохранена: ${domainLabel(box.domain)}`);
  } else {
    box.domain = domainFor(box); state.manualBoxes.push(box); saveAnchors();
    toast(`Ручная рамка сохранена: ${domainLabel(box.domain)}`);
  }
});

function confirmPreview() {
  const selected = state.preview?.item || selectionAt(video.currentTime)?.item;
  if (!selected) { toast("Сначала кликни по своей рамке"); return; }
  const item = selected, time = video.currentTime;
  if (state.mode === "evaluation") {
    addGroundTruthKeyframe(item);
    state.preview = null; $("#confirmBox").disabled = true; $("#stageConfirm").hidden = true;
    toast(`Human GT keyframe: ${domainLabel(evaluationDomain())}`);
    return;
  }
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
  if (state.tracksName) { loadAnchors(); if (state.mode === "evaluation") loadEvaluation(); }
});
$("#tracksFile").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  state.tracksName = `${file.name}:${file.size}:${file.lastModified}`;
  buildChunks(parseCsv(await file.text())); loadAnchors();
  if (state.mode === "evaluation") loadEvaluation();
  setStatus("lost", "Рамки загружены", "Запусти видео и кликни по себе.");
});
$("#scoresFile").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  const rows = parseCsv(await file.text());
  state.scores = new Map(rows.map((row) => [Number(row.track_id), Number(row.margin)]));
  state.chunkScores = new Map(rows.filter((row) => row.key).map((row) => [row.key, Number(row.margin)]));
  rebuildRoute(); toast("ReID-оценки загружены");
});
$("#annotationFile").addEventListener("change", async (event) => {
  const file = event.target.files[0]; if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    if (payload.schema === VolleyTagEvaluation.SCHEMA) {
      $("#annotationMode").value = "evaluation"; setAnnotationMode("evaluation");
      state.evaluation = {
        reviewed_ranges: payload.reviewed_ranges || [], presence_intervals: payload.presence_intervals || [],
        trajectory_intervals: payload.trajectory_intervals || [], validated_gaps: payload.validated_gaps || [],
        corrections: payload.corrections || [], review: payload.review || { complete: false },
        inputs: payload.inputs || {}, activePresenceId: null, activeReviewedId: null, activeGapId: null,
      };
      $("#evalVideoHash").value = payload.inputs?.video_sha256 || "";
      $("#evalTracksHash").value = payload.inputs?.tracks_sha256 || "";
      $("#evalRouteHash").value = payload.inputs?.route_or_model_sha256 || "";
      $("#evalCommit").value = payload.inputs?.code_commit || "";
      $("#reviewComplete").checked = state.evaluation.review.complete === true;
      $("#evalAnnotator").value = state.evaluation.review.annotator || "";
      saveEvaluation();
      toast("Evaluation v4 импортирован");
    } else if (payload.version === 3) {
      state.anchors = payload.anchors || []; state.negatives = payload.negatives || [];
      state.segments = payload.segments || []; state.lostEvents = payload.lostEvents || [];
      state.manualBoxes = payload.manualBoxes || []; state.activeSegmentIndex = null;
      $("#annotationMode").value = "training"; setAnnotationMode("training"); rebuildRoute(); saveAnchors();
      toast("Training/review v3 импортирован");
    } else throw new Error("Неизвестная версия JSON");
  } catch (error) { toast(`Ошибка импорта: ${error.message}`); }
  event.target.value = "";
});
$("#annotationMode").addEventListener("change", (event) => setAnnotationMode(event.target.value));
document.querySelectorAll("[data-presence]").forEach((button) => button.addEventListener("click", () => startPresence(button.dataset.presence)));
$("#toggleReviewed").addEventListener("click", () => toggleSimpleEvaluationInterval("reviewed_ranges", "activeReviewedId", "reviewed"));
$("#toggleGap").addEventListener("click", () => toggleSimpleEvaluationInterval("validated_gaps", "activeGapId", "gap"));
$("#reviewComplete").addEventListener("change", (event) => { state.evaluation.review.complete = event.target.checked; saveEvaluation(); });
$("#evalAnnotator").addEventListener("change", (event) => { state.evaluation.review.annotator = event.target.value.trim(); saveEvaluation(); });
$("#addCorrection").addEventListener("click", () => {
  const after = state.preview?.item;
  if (!after) { toast("Для correction сначала выбери правильную фиолетовую рамку"); return; }
  const before = selectionAt(video.currentTime)?.item || null;
  if (!before || before.key === after.key) { toast("Correction требует отличающуюся исходную рамку before"); return; }
  const rawStart = prompt("error_start_s", String(video.currentTime));
  if (rawStart === null) return;
  const errorStart = Number(rawStart);
  if (!Number.isFinite(errorStart) || errorStart < 0 || errorStart > video.currentTime) { toast("Некорректный error_start_s"); return; }
  const reason = prompt("Причина correction", "identity-switch")?.trim();
  if (!reason) { toast("Correction требует причину"); return; }
  state.evaluation.corrections.push({
    id: nextEvaluationId("correction"), error_start_s: errorStart, corrected_at_s: video.currentTime,
    domain: evaluationDomain(), reason,
    before: { identity: { key: before.key, track_id: before.id }, box: [before.x1, before.y1, before.x2, before.y2] },
    after: { identity: { key: after.key, track_id: after.id }, box: [after.x1, after.y1, after.x2, after.y2] },
  });
  addGroundTruthKeyframe(after); toast("Correction сохранён");
});
$("#play").addEventListener("click", () => { if (video.paused) video.play(); else video.pause(); });
$("#playbackRate").addEventListener("change", (event) => {
  const rate = Number(event.target.value) || 1;
  video.playbackRate = rate;
  localStorage.setItem("volleytag-playback-rate", String(rate));
  toast(`Скорость: ${String(rate).replace(".", ",")}×`);
});
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
$("#manualBox").addEventListener("click", () => {
  state.manualMode = !state.manualMode; state.manualStart = null; state.manualDraft = null;
  $("#stage").classList.toggle("manual-draw", state.manualMode);
  $("#manualBox").classList.toggle("active", state.manualMode);
  if (state.manualMode) { video.pause(); toast("Обведи себя прямоугольником на видео"); }
});
$("#fullscreen").addEventListener("click", () => { $("#stage").requestFullscreen?.(); });
$("#undoAnchor").addEventListener("click", () => {
  state.anchors.pop();
  if (state.segments.length) state.segments.pop();
  state.activeSegmentIndex = null; rebuildRoute(); saveAnchors();
});
$("#clear").addEventListener("click", () => { state.anchors = []; state.negatives = []; state.segments = []; state.lostEvents = []; state.manualBoxes = []; state.activeSegmentIndex = null; rebuildRoute(); saveAnchors(); });
$("#export").addEventListener("click", () => {
  const segments = state.segments.map((segment, index) => {
    const end = segment.end ?? (index === state.activeSegmentIndex ? video.currentTime : segment.start);
    const routeKeys = [...new Set(state.route.filter((entry) =>
      entry.item.time >= segment.start && entry.item.time <= end).map((entry) => entry.item.key))];
    return { ...segment, end, routeKeys };
  });
  const payload = VolleyTagEvaluation.trainingExport(state, {
    video: state.videoName, tracks: state.tracksName,
    roi: [Number($("#roiX").value), Number($("#roiY").value)], segments,
  });
  const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })); link.download = "player-51-training-data.json"; link.click(); URL.revokeObjectURL(link.href);
});
$("#exportEvaluation").addEventListener("click", () => {
  closeEvaluationActive(video.currentTime); saveEvaluation();
  const payload = evaluationPayload();
  const result = VolleyTagEvaluation.validateEvaluation(payload);
  if (!result.valid) { alert(`Экспорт остановлен:\n${result.errors.join("\n")}`); return; }
  if (result.warnings.length && !confirm(`Предупреждения:\n${result.warnings.join("\n")}\n\nЭкспортировать?`)) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
  link.download = "player-51-evaluation.json"; link.click(); URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.target.matches("input")) { event.preventDefault(); if (video.paused) video.play(); else video.pause(); }
  if (event.code === "ArrowLeft") seekTo(video.currentTime - 5);
  if (event.code === "ArrowRight") seekTo(video.currentTime + 5);
  if (event.code === "Enter" && state.preview) confirmPreview();
  if (event.code === "KeyQ" && !event.target.matches("input,select")) { event.preventDefault(); confirmPreview(); }
  if (state.mode === "evaluation" && !event.target.matches("input,select")) {
    if (["Digit1", "Digit2", "Digit3", "Digit4"].includes(event.code)) {
      event.preventDefault(); startPresence({ Digit1: "visible", Digit2: "occluded", Digit3: "out_of_frame", Digit4: "uncertain" }[event.code]);
    }
    if (event.code === "KeyB") { event.preventDefault(); $("#toggleReviewed").click(); }
    if (event.code === "KeyG") { event.preventDefault(); $("#toggleGap").click(); }
    if (event.code === "KeyC") { event.preventDefault(); $("#addCorrection").click(); }
  }
  if (state.mode === "training" && event.code === "KeyW" && !event.target.matches("input,select")) { event.preventDefault(); $("#rejectCurrent").click(); }
  if (state.mode === "training" && event.code === "KeyE" && !event.target.matches("input,select")) { event.preventDefault(); $("#markLost").click(); }
  if (event.code === "KeyR" && !event.target.matches("input,select")) { event.preventDefault(); $("#manualBox").click(); }
});
video.addEventListener("play", () => { $("#play").textContent = "❚❚ Пауза"; });
video.addEventListener("pause", () => { $("#play").textContent = "▶ Смотреть"; });
video.addEventListener("loadedmetadata", () => {
  $("#timeline").max = video.duration || 1;
  const params = new URLSearchParams(location.search);
  const start = Number(params.get("start") || 0);
  const requestedRate = Number(params.get("speed") || localStorage.getItem("volleytag-playback-rate") || 1);
  const rate = [1, 1.25, 1.5, 2].includes(requestedRate) ? requestedRate : 1;
  video.playbackRate = rate;
  $("#playbackRate").value = String(rate);
  localStorage.setItem("volleytag-playback-rate", String(rate));
  if (start > 0 && start < video.duration) video.currentTime = start;
  renderEvaluation();
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
    if (VolleyTagEvaluation.shouldRestoreQueryEvaluation(state.mode, videoUrl, tracksUrl)) loadEvaluation();
    setStatus("lost", "Рамки загружены", "Запусти видео и кликни по себе.");
  }
}

loadQueryFiles().catch((error) => {
  setStatus("lost", "Ошибка загрузки", error.message);
});
setAnnotationMode(new URLSearchParams(location.search).get("mode") === "evaluation" ? "evaluation" : "training");
requestAnimationFrame(render);
