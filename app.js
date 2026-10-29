const video = document.querySelector('#video');
const els = Object.fromEntries([...document.querySelectorAll('[id]')].map(el => [el.id, el]));
const STORAGE_KEY = 'volleytag-51-events-v2';
const SETTINGS_KEY = 'volleytag-51-settings-v2';
const SHEETS_KEY = 'volleytag-sheets-settings-v1';
const MATCH_ID_KEY = 'volleytag-current-match-id-v1';

const labels = {
  reception: { label: 'Приём', perfect: 'идеальный', positive: 'позитивный', negative: 'негативный', error: 'ошибка' },
  attack: { label: 'Атака', kill: 'очко', in_play: 'в игре', blocked: 'блок', error: 'ошибка' },
  serve: { label: 'Подача', ace: 'эйс', out_of_system: 'выбитый приём', in_play: 'в игре', error: 'ошибка' },
  block: { label: 'Блок', point: 'очко', touch: 'касание', blockout: 'блокаут' },
  defense: { label: 'Защита', perfect: 'в доводку', in_play: 'в игре', error: 'не поднял' }
};

let events = loadJson(STORAGE_KEY, []);
let objectUrl = null;
let toastTimer = null;

function loadJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}

function pad(number, size = 2) { return String(number).padStart(size, '0'); }
function formatTime(seconds, decimal = false) {
  if (!Number.isFinite(seconds)) seconds = 0;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}${decimal ? `.${Math.floor((seconds % 1) * 10)}` : ''}`;
}

function percent(value, total) { return total ? `${Math.round(value / total * 100)}%` : '—'; }
function count(action, result) { return events.filter(e => e.action === action && (!result || e.result === result)).length; }

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(events));
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ player: els.playerNumber.value, set: els.setNumber.value, pause: els.pauseAfterTag.checked }));
  els.saveState.textContent = 'Сохранено локально';
}

function saveSheetsSettings() {
  localStorage.setItem(SHEETS_KEY, JSON.stringify({
    date: els.matchDate.value, competition: els.competition.value.trim(), opponent: els.opponent.value.trim(),
    sets: els.setsPlayed.value, endpoint: els.sheetsEndpoint.value.trim(), token: els.sheetsToken.value
  }));
}

function currentMatchId() {
  let id = localStorage.getItem(MATCH_ID_KEY);
  if (!id) { id = crypto.randomUUID(); localStorage.setItem(MATCH_ID_KEY, id); }
  return id;
}

async function sendToSheets() {
  const endpoint = els.sheetsEndpoint.value.trim();
  const token = els.sheetsToken.value;
  if (!events.length) return showToast('Сначала разметь хотя бы одно действие');
  if (!endpoint || !token) return showToast('Добавь URL и ключ подключения');
  if (!els.matchDate.value || !els.opponent.value.trim()) return showToast('Укажи дату и соперника');
  saveSheetsSettings();
  els.sendToSheets.disabled = true;
  els.sendToSheets.textContent = 'Отправляю…';
  try {
    await fetch(endpoint, {
      method: 'POST', mode: 'no-cors', headers: { 'Content-Type': 'text/plain;charset=utf-8' },
      body: JSON.stringify({
        token, match: { id: currentMatchId(), date: els.matchDate.value, competition: els.competition.value.trim(),
          opponent: els.opponent.value.trim(), sets: Number(els.setsPlayed.value) || 1,
          player: Number(els.playerNumber.value) || 51 }, events
      })
    });
    els.sheetsHint.textContent = 'Запрос отправлен. Матч появится на листах «Матчи» и «События».';
    showToast('Матч отправлен в Google Sheets');
  } catch {
    els.sheetsHint.textContent = 'Не удалось отправить. Проверь URL веб-приложения и интернет.';
    showToast('Ошибка отправки в таблицу');
  } finally {
    els.sendToSheets.disabled = false;
    els.sendToSheets.textContent = 'Отправить матч в таблицу';
  }
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), 1200);
}

function seekBy(seconds) {
  if (video.readyState < 1 || !Number.isFinite(video.duration)) {
    showToast('Видео ещё загружается — секунду');
    return;
  }
  video.currentTime = Math.min(video.duration, Math.max(0, video.currentTime + seconds));
}

function addEvent(action, result) {
  const event = {
    id: crypto.randomUUID(), player: Number(els.playerNumber.value) || 51, timestamp: Number(video.currentTime.toFixed(2)),
    set: Number(els.setNumber.value), score: els.score.value.trim(), action, result,
    note: els.note.value.trim(), createdAt: new Date().toISOString()
  };
  events.push(event);
  if (els.pauseAfterTag.checked) video.pause();
  els.note.value = '';
  save();
  render();
  showToast(`${labels[action].label}: ${labels[action][result]}`);
}

function removeEvent(id) {
  events = events.filter(event => event.id !== id);
  save(); render();
}

function undo() {
  if (!events.length) return;
  const removed = events.pop();
  save(); render();
  showToast(`Отменено: ${labels[removed.action].label.toLowerCase()}`);
}

function render() {
  els.playerLabel.textContent = els.playerNumber.value || '51';
  const receptionTotal = count('reception');
  const perfect = count('reception', 'perfect');
  const positive = count('reception', 'positive');
  const attackTotal = count('attack');
  const kills = count('attack', 'kill');
  const attackErrors = count('attack', 'error') + count('attack', 'blocked');
  const serveTotal = count('serve');
  const aces = count('serve', 'ace');
  const outOfSystem = count('serve', 'out_of_system');
  const serveErrors = count('serve', 'error');
  const blockPoints = count('block', 'point');
  const defenseTotal = count('defense');
  const perfectDefense = count('defense', 'perfect');
  const playableDefense = count('defense', 'in_play');

  els.totalPoints.textContent = kills + aces + blockPoints;
  els.positiveReception.textContent = percent(perfect + positive, receptionTotal);
  els.positiveReceptionDetail.textContent = `${perfect + positive} из ${receptionTotal}`;
  els.perfectReception.textContent = percent(perfect, receptionTotal);
  els.perfectReceptionDetail.textContent = `${perfect} из ${receptionTotal}`;
  els.attackRate.textContent = percent(kills, attackTotal);
  els.attackRateDetail.textContent = `${kills} из ${attackTotal}`;
  els.attackEfficiency.textContent = attackTotal ? `${Math.round((kills - attackErrors) / attackTotal * 100)}%` : '—';
  els.attackEfficiencyDetail.textContent = `${kills} − ${attackErrors} / ${attackTotal}`;
  els.servePressure.textContent = percent(aces + outOfSystem, serveTotal);
  els.servePressureDetail.textContent = `${aces + outOfSystem} из ${serveTotal} · эйсы + выбитые`;
  els.serveErrors.textContent = serveErrors;
  els.serveErrorsDetail.textContent = `${serveTotal} подач`;
  els.defenseRate.textContent = percent(perfectDefense + playableDefense, defenseTotal);
  els.defenseRateDetail.textContent = `${perfectDefense + playableDefense} из ${defenseTotal}`;
  els.perfectDefense.textContent = percent(perfectDefense, defenseTotal);
  els.perfectDefenseDetail.textContent = `${perfectDefense} из ${defenseTotal}`;
  els.eventCount.textContent = events.length;
  els.undoButton.disabled = !events.length;

  if (!events.length) {
    els.eventLog.innerHTML = '<div class="empty-log">Первая отметка появится здесь.<br>Запусти видео и нажми кнопку действия.</div>';
    return;
  }

  els.eventLog.innerHTML = [...events].reverse().map(event => `
    <div class="event" data-id="${event.id}">
      <button class="event-time ghost" data-jump="${event.timestamp}" title="Перейти к моменту">${formatTime(event.timestamp)}</button>
      <div class="event-main"><b>${labels[event.action].label} · ${labels[event.action][event.result]}</b><small>Сет ${event.set}${event.score ? ` · ${escapeHtml(event.score)}` : ''}${event.note ? ` · ${escapeHtml(event.note)}` : ''}</small></div>
      <button class="delete-event" data-delete="${event.id}" title="Удалить">×</button>
    </div>`).join('');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function download(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportCsv() {
  const header = ['timestamp','seconds','set','score','player','action','result','note'];
  const rows = events.map(e => [formatTime(e.timestamp, true), e.timestamp, e.set, e.score, e.player, e.action, e.result, e.note]);
  download(`volleyball-${els.playerNumber.value || 51}-events.csv`, '\ufeff' + [header, ...rows].map(row => row.map(csvCell).join(',')).join('\n'), 'text/csv;charset=utf-8');
}

document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => addEvent(button.dataset.action, button.dataset.result)));
document.querySelectorAll('[data-seek]').forEach(button => button.addEventListener('click', () => seekBy(Number(button.dataset.seek))));

els.playButton.addEventListener('click', () => video.paused ? video.play() : video.pause());
video.addEventListener('play', () => els.playButton.textContent = 'Ⅱ Пауза');
video.addEventListener('pause', () => els.playButton.textContent = '▶ Смотреть');
els.videoToggle.addEventListener('click', () => video.paused ? video.play() : video.pause());
video.addEventListener('timeupdate', () => els.timecode.textContent = formatTime(video.currentTime, true));
video.addEventListener('error', () => { els.videoEmpty.hidden = false; });
video.addEventListener('loadedmetadata', () => { els.videoEmpty.hidden = true; });

els.videoFile.addEventListener('change', event => {
  const file = event.target.files[0]; if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file); video.src = objectUrl; video.load();
});

els.eventLog.addEventListener('click', event => {
  const jump = event.target.closest('[data-jump]');
  const del = event.target.closest('[data-delete]');
  if (jump) { video.currentTime = Number(jump.dataset.jump); video.pause(); }
  if (del) removeEvent(del.dataset.delete);
});

els.undoButton.addEventListener('click', undo);
els.exportCsv.addEventListener('click', exportCsv);
els.sendToSheets.addEventListener('click', sendToSheets);
els.exportJson.addEventListener('click', () => download(`volleyball-${els.playerNumber.value || 51}-backup.json`, JSON.stringify({ version: 1, player: Number(els.playerNumber.value) || 51, exportedAt: new Date().toISOString(), events }, null, 2), 'application/json'));
els.importJson.addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (!Array.isArray(data.events)) throw new Error('Формат');
    events = data.events; save(); render(); showToast(`Загружено событий: ${events.length}`);
  } catch { showToast('Не удалось прочитать JSON'); }
  event.target.value = '';
});
els.clearAll.addEventListener('click', () => {
  if (!events.length || !confirm(`Удалить все события (${events.length})? Сначала можно скачать резервную копию JSON.`)) return;
  events = []; localStorage.removeItem(MATCH_ID_KEY); save(); render(); showToast('Разметка очищена · начат новый матч');
});
[els.playerNumber, els.setNumber, els.pauseAfterTag].forEach(el => el.addEventListener('change', () => { save(); render(); }));
[els.matchDate, els.competition, els.opponent, els.setsPlayed, els.sheetsEndpoint, els.sheetsToken]
  .forEach(el => el.addEventListener('change', saveSheetsSettings));

document.addEventListener('keydown', event => {
  const typing = ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName);
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo(); return; }
  if (typing) return;
  if (event.code === 'Space') { event.preventDefault(); video.paused ? video.play() : video.pause(); return; }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); seekBy(event.key === 'ArrowLeft' ? -5 : 5); return; }
  const button = document.querySelector(`[data-key="${event.key.toUpperCase()}"]`);
  if (button) { event.preventDefault(); button.click(); button.classList.add('flash'); setTimeout(() => button.classList.remove('flash'), 150); }
});

const settings = loadJson(SETTINGS_KEY, {});
if (settings.player) els.playerNumber.value = settings.player;
if (settings.set) els.setNumber.value = settings.set;
if (typeof settings.pause === 'boolean') els.pauseAfterTag.checked = settings.pause;
const sheetsSettings = loadJson(SHEETS_KEY, {});
els.matchDate.value = sheetsSettings.date || new Date().toISOString().slice(0, 10);
els.competition.value = sheetsSettings.competition || '';
els.opponent.value = sheetsSettings.opponent || '';
els.setsPlayed.value = sheetsSettings.sets || 3;
els.sheetsEndpoint.value = sheetsSettings.endpoint || '';
els.sheetsToken.value = sheetsSettings.token || '';
render();
