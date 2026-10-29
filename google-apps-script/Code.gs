const MATCH_HEADERS = ['match_id','дата','турнир','соперник','игрок','сеты','очки','атаки','очки_атакой','блок_соперника','ошибки_атаки','приёмы','идеальный_приём','позитивный_приём','ошибки_приёма','подачи','эйсы','выбитый_приём','ошибки_подачи','действия_на_блоке','очки_блоком','блокауты','защита','защита_в_доводку','защита_в_игре','комментарий'];
const EVENT_HEADERS = ['match_id','timestamp','seconds','set','score','player','action','result','detail','rally_outcome','source','transcript','note'];

function doPost(e) {
  try {
    const payload = JSON.parse(e.postData.contents || '{}');
    const expected = PropertiesService.getScriptProperties().getProperty('VOLLEYTAG_TOKEN');
    if (!expected || payload.token !== expected) return json_({ ok: false, error: 'unauthorized' });
    if (!payload.match || !Array.isArray(payload.events)) return json_({ ok: false, error: 'bad_payload' });
    const lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try { upsertMatch_(payload.match, payload.events); } finally { lock.releaseLock(); }
    return json_({ ok: true, matchId: payload.match.id, events: payload.events.length });
  } catch (error) { return json_({ ok: false, error: String(error) }); }
}

function upsertMatch_(match, events) {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const matches = sheet_(ss, 'Матчи', MATCH_HEADERS);
  const eventSheet = sheet_(ss, 'События', EVENT_HEADERS);
  const matchId = String(match.id);
  if (eventSheet.getLastRow() > 1) {
    const ids = eventSheet.getRange(2, 1, eventSheet.getLastRow() - 1, 1).getValues().flat();
    for (let i = ids.length - 1; i >= 0; i--) if (String(ids[i]) === matchId) eventSheet.deleteRow(i + 2);
  }
  const eventRows = events.map(x => [matchId, time_(x.timestamp), Number(x.timestamp) || 0, Number(x.set) || 1,
    x.score || '', Number(x.player || match.player) || 51, x.action || '', x.result || '', x.detail || '',
    x.rallyOutcome || '', x.source || 'manual', x.transcript || '', x.note || '']);
  if (eventRows.length) eventSheet.getRange(eventSheet.getLastRow() + 1, 1, eventRows.length, EVENT_HEADERS.length).setValues(eventRows);
  const n = (action, result) => events.filter(x => x.action === action && (!result || x.result === result)).length;
  const kills = n('attack', 'kill'), aces = n('serve', 'ace'), blockPoints = n('block', 'point');
  const row = [matchId, match.date || '', match.competition || '', match.opponent || '', Number(match.player) || 51,
    Number(match.sets) || 1, kills + aces + blockPoints, n('attack'), kills, n('attack', 'blocked'), n('attack', 'error'),
    n('reception'), n('reception', 'perfect'), n('reception', 'positive'), n('reception', 'error'), n('serve'), aces,
    n('serve', 'out_of_system'), n('serve', 'error'), n('block'), blockPoints, n('block', 'blockout'), n('defense'),
    n('defense', 'perfect'), n('defense', 'in_play'), 'Отправлено из VolleyTag'];
  const last = matches.getLastRow();
  const ids = last > 1 ? matches.getRange(2, 1, last - 1, 1).getValues().flat().map(String) : [];
  const existing = ids.indexOf(matchId);
  matches.getRange(existing >= 0 ? existing + 2 : last + 1, 1, 1, MATCH_HEADERS.length).setValues([row]);
}

function sheet_(ss, name, headers) {
  const sheet = ss.getSheetByName(name) || ss.insertSheet(name);
  if (sheet.getLastRow() === 0) sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  return sheet;
}

function time_(seconds) {
  seconds = Number(seconds) || 0;
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = seconds % 60;
  return [h, m, s].map((v, i) => i === 2 ? v.toFixed(1).padStart(4, '0') : String(v).padStart(2, '0')).join(':');
}

function json_(value) {
  return ContentService.createTextOutput(JSON.stringify(value)).setMimeType(ContentService.MimeType.JSON);
}
