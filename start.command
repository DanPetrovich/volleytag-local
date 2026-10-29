#!/bin/zsh
cd "$(dirname "$0")" || exit 1
python3 range_server.py 8765 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null' EXIT INT TERM

for attempt in {1..50}; do
  if curl --silent --fail "http://127.0.0.1:8765/" >/dev/null; then
    open "http://127.0.0.1:8765/"
    wait "$server_pid"
    exit $?
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Не удалось запустить VolleyTag. Возможно, старое окно разметчика ещё открыто."
    echo "Закрой старое окно Терминала и запусти start.command ещё раз."
    read -k 1 "?Нажми любую клавишу, чтобы закрыть окно..."
    exit 1
  fi
  sleep 0.1
done

echo "Сервер запущен, но браузер не смог к нему подключиться."
read -k 1 "?Нажми любую клавишу, чтобы закрыть окно..."
exit 1
