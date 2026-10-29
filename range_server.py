#!/usr/bin/env python3
"""Small local HTTP server with byte-range support for large video files."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid


VOICE_SYSTEM_PROMPT = """Ты преобразуешь короткую русскую голосовую команду разметки волейбола в JSON.
Верни только объект с полями action, result, detail, rallyOutcome, confidence.
action: reception|attack|serve|block|defense|null.
result: reception=perfect|positive|negative|error; attack=kill|in_play|blocked|error; serve=ace|out_of_system|in_play|error; block=point|touch|blockout; defense=perfect|in_play|error.
Строго различай reception и defense по источнику мяча, а не по глаголу «принял»:
- reception — ТОЛЬКО приём подачи соперника. Если явно или по короткой команде говорится о приёме подачи: «приём», «принял подачу», «подачу довёл».
- defense — любое действие с мячом НЕ после подачи: защита от атаки, скидка, накат, свободный мяч, фрибол, переходящий мяч, страховка. Слова «переходящий мяч», «свободный мяч», «скидка», «атака соперника», «в защите» всегда имеют приоритет и означают defense, даже если человек говорит «принял».
Для защиты: perfect — только если явно сказано «довёл до связки» или «точная доводка»; in_play — «поднял», «просто поднял», «оставил в игре», «игра продолжилась»; error — «не поднял», «не достал», мяч упал после атаки соперника.
detail: blockout только когда атакующий попал от блока соперника в аут, иначе null.
rallyOutcome: won|lost|null. Очко атакой, эйс и блок-очко логически означают won. Ошибка подачи, ошибка приёма, ошибка атаки и атака в блок с очком соперника означают lost. Если сказано только что мяч подняли или действие продолжилось, итог неизвестен.
Не выдумывай неизвестные поля. Примеры: «забил» => attack/kill/null/won; «атака блокаут» => attack/kill/blockout/won; «эйс» => serve/ace/null/won; «атаку подняли» => attack/in_play/null/null; «приём подачи позитивный» => reception/positive/null/null; «принял переходящий мяч, довёл до связки» => defense/perfect/null/null; «свободный мяч просто поднял» => defense/in_play/null/null."""


class RangeRequestHandler(SimpleHTTPRequestHandler):
    range_to_send = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        if self.path not in ("/api/parse-voice", "/api/transcribe-voice"):
            self.send_error(404)
            return
        transcript = ""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            if self.path == "/api/transcribe-voice":
                if not raw_body:
                    raise ValueError("Пустая аудиозапись")
                transcript = self.transcribe_with_groq(raw_body, self.headers.get("Content-Type", "audio/webm"))
            else:
                payload = json.loads(raw_body or b"{}")
                transcript = str(payload.get("text", "")).strip()
            if not transcript:
                raise ValueError("Пустая команда")
            result = self.parse_voice_with_groq(transcript)
            result = self.normalize_voice_event(transcript, result)
            self.write_voice_log({"status": "ok", "transcript": transcript, "event": result})
            self.send_json(200, {"ok": True, "transcript": transcript, "event": result})
        except FileNotFoundError:
            self.send_json(503, {"ok": False, "error": "Добавь ключ Groq в файл .groq-key"})
        except Exception as error:
            self.write_voice_log({"status": "error", "transcript": transcript, "error": str(error)})
            self.send_json(400, {"ok": False, "error": str(error)})

    def write_voice_log(self, record):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        path = Path(__file__).resolve().parent / "volleytag-voice.log"
        with path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

    def normalize_voice_event(self, transcript, event):
        text = transcript.lower().replace("ё", "е")
        defense_context = re.search(
            r"переходящ|свободн(ый|ого|ому)? мяч|фрибол|фри бол|скидк|накат|страховк|атак[аиуы]? соперник|в защит[еуы]",
            text,
        )
        if defense_context:
            event["action"] = "defense"
        if event.get("action") == "defense":
            if re.search(r"\b(не поднял|не достал|не подняла|не достала)\b", text):
                event["result"] = "error"
                event["rallyOutcome"] = "lost"
            elif "связ" in text or "довел" in text or "точная доводка" in text:
                event["result"] = "perfect"
            elif "поднял" in text or "подняла" in text or "игра продолж" in text or "оставил в игре" in text:
                event["result"] = "in_play"
        return event

    def send_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def parse_voice_with_groq(self, transcript):
        key_path = Path(__file__).resolve().parent / ".groq-key"
        api_key = os.environ.get("GROQ_API_KEY") or key_path.read_text(encoding="utf-8").strip()
        request_body = json.dumps({
            "model": "openai/gpt-oss-20b",
            "temperature": 0,
            "max_completion_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": VOICE_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VolleyTag-Local/2.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq ответил {error.code}: {detail[:300]}") from error
        return json.loads(data["choices"][0]["message"]["content"])

    def transcribe_with_groq(self, audio, mime_type):
        key_path = Path(__file__).resolve().parent / ".groq-key"
        api_key = os.environ.get("GROQ_API_KEY") or key_path.read_text(encoding="utf-8").strip()
        boundary = f"----VolleyTag{uuid.uuid4().hex}"
        extension = "mp4" if "mp4" in mime_type else "wav" if "wav" in mime_type else "webm"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nru\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice.{extension}\"\r\nContent-Type: {mime_type}\r\n\r\n".encode(),
            audio,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            data=b"".join(parts),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "VolleyTag-Local/2.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq Speech-to-Text ответил {error.code}: {detail[:300]}") from error
        return str(data.get("text", "")).strip()

    def send_head(self):
        range_header = self.headers.get("Range")
        if not range_header:
            self.range_to_send = None
            return super().send_head()

        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.range_to_send = None
            return super().send_head()

        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            self.send_error(416, "Invalid byte range")
            return None

        file_size = path.stat().st_size
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        else:
            suffix_length = int(end_text or 0)
            start = max(0, file_size - suffix_length)
            end = file_size - 1

        if start >= file_size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        end = min(end, file_size - 1)
        file = path.open("rb")
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        file.seek(start)
        self.range_to_send = end - start + 1
        return file

    def copyfile(self, source, outputfile):
        if self.range_to_send is None:
            return super().copyfile(source, outputfile)

        remaining = self.range_to_send
        try:
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # The browser cancels an older range request when the user seeks again.
            pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.chdir(Path(__file__).resolve().parent)
    server = ThreadingHTTPServer(("127.0.0.1", port), RangeRequestHandler)
    print(f"VolleyTag открыт: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
