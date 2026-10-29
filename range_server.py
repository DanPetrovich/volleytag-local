#!/usr/bin/env python3
"""Small local HTTP server with byte-range support for large video files."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import re
import sys


class RangeRequestHandler(SimpleHTTPRequestHandler):
    range_to_send = None

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

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
