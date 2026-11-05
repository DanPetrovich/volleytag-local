#!/usr/bin/env python3
"""Extract the latest VolleyTag evaluation JSON from Chromium LevelDB logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("leveldb", type=Path)
    parser.add_argument("key")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    decoder = json.JSONDecoder()
    latest = None
    marker = args.key.encode()
    for path in sorted(args.leveldb.glob("*"), key=lambda item: item.stat().st_mtime):
        if not path.is_file():
            continue
        data = path.read_bytes()
        offset = 0
        while True:
            position = data.find(marker, offset)
            if position < 0:
                break
            brace = data.find(b"{", position + len(marker))
            if brace >= 0:
                text = data[brace:].decode("utf-8", errors="ignore")
                try:
                    value, _ = decoder.raw_decode(text)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, dict) and (
                        "trajectory_intervals" in value or "samples" in value
                    ):
                        latest = value
            offset = position + len(marker)

    if latest is None:
        raise SystemExit(f"No evaluation state found for key: {args.key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n")
    if "samples" in latest:
        print(f"saved={args.output} samples={len(latest['samples'])}")
    else:
        keyframes = sum(
            len(interval.get("keyframes", []))
            for interval in latest.get("trajectory_intervals", [])
        )
        print(f"saved={args.output} keyframes={keyframes}")


if __name__ == "__main__":
    main()
