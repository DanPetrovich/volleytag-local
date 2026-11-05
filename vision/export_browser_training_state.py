#!/usr/bin/env python3
"""Recover the current training annotation from Codex IAB localStorage.

The review UI persists every edit in Chromium LevelDB before offering a JSON
download. This helper makes that saved state reproducible even when the in-app
browser download surface is unavailable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def next_positive_start(segments: list[dict], manual_boxes: list[dict], time_s: float) -> float | None:
    starts = [float(item["start"]) for item in segments
              if item.get("label") == "positive" and float(item["start"]) > time_s]
    starts.extend(float(item["time"]) for item in manual_boxes if float(item["time"]) > time_s)
    starts.sort()
    return starts[0] if starts else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leveldb-log", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = subprocess.run(
        ["strings", "-a", str(args.leveldb_log)], check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    states = [json.loads(line) for line in lines
              if line.startswith('{"version":3,"anchors":')]
    if not states:
        raise SystemExit("training state was not found in LevelDB log")
    state = states[-1]
    segments = state.get("segments", [])
    active = state.get("activeSegmentIndex")
    if isinstance(active, int) and 0 <= active < len(segments) and segments[active].get("end") is None:
        segments[active]["end"] = args.duration
        segments[active]["endReason"] = "video-end"

    normalized_lost = []
    manual_boxes = state.get("manualBoxes", [])
    for event in state.get("lostEvents", []):
        start = float(event.get("start", event["time"]))
        following = next_positive_start(segments, manual_boxes, start)
        # Legacy E presses were point events without an explicit start. Some
        # browser versions later closed all of them at the same distant Q press.
        # Their correct end is the next positive confirmation.
        if "start" not in event:
            end = following if following is not None else min(args.duration, start + .25)
        else:
            raw_end = event.get("end")
            end = float(raw_end) if raw_end is not None else (
                following if following is not None else args.duration)
            if following is not None:
                end = min(end, following)
        normalized_lost.append({**event, "start": start, "end": max(start, min(args.duration, end))})

    exclusions = json.loads(args.exclude.read_text(encoding="utf-8")).get("excluded_intervals", [])
    payload = {
        "version": 3,
        "video": args.video,
        "tracks": args.tracks,
        "roi": [0, 0],
        "domains": {"near_back": "ближняя сторона, спиной", "far_front": "дальняя сторона, лицом"},
        "segments": segments,
        "anchors": state.get("anchors", []),
        "negatives": state.get("negatives", []),
        "lostEvents": normalized_lost,
        "manualBoxes": manual_boxes,
        "excluded_intervals": exclusions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "segments": len(segments), "anchors": len(payload["anchors"]),
        "negatives": len(payload["negatives"]), "lost": len(normalized_lost),
        "manual": len(payload["manualBoxes"]), "excluded": len(exclusions),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
