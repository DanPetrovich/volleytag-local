#!/usr/bin/env python3
"""Convert an independent benchmark queue and human decisions to GT samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    decisions = json.loads(args.results.read_text())["decisions"]
    intervals = []
    if args.exclude:
        payload = json.loads(args.exclude.read_text())
        intervals = payload if isinstance(payload, list) else payload.get("intervals", [])
    samples = []
    for item in queue["items"]:
        if any(float(interval["start_s"]) <= item["time_s"] <= float(interval["end_s"])
               for interval in intervals):
            continue
        decision = decisions.get(item["id"])
        if not decision or decision["verdict"] not in {"selected_candidate", "corrected_box"}:
            continue
        frame = cv2.imread(item["image"])
        if frame is None:
            continue
        height, width = frame.shape[:2]
        _, cx, cy, box_width, box_height = map(float, decision["correctedLabel"])
        samples.append({
            "id": item["id"], "time_s": float(item["time_s"]), "state": "visible",
            "box": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width, (cy + box_height / 2) * height],
        })
    payload = {"schema": "volleytag.player51-uniform-evaluation.v1",
               "source": "independent-human-benchmark", "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"samples": len(samples), "output": str(args.output)}))


if __name__ == "__main__":
    main()
