#!/usr/bin/env python3
"""Build a label-free action queue around reviewed trajectory breaks and manual fixes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2


OFFSETS = (-.6, -.3, 0.0, .3, .6)


def load_tracks(path: Path):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "time_s": float(row["time_s"]), "track_id": int(row["track_id"]),
                "confidence": float(row["confidence"]),
                "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            })
    return rows


def candidates_at(rows, timestamp, tolerance=.13):
    nearby = [row for row in rows if abs(row["time_s"] - timestamp) <= tolerance]
    if not nearby:
        return []
    nearest = min(abs(row["time_s"] - timestamp) for row in nearby)
    return [row for row in nearby if abs(abs(row["time_s"] - timestamp) - nearest) < 1e-5]


def seed_times(spec):
    seeds = []
    for index, manual in enumerate(spec.get("manualBoxes", [])):
        seeds.append((float(manual["time"]), 5, f"ручная рамка {index + 1}: tracker потерял Даню"))
    for index, event in enumerate(spec.get("lostEvents", [])):
        for key in ("time", "start", "end"):
            value = float(event.get(key, 0) or 0)
            if value > 0:
                seeds.append((value, 5, f"потеря траектории {index + 1}"))
    for index, segment in enumerate(spec.get("segments", [])):
        if segment.get("label") != "positive":
            continue
        priority = 5 if segment.get("endReason") == "lost" else 2
        seeds.append((float(segment["start"]), priority, f"начало подтверждённого участка {index + 1}"))
        seeds.append((float(segment["end"]), priority, f"конец участка: {segment.get('endReason', 'transition')}"))
    expanded = []
    for timestamp, priority, reason in seeds:
        for offset in OFFSETS:
            if timestamp + offset >= 0:
                expanded.append({"time_s": timestamp + offset, "priority": priority,
                                 "reason": reason, "offset": offset})
    return expanded


def balanced_times(values, count, reviewed_times):
    values = sorted(values, key=lambda row: (-row["priority"], row["time_s"], abs(row["offset"])))
    chosen = []
    for row in values:
        if any(abs(row["time_s"] - value) < .25 for value in reviewed_times):
            continue
        if any(abs(row["time_s"] - other["time_s"]) < .24 for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda row: row["time_s"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--previous-queue", type=Path)
    parser.add_argument("--previous-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)

    previous_times = {}
    if args.previous_queue and args.previous_results and args.previous_results.exists():
        queue = json.loads(args.previous_queue.read_text())
        results = json.loads(args.previous_results.read_text()).get("decisions", {})
        by_id = {item["id"]: item for item in queue["items"]}
        for item_id in results:
            item = by_id.get(item_id)
            if item:
                previous_times.setdefault(int(item["source"]), []).append(float(item["time_s"]))

    items = []
    for source, spec_path in enumerate(args.specs):
        spec = json.loads(spec_path.read_text())
        selected = balanced_times(seed_times(spec), args.per_source,
                                  previous_times.get(source, []))
        tracks = load_tracks(Path(spec["tracks"]))
        # track_players.py stores boxes in coordinates of the cropped ROI.
        # Queue images are full video frames, so convert every candidate back
        # to full-frame coordinates before normalizing it for the reviewer.
        roi_x, roi_y = [float(value) for value in spec.get("roi", [0, 0])]
        cap = cv2.VideoCapture(spec["video"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for selection in selected:
            timestamp = selection["time_s"]
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            candidates = []
            for row in candidates_at(tracks, timestamp):
                x1, y1, x2, y2 = row["box"]
                x1 = max(0.0, min(float(width), x1 + roi_x))
                x2 = max(0.0, min(float(width), x2 + roi_x))
                y1 = max(0.0, min(float(height), y1 + roi_y))
                y2 = max(0.0, min(float(height), y2 + roi_y))
                candidates.append({
                    "label": [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                              (x2 - x1) / width, (y2 - y1) / height],
                    "confidence": row["confidence"], "track_id": row["track_id"],
                })
            item_id = hashlib.sha1(
                f"action-select-v2:{source}:{timestamp:.3f}".encode()
            ).hexdigest()[:16]
            image_path = frame_dir / f"s{source}-t{timestamp:.3f}-{item_id}.jpg"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            domain = "unknown"
            containing = [segment for segment in spec.get("segments", [])
                          if float(segment.get("start", 0)) - .7 <= timestamp <=
                          float(segment.get("end", 0)) + .7]
            if containing:
                domain = containing[0].get("domain", "unknown")
            items.append({
                "id": item_id, "image": str(image_path), "video": spec["video"],
                "source": source, "split": "action_active_learning_v2",
                "kind": "action_selection", "domain": domain,
                "time_s": timestamp, "frame": int(round(timestamp * fps)),
                "labels": [], "candidates": candidates,
                "risk_reasons": [selection["reason"],
                                 "разметка отсутствует: выбери себя среди всех людей"],
            })
        cap.release()
        print(f"source={source} selected={sum(item['source'] == source for item in items)}")

    payload = {
        "schema_version": 1, "mode": "action_selection_v2", "total": len(items),
        "training_only": True, "inherited_ground_truth": False,
        "previous_reviewed_frames_excluded": sum(len(value) for value in previous_times.values()),
        "items": items,
    }
    (args.output / "queue.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
