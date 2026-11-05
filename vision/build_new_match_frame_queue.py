#!/usr/bin/env python3
"""Build a compact frame-based identity audit for a newly added match.

The queue mixes timeline coverage with detector/tracker-heavy motion moments.  It
uses the personal detector only as a proposal: the human review remains ground
truth, and no video-level identity tracing is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from build_person_selection_queue import deduplicate_candidates


def excluded(timestamp: float, intervals: list[dict]) -> bool:
    return any(float(row["start_s"]) <= timestamp <= float(row["end_s"])
               for row in intervals)


def evenly_spaced(values: list[float], count: int) -> list[float]:
    if len(values) <= count:
        return values
    return [values[round(index * (len(values) - 1) / (count - 1))]
            for index in range(count)]


def motion_times(path: Path) -> list[tuple[float, float]]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_track[int(row["track_id"])].append({
                "time": float(row["time_s"]),
                "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            })
    scores: dict[float, float] = {}
    for rows in by_track.values():
        rows.sort(key=lambda row: row["time"])
        for previous, current in zip(rows, rows[1:]):
            dt = current["time"] - previous["time"]
            if not 0.05 <= dt <= 0.25:
                continue
            a, b = previous["box"], current["box"]
            ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            scale = max(20.0, ((a[3] - a[1]) + (b[3] - b[1])) / 2)
            speed = math.hypot(bx - ax, by - ay) / scale / dt
            bucket = round(current["time"], 1)
            scores[bucket] = max(scores.get(bucket, 0.0), speed)
    return list(scores.items())


def select_motion(values: list[tuple[float, float]], count: int,
                  existing: list[float], minimum_gap: float = 2.0) -> list[float]:
    selected: list[float] = []
    for timestamp, _ in sorted(values, key=lambda row: row[1], reverse=True):
        if any(abs(timestamp - other) < minimum_gap for other in existing + selected):
            continue
        selected.append(timestamp)
        if len(selected) >= count:
            break
    return selected


def yolo_label(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--source-info", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uniform-per-set", type=int, default=100)
    parser.add_argument("--motion-per-set", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source_info = json.loads(args.source_info.read_text(encoding="utf-8"))
    intervals = json.loads(args.exclude.read_text(encoding="utf-8"))["excluded_intervals"]
    motion = motion_times(args.tracks)
    selections = []
    domains = ["near_back", "far_front"]
    for source, set_info in enumerate(source_info["sets"]):
        start, end = float(set_info["start_s"]), float(set_info["end_s"])
        timeline = [start + index for index in range(1, max(1, int(end - start)))
                    if not excluded(start + index, intervals)]
        uniform = evenly_spaced(timeline, args.uniform_per_set)
        motion_pool = [(timestamp, score) for timestamp, score in motion
                       if start <= timestamp <= end and not excluded(timestamp, intervals)]
        dynamic = select_motion(motion_pool, args.motion_per_set, uniform)
        for timestamp in sorted(uniform + dynamic):
            selections.append({"source": source, "set": int(set_info["set"]),
                               "time_s": timestamp, "domain": domains[source]})

    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    frames, selected = [], []
    for row in selections:
        capture.set(cv2.CAP_PROP_POS_MSEC, row["time_s"] * 1000.0)
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
            selected.append(row)
    capture.release()

    model = YOLO(str(args.model))
    started = time.monotonic()
    predictions = model.predict(frames, imgsz=960, conf=.03, device=args.device,
                                batch=8, verbose=False, stream=True)
    items = []
    for row, frame, result in zip(selected, frames, predictions):
        height, width = frame.shape[:2]
        candidates = []
        if result.boxes is not None:
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist()):
                candidates.append({"label": yolo_label(box, width, height),
                                   "confidence": round(float(confidence), 4)})
        candidates = deduplicate_candidates(candidates)[:3]
        marker = f"keaz-frame-v1:{row['set']}:{row['time_s']:.3f}"
        item_id = hashlib.sha1(marker.encode()).hexdigest()[:16]
        image_path = frame_dir / f"set{row['set']}-t{row['time_s']:.3f}-{item_id}.jpg"
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items.append({
            "id": item_id, "image": str(image_path), "video": str(args.video),
            "source": row["source"], "set": row["set"], "split": "train",
            "kind": "new_match_frame_audit", "domain": row["domain"],
            "time_s": round(row["time_s"], 3),
            "frame": int(round(row["time_s"] * 60)),
            "labels": [candidates[0]["label"]] if candidates else [],
            "candidates": candidates,
            "risk_reasons": [
                f"КЭАЗ — X5 Tech, партия {row['set']}",
                "покадровая проверка нового матча",
                "зелёная рамка — предложение v8; исправь её или отметь отсутствие/перекрытие",
            ],
        })
    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1, "mode": "new_match_frame_audit",
        "match": source_info["match"], "match_date": source_info["match_date"],
        "video": str(args.video), "total": len(items),
        "training_only": True, "frozen_tests_used": False,
        "personal_model": str(args.model), "elapsed_s": round(elapsed, 3),
        "items": items,
    }
    (args.output / "queue.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "frames": len(items), "by_set": {
            str(index): sum(item["set"] == index for item in items) for index in (1, 2)},
        "with_v8_proposal": sum(bool(item["labels"]) for item in items),
        "elapsed_s": round(elapsed, 2),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
