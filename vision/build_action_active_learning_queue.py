#!/usr/bin/env python3
"""Mine action-heavy, detector-hard frames from reviewed training matches only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_fused_detectors import iou
from export_danya_detector_dataset import read_tracks, split_chunks


def center(row):
    x1, y1, x2, y2 = row["box"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def height(row):
    return max(20.0, row["box"][3] - row["box"][1])


def motion_features(rows: list[dict], index: int) -> dict:
    row = rows[index]
    before = max(0, index - 5)
    after = min(len(rows) - 1, index + 5)
    first, last = rows[before], rows[after]
    dt = max(.08, last["time_s"] - first["time_s"])
    x0, y0 = center(first)
    x1, y1 = center(last)
    scale = max(30.0, (height(first) + height(last)) / 2)
    vx = (x1 - x0) / scale / dt
    vy = (y1 - y0) / scale / dt
    speed = math.hypot(vx, vy)
    box = row["box"]
    aspect = (box[2] - box[0]) / max(1.0, box[3] - box[1])
    return {"speed": speed, "vx": vx, "vy": vy, "aspect": aspect}


def reviewed_rows(spec: dict) -> list[dict]:
    chunks = split_chunks(read_tracks(Path(spec["tracks"])))
    values = []
    for segment_index, segment in enumerate(spec.get("segments", [])):
        if segment.get("label") != "positive":
            continue
        start, end = float(segment["start"]), float(segment["end"])
        rows = [row for key in (segment.get("routeKeys") or [segment.get("key")])
                for row in chunks.get(key, []) if start <= row["time_s"] <= end]
        rows = sorted({row["frame"]: row for row in rows}.values(), key=lambda row: row["time_s"])
        for index, row in enumerate(rows):
            if len(rows) < 3:
                continue
            values.append({**row, **motion_features(rows, index),
                           "domain": segment.get("domain", "unknown"),
                           "origin": f"segment-{segment_index}"})
    return values


def diverse_top(rows: list[dict], count: int, separation: float = .65) -> list[dict]:
    ranked = sorted(rows, key=lambda row: row["motion_priority"], reverse=True)
    selected = []
    for row in ranked:
        if any(abs(row["time_s"] - other["time_s"]) < separation for other in selected):
            continue
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def yolo_label(box, width, height):
    x1, y1, x2, y2 = box
    return [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=60)
    parser.add_argument("--pool-per-source", type=int, default=140)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)

    metadata = []
    model = YOLO(str(args.model))
    for source, spec_path in enumerate(args.specs):
        spec = json.loads(spec_path.read_text())
        rows = reviewed_rows(spec)
        for row in rows:
            # Favor fast movement, vertical movement, and non-upright poses.
            row["motion_priority"] = (row["speed"] + .8 * abs(row["vy"]) +
                                      4.0 * max(0.0, row["aspect"] - .58))
        pool = diverse_top(rows, args.pool_per_source, separation=.45)
        cap = cv2.VideoCapture(spec["video"])
        source_frames, source_rows = [], []
        for row in sorted(pool, key=lambda value: value["frame"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ok, frame = cap.read()
            if ok:
                source_frames.append(frame)
                source_rows.append({**row, "source": source, "video": spec["video"]})
        cap.release()
        print(f"source={source} motion_pool={len(source_rows)}", flush=True)
        predictions = model.predict(source_frames, imgsz=args.imgsz, conf=args.conf,
                                    device="cpu", batch=8, verbose=False, stream=True)
        for row, result in zip(source_rows, predictions):
            boxes = (result.boxes.xyxy.cpu().numpy() if result.boxes is not None
                     else np.empty((0, 4)))
            scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                      else np.empty(0))
            overlaps = [iou(row["box"], box.tolist()) for box in boxes]
            matching = [float(score) for score, overlap in zip(scores, overlaps) if overlap >= .30]
            row["v7_confidence"] = max(matching, default=0.0)
            row["v7_hit"] = bool(matching)
            row["active_priority"] = (row["motion_priority"] +
                                      (4.0 if not matching else 0.0) +
                                      2.0 * (1.0 - row["v7_confidence"]))
        metadata.extend(source_rows)
        del source_frames

    chosen = []
    for source in range(len(args.specs)):
        source_rows = [row for row in metadata if row["source"] == source]
        # Final queue stays source-balanced and temporally diverse.
        ranked = sorted(source_rows, key=lambda row: row["active_priority"], reverse=True)
        selected = []
        for row in ranked:
            if any(abs(row["time_s"] - other["time_s"]) < .65 for other in selected):
                continue
            selected.append(row)
            if len(selected) >= args.per_source:
                break
        chosen.extend(selected)

    items = []
    captures = {}
    for index, row in enumerate(chosen):
        video = row["video"]
        cap = captures.setdefault(video, cv2.VideoCapture(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = cap.read()
        if not ok:
            continue
        height_px, width_px = frame.shape[:2]
        item_id = hashlib.sha1(
            f"action-v1:{row['source']}:{row['frame']}".encode()
        ).hexdigest()[:16]
        image_path = frame_dir / f"s{row['source']}-f{row['frame']}-{item_id}.jpg"
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        reasons = []
        if row["speed"] >= 1.2:
            reasons.append("быстрое движение")
        if abs(row["vy"]) >= .7:
            reasons.append("вертикальное движение / прыжок")
        if row["aspect"] >= .58:
            reasons.append("необычная низкая или вытянутая поза")
        if not row["v7_hit"]:
            reasons.append("v7 не нашёл Даню")
        elif row["v7_confidence"] < .55:
            reasons.append("низкая уверенность v7")
        items.append({
            "id": item_id, "image": str(image_path), "video": row["video"],
            "source": row["source"], "split": "action_active_learning",
            "kind": "positive", "domain": row["domain"],
            "time_s": row["time_s"], "frame": row["frame"],
            "track_id": row["track_id"],
            "labels": [yolo_label(row["box"], width_px, height_px)],
            "risk_reasons": reasons,
            "action_metrics": {key: round(float(row[key]), 4) for key in
                               ("speed", "vx", "vy", "aspect", "v7_confidence",
                                "motion_priority", "active_priority")},
        })
    for cap in captures.values():
        cap.release()
    queue = {
        "schema_version": 1, "mode": "action_active_learning",
        "model": str(args.model), "total": len(items),
        "training_only": True, "frozen_tests_used": False,
        "items": items,
    }
    (args.output / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    summary = {
        "total": len(items),
        "by_source": {str(source): sum(item["source"] == source for item in items)
                      for source in range(len(args.specs))},
        "v7_misses": sum(not item["action_metrics"]["v7_confidence"] for item in items),
        "v7_low_confidence_below_055": sum(
            0 < item["action_metrics"]["v7_confidence"] < .55 for item in items),
        "fast_frames": sum(item["action_metrics"]["speed"] >= 1.2 for item in items),
        "vertical_frames": sum(abs(item["action_metrics"]["vy"]) >= .7 for item in items),
        "unusual_pose_frames": sum(item["action_metrics"]["aspect"] >= .58 for item in items),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
