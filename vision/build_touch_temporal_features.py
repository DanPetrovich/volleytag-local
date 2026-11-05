#!/usr/bin/env python3
"""Build cached short-clip features for touch/no-touch temporal sanity checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def excluded_intervals(video: Path) -> list[tuple[float, float]]:
    directory = video.parent
    names = (["far-non-match-intervals.json", "non-match-intervals.json"]
             if "far-source" in video.name else ["non-match-intervals.json"])
    for name in names:
        path = directory / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [(float(row.get("start_s", row.get("start", 0))),
                     float(row.get("end_s", row.get("end", 0))))
                    for row in payload.get("excluded_intervals", [])]
    return []


def inside_excluded(value: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start <= value <= end for start, end in intervals)


def sample_negative_candidates(events: list[dict], multiplier: int) -> list[dict]:
    by_source: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        by_source[int(event["source"])].append(event)
    rows = []
    for source, source_events in sorted(by_source.items()):
        video = Path(source_events[0]["video"])
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        times = sorted(float(row["time_s"]) for row in source_events)
        start, end = max(1.0, times[0] - 12.0), min(duration - 1.0, times[-1] + 12.0)
        exclusions = excluded_intervals(video)
        candidates = []
        for value in np.arange(start, end, 1.25):
            timestamp = float(round(value, 3))
            if min(abs(timestamp - event_time) for event_time in times) < 2.0:
                continue
            if any(inside_excluded(timestamp + offset, exclusions) for offset in OFFSETS):
                continue
            candidates.append({"source": source, "video": str(video),
                               "time_s": timestamp, "label": 0,
                               "action": "non_touch"})
        target = min(len(candidates), len(source_events) * multiplier)
        if target:
            indexes = np.linspace(0, len(candidates) - 1, target, dtype=int)
            rows.extend(candidates[index] for index in indexes)
    return rows


def read_clip(video: str, timestamp: float) -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(video)
    frames = []
    for offset in OFFSETS:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp + offset) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(frame)
    cap.release()
    return frames


def crop_person(frame: np.ndarray, box: list[float] | None) -> tuple[np.ndarray, list[float]]:
    height, width = frame.shape[:2]
    if box is None:
        return np.zeros((96, 64, 3), dtype=np.uint8), [0.0] * 6
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1, x2 = max(0, int(x1 - .35 * bw)), min(width, int(x2 + .35 * bw))
    y1, y2 = max(0, int(y1 - .20 * bh)), min(height, int(y2 + .15 * bh))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((96, 64, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (64, 96), interpolation=cv2.INTER_AREA)
    return crop, [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                  (x2 - x1) / width, (y2 - y1) / height, 1.0, bh / height]


def clip_feature(frames: list[np.ndarray], results: list) -> tuple[np.ndarray, float, int]:
    crops, geometry, confidences = [], [], []
    for frame, result in zip(frames, results):
        box, confidence = None, 0.0
        if result.boxes is not None and len(result.boxes):
            scores = result.boxes.conf.detach().cpu().numpy()
            index = int(np.argmax(scores))
            box = result.boxes.xyxy[index].detach().cpu().tolist()
            confidence = float(scores[index])
        crop, geom = crop_person(frame, box)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
        geometry.append([*geom, confidence])
        confidences.append(confidence)
    tiny = [cv2.resize(crop, (16, 24), interpolation=cv2.INTER_AREA).reshape(-1)
            for crop in crops]
    differences = [np.abs(tiny[index + 1] - tiny[index])
                   for index in range(len(tiny) - 1)]
    difference_stats = []
    for delta in differences:
        difference_stats.extend([float(delta.mean()), float(delta.std()),
                                 float(np.quantile(delta, .75)), float(delta.max())])
    geom = np.asarray(geometry, dtype=np.float32)
    velocity = np.diff(geom[:, :4], axis=0).reshape(-1)
    acceleration = np.diff(geom[:, :4], n=2, axis=0).reshape(-1)
    appearance = np.concatenate([*tiny, *differences])
    feature = np.concatenate([geom.reshape(-1), velocity, acceleration,
                              np.asarray(difference_stats, dtype=np.float32), appearance])
    motion = float(np.mean([delta.mean() for delta in differences]) +
                   2.0 * np.mean(np.abs(velocity)))
    return feature.astype(np.float32), motion, sum(value > 0 for value in confidences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-multiplier", type=int, default=4)
    parser.add_argument("--batch-clips", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    positives = [{"source": int(row["source"]), "video": row["video"],
                  "time_s": float(row["time_s"]), "label": 1,
                  "action": row.get("action") or "unknown", "id": row["id"]}
                 for row in payload["events"]]
    negatives = sample_negative_candidates(positives, args.negative_multiplier)
    rows = positives + negatives
    model = YOLO(str(args.model))
    device = device_name(args.device)
    features, metadata = [], []
    for start in range(0, len(rows), args.batch_clips):
        batch_rows = rows[start:start + args.batch_clips]
        valid = []
        flat_frames = []
        for row in batch_rows:
            clip = read_clip(row["video"], row["time_s"])
            if clip is not None:
                valid.append((row, clip))
                flat_frames.extend(clip)
        if not valid:
            continue
        results = model.predict(flat_frames, imgsz=960, conf=.05, device=device,
                                batch=min(20, len(flat_frames)), verbose=False)
        pointer = 0
        for row, clip in valid:
            count = len(clip)
            feature, motion, detected = clip_feature(clip, results[pointer:pointer + count])
            pointer += count
            features.append(feature)
            metadata.append({**row, "motion": motion, "detected_frames": detected})
        print(f"processed={min(start + args.batch_clips, len(rows))}/{len(rows)}", flush=True)

    # Keep one highest-motion negative per positive for every whole source.
    selected = []
    by_source: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_source[int(row["source"])].append(index)
    for source, indexes in sorted(by_source.items()):
        positive = [index for index in indexes if metadata[index]["label"] == 1]
        negative = sorted((index for index in indexes if metadata[index]["label"] == 0),
                          key=lambda index: metadata[index]["motion"], reverse=True)
        selected.extend(positive)
        selected.extend(negative[:len(positive)])
    selected.sort(key=lambda index: (metadata[index]["source"], metadata[index]["time_s"],
                                     metadata[index]["label"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        features=np.stack([features[index] for index in selected]),
                        labels=np.asarray([metadata[index]["label"] for index in selected]),
                        sources=np.asarray([metadata[index]["source"] for index in selected]))
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "schema_version": 1, "offsets_s": OFFSETS,
        "candidates_before_hard_negative_selection": len(metadata),
        "samples": [metadata[index] for index in selected],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(selected), "positives": len(positives),
                      "feature_dimensions": int(features[0].shape[0]),
                      "device": device}, indent=2))


if __name__ == "__main__":
    main()
