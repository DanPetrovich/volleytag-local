#!/usr/bin/env python3
"""Build a source-balanced referee-gesture review queue from calibrated ROIs."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


TRAINING_SOURCES = (0, 1, 2, 3, 4, 7, 9)


def expanded_crop(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    x1, x2 = max(0, x1 - .5 * box_width), min(1, x2 + .5 * box_width)
    y1, y2 = max(0, y1 - .25 * box_height), min(1, y2 + .35 * box_height)
    left, top = round(x1 * width), round(y1 * height)
    crop_width, crop_height = round((x2 - x1) * width), round((y2 - y1) * height)
    left, top = left - left % 2, top - top % 2
    crop_width, crop_height = max(2, crop_width - crop_width % 2), max(2, crop_height - crop_height % 2)
    return left, top, crop_width, crop_height


def roi_motion(video: str, box: list[float], fps: int = 4) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    left, top, crop_width, crop_height = expanded_crop(box, width, height)
    command = [
        "ffmpeg", "-v", "error", "-i", video,
        "-vf", f"crop={crop_width}:{crop_height}:{left}:{top},fps={fps},scale=64:96,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    output = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    frame_size = 64 * 96
    frames = np.frombuffer(output[:len(output) // frame_size * frame_size], dtype=np.uint8)
    frames = frames.reshape(-1, 96, 64).astype(np.float32) / 255
    differences = np.abs(np.diff(frames, axis=0))
    scores = differences.mean(axis=(1, 2)) + differences[:, 12:72].std(axis=(1, 2))
    return scores, duration


def separated_peaks(scores: np.ndarray, fps: int, count: int,
                    occupied: list[float], separation_s: float = 8.0) -> list[tuple[float, float]]:
    result = []
    for index in np.argsort(scores)[::-1]:
        timestamp = (int(index) + 1) / fps
        if timestamp < 5 or all(abs(timestamp - value) >= separation_s
                                for value in occupied + [row[0] for row in result]):
            result.append((timestamp, float(scores[index])))
            if len(result) == count:
                break
    return sorted(result)


def item(source: int, video: str, timestamp: float, roi: list[float],
         reason: str, score: float | None = None) -> dict:
    row = {
        "id": f"referee-{reason}-s{source}-t{timestamp:.3f}",
        "source": source, "video": video, "time_s": timestamp,
        "window_start_s": max(0, timestamp - 3), "window_end_s": timestamp + 3,
        "roi_box_norm": roi, "selection_reason": reason,
    }
    if score is not None:
        row["motion_score"] = score
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("roi_queue", type=Path)
    parser.add_argument("roi_results", type=Path)
    parser.add_argument("whistle_dir", type=Path)
    parser.add_argument("whistle_queue", type=Path)
    parser.add_argument("whistle_results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-per-source", type=int, default=4)
    parser.add_argument("--random-per-source", type=int, default=2)
    args = parser.parse_args()
    event_payload = json.loads(args.events.read_text())
    roi_items = {int(row["source"]): row for row in json.loads(args.roi_queue.read_text())["items"]}
    roi_answers = json.loads(args.roi_results.read_text())["answers"]
    roi = {source: roi_answers[roi_items[source]["id"]]["box_norm"]
           for source in TRAINING_SOURCES}
    videos = {index: row["video"] for index, row in enumerate(event_payload["playlist"])}
    serves: dict[int, list[float]] = defaultdict(list)
    for row in event_payload["events"]:
        source = int(row["source"])
        if source in TRAINING_SOURCES and row.get("action") == "serve":
            serves[source].append(float(row["time_s"]))
    items, occupied = [], defaultdict(list)
    for source in TRAINING_SOURCES:
        whistles = json.loads((args.whistle_dir / f"source-{source}-whistles.json").read_text())
        whistle_times = [float(row["time_s"]) for row in whistles["candidates"]]
        for serve in serves[source]:
            eligible = [time for time in whistle_times if serve - 8 <= time <= serve + 1]
            timestamp = min(eligible, key=lambda time: abs(time - serve)) if eligible else serve - 3
            occupied[source].append(timestamp)
            items.append(item(source, videos[source], timestamp, roi[source], "serve_seed"))

    whistle_review = {row["id"]: row for row in json.loads(args.whistle_queue.read_text())["items"]}
    whistle_answers = json.loads(args.whistle_results.read_text())["answers"]
    for answer_id, answer in whistle_answers.items():
        if answer["decision"] != "rally_end_whistle":
            continue
        row = whistle_review[answer_id]
        source, timestamp = int(row["source"]), float(row["time_s"])
        if source in TRAINING_SOURCES:
            occupied[source].append(timestamp)
            items.append(item(source, videos[source], timestamp, roi[source], "reviewed_end_seed"))

    durations = {}
    for source in TRAINING_SOURCES:
        scores, duration = roi_motion(videos[source], roi[source])
        durations[source] = duration
        peaks = separated_peaks(scores, 4, args.motion_per_source, occupied[source])
        for timestamp, score in peaks:
            occupied[source].append(timestamp)
            items.append(item(source, videos[source], timestamp, roi[source], "roi_motion", score))
        print(f"source={source} duration={duration:.1f}s motion_peaks={len(peaks)}", flush=True)

    rng = random.Random(51)
    for source in TRAINING_SOURCES:
        chosen, attempts = [], 0
        while len(chosen) < args.random_per_source and attempts < 10000:
            attempts += 1
            timestamp = rng.uniform(5, durations[source] - 5)
            if all(abs(timestamp - value) >= 8 for value in occupied[source] + chosen):
                chosen.append(timestamp)
        for timestamp in chosen:
            items.append(item(source, videos[source], timestamp, roi[source], "random"))
    items.sort(key=lambda row: (row["source"], row["time_s"]))
    result = {
        "schema_version": 1, "kind": "volleytag.referee-gesture-review-queue",
        "frozen_tests_used": False,
        "labels": ["serve_authorization", "rally_end_team_to_serve", "other_signal",
                   "no_signal", "uncertain"],
        "counts": {reason: sum(row["selection_reason"] == reason for row in items)
                   for reason in ("serve_seed", "reviewed_end_seed", "roi_motion", "random")},
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": len(items), **result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
