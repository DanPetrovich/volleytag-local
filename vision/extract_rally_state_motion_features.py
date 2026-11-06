#!/usr/bin/env python3
"""Extract cheap full-court motion features for rally-state clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


OFFSETS = (-3, -2, -1, 0, 1, 2)


def motion_features(video: str, timestamp: float,
                    frame_cache: dict[tuple[str, float], np.ndarray] | None = None,
                    capture_cache: dict[str, cv2.VideoCapture] | None = None) -> np.ndarray | None:
    frame_cache = frame_cache if frame_cache is not None else {}
    capture_cache = capture_cache if capture_cache is not None else {}
    capture = capture_cache.setdefault(video, cv2.VideoCapture(video))
    frames = []
    for offset in OFFSETS:
        frame_time = round(max(0, timestamp + offset), 3)
        key = (video, frame_time)
        if key not in frame_cache:
            capture.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            ok, frame = capture.read()
            if not ok:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_cache[key] = (cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
                                .astype(np.float32) / 255)
        frames.append(frame_cache[key])
    differences = np.asarray([np.abs(frames[index + 1] - frames[index])
                              for index in range(len(frames) - 1)])
    result = []
    for difference in differences:
        result.extend([difference.mean(), difference.std(),
                       np.quantile(difference, .75), np.quantile(difference, .95)])
    for difference in differences:
        result.extend(part.mean() for part in np.array_split(difference, 3, axis=0))
    result.extend([differences[:2].mean(), differences[2:].mean(),
                   differences[-2:].mean() - differences[:2].mean()])
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"] if args.results else None
    vectors, labels, sources, item_ids = [], [], [], []
    frame_cache: dict[tuple[str, float], np.ndarray] = {}
    capture_cache: dict[str, cv2.VideoCapture] = {}
    for index, item in enumerate(items, 1):
        if answers and item["id"] not in answers:
            raise ValueError(f"missing answer for {item['id']}")
        vector = motion_features(item["video"], float(item["time_s"]),
                                 frame_cache, capture_cache)
        if vector is None:
            raise RuntimeError(f"could not decode {item['id']}")
        vectors.append(vector)
        labels.append(answers[item["id"]]["decision"] if answers else "")
        sources.append(int(item["source"]))
        item_ids.append(item["id"])
        print(f"processed={index}/{len(items)}", flush=True)
    for capture in capture_cache.values():
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(vectors),
                        labels=np.asarray(labels), sources=np.asarray(sources),
                        item_ids=np.asarray(item_ids))
    print(json.dumps({"samples": len(items), "dimensions": len(vectors[0])}, indent=2))


if __name__ == "__main__":
    main()
