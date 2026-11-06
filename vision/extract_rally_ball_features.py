#!/usr/bin/env python3
"""Extract short full-frame ball-presence and trajectory features for rally state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def read_frames(video: str, timestamp: float,
                capture_cache: dict[str, cv2.VideoCapture]) -> list[np.ndarray] | None:
    capture = capture_cache.setdefault(video, cv2.VideoCapture(video))
    frames = []
    for offset in OFFSETS:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp + offset) * 1000)
        ok, frame = capture.read()
        if not ok:
            return None
        frames.append(frame)
    return frames


def normalized_detections(result, width: int, height: int) -> np.ndarray:
    """Return confidence, center x/y and relative area for every ball candidate."""
    if result.boxes is None or not len(result.boxes):
        return np.empty((0, 4), dtype=np.float32)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidence = result.boxes.conf.detach().cpu().numpy()
    center_x = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
    center_y = (boxes[:, 1] + boxes[:, 3]) / (2 * height)
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            / max(width * height, 1))
    return np.column_stack((confidence, center_x, center_y, area)).astype(np.float32)


def trajectory_features(frames: list[np.ndarray]) -> np.ndarray:
    """Describe visibility plus both top-confidence and moving-candidate paths."""
    counts, top = [], []
    for detections in frames:
        counts.append(float(len(detections)))
        if len(detections):
            top.append(detections[int(np.argmax(detections[:, 0]))])
        else:
            top.append(np.zeros(4, dtype=np.float32))
    counts_array = np.asarray(counts, dtype=np.float32)
    top_array = np.asarray(top, dtype=np.float32)
    visible = counts_array > 0
    top_motion, moving_motion, moving_support = [], [], []
    for first, second, first_visible, second_visible in zip(
            frames[:-1], frames[1:], visible[:-1], visible[1:]):
        if first_visible and second_visible:
            top_delta = top_array[len(top_motion) + 1, 1:3] - top_array[len(top_motion), 1:3]
            top_motion.append(float(np.linalg.norm(top_delta)))
        else:
            top_motion.append(0.0)
        if not len(first) or not len(second):
            moving_motion.append(0.0); moving_support.append(0.0)
            continue
        delta = first[:, None, 1:3] - second[None, :, 1:3]
        distances = np.linalg.norm(delta, axis=2)
        confidence = np.minimum(first[:, None, 0], second[None, :, 0])
        plausible = (distances >= .004) & (distances <= .45)
        score = np.where(plausible, confidence * np.sqrt(distances), -1.0)
        if np.max(score) < 0:
            moving_motion.append(0.0); moving_support.append(0.0)
            continue
        index = np.unravel_index(int(np.argmax(score)), score.shape)
        moving_motion.append(float(distances[index]))
        moving_support.append(float(confidence[index]))
    top_motion_array = np.asarray(top_motion, dtype=np.float32)
    moving_motion_array = np.asarray(moving_motion, dtype=np.float32)
    moving_support_array = np.asarray(moving_support, dtype=np.float32)
    summary = np.asarray([
        visible.mean(), counts_array.mean(), counts_array.max(),
        top_array[:, 0].mean(), top_array[:, 0].max(),
        top_motion_array.mean(), top_motion_array.max(), top_motion_array.std(),
        moving_motion_array.mean(), moving_motion_array.max(),
        moving_support_array.mean(), moving_support_array.max(),
        float(np.sum(visible[1:] != visible[:-1])),
        float(np.sum(moving_motion_array > .01)),
    ], dtype=np.float32)
    return np.concatenate((counts_array, top_array.reshape(-1), top_motion_array,
                           moving_motion_array, moving_support_array, summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-clips", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=.05)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    answers = json.loads(args.results.read_text())["answers"]
    items = queue["items"][::args.stride]
    if args.limit is not None:
        items = items[:args.limit]
    missing = [row["id"] for row in items if row["id"] not in answers]
    if missing:
        raise ValueError(f"missing {len(missing)} review answers")
    model = YOLO(str(args.model))
    vectors, labels, sources, item_ids = [], [], [], []
    capture_cache: dict[str, cv2.VideoCapture] = {}
    for start in range(0, len(items), args.batch_clips):
        batch = items[start:start + args.batch_clips]
        flat_frames, retained = [], []
        for item in batch:
            frames = read_frames(item["video"], float(item["time_s"]), capture_cache)
            if frames is None:
                raise RuntimeError(f"could not decode {item['id']}")
            flat_frames.extend(frames); retained.append((item, frames))
        predictions = model.predict(
            flat_frames, imgsz=args.imgsz, conf=args.confidence, classes=[0],
            device="cpu", batch=len(flat_frames), verbose=False,
        )
        pointer = 0
        for item, frames in retained:
            detections = [
                normalized_detections(predictions[pointer + index], frame.shape[1], frame.shape[0])
                for index, frame in enumerate(frames)
            ]
            pointer += len(frames)
            vectors.append(trajectory_features(detections))
            labels.append(answers[item["id"]]["decision"])
            sources.append(int(item["source"])); item_ids.append(item["id"])
        print(f"processed={min(start + args.batch_clips, len(items))}/{len(items)}", flush=True)
    for capture in capture_cache.values():
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(vectors),
                        labels=np.asarray(labels), sources=np.asarray(sources),
                        item_ids=np.asarray(item_ids))
    matrix = np.stack(vectors)
    visible_fraction = matrix[:, -14]
    by_label = {
        label: {
            "samples": int(np.sum(np.asarray(labels) == label)),
            "mean_visible_fraction": float(visible_fraction[np.asarray(labels) == label].mean()),
        }
        for label in sorted(set(labels))
    }
    print(json.dumps({"samples": len(items), "dimensions": int(matrix.shape[1]),
                      "offsets": OFFSETS, "by_label": by_label}, indent=2))


if __name__ == "__main__":
    main()
