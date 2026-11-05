#!/usr/bin/env python3
"""Augment cached temporal clips with player-pose and sports-ball features."""

from __future__ import annotations

import argparse
import json
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


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def top_box(result) -> list[float] | None:
    if result.boxes is None or not len(result.boxes):
        return None
    scores = result.boxes.conf.detach().cpu().numpy()
    return result.boxes.xyxy[int(np.argmax(scores))].detach().cpu().tolist()


def pose_frame_feature(danya_box: list[float] | None, result) -> np.ndarray:
    output = np.zeros((17, 3), dtype=np.float32)
    if danya_box is None or result.boxes is None or not len(result.boxes) or result.keypoints is None:
        return output.reshape(-1)
    pose_boxes = result.boxes.xyxy.detach().cpu().tolist()
    overlaps = [iou(danya_box, box) for box in pose_boxes]
    index = int(np.argmax(overlaps))
    if overlaps[index] < .10:
        return output.reshape(-1)
    points = result.keypoints.xy[index].detach().cpu().numpy()
    confidences = result.keypoints.conf[index].detach().cpu().numpy()
    x1, y1, x2, y2 = danya_box
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    output[:, 0] = (points[:, 0] - x1) / width
    output[:, 1] = (points[:, 1] - y1) / height
    output[:, 2] = confidences
    output[confidences < .10] = 0.0
    return output.reshape(-1)


def person_context_crop(frame: np.ndarray, danya_box: list[float] | None):
    if danya_box is None:
        return np.zeros((240, 240, 3), dtype=np.uint8), None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = danya_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(240.0, (y2 - y1) * 3.0)
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros((240, 240, 3), dtype=np.uint8), None
    return crop, (left, top)


def ball_frame_feature(danya_box: list[float] | None, result, crop_origin) -> np.ndarray:
    output = np.zeros(5, dtype=np.float32)
    if danya_box is None or crop_origin is None or result.boxes is None or not len(result.boxes):
        return output
    x1, y1, x2, y2 = danya_box
    center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = max(20.0, y2 - y1)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2 + crop_origin[0],
                               (boxes[:, 1] + boxes[:, 3]) / 2 + crop_origin[1]))
    distances = np.linalg.norm(centers - center, axis=1) / scale
    index = int(np.argmin(distances))
    delta = (centers[index] - center) / scale
    output[:] = [1.0, float(scores[index]), float(delta[0]), float(delta[1]),
                 float(distances[index])]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_features", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("personal_model", type=Path)
    parser.add_argument("pose_model", type=Path)
    parser.add_argument("ball_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-clips", type=int, default=6)
    parser.add_argument("--ball-class", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    base = np.load(args.base_features)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))["samples"]
    if len(metadata) != len(base["features"]):
        raise SystemExit("feature/metadata length mismatch")
    personal, pose, ball = YOLO(str(args.personal_model)), YOLO(str(args.pose_model)), YOLO(str(args.ball_model))
    device, augmentations = device_name(args.device), []
    for start in range(0, len(metadata), args.batch_clips):
        rows = metadata[start:start + args.batch_clips]
        flat_frames, valid_counts = [], []
        for row in rows:
            clip = read_clip(row["video"], float(row["time_s"]))
            valid_counts.append(len(clip) if clip is not None else 0)
            if clip is not None:
                flat_frames.extend(clip)
        if not flat_frames:
            augmentations.extend([np.zeros(526, dtype=np.float32) for _ in rows])
            continue
        personal_results = personal.predict(flat_frames, imgsz=960, conf=.05, device=device,
                                            batch=min(18, len(flat_frames)), verbose=False)
        pose_results = pose.predict(flat_frames, imgsz=960, conf=.10, device=device,
                                    batch=min(18, len(flat_frames)), verbose=False)
        danya_boxes = [top_box(result) for result in personal_results]
        ball_crops_and_origins = [person_context_crop(frame, box)
                                  for frame, box in zip(flat_frames, danya_boxes)]
        ball_results = ball.predict([row[0] for row in ball_crops_and_origins], imgsz=960, conf=.05,
                                    classes=[args.ball_class], device=device,
                                    batch=min(18, len(flat_frames)), verbose=False)
        pointer = 0
        for count in valid_counts:
            if not count:
                augmentations.append(np.zeros(526, dtype=np.float32))
                continue
            pose_frames, ball_frames = [], []
            for offset in range(count):
                box = danya_boxes[pointer + offset]
                pose_frames.append(pose_frame_feature(box, pose_results[pointer + offset]))
                ball_frames.append(ball_frame_feature(
                    box, ball_results[pointer + offset], ball_crops_and_origins[pointer + offset][1]))
            pointer += count
            pose_array = np.asarray(pose_frames, dtype=np.float32).reshape(count, 17, 3)
            pose_velocity = np.diff(pose_array[:, :, :2], axis=0).reshape(-1)
            pose_acceleration = np.diff(pose_array[:, :, :2], n=2, axis=0).reshape(-1)
            ball_array = np.asarray(ball_frames, dtype=np.float32)
            ball_velocity = np.diff(ball_array[:, 2:4], axis=0).reshape(-1)
            augmentations.append(np.concatenate([pose_array.reshape(-1), pose_velocity,
                                                 pose_acceleration, ball_array.reshape(-1),
                                                 ball_velocity]).astype(np.float32))
        print(f"processed={min(start + args.batch_clips, len(metadata))}/{len(metadata)}", flush=True)
    augmented = np.concatenate([base["features"], np.stack(augmentations)], axis=1)
    np.savez_compressed(args.output, features=augmented, labels=base["labels"], sources=base["sources"])
    print(json.dumps({"samples": len(augmented), "base_dimensions": int(base["features"].shape[1]),
                      "pose_ball_dimensions": int(len(augmentations[0])),
                      "combined_dimensions": int(augmented.shape[1]), "device": device}, indent=2))


if __name__ == "__main__":
    main()
