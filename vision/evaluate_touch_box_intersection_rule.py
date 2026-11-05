#!/usr/bin/env python3
"""Evaluate a fixed temporal intersection rule using actual Danya and ball boxes."""

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


def read_clip(video: str, timestamp: float, offsets):
    cap = cv2.VideoCapture(video)
    frames = []
    for offset in offsets:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp + offset) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(frame)
    cap.release()
    return frames


def top_box(result):
    if result.boxes is None or not len(result.boxes):
        return None
    confidence = result.boxes.conf.detach().cpu().numpy()
    return result.boxes.xyxy[int(np.argmax(confidence))].detach().cpu().tolist()


def context_crop(frame, box):
    if box is None:
        return np.zeros((240, 240, 3), dtype=np.uint8), None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(240.0, (y2 - y1) * 3.0)
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    crop = frame[top:bottom, left:right]
    return (crop if crop.size else np.zeros((240, 240, 3), dtype=np.uint8),
            (left, top) if crop.size else None)


def intersects_expanded(danya, ball, expansion: float) -> bool:
    x1, y1, x2, y2 = danya
    width, height = x2 - x1, y2 - y1
    x1, y1, x2, y2 = (x1 - expansion * width, y1 - expansion * height,
                      x2 + expansion * width, y2 + expansion * height)
    return min(x2, ball[2]) > max(x1, ball[0]) and min(y2, ball[3]) > max(y1, ball[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("personal_model", type=Path)
    parser.add_argument("ball_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expansion", type=float, default=0.25)
    parser.add_argument("--batch-clips", type=int, default=6)
    parser.add_argument("--dense-fps", type=float, default=0.0)
    parser.add_argument("--window", type=float, default=0.75)
    parser.add_argument("--samples-per-label-source", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = json.loads(args.metadata.read_text(encoding="utf-8"))["samples"]
    if args.samples_per_label_source:
        selected = []
        for source in sorted({int(row["source"]) for row in rows}):
            for label in (0, 1):
                group = sorted((row for row in rows if int(row["source"]) == source
                                and int(row["label"]) == label),
                               key=lambda row: (float(row["time_s"]), str(row.get("id", ""))))
                count = min(args.samples_per_label_source, len(group))
                indexes = np.linspace(0, len(group) - 1, count, dtype=int)
                selected.extend(group[index] for index in indexes)
        rows = selected
    offsets = (tuple(np.arange(-args.window, args.window + 1e-9, 1.0 / args.dense_fps))
               if args.dense_fps else OFFSETS)
    device = device_name(args.device)
    personal, ball = YOLO(str(args.personal_model)), YOLO(str(args.ball_model))
    details = []
    for start in range(0, len(rows), args.batch_clips):
        batch_rows = rows[start:start + args.batch_clips]
        clips = [read_clip(row["video"], float(row["time_s"]), offsets) for row in batch_rows]
        frames = [frame for clip in clips if clip for frame in clip]
        personal_results = personal.predict(frames, imgsz=960, conf=.05, device=device,
                                            batch=min(18, len(frames)), verbose=False)
        danya_boxes = [top_box(result) for result in personal_results]
        crops_and_origins = [context_crop(frame, box) for frame, box in zip(frames, danya_boxes)]
        ball_results = ball.predict([row[0] for row in crops_and_origins], imgsz=960, conf=.05,
                                    classes=[0], device=device, batch=min(18, len(frames)), verbose=False)
        pointer = 0
        for row, clip in zip(batch_rows, clips):
            frame_hits, ball_frames, danya_frames = [], 0, 0
            for _ in clip or []:
                danya_box, result, origin = danya_boxes[pointer], ball_results[pointer], crops_and_origins[pointer][1]
                pointer += 1
                if danya_box is not None:
                    danya_frames += 1
                boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
                if boxes:
                    ball_frames += 1
                full_boxes = ([[box[0] + origin[0], box[1] + origin[1],
                                box[2] + origin[0], box[3] + origin[1]] for box in boxes]
                              if origin else [])
                frame_hits.append(bool(danya_box and any(
                    intersects_expanded(danya_box, box, args.expansion) for box in full_boxes)))
            details.append({"id": row.get("id"), "source": int(row["source"]),
                            "label": int(row["label"]), "predicted": int(any(frame_hits)),
                            "hit_frames": sum(frame_hits), "ball_frames": ball_frames,
                            "danya_frames": danya_frames})
        print(f"processed={min(start + args.batch_clips, len(rows))}/{len(rows)}", flush=True)

    labels = np.asarray([row["label"] for row in details])
    predicted = np.asarray([row["predicted"] for row in details])
    positive, negative = labels == 1, labels == 0
    recall = float((predicted[positive] == 1).mean())
    specificity = float((predicted[negative] == 0).mean())
    report = {"samples": len(details), "positive_recall": recall,
              "negative_specificity": specificity, "passed": recall >= .70 and specificity >= .70,
              "rule": {"expansion": args.expansion, "offsets_s": offsets,
                       "dense_fps": args.dense_fps or None,
                       "samples_per_label_source": args.samples_per_label_source or None},
              "frozen_tests_used": False, "details": details}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
