#!/usr/bin/env python3
"""Detect people only on the fixed timestamps used by Uniform test v1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    targets = []
    timestamp = args.step / 2
    while timestamp < duration:
        targets.append(round(timestamp, 3))
        timestamp += args.step

    model = YOLO(args.model)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "frame", "track_id", "confidence", "x1", "y1", "x2", "y2"))
        index = 0
        source_frame = 0
        first_pts = None
        while index < len(targets):
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if first_pts is None:
                first_pts = pts
            timestamp = max(0.0, pts - first_pts)
            target = targets[index]
            if timestamp + 0.02 < target:
                source_frame += 1
                continue
            result = model.predict(frame, classes=[0], imgsz=args.imgsz, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for candidate, (box, confidence) in enumerate(zip(boxes, confidences)):
                    writer.writerow((f"{target:.3f}", source_frame, index * 1000 + candidate,
                                     f"{float(confidence):.4f}", *[int(value) for value in box]))
            index += 1
            source_frame += 1
            if (index + 1) % 25 == 0:
                print(f"processed={index + 1}/{len(targets)}", flush=True)
    cap.release()
    print(f"samples={len(targets)} output={args.output}")


if __name__ == "__main__":
    main()
