#!/usr/bin/env python3
"""Evaluate a direct one-class Danya detector on frozen human keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def keyframes(payload):
    if payload.get("schema") == "volleytag.player51-uniform-evaluation.v1":
        return [row for row in payload.get("samples", [])
                if row.get("state") == "visible" and len(row.get("box", [])) == 4]
    return [row for trajectory in payload.get("trajectory_intervals", [])
            for row in trajectory.get("keyframes", []) if len(row.get("box", [])) == 4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--iou", type=float, default=.30)
    args = parser.parse_args()

    rows = keyframes(json.loads(args.ground_truth.read_text(encoding="utf-8")))
    cap, frames, valid = cv2.VideoCapture(str(args.video)), [], []
    for row in rows:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(row["time_s"]) * 1000)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            valid.append(row)
    cap.release()

    model = YOLO(str(args.model))
    results = model.predict(frames, imgsz=args.imgsz, conf=args.conf, verbose=False,
                            device="cpu", batch=1)
    details, hits, top1_hits, predictions = [], 0, 0, 0
    for row, result in zip(valid, results):
        gt = list(map(float, row["box"]))
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        overlaps = [iou(gt, box) for box in boxes]
        hit = any(value >= args.iou for value in overlaps)
        top1 = bool(overlaps and overlaps[int(np.argmax(scores))] >= args.iou)
        hits += int(hit); top1_hits += int(top1); predictions += len(boxes)
        details.append({"time_s": float(row["time_s"]), "predictions": int(len(boxes)),
                        "best_iou": float(max(overlaps, default=0.)),
                        "hit": bool(hit), "top1_hit": bool(top1)})
    total = len(valid)
    metrics = {"model": str(args.model), "keyframes": total,
               "recall_any_prediction": hits / total if total else 0.,
               "top1_recall": top1_hits / total if total else 0.,
               "predictions_per_frame": predictions / total if total else 0.,
               "confidence": args.conf, "iou_threshold": args.iou, "imgsz": args.imgsz}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
