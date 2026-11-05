#!/usr/bin/env python3
"""Evaluate a ball detector on reviewed positive and negative crop frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-class", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    items = [row for row in json.loads(args.manifest.read_text(encoding="utf-8"))["items"]
             if row["split"] == args.split]
    model, device = YOLO(str(args.model)), device_name(args.device)
    predictions = model.predict([row["image"] for row in items], imgsz=args.imgsz, conf=args.conf,
                                classes=[args.model_class], device=device, batch=8, verbose=False)
    positive, negative, matched, total_predictions, clean_negative, best_ious, details = 0, 0, 0, 0, 0, [], []
    for row, result in zip(items, predictions):
        label_text = Path(row["label"]).read_text().strip()
        gt = None
        if label_text:
            _, cx, cy, width, height = map(float, label_text.split())
            image_height, image_width = result.orig_shape
            gt = [(cx - width / 2) * image_width, (cy - height / 2) * image_height,
                  (cx + width / 2) * image_width, (cy + height / 2) * image_height]
            positive += 1
        else:
            negative += 1
        boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
        total_predictions += len(boxes)
        best = max((iou(gt, box) for box in boxes), default=0.0) if gt else 0.0
        if gt:
            best_ious.append(best)
            if best >= args.iou:
                matched += 1
        elif not boxes:
            clean_negative += 1
        details.append({"id": row["id"], "positive": bool(gt), "predictions": len(boxes),
                        "best_iou": round(best, 6), "matched": bool(gt and best >= args.iou)})
    report = {
        "model": str(args.model), "split": args.split, "images": len(items),
        "positive_images": positive, "negative_images": negative,
        "positive_frame_recall": matched / positive if positive else None,
        "negative_frame_specificity": clean_negative / negative if negative else None,
        "matched_positive_frames": matched, "total_predictions": total_predictions,
        "frame_detection_precision": matched / total_predictions if total_predictions else 0.0,
        "mean_best_iou_positive": float(np.mean(best_ious)) if best_ious else None,
        "protocol": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                     "model_class": args.model_class, "device": device},
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
