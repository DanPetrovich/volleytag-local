#!/usr/bin/env python3
"""Measure whether two one-class detectors make complementary errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def xyxy(label: list[float], width: int, height: int) -> list[float]:
    _, cx, cy, bw, bh = label
    return [(cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height]


def iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.


def detections(result) -> list[dict]:
    if result.boxes is None:
        return []
    rows = [{"box": [float(value) for value in box], "confidence": float(confidence)}
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist())]
    return sorted(rows, key=lambda row: row["confidence"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--iou", type=float, default=.30)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest if row["split"] == args.split]
    paths = [args.dataset / row["image"] for row in rows]
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise SystemExit("A dataset image could not be decoded")

    predictions_a = list(YOLO(str(args.model_a)).predict(
        [str(path) for path in paths], imgsz=args.imgsz, conf=args.conf,
        device="cpu", batch=8, verbose=False, stream=True))
    predictions_b = list(YOLO(str(args.model_b)).predict(
        [str(path) for path in paths], imgsz=args.imgsz, conf=args.conf,
        device="cpu", batch=8, verbose=False, stream=True))

    details = []
    for row, frame, result_a, result_b in zip(rows, frames, predictions_a, predictions_b):
        first, second = detections(result_a), detections(result_b)
        label = row.get("box_yolo")
        gt = xyxy(label, frame.shape[1], frame.shape[0]) if label else None
        hit_a = bool(gt and first and iou(gt, first[0]["box"]) >= args.iou)
        hit_b = bool(gt and second and iou(gt, second[0]["box"]) >= args.iou)
        candidate_a = bool(gt and any(iou(gt, item["box"]) >= args.iou for item in first))
        candidate_b = bool(gt and any(iou(gt, item["box"]) >= args.iou for item in second))
        details.append({
            "image": row["image"], "source": row["source"], "action": row.get("action"),
            "positive": gt is not None, "a_top1": hit_a, "b_top1": hit_b,
            "a_candidate": candidate_a, "b_candidate": candidate_b,
            "a_count": len(first), "b_count": len(second),
            "a_confidence": first[0]["confidence"] if first else 0.,
            "b_confidence": second[0]["confidence"] if second else 0.,
        })

    visible = [row for row in details if row["positive"]]
    negatives = [row for row in details if not row["positive"]]
    ratio = lambda key: sum(row[key] for row in visible) / len(visible) if visible else 0.
    summary = {
        "frames": len(details), "visible": len(visible), "negatives": len(negatives),
        "a_top1": ratio("a_top1"), "b_top1": ratio("b_top1"),
        "a_candidate": ratio("a_candidate"), "b_candidate": ratio("b_candidate"),
        "union_candidate": sum(row["a_candidate"] or row["b_candidate"] for row in visible) / len(visible),
        "both_top1_correct": sum(row["a_top1"] and row["b_top1"] for row in visible),
        "only_a_top1_correct": sum(row["a_top1"] and not row["b_top1"] for row in visible),
        "only_b_top1_correct": sum(row["b_top1"] and not row["a_top1"] for row in visible),
        "neither_top1_correct": sum(not row["a_top1"] and not row["b_top1"] for row in visible),
        "only_b_candidate_recovery": sum(row["b_candidate"] and not row["a_candidate"] for row in visible),
        "only_a_candidate_recovery": sum(row["a_candidate"] and not row["b_candidate"] for row in visible),
        "a_absent_detection_rate": sum(row["a_count"] > 0 for row in negatives) / len(negatives) if negatives else 0.,
        "b_absent_detection_rate": sum(row["b_count"] > 0 for row in negatives) / len(negatives) if negatives else 0.,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "details": details}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
