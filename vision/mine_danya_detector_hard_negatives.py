#!/usr/bin/env python3
"""Mine confident wrong-player detections from the detector training split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def label_boxes(dataset: Path, row: dict, width: int, height: int) -> list[list[float]]:
    label = dataset / row["image"].replace("images/", "labels/").rsplit(".", 1)[0]
    label = label.with_suffix(".txt")
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        _cls, cx, cy, bw, bh = map(float, line.split())
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height])
    return boxes


def balanced(rows: list[dict], limit: int) -> list[dict]:
    if not limit or len(rows) <= limit:
        return rows
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["source"], row.get("domain") or "unknown")].append(row)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--positive-iou", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest
            if row["split"] == "train" and row["kind"] in {"positive", "manual_positive"}]
    rows = balanced(rows, args.max_images)
    paths = [str(args.dataset / row["image"]) for row in rows]
    predictions = YOLO(str(args.model)).predict(paths, imgsz=args.imgsz, conf=args.conf,
                                                verbose=False, device="cpu", batch=1)
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images" / "train"
    labels_dir = args.output / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    mined = []
    for row, result in zip(rows, predictions):
        image = cv2.imread(str(args.dataset / row["image"]))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = label_boxes(args.dataset, row, width, height)
        boxes = (result.boxes.xyxy.cpu().numpy() if result.boxes is not None
                 else np.empty((0, 4)))
        scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                  else np.empty(0))
        wrong = []
        for box, score in zip(boxes, scores):
            overlap = max((iou(list(box), target) for target in gt), default=0.0)
            if overlap < args.positive_iou:
                wrong.append((float(score), list(map(float, box))))
        # One strongest mistake per source frame limits near-duplicate players.
        if not wrong:
            continue
        score, box = max(wrong)
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(round(x1 - args.padding * bw)))
        x2 = min(width, int(round(x2 + args.padding * bw)))
        y1 = max(0, int(round(y1 - args.padding * bh)))
        y2 = min(height, int(round(y2 + args.padding * bh)))
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < 32 or crop.shape[1] < 16:
            continue
        stem = f"s{row['source']}-f{row['frame']}-auto-negative"
        image_rel = Path("images/train") / f"{stem}.jpg"
        label_rel = Path("labels/train") / f"{stem}.txt"
        cv2.imwrite(str(args.output / image_rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (args.output / label_rel).write_text("", encoding="utf-8")
        mined.append({
            "source": row["source"], "domain": row.get("domain") or "unknown",
            "time_s": row["time_s"], "frame": row["frame"], "confidence": score,
            "box": box, "image": str(image_rel), "label": str(label_rel),
        })

    report = {
        "scanned_images": len(rows),
        "mined_negatives": len(mined),
        "yield": len(mined) / len(rows) if rows else 0.0,
        "by_source": dict(Counter(str(row["source"]) for row in mined)),
        "by_domain": dict(Counter(row["domain"] for row in mined)),
        "confidence": args.conf,
        "positive_iou_ceiling": args.positive_iou,
    }
    (args.output / "manifest.json").write_text(json.dumps(mined, indent=2) + "\n",
                                                encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
