#!/usr/bin/env python3
"""Compare full-frame and zoomed YOLO person detection on frozen uniform GT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def overlap(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def parse_crop(value: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not value:
        return 0, 0, width, height
    crop = tuple(int(part) for part in value.split(","))
    if len(crop) != 4:
        raise SystemExit("--crop must be x1,y1,x2,y2")
    x1, y1, x2, y2 = crop
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(f"crop {crop} is outside {width}x{height}")
    return crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--crop", help="Detection crop x1,y1,x2,y2; boxes are restored to frame coordinates")
    parser.add_argument("--time-range", help="Only evaluate start,end seconds")
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    samples = [sample for sample in payload.get("samples", [])
               if sample.get("state") == "visible" and len(sample.get("box", [])) == 4]
    if args.time_range:
        start, end = (float(value) for value in args.time_range.split(","))
        samples = [sample for sample in samples if start <= float(sample["time_s"]) < end]

    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1, x2, y2 = parse_crop(args.crop, width, height)
    model = YOLO(args.model)
    rows = []
    started = time.perf_counter()
    # Decode sequentially. Random CAP_PROP_POS_MSEC seeking is inaccurate on the
    # concatenated/VFR test video and can silently evaluate a neighbouring scene.
    sample_index = 0
    first_pts = None
    while sample_index < len(samples):
        ok, frame = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_pts is None:
            first_pts = pts
        timestamp = max(0.0, pts - first_pts)
        sample = samples[sample_index]
        time_s = float(sample["time_s"])
        if timestamp + 0.02 < time_s:
            continue
        boxes: list[list[float]] = []
        result = model.predict(frame[y1:y2, x1:x2], classes=[0], imgsz=args.imgsz,
                               conf=args.conf, device=args.device, verbose=False)[0]
        if result.boxes is not None:
            boxes = [[float(box[0]) + x1, float(box[1]) + y1,
                      float(box[2]) + x1, float(box[3]) + y1]
                     for box in result.boxes.xyxy.cpu().numpy()]
        gt_box = [float(value) for value in sample["box"]]
        best_iou = max((overlap(gt_box, box) for box in boxes), default=0.0)
        rows.append({"time_s": time_s, "candidate_count": len(boxes),
                     "best_iou": best_iou, "detected": best_iou >= args.iou})
        sample_index += 1
        if sample_index % 20 == 0:
            print(f"processed={sample_index}/{len(samples)}", flush=True)
    cap.release()

    detected = sum(row["detected"] for row in rows)
    metrics = {
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "crop": [x1, y1, x2, y2],
        "samples": len(rows),
        "detected": detected,
        "recall": detected / len(rows) if rows else 0.0,
        "iou_threshold": args.iou,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "samples.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
