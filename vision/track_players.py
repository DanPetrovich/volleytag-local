#!/usr/bin/env python3
"""Local-first player detection and tracking pilot for VolleyTag.

This is deliberately only the first stage of the pipeline. It finds people,
assigns temporary track IDs and exports representative crops. A later ReID
stage can join fragmented tracks and decide which ones belong to player #51.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class BestCrop:
    score: float
    image: np.ndarray
    timestamp: float


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_roi(value: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not value:
        return 0, 0, width, height
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI {value} is outside {width}x{height}")
    return x1, y1, x2, y2


def contact_sheet(crops: dict[int, BestCrop], output: Path) -> None:
    if not crops:
        return
    tile_w, tile_h = 220, 300
    cols = 6
    rows = (len(crops) + cols - 1) // cols
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 238, dtype=np.uint8)
    for index, (track_id, crop) in enumerate(sorted(crops.items())):
        row, col = divmod(index, cols)
        image = crop.image
        scale = min((tile_w - 16) / image.shape[1], (tile_h - 48) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale)
        y = row * tile_h + 34
        x = col * tile_w + (tile_w - resized.shape[1]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        label = f"ID {track_id}  {crop.timestamp:.1f}s"
        cv2.putText(sheet, label, (col * tile_w + 8, row * tile_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision/runs/latest"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--stride", type=int, default=3,
                        help="Analyze every Nth source frame")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--roi", help="Optional court crop: x1,y1,x2,y2")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x1, y1, x2, y2 = parse_roi(args.roi, width, height)
    roi_w, roi_h = x2 - x1, y2 - y1
    device = choose_device(args.device)

    output_video = args.output / "tracked.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (roi_w, roi_h)
    )
    model = YOLO(args.model)
    csv_file = (args.output / "tracks.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(("time_s", "frame", "track_id", "confidence", "x1", "y1", "x2", "y2"))

    best_crops: dict[int, BestCrop] = {}
    read_frames = processed_frames = detections = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_frame = read_frames
            read_frames += 1
            timestamp = source_frame / fps
            if args.max_seconds and timestamp >= args.max_seconds:
                break
            if source_frame % args.stride:
                continue

            court = frame[y1:y2, x1:x2]
            results = model.track(
                court,
                persist=True,
                classes=[0],
                tracker=args.tracker,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )[0]
            annotated = court.copy()
            boxes = results.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for box, track_id, confidence in zip(xyxy, ids, confidences):
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                    detections += 1
                    csv_writer.writerow((f"{timestamp:.3f}", source_frame, track_id,
                                         f"{confidence:.4f}", bx1, by1, bx2, by2))
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (40, 220, 70), 3)
                    cv2.putText(annotated, f"ID {track_id}", (bx1, max(24, by1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 220, 70), 2, cv2.LINE_AA)
                    crop = court[max(0, by1):min(roi_h, by2), max(0, bx1):min(roi_w, bx2)]
                    if crop.size:
                        area = crop.shape[0] * crop.shape[1]
                        score = area * max(0.1, confidence)
                        if track_id not in best_crops or score > best_crops[track_id].score:
                            best_crops[track_id] = BestCrop(score, crop.copy(), timestamp)

            writer.write(annotated)
            processed_frames += 1
    finally:
        cap.release()
        writer.release()
        csv_file.close()

    elapsed = time.perf_counter() - started
    contact_sheet(best_crops, args.output / "tracks.jpg")
    source_seconds = min(read_frames / fps, args.max_seconds or total_frames / fps)
    print(f"device={device}")
    print(f"source_seconds={source_seconds:.1f}")
    print(f"processed_frames={processed_frames}")
    print(f"temporary_track_ids={len(best_crops)}")
    print(f"detections={detections}")
    print(f"elapsed_seconds={elapsed:.1f}")
    print(f"source_speed={source_seconds / elapsed:.2f}x realtime")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
