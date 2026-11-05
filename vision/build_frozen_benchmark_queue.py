#!/usr/bin/env python3
"""Build an independent, uniformly sampled benchmark annotation queue.

No previous identity boxes or tracks are consumed. Each sampled source frame is
decoded directly and receives fresh, per-frame generic-person proposals only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from build_person_selection_queue import deduplicate_candidates


def sample_times(duration_s: float, count: int, margin_s: float) -> list[float]:
    start = min(margin_s, duration_s / 4)
    end = max(start, duration_s - margin_s)
    if count <= 1:
        return [(start + end) / 2]
    return [start + index * (end - start) / (count - 1) for index in range(count)]


def is_excluded(timestamp: float, intervals: list[dict]) -> bool:
    return any(float(row["start_s"]) <= timestamp <= float(row["end_s"])
               for row in intervals)


def valid_sample_times(duration_s: float, count: int, margin_s: float,
                       intervals: list[dict]) -> list[float]:
    if not intervals:
        return sample_times(duration_s, count, margin_s)
    # Start with a denser timeline, remove replay/non-match intervals, then
    # retain exactly `count` evenly distributed valid instants.
    dense = [value for value in sample_times(duration_s, max(count * 3, count), margin_s)
             if not is_excluded(value, intervals)]
    if len(dense) <= count:
        return dense
    return [dense[round(index * (len(dense) - 1) / (count - 1))]
            for index in range(count)]


def on_court(label: list[float]) -> bool:
    """Camera-specific trapezoid shared by the two straight-view benchmarks."""
    _, cx, cy, width, height = label
    foot_y = cy + height / 2
    if foot_y < .23 or foot_y > 1.02:
        return False
    progress = min(1., max(0., (foot_y - .23) / .73))
    left = .30 - .22 * progress
    right = .70 + .22 * progress
    return left <= cx <= right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--domain", default="unknown")
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--conf", type=float, default=.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margin", type=float, default=1.)
    parser.add_argument("--exclude", type=Path,
                        help="Optional non-match/replay interval manifest")
    parser.add_argument("--court-filter", action="store_true",
                        help="Keep only people whose foot point lies on the court")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps
    args.output_dir.mkdir(parents=True, exist_ok=True)

    intervals: list[dict] = []
    if args.exclude:
        payload = json.loads(args.exclude.read_text(encoding="utf-8"))
        intervals = payload.get("excluded_intervals", payload)

    decoded = []
    for index, time_s in enumerate(valid_sample_times(duration_s, args.count,
                                                       args.margin, intervals)):
        ok, frame = False, None
        decoded_time = time_s
        # Some MP4 indices have an isolated seek hole. Retry immediately around
        # the requested instant while retaining one stable sample per slot.
        for offset_s in (0., -.12, .12, -.30, .30):
            decoded_time = max(0., min(duration_s - .5, time_s + offset_s))
            cap.set(cv2.CAP_PROP_POS_MSEC, decoded_time * 1000.)
            ok, frame = cap.read()
            if ok:
                break
        if not ok:
            continue
        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        image_path = args.output_dir / f"frame-{index:04d}-{actual_frame:07d}.jpg"
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            continue
        decoded.append((index, decoded_time, actual_frame, image_path))
    cap.release()

    model = YOLO(str(args.model))
    started = time.monotonic()
    predictions = model.predict([str(row[3]) for row in decoded], imgsz=args.imgsz,
                                conf=args.conf, device=args.device, batch=8,
                                verbose=False, stream=True)
    items = []
    for (index, time_s, frame_number, image_path), result in zip(decoded, predictions):
        height, width = result.orig_shape
        candidates = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                                  result.boxes.conf.cpu().tolist(),
                                                  result.boxes.cls.cpu().tolist()):
                if int(class_id) != 0:
                    continue
                x1, y1, x2, y2 = box
                candidates.append({
                    "label": [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                              (x2 - x1) / width, (y2 - y1) / height],
                    "confidence": round(float(confidence), 4),
                })
        candidates = deduplicate_candidates(candidates)
        if args.court_filter:
            candidates = [candidate for candidate in candidates
                          if on_court(candidate["label"])]
        stable_id = hashlib.sha1(f"{args.video}:{frame_number}".encode()).hexdigest()[:16]
        items.append({
            "id": stable_id,
            "image": str(image_path),
            "video": str(args.video),
            "source": args.source,
            "split": "frozen_benchmark",
            "kind": "benchmark_frame",
            "domain": args.domain,
            "time_s": time_s,
            "frame": frame_number,
            "labels": [],
            "candidates": candidates,
            "risk_reasons": ["uniform_fresh_frame_no_previous_ground_truth"],
        })
    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1,
        "mode": "frozen_benchmark",
        "video": str(args.video),
        "total": len(items),
        "model": str(args.model),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "sampling": "uniform_direct_decode",
        "exclude": str(args.exclude) if args.exclude else None,
        "excluded_intervals": len(intervals),
        "uses_previous_ground_truth": False,
        "uses_tracking": False,
        "court_filter": args.court_filter,
        "candidate_counts": dict(sorted(Counter(len(row["candidates"]) for row in items).items())),
        "elapsed_s": elapsed,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"frames": len(items), "duration_s": duration_s,
                      "inference_s": round(elapsed, 2),
                      "with_candidates": sum(bool(row["candidates"]) for row in items),
                      "candidate_counts": payload["candidate_counts"]}, indent=2))


if __name__ == "__main__":
    main()
