#!/usr/bin/env python3
"""Evaluate a fixed soft 5-1 court-position prior without tuning on frozen tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ADJACENT = {
    1: {2, 6}, 2: {1, 3}, 3: {2, 4, 6},
    4: {3, 5}, 5: {4, 6}, 6: {1, 3, 5},
}


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def expected_zones(interval: dict) -> set[int]:
    zones: set[int] = set()
    for key in ("pre_serve", "playing_base"):
        value = interval.get(key)
        if isinstance(value, list):
            zones.update(int(item) for item in value)
    return zones


def zone_for_box(box: list[float], polygons: dict[int, np.ndarray]) -> int | None:
    point = ((box[0] + box[2]) / 2.0, box[3])
    for zone, polygon in polygons.items():
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return zone
    return None


def logit(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return math.log(score / (1.0 - score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("rotation", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--expected-bonus", type=float, default=0.20)
    parser.add_argument("--adjacent-bonus", type=float, default=0.08)
    args = parser.parse_args()

    rotation = json.loads(args.rotation.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    cycle_end = float(rotation["first_full_cycle_at_s"])
    intervals = [row for row in rotation["intervals"] if float(row["start_s"]) < cycle_end]
    polygons = {
        int(zone): np.asarray(points, dtype=np.float32)
        for zone, points in calibration["zones"].items()
    }
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest
            if int(row["source"]) == 0 and row["kind"] != "negative"
            and float(row["time_s"]) < cycle_end]
    rows.sort(key=lambda row: float(row["time_s"]))
    image_paths = [str(args.dataset / row["image"]) for row in rows]

    model = YOLO(str(args.model))
    predictions = model.predict(image_paths, imgsz=args.imgsz, conf=args.conf,
                                verbose=False, device="cpu", batch=1)
    details = []
    for row, result in zip(rows, predictions):
        time_s = float(row["time_s"])
        interval = next(item for item in reversed(intervals)
                        if float(item["start_s"]) <= time_s)
        expected = expected_zones(interval)
        adjacent = set().union(*(ADJACENT[zone] for zone in expected)) - expected
        boxes = (result.boxes.xyxy.cpu().numpy() if result.boxes is not None
                 else np.empty((0, 4)))
        scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                  else np.empty(0))
        overlaps = [iou(row["box"], list(map(float, box))) for box in boxes]
        candidate_zones = [zone_for_box(list(map(float, box)), polygons) for box in boxes]
        bonuses = [args.expected_bonus if zone in expected else
                   args.adjacent_bonus if zone in adjacent else 0.0
                   for zone in candidate_zones]
        adjusted = [logit(float(score)) + bonus
                    for score, bonus in zip(scores, bonuses)]
        baseline_index = int(np.argmax(scores)) if len(scores) else -1
        prior_index = int(np.argmax(adjusted)) if adjusted else -1
        baseline_hit = baseline_index >= 0 and overlaps[baseline_index] >= args.iou
        prior_hit = prior_index >= 0 and overlaps[prior_index] >= args.iou
        details.append({
            "time_s": time_s,
            "rotation": interval["rotation"],
            "serve_owner": interval["serve_owner"],
            "expected_zones": sorted(expected),
            "adjacent_zones": sorted(adjacent),
            "candidate_count": len(boxes),
            "candidate_hit": any(value >= args.iou for value in overlaps),
            "baseline_top1_hit": bool(baseline_hit),
            "prior_top1_hit": bool(prior_hit),
            "changed_selection": baseline_index != prior_index,
            "recovered": bool(prior_hit and not baseline_hit),
            "lost": bool(baseline_hit and not prior_hit),
            "candidates": [
                {"confidence": float(score), "zone": zone, "bonus": bonus,
                 "adjusted_logit": adjusted_score, "iou": float(overlap)}
                for score, zone, bonus, adjusted_score, overlap in
                zip(scores, candidate_zones, bonuses, adjusted, overlaps)
            ],
        })

    total = len(details)
    baseline_hits = sum(row["baseline_top1_hit"] for row in details)
    prior_hits = sum(row["prior_top1_hit"] for row in details)
    recovered = sum(row["recovered"] for row in details)
    lost = sum(row["lost"] for row in details)
    metrics = {
        "model": str(args.model),
        "samples": total,
        "cycle_end_s": cycle_end,
        "candidate_recall": sum(row["candidate_hit"] for row in details) / total,
        "baseline_top1_recall": baseline_hits / total,
        "position_prior_top1_recall": prior_hits / total,
        "absolute_gain": (prior_hits - baseline_hits) / total,
        "changed_selections": sum(row["changed_selection"] for row in details),
        "recovered_baseline_errors": recovered,
        "introduced_errors": lost,
        "passed": prior_hits >= baseline_hits and recovered >= 2,
        "protocol": {
            "imgsz": args.imgsz, "confidence": args.conf, "iou": args.iou,
            "expected_zone_logit_bonus": args.expected_bonus,
            "adjacent_zone_logit_bonus": args.adjacent_bonus,
            "hard_filter": False,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
