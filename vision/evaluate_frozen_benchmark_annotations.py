#!/usr/bin/env python3
"""Evaluate direct and fused detections on independently annotated frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_fused_detectors import iou, merge_candidates


VISIBLE = {"selected_candidate", "corrected_box"}
EXCLUDED = {"unusable", "occluded", "uncertain"}


def xyxy(label: list[float], width: int, height: int) -> list[float]:
    _, cx, cy, box_width, box_height = map(float, label)
    return [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
            (cx + box_width / 2) * width, (cy + box_height / 2) * height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--iou", type=float, default=.30)
    parser.add_argument("--dedupe-iou", type=float, default=.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text())
    decisions = json.loads(args.results.read_text())["decisions"]
    intervals = []
    if args.exclude:
        payload = json.loads(args.exclude.read_text())
        intervals = payload if isinstance(payload, list) else payload.get("intervals", [])
    items = [item for item in queue["items"]
             if item["id"] in decisions and not any(
                 float(interval.get("start_s", interval.get("start"))) <= item["time_s"] <=
                 float(interval.get("end_s", interval.get("end"))) for interval in intervals)]
    if args.limit:
        # Fixed uniform subset, not the first consecutive frames.
        indexes = np.linspace(0, len(items) - 1, min(args.limit, len(items)), dtype=int)
        items = [items[index] for index in indexes]

    frames, valid = [], []
    for item in items:
        frame = cv2.imread(item["image"])
        if frame is not None:
            frames.append(frame)
            valid.append(item)
    started = time.monotonic()
    predictions = YOLO(str(args.model)).predict(
        frames, imgsz=args.imgsz, conf=args.conf, device=args.device,
        batch=8, verbose=False, stream=True)
    details = []
    for item, frame, result in zip(valid, frames, predictions):
        decision = decisions[item["id"]]
        verdict = decision["verdict"]
        height, width = frame.shape[:2]
        direct_boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        direct_scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        direct = [{"box": [float(value) for value in box], "confidence": float(score),
                   "direct_confidence": float(score)} for box, score in zip(direct_boxes, direct_scores)]
        generic = [{"box": xyxy(candidate["label"], width, height),
                    "confidence": float(candidate["confidence"]), "track_id": None}
                   for candidate in item.get("candidates", [])]
        merged = merge_candidates(generic, direct, args.dedupe_iou)
        ranked = sorted(merged, key=lambda row: (
            row["direct_confidence"] > 0.0,
            row["direct_confidence"] if row["direct_confidence"] > 0.0 else row["confidence"],
        ), reverse=True)
        row = {"id": item["id"], "time_s": item["time_s"], "verdict": verdict,
               "direct_candidates": len(direct), "generic_candidates": len(generic),
               "fused_candidates": len(merged)}
        if verdict in VISIBLE:
            gt = xyxy(decision["correctedLabel"], width, height)
            direct_overlaps = [iou(gt, candidate["box"]) for candidate in direct]
            generic_overlaps = [iou(gt, candidate["box"]) for candidate in generic]
            fused_overlaps = [iou(gt, candidate["box"]) for candidate in merged]
            row.update({
                "direct_hit": any(value >= args.iou for value in direct_overlaps),
                "direct_top1_hit": bool(direct and direct_overlaps[int(np.argmax(direct_scores))] >= args.iou),
                "generic_hit": any(value >= args.iou for value in generic_overlaps),
                "fused_hit": any(value >= args.iou for value in fused_overlaps),
                "fused_top1_hit": bool(ranked and iou(gt, ranked[0]["box"]) >= args.iou),
                "best_direct_iou": max(direct_overlaps, default=0.),
            })
        details.append(row)

    visible = [row for row in details if row["verdict"] in VISIBLE]
    negatives = [row for row in details if row["verdict"] in {"not_visible", "not_in_box"}]
    ratio = lambda key: sum(bool(row[key]) for row in visible) / len(visible) if visible else 0.
    elapsed = time.monotonic() - started
    metrics = {
        "model": str(args.model), "annotated_frames": len(details),
        "visible_frames": len(visible), "negative_frames": len(negatives),
        "excluded_state_frames": sum(row["verdict"] in EXCLUDED for row in details),
        "generic_recall": ratio("generic_hit"),
        "direct_candidate_recall": ratio("direct_hit"),
        "direct_top1_recall": ratio("direct_top1_hit"),
        "fused_candidate_recall": ratio("fused_hit"),
        "fused_top1_recall": ratio("fused_top1_hit"),
        "direct_false_positive_rate_on_absent": (
            sum(row["direct_candidates"] > 0 for row in negatives) / len(negatives) if negatives else 0.),
        "direct_candidates_per_visible_frame": (
            sum(row["direct_candidates"] for row in visible) / len(visible) if visible else 0.),
        "confidence": args.conf, "iou_threshold": args.iou,
        "dedupe_iou": args.dedupe_iou, "imgsz": args.imgsz,
        "device": args.device, "elapsed_s": elapsed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
