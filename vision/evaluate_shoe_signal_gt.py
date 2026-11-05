#!/usr/bin/env python3
"""Evaluate the orange-shoe cue on human GT without tuning its thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from identify_player import Observation
from score_shoe_color import orange_shoe_fraction


def iou(a: list[float], b: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def ground_truth(payload: dict) -> list[dict]:
    if payload.get("schema") == "volleytag.player51-uniform-evaluation.v1":
        return [sample for sample in payload.get("samples", [])
                if sample.get("state") == "visible" and len(sample.get("box", [])) == 4]
    return [keyframe for trajectory in payload.get("trajectory_intervals", [])
            for keyframe in trajectory.get("keyframes", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=.30)
    parser.add_argument("--max-time-delta", type=float, default=.12)
    args = parser.parse_args()

    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    roi = tuple(payload.get("video", {}).get("roi", [40, 70, 1900, 1020]))
    gt_samples = ground_truth(payload)
    by_time: dict[float, list[Observation]] = defaultdict(list)
    with args.tracks.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = Observation(float(row["time_s"]), int(row["frame"]), int(row["track_id"]),
                               float(row["confidence"]), tuple(int(float(row[key])) for key in ("x1", "y1", "x2", "y2")))
            by_time[item.time_s].append(item)
    times = np.array(sorted(by_time))

    rows, labels, scores, gt_scores = [], [], [], []
    cap = cv2.VideoCapture(str(args.video))
    for index, gt in enumerate(gt_samples):
        time_s, gt_box = float(gt["time_s"]), [float(value) for value in gt["box"]]
        nearest = float(times[np.argmin(np.abs(times - time_s))])
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gt_item = Observation(time_s, -1, -1, 1.0, tuple(round(value) for value in gt_box))
        gt_fraction, gt_pixels = orange_shoe_fraction(frame, gt_item, roi)
        gt_scores.append(gt_fraction)
        candidates = by_time[nearest] if abs(nearest - time_s) <= args.max_time_delta else []
        ranked = []
        for candidate in candidates:
            fraction, pixels = orange_shoe_fraction(frame, candidate, roi)
            overlap = iou(gt_box, candidate.box)
            label = int(overlap >= args.iou)
            labels.append(label); scores.append(fraction)
            ranked.append((fraction, overlap, pixels))
        ranked.sort(reverse=True)
        rows.append({
            "gt_index": index, "time_s": time_s, "gt_orange_fraction": gt_fraction,
            "gt_orange_pixels": gt_pixels, "candidate_count": len(ranked),
            "detectable": any(overlap >= args.iou for _, overlap, _ in ranked),
            "shoe_top1": bool(ranked and ranked[0][1] >= args.iou),
            "shoe_top3": any(overlap >= args.iou for _, overlap, _ in ranked[:3]),
        })
    cap.release()

    positives = sum(labels)
    metrics = {
        "shoe_policy": payload.get("appearance_cues", {}).get("shoe", "unknown"),
        "visible_gt": len(rows),
        "detectable_gt": sum(row["detectable"] for row in rows),
        "shoe_top1_recall": np.mean([row["shoe_top1"] for row in rows]) if rows else 0,
        "shoe_top3_recall": np.mean([row["shoe_top3"] for row in rows]) if rows else 0,
        "gt_orange_median": float(np.median(gt_scores)) if gt_scores else 0,
        "gt_orange_p75": float(np.quantile(gt_scores, .75)) if gt_scores else 0,
        "candidate_positive_count": positives,
        "candidate_negative_count": len(labels) - positives,
        "candidate_roc_auc": float(roc_auc_score(labels, scores)) if positives and positives < len(labels) else None,
        "candidate_average_precision": float(average_precision_score(labels, scores)) if positives else None,
        "fixed_threshold_gt_recall": float(np.mean(np.array(gt_scores) >= .035)) if gt_scores else 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (args.output / "keyframes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
