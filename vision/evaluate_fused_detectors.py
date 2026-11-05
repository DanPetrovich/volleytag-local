#!/usr/bin/env python3
"""Evaluate the union of generic person tracks and the direct Danya detector.

The direct detector is the primary identity signal. Generic person boxes are a
fallback and can also replace an overlapping direct box when their localization
is tighter. The frozen test set must not be used to tune these defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def keyframes(payload: dict) -> list[dict]:
    if payload.get("schema") == "volleytag.player51-uniform-evaluation.v1":
        return [row for row in payload.get("samples", [])
                if row.get("state") == "visible" and len(row.get("box", [])) == 4]
    return [row for trajectory in payload.get("trajectory_intervals", [])
            for row in trajectory.get("keyframes", []) if len(row.get("box", [])) == 4]


def load_tracks(path: Path) -> tuple[np.ndarray, dict[float, list[dict]]]:
    by_time: dict[float, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_s = float(row["time_s"])
            by_time.setdefault(time_s, []).append({
                "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
                "confidence": float(row["confidence"]),
                "track_id": int(row["track_id"]),
            })
    return np.asarray(sorted(by_time)), by_time


def merge_candidates(generic: list[dict], direct: list[dict], threshold: float) -> list[dict]:
    """Attach direct identity evidence to generic boxes and retain unmatched boxes."""
    merged = [{**row, "source": "generic", "direct_confidence": 0.0}
              for row in generic]
    for candidate in direct:
        overlaps = [iou(candidate["box"], row["box"]) for row in merged]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] >= threshold:
            # The personal detector owns identity *and* localization. Retaining
            # the generic box here can expand a correct Danya box onto a nearby
            # player. Preserve only its tracking metadata as supporting evidence.
            generic_row = merged[best]
            if generic_row.get("direct_confidence", 0.0) > 0.0:
                # More than one personal proposal may overlap the same broad
                # person box. Keep both proposals; otherwise the later one can
                # overwrite the higher-confidence identity observation.
                merged.append({**candidate, "source": "direct", "track_id": None})
            else:
                merged[best] = {
                    **candidate,
                    "source": "both",
                    "track_id": generic_row["track_id"],
                    "generic_confidence": generic_row["confidence"],
                }
        else:
            merged.append({**candidate, "source": "direct", "track_id": None})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--dedupe-iou", type=float, default=0.50)
    parser.add_argument("--max-time-delta", type=float, default=0.12)
    args = parser.parse_args()

    samples = keyframes(json.loads(args.ground_truth.read_text(encoding="utf-8")))
    track_times, by_time = load_tracks(args.tracks)
    cap = cv2.VideoCapture(str(args.video))
    frames, valid = [], []
    for sample in samples:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sample["time_s"]) * 1000.0)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            valid.append(sample)
    cap.release()

    model = YOLO(str(args.model))
    results = model.predict(frames, imgsz=args.imgsz, conf=args.conf,
                            verbose=False, device="cpu", batch=1)
    details = []
    for sample, result in zip(valid, results):
        time_s = float(sample["time_s"])
        gt = [float(value) for value in sample["box"]]
        nearest = float(track_times[np.argmin(np.abs(track_times - time_s))])
        generic = by_time[nearest] if abs(nearest - time_s) <= args.max_time_delta else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        direct = [{"box": [float(value) for value in box],
                   "confidence": float(score), "direct_confidence": float(score)}
                  for box, score in zip(boxes, scores)]
        merged = merge_candidates(generic, direct, args.dedupe_iou)
        # A direct-Danya observation is stronger identity evidence than an
        # unmatched generic-person box. Generic confidence is only the fallback.
        ranked = sorted(merged, key=lambda row: (
            row["direct_confidence"] > 0.0,
            row["direct_confidence"] if row["direct_confidence"] > 0.0 else row["confidence"],
        ), reverse=True)
        direct_hit = any(iou(gt, row["box"]) >= args.iou for row in direct)
        generic_hit = any(iou(gt, row["box"]) >= args.iou for row in generic)
        fused_hit = any(iou(gt, row["box"]) >= args.iou for row in merged)
        top1_hit = bool(ranked and iou(gt, ranked[0]["box"]) >= args.iou)
        details.append({
            "time_s": time_s,
            "generic_candidates": len(generic),
            "direct_candidates": len(direct),
            "fused_candidates": len(merged),
            "generic_hit": bool(generic_hit),
            "direct_hit": bool(direct_hit),
            "fused_hit": bool(fused_hit),
            "top1_hit": bool(top1_hit),
            "top1_source": ranked[0]["source"] if ranked else None,
        })

    total = len(details)
    ratio = lambda key: sum(row[key] for row in details) / total if total else 0.0
    metrics = {
        "keyframes": total,
        "generic_recall": ratio("generic_hit"),
        "direct_recall": ratio("direct_hit"),
        "fused_candidate_recall": ratio("fused_hit"),
        "fused_top1_recall": ratio("top1_hit"),
        "generic_candidates_per_frame": sum(row["generic_candidates"] for row in details) / total,
        "direct_candidates_per_frame": sum(row["direct_candidates"] for row in details) / total,
        "fused_candidates_per_frame": sum(row["fused_candidates"] for row in details) / total,
        "confidence": args.conf,
        "iou_threshold": args.iou,
        "dedupe_iou": args.dedupe_iou,
        "max_time_delta_s": args.max_time_delta,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
