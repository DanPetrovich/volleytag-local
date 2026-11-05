#!/usr/bin/env python3
"""Compare personal detectors by camera domain on held-out training validation.

The manifest already contains whole-block train/validation assignments.  This
script further alternates validation blocks into calibration and test folds so
that a domain router is selected without looking at its reported test fold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def ground_truth(dataset: Path, row: dict) -> list[list[float]]:
    image_path = dataset / row["image"]
    label_path = dataset / row["image"].replace("images/", "labels/").rsplit(".", 1)[0]
    label_path = label_path.with_suffix(".txt")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")
    height, width = image.shape[:2]
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _cls, cx, cy, bw, bh = map(float, line.split())
        boxes.append([
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ])
    return boxes


def summarize(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    return {
        "samples": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "candidate_recall": (sum(row["hit"] for row in positives) / len(positives)
                             if positives else None),
        "top1_recall": (sum(row["top1_hit"] for row in positives) / len(positives)
                        if positives else None),
        "negative_false_positive_rate": (
            sum(row["predictions"] > 0 for row in negatives) / len(negatives)
            if negatives else None
        ),
        "predictions_per_image": (sum(row["predictions"] for row in rows) / len(rows)
                                  if rows else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--selection-block", type=float, default=30.0,
                        help="Whole sub-block size for calibration/test alternation")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest if row["split"] == "val"]
    for row in rows:
        block = int(float(row["time_s"]) // args.selection_block)
        row["fold"] = "calibration" if block % 2 == 0 else "test"
        row["domain"] = row.get("domain") or "unknown"
    # Deterministic, domain-balanced sanity subset. Full runs leave this disabled.
    if args.max_samples and len(rows) > args.max_samples:
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[(row["fold"], row["domain"], row["kind"])].append(row)
        selected = []
        while len(selected) < args.max_samples and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(selected) < args.max_samples:
                    selected.append(buckets[key].pop(0))
        rows = selected

    for row in rows:
        row["ground_truth"] = ground_truth(args.dataset, row)

    image_paths = [str(args.dataset / row["image"]) for row in rows]
    details: dict[str, list[dict]] = {}
    for model_path in args.models:
        name = model_path.stem
        model = YOLO(str(model_path))
        predictions = model.predict(image_paths, imgsz=args.imgsz, conf=args.conf,
                                    verbose=False, device="cpu", batch=1)
        model_rows = []
        for row, result in zip(rows, predictions):
            boxes = (result.boxes.xyxy.cpu().numpy() if result.boxes is not None
                     else np.empty((0, 4)))
            scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                      else np.empty(0))
            gt = row["ground_truth"]
            overlaps = [max((iou(target, box) for target in gt), default=0.0)
                        for box in boxes]
            top = int(np.argmax(scores)) if len(scores) else -1
            model_rows.append({
                "image": row["image"],
                "source": row["source"],
                "time_s": row["time_s"],
                "domain": row["domain"],
                "kind": row["kind"],
                "fold": row["fold"],
                "positive": bool(gt),
                "predictions": int(len(boxes)),
                "hit": bool(any(value >= args.iou for value in overlaps)),
                "top1_hit": bool(top >= 0 and overlaps[top] >= args.iou),
                "candidates": [
                    {"confidence": float(score), "overlap": float(overlap)}
                    for score, overlap in zip(scores, overlaps)
                ],
            })
        details[name] = model_rows

    metrics = {"models": {}, "router": {}}
    for name, model_rows in details.items():
        groups = {"all": summarize(model_rows)}
        for fold in ("calibration", "test"):
            fold_rows = [row for row in model_rows if row["fold"] == fold]
            groups[fold] = summarize(fold_rows)
            for domain in sorted({row["domain"] for row in fold_rows}):
                groups[f"{fold}:{domain}"] = summarize(
                    [row for row in fold_rows if row["domain"] == domain]
                )
        metrics["models"][name] = groups

    names = list(details)
    domains = sorted({row["domain"] for row in rows})
    choices = {}
    for domain in domains:
        # Select only by calibration top-1.  Negatives are reported as a guardrail,
        # never used to tune a confidence threshold here.
        def calibration_score(name: str) -> float:
            group = metrics["models"][name].get(f"calibration:{domain}", {})
            return group.get("top1_recall") or -1.0
        choices[domain] = max(names, key=calibration_score)

    routed = []
    indices = {name: {row["image"]: row for row in model_rows}
               for name, model_rows in details.items()}
    for row in rows:
        if row["fold"] != "test":
            continue
        routed.append(indices[choices[row["domain"]]][row["image"]])
    metrics["router"] = {
        "choices_from_calibration": choices,
        "test": summarize(routed),
        "test_by_domain": {
            domain: summarize([row for row in routed if row["domain"] == domain])
            for domain in domains
        },
    }
    ensemble_rows = []
    for row in rows:
        combined = []
        for name in names:
            combined.extend(indices[name][row["image"]]["candidates"])
        best = max(combined, key=lambda candidate: candidate["confidence"], default=None)
        ensemble_rows.append({
            "image": row["image"],
            "domain": row["domain"],
            "fold": row["fold"],
            "positive": bool(row["ground_truth"]),
            "predictions": len(combined),
            "hit": any(candidate["overlap"] >= args.iou for candidate in combined),
            "top1_hit": bool(best and best["overlap"] >= args.iou),
        })
    metrics["ensemble"] = {}
    for fold in ("calibration", "test"):
        fold_rows = [row for row in ensemble_rows if row["fold"] == fold]
        metrics["ensemble"][fold] = summarize(fold_rows)
        metrics["ensemble"][f"{fold}_by_domain"] = {
            domain: summarize([row for row in fold_rows if row["domain"] == domain])
            for domain in domains
        }
    metrics["config"] = {
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "iou_threshold": args.iou,
        "samples": len(rows),
        "max_samples": args.max_samples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                               encoding="utf-8")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
