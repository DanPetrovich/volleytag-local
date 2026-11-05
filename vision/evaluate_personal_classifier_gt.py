#!/usr/bin/env python3
"""Evaluate frozen personal classifiers on sparse human GT keyframes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def embed(crops, batch_size=32):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = [transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                     for crop in crops[start:start + batch_size]]
            values = model(torch.stack(batch).to(device)).float().cpu().numpy()
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            vectors.append(values)
    return np.vstack(vectors), device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("near_classifier", type=Path)
    parser.add_argument("far_classifier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time-delta", type=float, default=.12)
    parser.add_argument("--iou", type=float, default=.30)
    args = parser.parse_args()

    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if payload.get("schema") == "volleytag.player51-uniform-evaluation.v1":
        keyframes = [sample for sample in payload.get("samples", [])
                     if sample.get("state") == "visible" and len(sample.get("box", [])) == 4]
    else:
        keyframes = [keyframe for trajectory in payload["trajectory_intervals"]
                     for keyframe in trajectory.get("keyframes", [])]
    with args.tracks.open(encoding="utf-8") as handle:
        tracks = [{**row, "time_s": float(row["time_s"]),
                   "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")]} for row in csv.DictReader(handle)]
    times = np.array(sorted({row["time_s"] for row in tracks}))
    by_time = {}
    for row in tracks:
        by_time.setdefault(row["time_s"], []).append(row)

    cap = cv2.VideoCapture(str(args.video))
    samples, crops = [], []
    for gt_index, gt in enumerate(keyframes):
        time_s, gt_box = float(gt["time_s"]), list(map(float, gt["box"]))
        nearest = float(times[np.argmin(np.abs(times - time_s))])
        if abs(nearest - time_s) > args.max_time_delta:
            samples.append({"gt_index": gt_index, "time_s": time_s, "gt_box": gt_box,
                            "candidate_indexes": [], "detectable": False})
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = cap.read()
        candidate_indexes = []
        if ok:
            height, width = frame.shape[:2]
            for row in by_time[nearest]:
                x1, y1, x2, y2 = [int(round(value)) for value in row["box"]]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 - x1 < 12 or y2 - y1 < 25:
                    continue
                candidate_indexes.append(len(crops))
                crops.append(frame[y1:y2, x1:x2])
                row["iou"] = iou(gt_box, row["box"])
                row["crop_index"] = len(crops) - 1
        candidates = [row for row in by_time.get(nearest, []) if "crop_index" in row]
        samples.append({"gt_index": gt_index, "time_s": time_s, "gt_box": gt_box,
                        "candidates": candidates, "candidate_indexes": candidate_indexes,
                        "detectable": any(row["iou"] >= args.iou for row in candidates)})
    cap.release()

    vectors, device = embed(crops)
    near = joblib.load(args.near_classifier).predict_proba(vectors)[:, 1]
    far = joblib.load(args.far_classifier).predict_proba(vectors)[:, 1]

    rows = []
    for sample in samples:
        candidates = sample.get("candidates", [])
        for candidate in candidates:
            index = candidate["crop_index"]
            box = candidate["box"]
            # Fixed court-geometry router: large/lower boxes use the near/back model.
            box_height, bottom = box[3] - box[1], box[3]
            use_near = box_height >= 240 or bottom >= 700
            candidate["near_probability"] = float(near[index])
            candidate["far_probability"] = float(far[index])
            candidate["routed_probability"] = float(near[index] if use_near else far[index])
        row = {"gt_index": sample["gt_index"], "time_s": sample["time_s"],
               "detectable": sample["detectable"], "candidate_count": len(candidates)}
        for name, score_key in (("near", "near_probability"), ("far", "far_probability"),
                                ("routed", "routed_probability")):
            ranked = sorted(candidates, key=lambda item: item[score_key], reverse=True)
            row[f"{name}_top1"] = bool(ranked and ranked[0]["iou"] >= args.iou)
            row[f"{name}_top3"] = any(item["iou"] >= args.iou for item in ranked[:3])
            row[f"{name}_best_iou"] = float(ranked[0]["iou"]) if ranked else 0.0
        rows.append(row)

    total = len(rows)
    detectable = sum(row["detectable"] for row in rows)
    metrics = {"device": device, "gt_keyframes": total, "detectable_keyframes": detectable,
               "detector_recall": detectable / total if total else 0.0,
               "iou_threshold": args.iou, "max_time_delta_s": args.max_time_delta}
    for name in ("near", "far", "routed"):
        top1 = sum(row[f"{name}_top1"] for row in rows)
        top3 = sum(row[f"{name}_top3"] for row in rows)
        metrics[name] = {"top1_recall": top1 / total if total else 0.0,
                         "top3_recall": top3 / total if total else 0.0,
                         "top1_given_detected": top1 / detectable if detectable else 0.0}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "keyframes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
