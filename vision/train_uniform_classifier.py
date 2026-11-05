#!/usr/bin/env python3
"""Train and temporally validate a camera-specific player-51 classifier."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def crop(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    pad_x, pad_y = max(2, round((x2 - x1) * .05)), max(2, round((y2 - y1) * .03))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    value = frame[y1:y2, x1:x2]
    return value.copy() if value.size and value.shape[0] >= 25 and value.shape[1] >= 10 else None


def orange_fraction(image):
    lower = image[int(image.shape[0] * .58):]
    if lower.size == 0:
        return 0.0
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((4, 85, 80)), np.array((28, 255, 255)))
    return float(np.count_nonzero(mask)) / mask.size


def embed(images, batch_size):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB)))
                                 for value in images[start:start + batch_size]]).to(device)
            value = model(batch).float().cpu().numpy()
            value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
            vectors.append(value)
    return np.vstack(vectors), device


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.35, class_weight="balanced", max_iter=3000, random_state=51,
    ))


def metrics(y, probabilities):
    predicted = probabilities >= .5
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision_at_0.5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, predicted, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-seconds", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iou", type=float, default=.30)
    parser.add_argument("--max-negatives-per-frame", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    samples = [sample for sample in gt.get("samples", [])
               if sample.get("state") == "visible" and len(sample.get("box", [])) == 4]
    by_time = defaultdict(list)
    with args.tracks.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = {"time_s": float(row["time_s"]), "track_id": int(row["track_id"]),
                    "confidence": float(row["confidence"]),
                    "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")]}
            by_time[item["time_s"]].append(item)
    track_times = np.array(sorted(by_time))

    rows, images = [], []
    cap = cv2.VideoCapture(str(args.video))
    for frame_group, sample in enumerate(samples):
        time_s, gt_box = float(sample["time_s"]), [float(v) for v in sample["box"]]
        nearest = float(track_times[np.argmin(np.abs(track_times - time_s))])
        if abs(nearest - time_s) > .12:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        candidates = []
        for item in by_time[nearest]:
            overlap = iou(gt_box, item["box"])
            candidate_crop = crop(frame, item["box"])
            if candidate_crop is not None:
                candidates.append((item, overlap, candidate_crop))
        positives = sorted([value for value in candidates if value[1] >= args.iou],
                           key=lambda value: value[1], reverse=True)[:2]
        negatives = sorted([value for value in candidates if value[1] < .08],
                           key=lambda value: value[0]["confidence"], reverse=True)[:args.max_negatives_per_frame]
        selected = positives + negatives
        # A manual GT crop repairs the rare detector miss but is not used in top-1 ranking.
        if not positives:
            manual = crop(frame, gt_box)
            if manual is not None:
                selected.append(({"track_id": -1, "confidence": 1.0, "box": gt_box}, 1.0, manual))
        for item, overlap, candidate_crop in selected:
            rows.append({"time_s": time_s, "frame_group": frame_group,
                         "block": int(time_s // args.block_seconds),
                         "track_id": item["track_id"], "iou": overlap,
                         "label": int(overlap >= args.iou), "ranking_candidate": item["track_id"] != -1,
                         "orange_fraction": orange_fraction(candidate_crop)})
            images.append(candidate_crop)
    cap.release()

    x, device = embed(images, args.batch_size)
    shoe = np.array([row["orange_fraction"] for row in rows])[:, None]
    variants = {"appearance": x, "appearance_plus_shoe": np.hstack([x, shoe])}
    y = np.array([row["label"] for row in rows])
    blocks = np.array([row["block"] for row in rows])
    report = {"model": "DINOv2 ViT-S/14 + logistic regression", "device": device,
              "samples": len(rows), "positives": int(y.sum()), "negatives": int((1-y).sum()),
              "visible_gt": len(samples), "temporal_blocks": sorted(set(map(int, blocks))), "variants": {}}

    for name, features in variants.items():
        probabilities = np.full(len(y), np.nan)
        held_out = []
        for block in sorted(set(blocks)):
            test, train = blocks == block, blocks != block
            if len(set(y[test])) < 2 or len(set(y[train])) < 2:
                continue
            probabilities[test] = classifier().fit(features[train], y[train]).predict_proba(features[test])[:, 1]
            held_out.append(int(block))
        valid = np.isfinite(probabilities)
        variant = metrics(y[valid], probabilities[valid])
        top1, detectable = 0, 0
        for group in sorted(set(row["frame_group"] for row in rows)):
            indexes = [i for i, row in enumerate(rows)
                       if row["frame_group"] == group and row["ranking_candidate"] and valid[i]]
            if not indexes:
                continue
            positives = [i for i in indexes if y[i] == 1]
            detectable += bool(positives)
            top1 += bool(positives and y[max(indexes, key=lambda i: probabilities[i])] == 1)
        variant.update({"held_out_blocks": held_out, "detectable_frames": detectable,
                        "top1_frames": top1,
                        "top1_recall_all_visible": top1 / len(samples) if samples else 0.0,
                        "top1_given_detected": top1 / detectable if detectable else 0.0})
        report["variants"][name] = variant
        fitted = classifier().fit(features, y)
        joblib.dump(fitted, args.output / f"player51-{name}.joblib")
        np.save(args.output / f"features-{name}.npy", features)
        for index, probability in enumerate(probabilities):
            rows[index][f"cv_probability_{name}"] = None if not np.isfinite(probability) else float(probability)

    with (args.output / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
