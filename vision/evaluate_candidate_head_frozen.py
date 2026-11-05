#!/usr/bin/env python3
"""Evaluate a frozen candidate identity head on independently annotated frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from ultralytics import YOLO

from evaluate_fused_detectors import iou, merge_candidates
from evaluate_frozen_benchmark_annotations import VISIBLE, EXCLUDED, xyxy


def padded_crop(frame: np.ndarray, box: list[float], padding: float = .08) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, round(x1 - bw * padding)), min(width, round(x2 + bw * padding))
    y1, y2 = max(0, round(y1 - bh * padding)), min(height, round(y2 + bh * padding))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.shape[0] >= 25 and crop.shape[1] >= 12 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detector", type=Path)
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("near_classifier", type=Path)
    parser.add_argument("far_classifier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-generic-candidates", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--iou", type=float, default=.30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    decisions = json.loads(args.results.read_text(encoding="utf-8"))["decisions"]
    items = [item for item in queue["items"] if item["id"] in decisions]
    if args.limit and args.limit < len(items):
        indexes = np.linspace(0, len(items) - 1, args.limit, dtype=int)
        items = [items[index] for index in indexes]
    frames, valid = [], []
    for item in items:
        frame = cv2.imread(item["image"])
        if frame is not None:
            frames.append(frame); valid.append(item)

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    detector_results = list(YOLO(str(args.detector)).predict(
        frames, imgsz=args.imgsz, conf=args.conf, device=device,
        batch=8, verbose=False, stream=True))

    candidate_sets, crops, refs = [], [], []
    for frame_index, (item, frame, result) in enumerate(zip(valid, frames, detector_results)):
        height, width = frame.shape[:2]
        generic = [{"box": xyxy(row["label"], width, height),
                    "confidence": float(row["confidence"]), "track_id": None}
                   for row in item.get("candidates", [])[:args.max_generic_candidates]]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty(0)
        direct = [{"box": box.tolist(), "confidence": float(score),
                   "direct_confidence": float(score)} for box, score in zip(boxes, scores)]
        merged = merge_candidates(generic, direct, .50)
        candidate_sets.append(merged)
        for candidate_index, row in enumerate(merged):
            crop = padded_crop(frame, row["box"])
            if crop is not None:
                crops.append(crop); refs.append((frame_index, candidate_index))

    embedder = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
    ).eval().to(device)
    transform = create_transform(**resolve_model_data_config(embedder), is_training=False)
    near_model, far_model = joblib.load(args.near_classifier), joblib.load(args.far_classifier)
    frame_scores = [[-float("inf")] * len(rows) for rows in candidate_sets]
    started = time.monotonic(); vectors_done = 0
    with torch.inference_mode():
        for start in range(0, len(crops), args.batch_size):
            batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                                 for crop in crops[start:start + args.batch_size]]).to(device)
            vectors = embedder(batch).float().cpu().numpy()
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            near_scores = near_model.decision_function(vectors)
            far_scores = far_model.decision_function(vectors)
            for offset, (near_score, far_score) in enumerate(zip(near_scores, far_scores)):
                frame_index, candidate_index = refs[start + offset]
                box = candidate_sets[frame_index][candidate_index]["box"]
                use_near = box[3] - box[1] >= 240 or box[3] >= 700
                frame_scores[frame_index][candidate_index] = float(near_score if use_near else far_score)
            vectors_done += len(vectors)
            if vectors_done >= 200 and vectors_done - len(vectors) < 200:
                elapsed = time.monotonic() - started
                estimate = len(crops) * elapsed / vectors_done
                print(json.dumps({"sanity_embeddings": vectors_done, "seconds": elapsed,
                                  "total_embeddings": len(crops),
                                  "estimated_total_seconds": estimate, "device": device}), flush=True)

    details = []
    for item, frame, candidates, scores in zip(valid, frames, candidate_sets, frame_scores):
        decision = decisions[item["id"]]; verdict = decision["verdict"]
        row = {"id": item["id"], "time_s": item["time_s"], "verdict": verdict,
               "candidates": len(candidates)}
        selected = int(np.argmax(scores)) if scores and np.isfinite(max(scores)) else None
        if verdict in VISIBLE:
            height, width = frame.shape[:2]
            gt = xyxy(decision["correctedLabel"], width, height)
            overlaps = [iou(gt, candidate["box"]) for candidate in candidates]
            row.update({
                "candidate_hit": any(value >= args.iou for value in overlaps),
                "top1_hit": selected is not None and overlaps[selected] >= args.iou,
                "selected_iou": overlaps[selected] if selected is not None else 0.0,
            })
        details.append(row)
    visible = [row for row in details if row["verdict"] in VISIBLE]
    metrics = {
        "head": "RBF SVM C=5 on frozen DINOv2 embeddings",
        "detector": str(args.detector), "annotated_frames": len(details),
        "visible_frames": len(visible),
        "excluded_state_frames": sum(row["verdict"] in EXCLUDED for row in details),
        "candidate_recall": sum(row["candidate_hit"] for row in visible) / len(visible),
        "top1_recall": sum(row["top1_hit"] for row in visible) / len(visible),
        "iou_threshold": args.iou, "max_generic_candidates": args.max_generic_candidates,
        "candidates_scored": len(crops), "device": device,
        "elapsed_s": time.monotonic() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
