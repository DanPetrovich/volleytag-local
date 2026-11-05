#!/usr/bin/env python3
"""Prioritize dataset QA with disagreement from an already-trained detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import gc

import torch
from ultralytics import YOLO


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def label_xyxy(label, width, height):
    _, cx, cy, bw, bh = label
    return [(cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conf", type=float, default=.05)
    args = parser.parse_args()

    payload = json.loads(args.queue.read_text(encoding="utf-8"))
    items = payload["items"][:args.limit or None]
    root = Path(__file__).resolve().parent.parent
    started = time.monotonic()
    scored = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        scored = previous.get("items", [])
    completed_images = {item["image"] for item in scored}
    pending = [item for item in items if item["image"] not in completed_images]
    model = YOLO(str(args.model))

    for chunk_start in range(0, len(pending), args.chunk_size):
        chunk = pending[chunk_start:chunk_start + args.chunk_size]
        paths = [str(root / item["image"]) for item in chunk]
        predictions = model.predict(
            paths, imgsz=960, conf=args.conf, device=args.device, batch=args.batch,
            verbose=False, stream=True,
        )
        for item, result in zip(chunk, predictions):
            if result.boxes is not None:
                classes = result.boxes.cls.cpu().tolist()
                raw_boxes = result.boxes.xyxy.cpu().tolist()
                raw_confidences = result.boxes.conf.cpu().tolist()
                selected = [(box, confidence) for box, confidence, class_id in
                            zip(raw_boxes, raw_confidences, classes) if int(class_id) == 0]
                pred_boxes = [box for box, _ in selected]
                confidences = [confidence for _, confidence in selected]
            else:
                pred_boxes, confidences = [], []
            gt_boxes = [label_xyxy(label, result.orig_shape[1], result.orig_shape[0]) for label in item.get("labels", [])]
            best_iou = max((iou(gt, pred) for gt in gt_boxes for pred in pred_boxes), default=0.0)
            max_conf = max(confidences, default=0.0)
            reasons = list(item.get("risk_reasons", []))
            risk = float(item.get("risk_score", 0))
            if gt_boxes and best_iou < .25:
                reasons.append("model_disagrees_with_positive")
                risk += 8 + 4 * max_conf
            if not gt_boxes and max_conf >= .25:
                reasons.append("model_finds_danya_in_negative")
                risk += 6 + 4 * max_conf
            scored.append({**item, "risk_reasons": reasons, "risk_score": round(risk, 3),
                           "audit_model_max_conf": round(max_conf, 4),
                           "audit_model_best_iou": round(best_iou, 4)})

        # MPS can retain inference buffers for a long stream. Checkpoint each
        # chunk so an interrupted audit resumes without losing completed work.
        elapsed = time.monotonic() - started
        checkpoint = {**payload, "model": str(args.model), "total": len(scored),
                      "target_total": len(items), "complete": len(scored) == len(items),
                      "elapsed_s": elapsed, "items": scored}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del predictions
        gc.collect()
        if args.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"checkpoint {len(scored)}/{len(items)}", flush=True)
    scored.sort(key=lambda item: (-item["risk_score"], item["source"], item["time_s"]))
    elapsed = time.monotonic() - started
    output = {**payload, "model": str(args.model), "total": len(scored),
              "target_total": len(items), "complete": len(scored) == len(items),
              "elapsed_s": elapsed, "items": scored}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": len(scored), "elapsed_s": round(elapsed, 2),
                      "items_per_s": round(len(scored) / elapsed, 2) if elapsed else None,
                      "flagged": sum(bool(item["risk_reasons"]) for item in scored)}, indent=2))


if __name__ == "__main__":
    main()
