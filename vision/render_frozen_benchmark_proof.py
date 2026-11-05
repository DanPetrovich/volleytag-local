#!/usr/bin/env python3
"""Render an auditable reel from frozen human annotations and model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_fused_detectors import iou


VISIBLE = {"selected_candidate", "corrected_box"}
NEGATIVE = {"not_visible", "not_in_box"}


def xyxy(label: list[float], width: int, height: int) -> list[float]:
    _, cx, cy, bw, bh = map(float, label)
    return [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ]


def draw_box(frame, box, color, label, thickness=4):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(frame, (x1, max(0, y1 - 32)), (min(frame.shape[1], x1 + 260), y1), color, -1)
    cv2.putText(frame, label, (x1 + 6, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                .72, (10, 10, 10), 2, cv2.LINE_AA)


def banner(frame, lines, color):
    height = 48 + 34 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), (10, 12, 18), -1)
    cv2.addWeighted(overlay, .84, frame, .16, 0, frame)
    cv2.rectangle(frame, (0, 0), (12, height), color, -1)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (28, 39 + index * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    .78 if index else .92, (240, 245, 250), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--iou", type=float, default=.30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text())
    decisions = json.loads(args.results.read_text())["decisions"]
    intervals = []
    if args.exclude:
        payload = json.loads(args.exclude.read_text())
        intervals = payload if isinstance(payload, list) else payload.get("intervals", [])
    items = [item for item in queue["items"] if item["id"] in decisions and not any(
        float(interval.get("start_s", interval.get("start"))) <= float(item["time_s"]) <=
        float(interval.get("end_s", interval.get("end"))) for interval in intervals
    )]
    frames, valid = [], []
    for item in items:
        frame = cv2.imread(item["image"])
        if frame is not None:
            frames.append(frame)
            valid.append(item)
    predictions = YOLO(str(args.model)).predict(
        frames, imgsz=args.imgsz, conf=args.conf, device=args.device,
        batch=8, verbose=False, stream=True,
    )

    rendered, errors, rows = [], [], []
    for item, source, prediction in zip(valid, frames, predictions):
        frame = source.copy()
        decision = decisions[item["id"]]
        verdict = decision["verdict"]
        boxes = (prediction.boxes.xyxy.cpu().numpy() if prediction.boxes is not None
                 else np.empty((0, 4)))
        scores = (prediction.boxes.conf.cpu().numpy() if prediction.boxes is not None
                  else np.empty(0))
        order = np.argsort(-scores) if len(scores) else np.empty(0, dtype=int)
        top_box = boxes[order[0]].tolist() if len(order) else None
        top_score = float(scores[order[0]]) if len(order) else 0.0
        gt = None
        hit = False
        category = "excluded"
        if verdict in VISIBLE:
            gt = xyxy(decision["correctedLabel"], frame.shape[1], frame.shape[0])
            hit = top_box is not None and iou(gt, top_box) >= args.iou
            candidate_hit = any(iou(gt, box.tolist()) >= args.iou for box in boxes)
            if hit:
                category = "hit"
            elif candidate_hit:
                category = "ranking_error"
            else:
                generic = [xyxy(row["label"], frame.shape[1], frame.shape[0])
                           for row in item.get("candidates", [])]
                category = ("generic_recovers" if any(iou(gt, box) >= args.iou for box in generic)
                            else "candidate_miss")
            draw_box(frame, gt, (70, 230, 90), "HUMAN GT")
            if top_box is not None:
                draw_box(frame, top_box, (255, 210, 30) if hit else (40, 55, 245),
                         f"V7 TOP1 {top_score:.2f}")
        elif verdict in NEGATIVE:
            hit = top_box is None
            category = "correct_absent" if hit else "false_positive"
            if top_box is not None:
                draw_box(frame, top_box, (40, 55, 245), f"FALSE POS {top_score:.2f}")
        else:
            category = "excluded"

        color = (70, 230, 90) if hit else ((100, 100, 100) if category == "excluded" else (40, 55, 245))
        banner(frame, [
            f"{category.upper()}  |  t={float(item['time_s']):.2f}s  |  id={item['id']}",
            "GREEN = independent human box   CYAN/RED = v7 top-1",
        ], color)
        target_width = 1280
        target_height = round(frame.shape[0] * target_width / frame.shape[1])
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        rendered.append(frame)
        if category not in {"hit", "correct_absent", "excluded"}:
            errors.extend([frame] * max(1, round(args.fps * 1.5)))
        rows.append({
            "id": item["id"], "time_s": float(item["time_s"]), "verdict": verdict,
            "category": category, "top1_hit": hit, "top1_confidence": top_score,
            "direct_candidates": int(len(boxes)),
        })

    def write(path: Path, values: list[np.ndarray]):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not values:
            return
        height, width = values[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (width, height))
        for value in values:
            writer.write(value)
        writer.release()

    write(args.output, rendered)
    write(args.errors_output, errors)
    counts = {category: sum(row["category"] == category for row in rows)
              for category in sorted({row["category"] for row in rows})}
    summary = {
        "model": str(args.model), "frames": len(rows), "counts": counts,
        "visible": sum(row["verdict"] in VISIBLE for row in rows),
        "visible_top1_accuracy": (
            sum(row["top1_hit"] for row in rows if row["verdict"] in VISIBLE) /
            max(1, sum(row["verdict"] in VISIBLE for row in rows))
        ),
        "rows": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
