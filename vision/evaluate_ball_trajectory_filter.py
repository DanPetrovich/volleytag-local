#!/usr/bin/env python3
"""Evaluate a fixed local trajectory-consistency filter on reviewed ball frames."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Keep the dense 15 FPS context, but also evaluate the exact human-labelled
# timestamps.  A one-frame temporal mismatch is large relative to a tiny,
# fast-moving volleyball and would make IoU evaluation invalid.
OFFSETS = np.asarray(sorted(set((np.arange(-11, 12, dtype=np.float32) / 15.0).tolist()
                                + [-0.3, 0.0, 0.3])), dtype=np.float32)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def read_clip(video: str, timestamp: float):
    cap = cv2.VideoCapture(video)
    frames = []
    for offset in OFFSETS:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp + float(offset)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(frame)
    cap.release()
    return frames


def top_box(result):
    if result.boxes is None or not len(result.boxes):
        return None
    scores = result.boxes.conf.detach().cpu().numpy()
    return result.boxes.xyxy[int(np.argmax(scores))].detach().cpu().tolist()


def context_crop(frame, box):
    if box is None:
        return np.zeros((240, 240, 3), dtype=np.uint8), None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(240.0, (y2 - y1) * 3.0)
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    crop = frame[top:bottom, left:right]
    return (crop if crop.size else np.zeros((240, 240, 3), dtype=np.uint8),
            (left, top) if crop.size else None)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def center(box):
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def select_supported(candidates, danya_boxes, frame_index: int):
    best = None
    for candidate in candidates[frame_index]:
        support_score = None
        for neighbour in range(max(0, frame_index - 2), min(len(candidates), frame_index + 3)):
            if neighbour == frame_index or not candidates[neighbour]:
                continue
            danya = danya_boxes[frame_index] or danya_boxes[neighbour]
            if danya is None:
                continue
            scale = max(20.0, danya[3] - danya[1])
            for other in candidates[neighbour]:
                distance = float(np.linalg.norm(center(candidate["box"]) - center(other["box"])) / scale)
                if distance <= 1.0:
                    score = candidate["confidence"] + other["confidence"] - 0.25 * distance
                    support_score = score if support_score is None else max(support_score, score)
        if support_score is not None and (best is None or support_score > best[0]):
            best = (support_score, candidate["box"])
    return best[1] if best else None


def gt_items(manifest_path, base_queue_path, correction_queue_path, split):
    manifest = [row for row in read(manifest_path)["items"] if row["split"] == split]
    base = {row["id"]: row for row in read(base_queue_path)["items"]}
    correction = {row["id"]: row for row in read(correction_queue_path)["items"]}
    rows = []
    for row in manifest:
        item = correction.get(row["id"], base.get(row["id"]))
        if item is None:
            raise RuntimeError(f"missing queue item {row['id']}")
        gt = None
        text = Path(row["label"]).read_text().strip()
        if text:
            _, cx, cy, width, height = map(float, text.split())
            left, top, right, bottom = item["crop_xyxy"]
            crop_width, crop_height = right - left, bottom - top
            gt = [left + (cx - width / 2) * crop_width,
                  top + (cy - height / 2) * crop_height,
                  left + (cx + width / 2) * crop_width,
                  top + (cy + height / 2) * crop_height]
        rows.append({**row, "queue_item": item, "gt": gt})
    return rows


def summarize(details, key):
    positive = [row for row in details if row["positive"]]
    negative = [row for row in details if not row["positive"]]
    predictions = [row for row in details if row[key] is not None]
    matched = sum(row[key] is not None and row[f"{key}_iou"] >= 0.5 for row in positive)
    clean_negative = sum(row[key] is None for row in negative)
    return {"positive_frame_recall": matched / len(positive),
            "frame_detection_precision": matched / len(predictions) if predictions else 0.0,
            "negative_frame_specificity": clean_negative / len(negative),
            "predictions": len(predictions), "matched": matched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-queue", type=Path, required=True)
    parser.add_argument("--correction-queue", type=Path, required=True)
    parser.add_argument("--personal-model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    gt_rows = gt_items(args.manifest, args.base_queue, args.correction_queue, args.split)
    grouped = defaultdict(list)
    for row in gt_rows:
        grouped[row["event_id"]].append(row)
    personal, ball, device = YOLO(str(args.personal_model)), YOLO(str(args.ball_model)), device_name(args.device)
    details = []
    for event_index, (event_id, rows) in enumerate(sorted(grouped.items())):
        event_time = float(rows[0]["time_s"]) - float(rows[0]["offset_s"])
        clip = read_clip(rows[0]["queue_item"]["video"], event_time)
        if clip is None:
            continue
        person_results = personal.predict(clip, imgsz=960, conf=.05, device=device, batch=16, verbose=False)
        danya_boxes = [top_box(result) for result in person_results]
        crops_origins = [context_crop(frame, box) for frame, box in zip(clip, danya_boxes)]
        ball_results = ball.predict([row[0] for row in crops_origins], imgsz=960, conf=.05,
                                    classes=[0], device=device, batch=16, verbose=False)
        candidates = []
        for result, (_, origin) in zip(ball_results, crops_origins):
            frame_candidates = []
            if result.boxes is not None and origin:
                for box, confidence in zip(result.boxes.xyxy.detach().cpu().tolist(),
                                           result.boxes.conf.detach().cpu().tolist()):
                    frame_candidates.append({"box": [box[0] + origin[0], box[1] + origin[1],
                                                       box[2] + origin[0], box[3] + origin[1]],
                                             "confidence": float(confidence)})
            candidates.append(frame_candidates)
        for row in rows:
            target = int(np.argmin(np.abs(OFFSETS - float(row["offset_s"]))))
            baseline = (max(candidates[target], key=lambda value: value["confidence"])["box"]
                        if candidates[target] else None)
            trajectory = select_supported(candidates, danya_boxes, target)
            gt = row["gt"]
            details.append({"id": row["id"], "event_id": event_id, "positive": gt is not None,
                            "baseline": baseline, "baseline_iou": iou(gt, baseline) if gt and baseline else 0.0,
                            "trajectory": trajectory,
                            "trajectory_iou": iou(gt, trajectory) if gt and trajectory else 0.0})
        print(f"events={event_index + 1}/{len(grouped)}", flush=True)

    baseline, trajectory = summarize(details, "baseline"), summarize(details, "trajectory")
    precision_gain = trajectory["frame_detection_precision"] - baseline["frame_detection_precision"]
    recall_drop = baseline["positive_frame_recall"] - trajectory["positive_frame_recall"]
    passed = (precision_gain >= .10 and recall_drop <= .10
              and trajectory["negative_frame_specificity"] >= .80)
    report = {"split": args.split, "frames": len(details), "events": len(grouped),
              "baseline": baseline, "trajectory": trajectory,
              "precision_gain": precision_gain, "recall_drop": recall_drop,
              "passed": passed, "frozen_tests_used": False, "details": details}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
