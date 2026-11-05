#!/usr/bin/env python3
"""Evaluate a fixed ball-trajectory change rule around Danya proximity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


FPS = 15.0
OFFSETS = np.arange(-15, 16, dtype=np.float32) / FPS


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


def center(box):
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def expanded_gap(danya, ball, expansion=0.25):
    x1, y1, x2, y2 = danya
    width, height = x2 - x1, y2 - y1
    x1, y1, x2, y2 = (x1 - expansion * width, y1 - expansion * height,
                      x2 + expansion * width, y2 + expansion * height)
    dx = max(x1 - ball[2], ball[0] - x2, 0.0)
    dy = max(y1 - ball[3], ball[1] - y2, 0.0)
    return math.hypot(dx, dy) / max(height, 20.0)


def trace(candidates, danya_boxes, seed_frame, seed_index, direction, maximum_gap):
    points = [(seed_frame, candidates[seed_frame][seed_index])]
    current_frame, current = seed_frame, candidates[seed_frame][seed_index]
    while True:
        chosen = None
        for gap in range(1, maximum_gap + 1):
            frame = current_frame + direction * gap
            if frame < 0 or frame >= len(candidates):
                continue
            danya = danya_boxes[current_frame] or danya_boxes[frame]
            if danya is None:
                continue
            scale = max(20.0, danya[3] - danya[1])
            options = []
            for index, candidate in enumerate(candidates[frame]):
                distance = float(np.linalg.norm(center(current["box"]) - center(candidate["box"])) / scale)
                if distance <= 1.25 * gap:
                    options.append((distance - 0.15 * candidate["confidence"], index, candidate))
            if options:
                _, _, candidate = min(options, key=lambda value: value[0])
                chosen = (frame, candidate)
                break
        if chosen is None:
            break
        points.append(chosen)
        current_frame, current = chosen
    return points


def angle_degrees(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def best_touch_signature(candidates, danya_boxes, maximum_gap, minimum_side_points):
    best = None
    for frame, frame_candidates in enumerate(candidates):
        danya = danya_boxes[frame]
        if danya is None:
            continue
        scale = max(20.0, danya[3] - danya[1])
        for index, candidate in enumerate(frame_candidates):
            gap = expanded_gap(danya, candidate["box"])
            if gap > 0.15:
                continue
            before = trace(candidates, danya_boxes, frame, index, -1, maximum_gap)[1:]
            after = trace(candidates, danya_boxes, frame, index, 1, maximum_gap)[1:]
            if len(before) < minimum_side_points or len(after) < minimum_side_points:
                continue
            before = sorted(before, key=lambda value: value[0])[-4:]
            after = sorted(after, key=lambda value: value[0])[:4]
            pivot = center(candidate["box"])
            in_frame, in_candidate = before[0]
            out_frame, out_candidate = after[-1]
            incoming = (pivot - center(in_candidate["box"])) / max((frame - in_frame) / FPS, 1e-6)
            outgoing = (center(out_candidate["box"]) - pivot) / max((out_frame - frame) / FPS, 1e-6)
            in_speed, out_speed = float(np.linalg.norm(incoming) / scale), float(np.linalg.norm(outgoing) / scale)
            direction_change = angle_degrees(incoming, outgoing)
            speed_ratio = max(in_speed, out_speed) / max(min(in_speed, out_speed), 0.05)
            motion = max(in_speed, out_speed)
            predicted = motion >= 0.5 and (direction_change >= 35.0 or speed_ratio >= 1.8)
            signature = {"frame": frame, "gap": gap, "points_before": len(before),
                         "points_after": len(after), "incoming_speed": in_speed,
                         "outgoing_speed": out_speed, "speed_ratio": speed_ratio,
                         "direction_change_degrees": direction_change, "motion": motion,
                         "predicted": predicted}
            rank = (int(predicted), direction_change + 20 * min(speed_ratio, 5), -gap)
            if best is None or rank > best[0]:
                best = (rank, signature)
    return best[1] if best else None


def select_rows(rows, per_label_source):
    selected = []
    for source in sorted({int(row["source"]) for row in rows}):
        for label in (0, 1):
            group = sorted((row for row in rows if int(row["source"]) == source
                            and int(row["label"]) == label), key=lambda row: float(row["time_s"]))
            count = min(per_label_source, len(group))
            indexes = np.linspace(0, len(group) - 1, count, dtype=int)
            selected.extend(group[index] for index in indexes)
    return selected


def sample_key(row):
    return (int(row["source"]), int(row["label"]), round(float(row["time_s"]), 3),
            str(row.get("id") or ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("personal_model", type=Path)
    parser.add_argument("ball_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-label-source", type=int, default=4)
    parser.add_argument("--maximum-gap", type=int, default=2)
    parser.add_argument("--minimum-side-points", type=int, default=2)
    parser.add_argument("--exclude-report", type=Path)
    parser.add_argument("--batch-clips", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = json.loads(args.metadata.read_text(encoding="utf-8"))["samples"]
    if args.exclude_report:
        # v1 used this exact deterministic sampler.  Reconstruct its rows from
        # metadata because generated negative clips intentionally have no id.
        excluded = {sample_key(row) for row in select_rows(rows, args.samples_per_label_source)}
        rows = [row for row in rows if sample_key(row) not in excluded]
    rows = select_rows(rows, args.samples_per_label_source) if args.samples_per_label_source else rows
    personal, ball, device = YOLO(str(args.personal_model)), YOLO(str(args.ball_model)), device_name(args.device)
    details = []
    for start in range(0, len(rows), args.batch_clips):
        batch_rows = rows[start:start + args.batch_clips]
        clips = [read_clip(row["video"], float(row["time_s"])) for row in batch_rows]
        frames = [frame for clip in clips if clip for frame in clip]
        person_results = personal.predict(frames, imgsz=960, conf=.05, device=device,
                                          batch=16, verbose=False)
        danya_flat = [top_box(result) for result in person_results]
        crops_origins = [context_crop(frame, box) for frame, box in zip(frames, danya_flat)]
        ball_results = ball.predict([row[0] for row in crops_origins], imgsz=960, conf=.05,
                                    classes=[0], device=device, batch=16, verbose=False)
        pointer = 0
        for row, clip in zip(batch_rows, clips):
            count = len(clip) if clip else 0
            danya_boxes = danya_flat[pointer:pointer + count]
            candidates = []
            for result, (_, origin) in zip(ball_results[pointer:pointer + count],
                                           crops_origins[pointer:pointer + count]):
                frame_candidates = []
                if result.boxes is not None and origin:
                    for box, confidence in zip(result.boxes.xyxy.detach().cpu().tolist(),
                                               result.boxes.conf.detach().cpu().tolist()):
                        frame_candidates.append({"box": [box[0] + origin[0], box[1] + origin[1],
                                                           box[2] + origin[0], box[3] + origin[1]],
                                                 "confidence": float(confidence)})
                candidates.append(frame_candidates)
            pointer += count
            signature = (best_touch_signature(candidates, danya_boxes, args.maximum_gap,
                                              args.minimum_side_points) if count else None)
            details.append({"id": row.get("id"), "source": int(row["source"]),
                            "time_s": float(row["time_s"]),
                            "label": int(row["label"]), "predicted": bool(signature and signature["predicted"]),
                            "signature": signature})
        print(f"processed={min(start + args.batch_clips, len(rows))}/{len(rows)}", flush=True)

    labels = np.asarray([row["label"] for row in details])
    predicted = np.asarray([row["predicted"] for row in details])
    positive, negative = labels == 1, labels == 0
    recall = float(predicted[positive].mean())
    specificity = float((~predicted[negative]).mean())
    report = {"samples": len(details), "positive_recall": recall,
              "negative_specificity": specificity, "passed": recall >= .70 and specificity >= .70,
              "maximum_gap": args.maximum_gap, "minimum_side_points": args.minimum_side_points,
              "excluded_prior_samples": bool(args.exclude_report),
              "frozen_tests_used": False, "details": details}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
