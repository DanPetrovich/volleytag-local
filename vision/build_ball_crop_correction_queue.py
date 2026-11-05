#!/usr/bin/env python3
"""Rebuild only ball frames whose crop was centred away from human-verified Danya."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

from build_ball_annotation_queue import crop_around_person, device_name, normalized_label


def human_labels(queue_path: Path, results_path: Path):
    queue = json.loads(queue_path.read_text(encoding="utf-8"))["items"]
    decisions = json.loads(results_path.read_text(encoding="utf-8"))["decisions"]
    labels = {}
    for item in queue:
        decision = decisions.get(item["id"], {})
        label = decision.get("correctedLabel")
        if not label and decision.get("verdict") == "in_box" and item.get("labels"):
            label = item["labels"][0]
        if label:
            labels[(item["action_event_id"], round(float(item["sample_offset_s"]), 1))] = label
    return labels


def interpolate_label(labels, event_id: str, offset: float):
    exact = labels.get((event_id, round(offset, 1)))
    if exact:
        return exact
    if round(offset, 1) != 0:
        return None
    left, right = labels.get((event_id, -0.3)), labels.get((event_id, 0.3))
    if not left or not right:
        return None
    return [0] + [(left[index] + right[index]) / 2 for index in range(1, 5)]


def pixel_box(label, width: int, height: int):
    _, cx, cy, bw, bh = label
    return [(cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height]


def crop_is_suspect(item, label) -> bool:
    width, height = item["full_image_size"]
    x1, y1, x2, y2 = pixel_box(label, width, height)
    hx, hy, hh = (x1 + x2) / 2, (y1 + y2) / 2, max(1.0, y2 - y1)
    left, top, right, bottom = item["crop_xyxy"]
    distance = math.hypot(hx - (left + right) / 2, hy - (top + bottom) / 2) / hh
    intersection = max(0.0, min(x2, right) - max(x1, left)) * max(0.0, min(y2, bottom) - max(y1, top))
    coverage = intersection / max((x2 - x1) * (y2 - y1), 1.0)
    return distance > 0.75 or coverage < 0.80


def project_old_ball(item, decision, new_crop):
    label = decision.get("correctedLabel")
    if not label:
        return None
    old_left, old_top, old_right, old_bottom = item["crop_xyxy"]
    old_width, old_height = old_right - old_left, old_bottom - old_top
    _, cx, cy, bw, bh = label
    full_cx, full_cy = old_left + cx * old_width, old_top + cy * old_height
    full_bw, full_bh = bw * old_width, bh * old_height
    left, top, right, bottom = new_crop
    if not (left <= full_cx <= right and top <= full_cy <= bottom):
        return None
    new_width, new_height = right - left, bottom - top
    return [0, (full_cx - left) / new_width, (full_cy - top) / new_height,
            full_bw / new_width, full_bh / new_height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball-queue", type=Path, required=True)
    parser.add_argument("--ball-results", type=Path, required=True)
    parser.add_argument("--person-queue", type=Path, required=True)
    parser.add_argument("--person-results", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ball_queue = json.loads(args.ball_queue.read_text(encoding="utf-8"))["items"]
    ball_decisions = json.loads(args.ball_results.read_text(encoding="utf-8"))["decisions"]
    verified = human_labels(args.person_queue, args.person_results)
    suspects = []
    for item in ball_queue:
        label = interpolate_label(verified, item["action_event_id"], float(item["sample_offset_s"]))
        if label and crop_is_suspect(item, label):
            suspects.append((item, label))

    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    device, model = device_name(args.device), YOLO(str(args.ball_model))
    pending = []
    for item, label in suspects:
        cap = cv2.VideoCapture(item["video"])
        cap.set(cv2.CAP_PROP_POS_MSEC, float(item["time_s"]) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        width, height = item["full_image_size"]
        crop, crop_xyxy, _ = crop_around_person(frame, pixel_box(label, width, height))
        pending.append((item, crop, crop_xyxy))

    results = model.predict([row[1] for row in pending], imgsz=960, conf=.01, classes=[32],
                            device=device, batch=8, verbose=False)
    items = []
    for (old, crop, crop_xyxy), result in zip(pending, results):
        height, width = crop.shape[:2]
        candidates = []
        if result.boxes is not None:
            for box, confidence in zip(result.boxes.xyxy.detach().cpu().tolist(),
                                       result.boxes.conf.detach().cpu().tolist()):
                candidates.append({"label": normalized_label(box, width, height),
                                   "confidence": round(float(confidence), 5), "origin": "coco"})
        prior = project_old_ball(old, ball_decisions.get(old["id"], {}), crop_xyxy)
        if prior:
            candidates.append({"label": prior, "confidence": 1.0, "origin": "previous_human_ball"})
        item_id = f"corrected-{old['id']}"
        image_path = frame_dir / f"{item_id}.jpg"
        cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
        items.append({**old, "id": item_id, "image": str(image_path), "labels": [],
                      "candidates": candidates, "crop_xyxy": crop_xyxy,
                      "original_item_id": old["id"], "person_crop_source": "human_verified",
                      "risk_reasons": ["исправленный кроп по твоему боксу Дани",
                                       f"действие: {old['action']}",
                                       f"смещение: {old['sample_offset_s']:+.2f} с"]})

    queue = {"schema_version": 1, "mode": "ball_crop_correction", "total": len(items),
             "training_only": True, "frozen_tests_used": False,
             "supersedes_items_in": str(args.ball_queue), "items": items}
    (args.output / "queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps({
        "schema_version": 1, "queue": str(args.output / "queue.json"),
        "kind": "volleytag.ball-crop-corrections", "decisions": {},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"suspect_frames": len(suspects), "correction_frames": len(items),
                      "human_verified_person_boxes": len(verified), "device": device}, indent=2))


if __name__ == "__main__":
    main()
