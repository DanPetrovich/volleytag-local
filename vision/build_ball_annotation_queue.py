#!/usr/bin/env python3
"""Build a training-only volleyball annotation queue around human touch marks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def normalized_label(box, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def top_box(result) -> list[float] | None:
    if result.boxes is None or not len(result.boxes):
        return None
    scores = result.boxes.conf.detach().cpu().numpy()
    return result.boxes.xyxy[int(np.argmax(scores))].detach().cpu().tolist()


def crop_around_person(frame: np.ndarray, box: list[float] | None):
    height, width = frame.shape[:2]
    if box is None:
        return frame, [0, 0, width, height], False
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(240.0, (y2 - y1) * 3.0)
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    if right - left < 80 or bottom - top < 80:
        return frame, [0, 0, width, height], False
    return frame[top:bottom, left:right], [left, top, right, bottom], True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--personal-model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--offsets", type=float, nargs="+", default=(-0.3, 0.0, 0.3))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    events = [event for event in payload.get("events", [])
              if event.get("split") in (None, "train")
              and event.get("action") and event.get("action") != "other"
              and "set3-unseen" not in str(event.get("video", ""))
              and "rasu" not in str(event.get("video", "")).lower()]
    rows = [{**event, "sample_offset_s": float(offset),
             "sample_time_s": max(0.0, float(event["time_s"]) + float(offset))}
            for event in events for offset in args.offsets]

    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    personal, ball = YOLO(str(args.personal_model)), YOLO(str(args.ball_model))
    device, items = device_name(args.device), []

    for start in range(0, len(rows), args.batch_size):
        batch_rows, frames = rows[start:start + args.batch_size], []
        valid_rows = []
        for row in batch_rows:
            cap = cv2.VideoCapture(row["video"])
            cap.set(cv2.CAP_PROP_POS_MSEC, row["sample_time_s"] * 1000.0)
            ok, frame = cap.read()
            cap.release()
            if ok:
                frames.append(frame)
                valid_rows.append(row)
        if not frames:
            continue
        personal_results = personal.predict(frames, imgsz=args.imgsz, conf=.03, device=device,
                                            batch=len(frames), verbose=False)
        crops, crop_meta = [], []
        for frame, result in zip(frames, personal_results):
            crop, crop_xyxy, person_found = crop_around_person(frame, top_box(result))
            crops.append(crop)
            crop_meta.append((crop_xyxy, person_found, frame.shape[1], frame.shape[0]))
        ball_results = ball.predict(crops, imgsz=args.imgsz, conf=.01, classes=[32], device=device,
                                    batch=len(crops), verbose=False)
        for row, crop, result, meta in zip(valid_rows, crops, ball_results, crop_meta):
            height, width = crop.shape[:2]
            candidates = []
            if result.boxes is not None:
                for box, confidence in zip(result.boxes.xyxy.detach().cpu().tolist(),
                                           result.boxes.conf.detach().cpu().tolist()):
                    candidates.append({"label": normalized_label(box, width, height),
                                       "confidence": round(float(confidence), 5)})
            marker = f"{row['id']}:{row['sample_offset_s']:+.3f}"
            item_id = hashlib.sha1(marker.encode()).hexdigest()[:16]
            image_path = frame_dir / f"s{row['source']}-t{row['sample_time_s']:.3f}-{item_id}.jpg"
            cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            crop_xyxy, person_found, full_width, full_height = meta
            items.append({
                "id": item_id, "image": str(image_path), "video": row["video"],
                "source": int(row["source"]), "split": "train", "kind": "ball_annotation",
                "domain": "touch_context", "time_s": row["sample_time_s"],
                "action_time_s": float(row["time_s"]), "action": row["action"],
                "action_event_id": row["id"], "sample_offset_s": row["sample_offset_s"],
                "labels": [], "candidates": candidates, "person_found": person_found,
                "crop_xyxy": crop_xyxy, "full_image_size": [full_width, full_height],
                "risk_reasons": [f"действие: {row['action']}",
                                 f"смещение: {row['sample_offset_s']:+.2f} с"],
            })
        print(f"processed={min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    queue = {
        "schema_version": 1, "mode": "ball_annotation", "total": len(items),
        "training_only": True, "frozen_tests_used": False,
        "source_events": str(args.events), "offsets_s": list(args.offsets),
        "personal_model": str(args.personal_model), "proposal_model": str(args.ball_model),
        "items": items,
    }
    (args.output / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "events": len(events), "frames": len(items),
        "frames_with_person_crop": sum(item["person_found"] for item in items),
        "frames_with_ball_candidates": sum(bool(item["candidates"]) for item in items),
        "candidate_boxes": sum(len(item["candidates"]) for item in items),
        "by_source": dict(Counter(str(item["source"]) for item in items)),
        "by_action": dict(Counter(item["action"] for item in items)),
        "frozen_tests_used": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
