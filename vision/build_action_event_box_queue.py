#!/usr/bin/env python3
"""Build a label-free person-selection queue around human-tagged actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from build_person_selection_queue import deduplicate_candidates, evenly_spaced


FRAME_OFFSETS = (0.0,)
ACTION_PRIORITY = ("attack", "defense", "block", "reception", "serve", "set")


def diverse_actions(events, count):
    """Prefer dynamic action types, then fill the sample across the timeline."""
    events = sorted(events, key=lambda event: float(event["time_s"]))
    selected = []
    selected_ids = set()
    for action in ACTION_PRIORITY:
        rows = [event for event in events if event.get("action") == action]
        if rows and len(selected) < count:
            event = rows[len(rows) // 2]
            selected.append(event)
            selected_ids.add(event["id"])
    for event in evenly_spaced(events, min(len(events), count * 3)):
        if len(selected) >= count:
            break
        if event["id"] not in selected_ids:
            selected.append(event)
            selected_ids.add(event["id"])
    return sorted(selected, key=lambda event: float(event["time_s"]))


def label(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--person-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--personal-model", type=Path, required=True)
    parser.add_argument("--events-per-source", type=int, default=5)
    parser.add_argument("--offsets", type=float, nargs="+", default=list(FRAME_OFFSETS),
                        help="Seconds relative to each human-tagged action")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exclude-queue", type=Path)
    parser.add_argument("--exclude-results", type=Path)
    parser.add_argument("--seed-personal-label", action="store_true",
                        help="Show the strongest personal-model prediction as the current label")
    args = parser.parse_args()

    reviewed: set[tuple[str, float]] = set()
    if (args.exclude_queue and args.exclude_results
            and args.exclude_queue.exists() and args.exclude_results.exists()):
        old_queue = json.loads(args.exclude_queue.read_text(encoding="utf-8"))
        old_results = json.loads(args.exclude_results.read_text(encoding="utf-8"))
        decisions = old_results.get("decisions", {})
        for item in old_queue.get("items", []):
            if item.get("id") in decisions:
                reviewed.add((str(item.get("action_event_id")),
                              round(float(item.get("sample_offset_s", 0.0)), 3)))

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    # Exact-contact metadata uses ``samples`` while the original action review
    # stores ``events``. Supporting both keeps the continuous temporal sanity
    # queue tied to human-confirmed contacts without duplicating extraction.
    for event in payload.get("events", payload.get("samples", [])):
        if (event.get("split") not in (None, "train")
                or (event.get("label") is not None and int(event["label"]) != 1)
                or not event.get("action") or event.get("action") == "other"):
            continue
        source = event.get("source")
        if source is None or not event.get("video"):
            continue
        grouped[int(source)].append(event)

    selected = []
    for source, events in sorted(grouped.items()):
        # Sample exact human-marked contact moments and keep action types diverse.
        for event in diverse_actions(events, min(args.events_per_source, len(events))):
            for offset in args.offsets:
                if (str(event["id"]), round(float(offset), 3)) in reviewed:
                    continue
                selected.append({**event, "sample_time_s": max(0.0, float(event["time_s"]) + offset),
                                 "sample_offset_s": offset})

    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)
    frames, rows = [], []
    captures = {}
    for row in selected:
        cap = captures.setdefault(row["video"], cv2.VideoCapture(row["video"]))
        cap.set(cv2.CAP_PROP_POS_MSEC, row["sample_time_s"] * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
        rows.append(row)
    for cap in captures.values():
        cap.release()

    person_model = YOLO(str(args.person_model))
    personal_model = YOLO(str(args.personal_model))
    started = time.monotonic()
    person_results = list(person_model.predict(
        frames, imgsz=args.imgsz, conf=.08, classes=[0], device=args.device,
        batch=8, verbose=False, stream=True))
    personal_results = list(personal_model.predict(
        frames, imgsz=args.imgsz, conf=.03, device=args.device,
        batch=8, verbose=False, stream=True))
    elapsed = time.monotonic() - started

    items = []
    for row, frame, person_result, personal_result in zip(rows, frames, person_results, personal_results):
        height, width = frame.shape[:2]
        candidates = []
        if person_result.boxes is not None:
            for box, confidence in zip(person_result.boxes.xyxy.cpu().tolist(),
                                       person_result.boxes.conf.cpu().tolist()):
                candidates.append({"label": label(box, width, height),
                                   "confidence": round(float(confidence), 4)})
        candidates = deduplicate_candidates(candidates)
        hidden_personal = []
        if personal_result.boxes is not None:
            for box, confidence in zip(personal_result.boxes.xyxy.cpu().tolist(),
                                       personal_result.boxes.conf.cpu().tolist()):
                hidden_personal.append({"label": label(box, width, height),
                                        "confidence": round(float(confidence), 4)})
        hidden_personal.sort(key=lambda value: value["confidence"], reverse=True)
        seeded_labels = ([hidden_personal[0]["label"]]
                         if args.seed_personal_label and hidden_personal else [])
        marker = f"{row['id']}:{row['sample_offset_s']:+.2f}"
        item_id = hashlib.sha1(marker.encode()).hexdigest()[:16]
        image_path = frame_dir / f"s{row['source']}-t{row['sample_time_s']:.3f}-{item_id}.jpg"
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        items.append({
            "id": item_id, "image": str(image_path), "video": row["video"],
            "source": row["source"], "split": "action_box_sanity",
            "kind": "action_person_selection", "domain": "unknown",
            "time_s": row["sample_time_s"], "action_time_s": row["time_s"],
            "action": row["action"], "action_event_id": row["id"],
            "sample_offset_s": row["sample_offset_s"],
            "labels": seeded_labels, "candidates": candidates,
            # Kept hidden from the reviewer. This is scored only after human GT.
            "personal_predictions": hidden_personal,
            "risk_reasons": [
                f"действие: {row['action']}",
                f"кадр {row['sample_offset_s']:+.2f} с относительно метки",
                "выбери Даню; 5 — вне кадра, 6 — закрыт",
            ],
        })

    queue = {
        "schema_version": 1, "mode": "action_box_sanity",
        "source_events": str(args.events), "total": len(items),
        "training_only": True, "frozen_tests_used": False,
        "person_model": str(args.person_model), "personal_model": str(args.personal_model),
        "success_criterion": {
            "generic_person_candidate_coverage_visible": 0.95,
            "full_queue_only_after_human_sanity": True,
        },
        "elapsed_s": round(elapsed, 3), "items": items,
        "reviewed_frames_excluded": len(reviewed),
        "personal_predictions_seeded": bool(args.seed_personal_label),
    }
    (args.output / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "events_sampled": len({item["action_event_id"] for item in items}),
        "frames": len(items), "elapsed_s": round(elapsed, 2),
        "fps": round(len(items) / elapsed, 2) if elapsed else None,
        "by_source": dict(Counter(str(item["source"]) for item in items)),
        "by_action": dict(Counter(item["action"] for item in items)),
        "generic_with_candidates": sum(bool(item["candidates"]) for item in items),
        "personal_with_predictions": sum(bool(item["personal_predictions"]) for item in items),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
