#!/usr/bin/env python3
"""Normalize the original browser-local action labels into current video timebases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAPPINGS = {
    1: {"source": 0, "offset": 0.0, "video": "vision/runs/training-set1/source.mp4", "split": "train"},
    2: {"source": 1, "offset": 1500.0, "video": "vision/runs/training-set2/source.mp4", "split": "train"},
    3: {"source": None, "offset": 2880.0, "video": "vision/runs/set3-unseen/source.mp4", "split": "frozen_eval"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--training-events", type=Path, required=True)
    parser.add_argument(
        "--full-match-video",
        type=Path,
        help="Optional source-video path recorded as provenance in the normalized report",
    )
    args = parser.parse_args()

    recovered = json.loads(args.input.read_text(encoding="utf-8"))
    normalized, quarantined = [], []
    for event in recovered.get("events", []):
        set_number = int(event.get("set", 0) or 0)
        mapping = MAPPINGS.get(set_number)
        original_time = float(event.get("timestamp", -1))
        if not mapping:
            quarantined.append({"reason": "unknown_set", "event": event})
            continue
        local_time = original_time - mapping["offset"]
        # The clipped source durations are 1500, 1380 and 1020 seconds.
        duration = {1: 1500.043, 2: 1380.062, 3: 1020.082}[set_number]
        if not 0 <= local_time <= duration:
            quarantined.append({"reason": "timestamp_outside_declared_set", "event": event})
            continue
        normalized.append({
            "id": f"legacy-{event['id']}",
            "source": mapping["source"],
            "video": mapping["video"],
            "split": mapping["split"],
            "time_s": round(local_time, 3),
            "start_s": round(max(0.0, local_time - 1.5), 3),
            "end_s": round(min(duration, local_time + 3.0), 3),
            "label": "verified_action",
            "action": event.get("action"),
            "result": event.get("result"),
            "detail": event.get("detail"),
            "rally_outcome": event.get("rallyOutcome"),
            "original_full_match_time_s": original_time,
            "legacy_source": event.get("source", "manual"),
            "transcript": event.get("transcript"),
            "created_at": event.get("createdAt"),
        })

    report = {
        "schema_version": 1,
        "source": str(args.input),
        "full_match_video": str(args.full_match_video) if args.full_match_video else None,
        "timebase_offsets_s": {"set1": 0, "set2": 1500, "set3": 2880},
        "events": normalized,
        "quarantined": quarantined,
    }
    args.normalized.parent.mkdir(parents=True, exist_ok=True)
    args.normalized.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    current = {"schema_version": 1, "kind": "volleytag.action-events", "events": []}
    if args.training_events.exists():
        current = json.loads(args.training_events.read_text(encoding="utf-8"))
    by_id = {event["id"]: event for event in current.get("events", [])}
    for event in normalized:
        if event["split"] == "train":
            by_id[event["id"]] = event
    current["events"] = sorted(by_id.values(), key=lambda event: (event.get("source", 999), event["time_s"]))
    current["legacy_import"] = {
        "normalized": str(args.normalized),
        "training_events": sum(event["split"] == "train" for event in normalized),
        "frozen_eval_events_not_imported": sum(event["split"] == "frozen_eval" for event in normalized),
        "quarantined": len(quarantined),
    }
    args.training_events.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(current["legacy_import"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
