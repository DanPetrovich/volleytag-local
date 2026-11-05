#!/usr/bin/env python3
"""Summarize uniform player-51 GT and separate manual drawing from detector misses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def percent(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--max-time-delta", type=float, default=0.12)
    parser.add_argument("--time-range", help="Optional start,end seconds")
    parser.add_argument("--ui-display", action="store_true",
                        help="Match the review UI: nearest frame plus short track interpolation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if args.time_range:
        start, end = (float(value) for value in args.time_range.split(","))
        samples = [sample for sample in samples if start <= float(sample["time_s"]) < end]
    by_time: dict[float, list[dict]] = defaultdict(list)
    with args.tracks.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_time[float(row["time_s"])].append({
                "track_id": int(row["track_id"]),
                "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            })
    times = sorted(by_time)

    states = Counter(sample.get("state") or "unlabeled" for sample in samples)
    visible = [sample for sample in samples if sample.get("state") == "visible"]
    sources = Counter(sample.get("source") or "unknown" for sample in visible)
    domains = Counter(sample.get("domain") or "unknown" for sample in samples)
    detectable = 0
    manual_detectable = 0
    manual_misses = 0
    manual = [sample for sample in visible if sample.get("source") == "human-drawn"]

    for sample in visible:
        time_s = float(sample["time_s"])
        box = [float(value) for value in sample["box"]]
        if args.ui_display:
            exact_times = [value for value in times if abs(value - time_s) <= .055]
            exact_time = min(exact_times, key=lambda value: abs(value - time_s)) if exact_times else None
            exact = by_time[exact_time] if exact_time is not None else []
            nearby_boxes = [item["box"] for item in exact]
            present = {item["track_id"] for item in exact}
            track_items: dict[int, list[tuple[float, list[float]]]] = defaultdict(list)
            for value in times:
                if abs(value - time_s) <= .22:
                    for item in by_time[value]:
                        track_items[item["track_id"]].append((value, item["box"]))
            for track_id, items in track_items.items():
                if track_id in present:
                    continue
                before = max((item for item in items if item[0] <= time_s), default=None)
                after = min((item for item in items if item[0] >= time_s), default=None)
                if not before or not after or not .001 < after[0] - before[0] <= .26:
                    continue
                alpha = (time_s - before[0]) / (after[0] - before[0])
                nearby_boxes.append([a + (b - a) * alpha for a, b in zip(before[1], after[1])])
        else:
            nearby_boxes = [item["box"] for value in times
                            if abs(value - time_s) <= args.max_time_delta
                            for item in by_time[value]]
        found = any(iou(box, candidate) >= args.iou for candidate in nearby_boxes)
        detectable += int(found)
        if sample.get("source") == "human-drawn":
            manual_detectable += int(found)
            manual_misses += int(not found)

    result = {
        "total_samples": len(samples),
        "completed_samples": sum(value for key, value in states.items() if key != "unlabeled"),
        "states": dict(states),
        "domains": dict(domains),
        "visible_samples": len(visible),
        "visible_sources": dict(sources),
        "manual_drawn": len(manual),
        "manual_drawn_pct_of_visible": percent(len(manual), len(visible)),
        "manual_drawn_pct_of_all": percent(len(manual), len(samples)),
        "detector_detectable_visible": detectable,
        "detector_recall_on_visible_pct": percent(detectable, len(visible)),
        "true_detector_misses_visible": len(visible) - detectable,
        "true_detector_miss_pct_of_visible": percent(len(visible) - detectable, len(visible)),
        "manual_draws_with_adequate_detector_box": manual_detectable,
        "manual_draws_that_were_true_detector_misses": manual_misses,
        "matching": {"iou_threshold": args.iou, "max_time_delta_s": args.max_time_delta,
                     "ui_display_interpolation": args.ui_display},
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
