#!/usr/bin/env python3
"""Evaluate a saved temporal route against sparse frozen human keyframes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_fused_detectors import iou, keyframes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--max-time-delta", type=float, default=0.25)
    args = parser.parse_args()

    samples = keyframes(json.loads(args.ground_truth.read_text(encoding="utf-8")))
    with args.route.open(encoding="utf-8") as handle:
        route = list(csv.DictReader(handle))
    times = np.asarray([float(row["time_s"]) for row in route])
    details = []
    for sample in samples:
        time_s = float(sample["time_s"])
        index = int(np.argmin(np.abs(times - time_s)))
        row = route[index]
        visible = row["state"] == "visible" and abs(times[index] - time_s) <= args.max_time_delta
        overlap = 0.0
        evaluated_box = None
        # The saved route is sampled every few tenths of a second. Interpolate
        # coordinates only inside one continuous track; nearest-neighbour boxes
        # otherwise penalize ordinary player motion rather than identity errors.
        right = int(np.searchsorted(times, time_s))
        left = right - 1
        if 0 <= left < len(route) and 0 <= right < len(route):
            before, after = route[left], route[right]
            same_track = (before.get("track_id") and
                          before.get("track_id") == after.get("track_id"))
            if before["state"] == after["state"] == "visible" and same_track:
                span = max(times[right] - times[left], 1e-9)
                weight = (time_s - times[left]) / span
                first = np.asarray([float(before[key]) for key in ("x1", "y1", "x2", "y2")])
                second = np.asarray([float(after[key]) for key in ("x1", "y1", "x2", "y2")])
                evaluated_box = ((1.0 - weight) * first + weight * second).tolist()
                visible = True
        if visible:
            box = evaluated_box or [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            overlap = iou([float(value) for value in sample["box"]], box)
        details.append({"time_s": time_s, "route_time_s": float(times[index]),
                        "visible": bool(visible), "iou": float(overlap),
                        "hit": bool(overlap >= args.iou),
                        "source": row.get("source") if visible else None})

    total = len(details)
    hits = sum(row["hit"] for row in details)
    metrics = {"keyframes": total, "hits": hits,
               "top1_recall": hits / total if total else 0.0,
               "visible_route_recall": sum(row["visible"] for row in details) / total if total else 0.0,
               "mean_iou": sum(row["iou"] for row in details) / total if total else 0.0,
               "iou_threshold": args.iou, "max_time_delta_s": args.max_time_delta}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
