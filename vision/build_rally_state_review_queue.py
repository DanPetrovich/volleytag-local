#!/usr/bin/env python3
"""Build a source-balanced visual queue for rally-state annotation."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


TRAINING_SOURCES = {0, 1, 2, 3, 4, 7, 9}
MIN_SEPARATION_S = 10.0


def validate_queue(items: list[dict]) -> None:
    ids = [row["id"] for row in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate queue item ids")
    unexpected = {int(row["source"]) for row in items} - TRAINING_SOURCES
    if unexpected:
        raise ValueError(f"non-training sources in queue: {sorted(unexpected)}")
    if any(float(row["window_start_s"]) < 0 for row in items):
        raise ValueError("negative review window")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("whistle_queue", type=Path)
    parser.add_argument("whistle_results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-per-source", type=int, default=4)
    parser.add_argument("--random-per-source", type=int, default=4)
    args = parser.parse_args()
    event_payload = json.loads(args.events.read_text())
    whistle_items = json.loads(args.whistle_queue.read_text())["items"]
    whistle_answers = json.loads(args.whistle_results.read_text())["answers"]
    videos = {index: row["video"] for index, row in enumerate(event_payload["playlist"])}
    allowed = TRAINING_SOURCES
    items = []
    occupied: dict[int, list[float]] = defaultdict(list)

    for row in event_payload["events"]:
        source = int(row["source"])
        if source in allowed and row.get("action") == "serve":
            timestamp = max(1.0, float(row["time_s"]) - 2.5)
            occupied[source].append(timestamp)
            items.append({"id": f"rally-state-serve-s{source}-t{timestamp:.3f}",
                          "source": source, "video": videos[source], "time_s": timestamp,
                          "window_start_s": max(0.0, timestamp - 4), "window_end_s": timestamp + 4,
                          "selection_reason": "reviewed_serve_seed"})

    by_source: dict[int, list[dict]] = defaultdict(list)
    for row in whistle_items:
        decision = whistle_answers[row["id"]]["decision"]
        if decision != "serve_whistle":
            by_source[int(row["source"])].append(row)
    for source in sorted(allowed):
        candidates = sorted(by_source[source], key=lambda row: row["score"], reverse=True)
        for row in candidates[:args.hard_per_source]:
            timestamp = float(row["time_s"])
            occupied[source].append(timestamp)
            items.append({"id": f"rally-state-hard-s{source}-t{timestamp:.3f}",
                          "source": source, "video": videos[source], "time_s": timestamp,
                          "window_start_s": max(0.0, timestamp - 4), "window_end_s": timestamp + 4,
                          "selection_reason": "nonserve_whistle_hard_negative"})

    durations = {}
    for source in allowed:
        path = args.whistle_queue.parent / f"source-{source}-whistles.json"
        durations[source] = float(json.loads(path.read_text())["duration_s"])
    rng = random.Random(51)
    for source in sorted(allowed):
        chosen, attempts = [], 0
        while len(chosen) < args.random_per_source and attempts < 10000:
            attempts += 1
            timestamp = rng.uniform(5, durations[source] - 5)
            if all(abs(timestamp - value) >= MIN_SEPARATION_S
                   for value in occupied[source] + chosen):
                chosen.append(timestamp)
        if len(chosen) != args.random_per_source:
            raise RuntimeError(f"could not sample source {source} without overlap")
        for timestamp in sorted(chosen):
            items.append({"id": f"rally-state-random-s{source}-t{timestamp:.3f}",
                          "source": source, "video": videos[source], "time_s": timestamp,
                          "window_start_s": timestamp - 4, "window_end_s": timestamp + 4,
                          "selection_reason": "uniform_random"})

    items.sort(key=lambda row: (row["source"], row["time_s"]))
    validate_queue(items)
    result = {"schema_version": 1, "kind": "volleytag.rally-state-review-queue",
              "frozen_tests_used": False,
              "counts": {"items": len(items), "sources": len(allowed),
                         "serve_seeds": sum(i["selection_reason"] == "reviewed_serve_seed" for i in items),
                         "hard_negatives": sum(i["selection_reason"] == "nonserve_whistle_hard_negative" for i in items),
                         "random": sum(i["selection_reason"] == "uniform_random" for i in items)},
              "items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
