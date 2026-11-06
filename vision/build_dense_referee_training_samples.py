#!/usr/bin/env python3
"""Derive inference-like referee clips from reviewed gesture intervals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


POSITIVE = {"serve_authorization", "rally_end_team_to_serve"}


def merge_intervals(rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(rows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(row[0], row[1]) for row in merged]


def interval_distance(time_s: float, start_s: float, end_s: float) -> float:
    return max(start_s - time_s, 0.0, time_s - end_s)


def build(queue: dict, results: dict, step_s: float = 1.0,
          positive_margin_s: float = 1.5, negative_margin_s: float = 3.0) -> tuple[dict, dict]:
    if step_s <= 0 or not 0 <= positive_margin_s < negative_margin_s:
        raise ValueError("invalid sampling margins")
    answers = results.get("answers", {})
    missing = [row["id"] for row in queue["items"] if row["id"] not in answers]
    if missing:
        raise ValueError(f"missing {len(missing)} reviewed answers")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in queue["items"]:
        key = (row["source_name"], int(row["source"]), row["video"],
               tuple(row["roi_box_norm"]))
        groups[key].append(row)
    items, dense_answers = [], {}
    for (source_name, source, video, roi), rows in sorted(groups.items()):
        coverage = merge_intervals([(float(row["window_start_s"]), float(row["window_end_s"]))
                                    for row in rows])
        uncertain = merge_intervals([
            (float(row["window_start_s"]), float(row["window_end_s"]))
            for row in rows if answers[row["id"]]["decision"] == "uncertain"
        ])
        events = []
        for row in rows:
            answer = answers[row["id"]]
            if answer["decision"] not in POSITIVE:
                continue
            events.append({
                "decision": answer["decision"],
                "start_s": float(answer["gesture_start_s"]),
                "end_s": float(answer["gesture_end_s"]),
            })
        seen = set()
        for cover_start, cover_end in coverage:
            time_s = math.ceil(cover_start / step_s) * step_s
            while time_s <= cover_end + 1e-6:
                rounded = round(time_s, 3)
                if rounded in seen or any(start <= time_s <= end for start, end in uncertain):
                    time_s += step_s
                    continue
                seen.add(rounded)
                ranked = sorted(((interval_distance(time_s, event["start_s"], event["end_s"]), event)
                                 for event in events), key=lambda row: row[0])
                if ranked and ranked[0][0] <= positive_margin_s:
                    decision = ranked[0][1]["decision"]
                elif not ranked or ranked[0][0] >= negative_margin_s:
                    decision = "no_signal"
                else:
                    time_s += step_s
                    continue
                item_id = f"referee-dense-samples-{source_name}-t{rounded:.3f}"
                items.append({
                    "id": item_id, "source": source, "source_name": source_name,
                    "video": video, "time_s": rounded,
                    "window_start_s": max(0.0, rounded - 2.0),
                    "window_end_s": rounded + 2.0,
                    "roi_box_norm": list(roi), "selection_reason": "dense_reviewed_timeline",
                })
                dense_answers[item_id] = {"item_id": item_id, "decision": decision}
                time_s += step_s
    output_queue = {
        "schema_version": 1, "kind": "volleytag.referee-gesture-review-queue",
        "frozen_tests_used": False,
        "selection": {"method": "dense samples derived from reviewed gesture intervals",
                      "step_s": step_s, "positive_margin_s": positive_margin_s,
                      "negative_margin_s": negative_margin_s,
                      "uncertain_windows_excluded": True},
        "items": sorted(items, key=lambda row: (row["source_name"], row["time_s"])),
    }
    output_results = {"schema_version": 1, "kind": "volleytag.referee-gesture-results",
                      "frozen_tests_used": False, "answers": dense_answers}
    return output_queue, output_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    parser.add_argument("--step-s", type=float, default=1.0)
    args = parser.parse_args()
    queue, results = build(json.loads(args.queue.read_text()),
                           json.loads(args.results.read_text()), step_s=args.step_s)
    for path, payload in ((args.queue_output, queue), (args.results_output, results)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    counts = {label: sum(row["decision"] == label for row in results["answers"].values())
              for label in ("serve_authorization", "rally_end_team_to_serve", "no_signal")}
    print(json.dumps({"samples": len(queue["items"]), **counts}, indent=2))


if __name__ == "__main__":
    main()
