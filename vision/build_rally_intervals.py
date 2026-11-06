#!/usr/bin/env python3
"""Build recall-safe rally intervals from accepted rally-end boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def accepted_boundaries(route: dict, queue: dict, results: dict) -> tuple[list[dict], list[float]]:
    decisions = {round(float(item["time_s"]), 3):
                 results["answers"][item["id"]]["decision"] for item in queue["items"]}
    endings, serve_hints = [], []
    for event in route["events"]:
        if event["reason"].startswith("scoreboard"):
            endings.append(event)
            continue
        decision = decisions.get(round(float(event["time_s"]), 3))
        if decision == "rally_end_team_to_serve":
            endings.append({**event, "review_decision": decision})
        elif decision == "serve_authorization":
            serve_hints.append(float(event["time_s"]))
    return sorted(endings, key=lambda row: row["time_s"]), sorted(serve_hints)


def build_intervals(route: dict, endings: list[dict], serve_hints: list[float]) -> list[dict]:
    intervals, previous_end = [], float(route["start_s"])
    for index, ending in enumerate(endings, 1):
        end = float(ending["time_s"])
        if end <= previous_end:
            raise ValueError(f"non-positive rally interval at {end:.3f}s")
        hints = [time for time in serve_hints if previous_end < time < end]
        intervals.append({
            "id": f"rally-{index:03d}",
            "safe_start_s": previous_end,
            "estimated_start_s": hints[-1] if hints else None,
            "end_s": end,
            "duration_s": end - previous_end,
            "end_reason": ending["reason"],
            "left_censored": index == 1,
        })
        previous_end = end
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("fallback_queue", type=Path)
    parser.add_argument("fallback_results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = json.loads(args.route.read_text())
    queue = json.loads(args.fallback_queue.read_text())
    results = json.loads(args.fallback_results.read_text())
    endings, serve_hints = accepted_boundaries(route, queue, results)
    intervals = build_intervals(route, endings, serve_hints)
    result = {
        "schema_version": 1,
        "kind": "volleytag.rally-intervals",
        "video": route["video"],
        "start_s": route["start_s"],
        "end_s": route["end_s"],
        "frozen_tests_used": bool(route["frozen_tests_used"]),
        "policy": ("safe_start is the previous accepted rally end; estimated_start is an "
                   "optional reviewed serve authorization and never trims recall-safe analysis"),
        "accepted_rally_ends": len(endings),
        "reviewed_serve_hints": len(serve_hints),
        "intervals": intervals,
        "right_censored_tail": {
            "start_s": endings[-1]["time_s"] if endings else route["start_s"],
            "end_s": route["end_s"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    durations = [row["duration_s"] for row in intervals]
    print(json.dumps({
        "intervals": len(intervals),
        "serve_hints": len(serve_hints),
        "minimum_duration_s": min(durations) if durations else None,
        "maximum_duration_s": max(durations) if durations else None,
        "mean_duration_s": sum(durations) / len(durations) if durations else None,
    }, indent=2))


if __name__ == "__main__":
    main()
