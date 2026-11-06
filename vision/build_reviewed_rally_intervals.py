#!/usr/bin/env python3
"""Build complete rally intervals from reviewed live-rally end timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_intervals(times: list[float]) -> list[dict]:
    ordered = sorted(map(float, times))
    if ordered != list(map(float, times)) or len(ordered) != len(set(ordered)):
        raise ValueError("rally end timestamps must be sorted and unique")
    return [{
        "id": f"reviewed-rally-{index:03d}",
        "safe_start_s": start,
        "estimated_start_s": None,
        "end_s": end,
        "duration_s": end - start,
        "end_reason": "reviewed_live_rally_end",
        "left_censored": False,
    } for index, (start, end) in enumerate(zip(ordered, ordered[1:]), 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    results = json.loads(args.results.read_text())
    times = list(map(float, results["rally_end_times_s"]))
    if any(time < float(queue["start_s"]) or time > float(queue["end_s"])
           for time in times):
        raise ValueError("reviewed timestamp outside queue interval")
    intervals = build_intervals(times)
    payload = {
        "schema_version": 1,
        "kind": "volleytag.reviewed-rally-intervals",
        "video": queue["video"],
        "start_s": queue["start_s"],
        "end_s": queue["end_s"],
        "frozen_tests_used": bool(queue["frozen_tests_used"]),
        "policy": "each complete interval lies between consecutive reviewed live-rally ends",
        "reviewed_rally_ends": len(times),
        "intervals": intervals,
        "left_censored_head": {"start_s": queue["start_s"], "end_s": times[0]},
        "right_censored_tail": {"start_s": times[-1], "end_s": queue["end_s"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    durations = [row["duration_s"] for row in intervals]
    print(json.dumps({"reviewed_ends": len(times), "complete_intervals": len(intervals),
                      "minimum_duration_s": min(durations),
                      "maximum_duration_s": max(durations),
                      "mean_duration_s": sum(durations) / len(durations)}, indent=2))


if __name__ == "__main__":
    main()
