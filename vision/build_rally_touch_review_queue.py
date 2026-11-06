#!/usr/bin/env python3
"""Build a review queue asking whether player 51 touched the ball in each rally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def trim_past_replays(start: float, end: float, exclusions: list[dict]) -> tuple[float, list[dict]]:
    replays = [row for row in exclusions
               if "replay_between_bumpers" in row.get("reasons", []) and
               start < float(row["end_s"]) < end and float(row["start_s"]) < end]
    if not replays:
        return start, []
    return max(start, max(float(row["end_s"]) for row in replays)), replays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("intervals", type=Path)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--replay-intervals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.intervals.read_text())
    exclusions = []
    if args.replay_intervals:
        replay_payload = json.loads(args.replay_intervals.read_text())
        exclusions = replay_payload["sources"][str(args.source)]["exclude_intervals"]
    items = []
    for row in payload["intervals"]:
        # The first interval of a bounded evaluation segment can begin in the
        # middle of an active rally.  It cannot support a reliable rally-level
        # no-touch label, so keep it out of this benchmark.
        if row.get("left_censored"):
            continue
        safe_start, end = float(row["safe_start_s"]), float(row["end_s"])
        start, removed_replays = trim_past_replays(safe_start, end, exclusions)
        if start >= end:
            raise ValueError(f"replay trim removed complete interval {row['id']}")
        items.append({
            "id": f"rally-touch-s{args.source}-{row['id']}",
            "source": args.source,
            "rally_id": row["id"],
            "video": payload["video"],
            "window_start_s": start,
            "window_end_s": end,
            "duration_s": end - start,
            "safe_start_s": safe_start,
            "replay_trimmed_s": start - safe_start,
            "removed_replay_intervals": len(removed_replays),
            "left_censored": bool(row["left_censored"]),
            "end_reason": row["end_reason"],
        })
    result = {
        "schema_version": 1,
        "kind": "volleytag.rally-touch-review-queue",
        "frozen_tests_used": bool(payload["frozen_tests_used"]),
        "labels": ["player51_touch", "no_player51_touch", "uncertain"],
        "instructions": ("Did player 51 touch the ball during the live rally ending at the "
                         "end of this interval? Count a contextually certain off-screen touch "
                         "such as player 51 serving; ignore broadcast replays and dead-ball handoffs."),
        "excluded_left_censored": sum(bool(row.get("left_censored"))
                                        for row in payload["intervals"]),
        "replay_trim_enabled": bool(args.replay_intervals),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": len(items),
                      "total_duration_s": sum(row["duration_s"] for row in items),
                      "frozen_tests_used": result["frozen_tests_used"]}, indent=2))


if __name__ == "__main__":
    main()
