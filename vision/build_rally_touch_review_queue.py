#!/usr/bin/env python3
"""Build a review queue asking whether player 51 touched the ball in each rally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("intervals", type=Path)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.intervals.read_text())
    items = []
    for row in payload["intervals"]:
        # The first interval of a bounded evaluation segment can begin in the
        # middle of an active rally.  It cannot support a reliable rally-level
        # no-touch label, so keep it out of this benchmark.
        if row.get("left_censored"):
            continue
        start, end = float(row["safe_start_s"]), float(row["end_s"])
        items.append({
            "id": f"rally-touch-s{args.source}-{row['id']}",
            "source": args.source,
            "rally_id": row["id"],
            "video": payload["video"],
            "window_start_s": start,
            "window_end_s": end,
            "duration_s": end - start,
            "left_censored": bool(row["left_censored"]),
            "end_reason": row["end_reason"],
        })
    result = {
        "schema_version": 1,
        "kind": "volleytag.rally-touch-review-queue",
        "frozen_tests_used": bool(payload["frozen_tests_used"]),
        "labels": ["player51_touch", "no_player51_touch", "uncertain"],
        "instructions": ("Did player 51 touch the ball during the live rally ending at the "
                         "end of this interval? Ignore broadcast replays of earlier rallies."),
        "excluded_left_censored": sum(bool(row.get("left_censored"))
                                        for row in payload["intervals"]),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": len(items),
                      "total_duration_s": sum(row["duration_s"] for row in items),
                      "frozen_tests_used": result["frozen_tests_used"]}, indent=2))


if __name__ == "__main__":
    main()
