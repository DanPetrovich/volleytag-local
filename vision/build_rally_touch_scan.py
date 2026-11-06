#!/usr/bin/env python3
"""Build fixed-rate scan windows inside reviewed rally-touch intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_samples(items: list[dict], step_s: float, margin_s: float) -> list[dict]:
    if step_s <= 0:
        raise ValueError("step_s must be positive")

    rows = []
    for item in items:
        start = float(item["window_start_s"]) + margin_s
        end = float(item["window_end_s"]) - margin_s
        timestamp = start
        while timestamp <= end + 1e-6:
            rows.append({
                "id": f"{item['id']}-t{timestamp:.3f}",
                "item_id": item["id"],
                "rally_id": item["rally_id"],
                "source": int(item["source"]),
                "video": item["video"],
                "time_s": round(float(timestamp), 3),
            })
            timestamp += step_s
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--margin-s", type=float, default=.8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    rows = build_samples(queue["items"], args.step_s, args.margin_s)
    if args.limit is not None:
        rows = rows[:args.limit]
    payload = {
        "schema_version": 1,
        "kind": "volleytag.rally-touch-scan",
        "frozen_tests_used": bool(queue["frozen_tests_used"]),
        "queue": str(args.queue),
        "step_s": args.step_s,
        "margin_s": args.margin_s,
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"samples": len(rows),
                      "rallies": len({row["rally_id"] for row in rows}),
                      "frozen_tests_used": payload["frozen_tests_used"]}, indent=2))


if __name__ == "__main__":
    main()
