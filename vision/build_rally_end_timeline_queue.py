#!/usr/bin/env python3
"""Build a continuous review task for exact live-rally end timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = json.loads(args.route.read_text())
    payload = {
        "schema_version": 1,
        "kind": "volleytag.rally-end-timeline-review-queue",
        "frozen_tests_used": bool(route.get("frozen_tests_used")),
        "video": route["video"],
        "start_s": float(route["start_s"]),
        "end_s": float(route["end_s"]),
        "instructions": ("Mark the moment each live rally ends. Ignore broadcast replays, "
                         "score-overlay changes, warm-up touches, and ball handoffs."),
    }
    if payload["end_s"] <= payload["start_s"]:
        raise ValueError("review interval must have positive duration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"duration_s": payload["end_s"] - payload["start_s"],
                      "frozen_tests_used": payload["frozen_tests_used"]}, indent=2))


if __name__ == "__main__":
    main()
