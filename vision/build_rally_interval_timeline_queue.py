#!/usr/bin/env python3
"""Build a continuous review task for exact live-rally start/end intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_queue(video: str, start_s: float, end_s: float, split: str) -> dict:
    if end_s <= start_s:
        raise ValueError("review interval must have positive duration")
    if split not in {"train", "calibration", "frozen_eval"}:
        raise ValueError("split must be train, calibration, or frozen_eval")
    return {
        "schema_version": 1,
        "kind": "volleytag.rally-interval-timeline-review-queue",
        "video": video,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "split": split,
        "frozen_tests_used": split == "frozen_eval",
        "boundary_definition": {
            "start": "server contacts the ball for the live serve",
            "end": "first instant the live ball becomes dead",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "frozen_eval"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_queue(args.video, args.start_s, args.end_s, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"duration_s": payload["end_s"] - payload["start_s"],
                      "split": payload["split"],
                      "frozen_tests_used": payload["frozen_tests_used"]}, indent=2))


if __name__ == "__main__":
    main()
