#!/usr/bin/env python3
"""Select a bounded number of strong referee-ROI motions for expensive scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from build_referee_gesture_queue import roi_motion


def load_excluded_intervals(path: Path, source: int, video: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if "sources" in payload:
        row = payload["sources"][str(source)]
        intervals = row["exclude_intervals"]
    else:
        row = payload
        intervals = row.get("excluded_intervals", [])
    if row["video"] != video:
        raise ValueError("replay metadata belongs to another video")
    return intervals


def select_motion_peaks(scores: np.ndarray, fps: int, start_s: float, end_s: float,
                        count: int, separation_s: float,
                        excluded: list[dict]) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    for index in np.argsort(scores)[::-1]:
        timestamp = (int(index) + 1) / fps
        if not start_s <= timestamp <= end_s:
            continue
        if any(row["start_s"] <= timestamp <= row["end_s"] for row in excluded):
            continue
        if all(abs(timestamp - row[0]) >= separation_s for row in selected):
            selected.append((timestamp, float(scores[index])))
            if len(selected) == count:
                break
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--roi", type=float, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--candidates-per-minute", type=float, default=3)
    parser.add_argument("--separation-s", type=float, default=8)
    parser.add_argument("--replay-intervals", type=Path)
    parser.add_argument("--replay-source", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", action="store_true")
    args = parser.parse_args()
    excluded = []
    if args.replay_intervals:
        if args.replay_source is None:
            parser.error("--replay-source is required with --replay-intervals")
        excluded = load_excluded_intervals(
            args.replay_intervals, args.replay_source, args.video)
    scores, duration = roi_motion(args.video, args.roi)
    if args.end_s > duration + .5:
        raise ValueError(f"end {args.end_s}s exceeds video duration {duration:.3f}s")
    count = math.ceil((args.end_s - args.start_s) / 60 * args.candidates_per_minute)
    peaks = select_motion_peaks(scores, 4, args.start_s, args.end_s, count,
                                args.separation_s, excluded)
    result = {
        "schema_version": 1,
        "kind": "volleytag.referee-motion-candidates",
        "video": args.video,
        "roi_box_norm": args.roi,
        "start_s": args.start_s,
        "end_s": args.end_s,
        "candidates_per_minute": args.candidates_per_minute,
        "separation_s": args.separation_s,
        "frozen_tests_used": args.frozen_evaluation,
        "items": [{"time_s": time, "motion_score": score} for time, score in peaks],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidates": len(peaks), "times_s": [row[0] for row in peaks]},
                     indent=2))


if __name__ == "__main__":
    main()
