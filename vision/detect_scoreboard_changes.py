#!/usr/bin/env python3
"""Detect stable scoreboard point changes as independent rally-end ground truth."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def separated_changes(scores: list[tuple[float, float]], threshold: float,
                      separation_s: float, excluded: list[dict]) -> list[tuple[float, float]]:
    selected = []
    for timestamp, score in sorted(scores, key=lambda row: row[1], reverse=True):
        if score < threshold:
            break
        if any(row["start_s"] <= timestamp <= row["end_s"] for row in excluded):
            continue
        if all(abs(timestamp - row[0]) >= separation_s for row in selected):
            selected.append((timestamp, score))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--score-roi", type=int, nargs=4, required=True,
                        metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--change-threshold", type=float, default=.03)
    parser.add_argument("--separation-s", type=float, default=5)
    parser.add_argument("--replay-intervals", type=Path)
    parser.add_argument("--replay-source", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", action="store_true")
    args = parser.parse_args()
    excluded = []
    if args.replay_intervals:
        if args.replay_source is None:
            parser.error("--replay-source is required with --replay-intervals")
        replay = json.loads(args.replay_intervals.read_text())["sources"][str(args.replay_source)]
        if replay["video"] != args.video:
            raise ValueError("replay metadata belongs to another video")
        excluded = replay["exclude_intervals"]
    x, y, width, height = args.score_roi
    command = [
        "ffmpeg", "-v", "error", "-ss", str(args.start_s), "-i", args.video,
        "-t", str(args.end_s - args.start_s),
        "-vf", f"crop={width}:{height}:{x}:{y},fps={args.fps},format=bgr24",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    output = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    frame_size = width * height * 3
    frames = np.frombuffer(output[:len(output) // frame_size * frame_size], dtype=np.uint8)
    frames = frames.reshape(-1, height, width, 3)
    masks = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks.append(((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 150)).astype(np.uint8))
    masks = np.asarray(masks)
    scores = []
    radius = args.fps
    for index in range(radius, len(masks) - radius):
        before = np.median(masks[index - radius:index], axis=0) > .5
        after = np.median(masks[index:index + radius], axis=0) > .5
        scores.append((args.start_s + index / args.fps,
                       float(np.mean(before != after))))
    changes = separated_changes(scores, args.change_threshold, args.separation_s, excluded)
    result = {
        "schema_version": 1,
        "kind": "volleytag.scoreboard-boundary-ground-truth",
        "video": args.video,
        "start_s": args.start_s,
        "end_s": args.end_s,
        "score_roi": args.score_roi,
        "change_threshold": args.change_threshold,
        "frozen_tests_used": args.frozen_evaluation,
        "events": [{"time_s": time, "change_score": score} for time, score in changes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"events": len(changes), "times_s": [row[0] for row in changes]},
                     indent=2))


if __name__ == "__main__":
    main()
