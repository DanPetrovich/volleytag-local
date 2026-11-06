#!/usr/bin/env python3
"""Detect purple broadcast bumpers and replay intervals in training videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def purple_runs(video: str, fps: int = 2, threshold: float = .35) -> tuple[list[dict], float]:
    width, height = 160, 90
    command = [
        "ffmpeg", "-v", "error", "-i", video,
        "-vf", f"fps={fps},scale={width}:{height},format=bgr24",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    output = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    frame_size = width * height * 3
    frames = np.frombuffer(output[:len(output) // frame_size * frame_size], dtype=np.uint8)
    frames = frames.reshape(-1, height, width, 3)
    fractions = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 0] >= 120) & (hsv[:, :, 0] <= 170) &
                (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 35))
        fractions.append(float(mask.mean()))
    selected = np.asarray(fractions) >= threshold
    runs, index = [], 0
    while index < len(selected):
        if not selected[index]:
            index += 1
            continue
        end = index + 1
        while end < len(selected) and (selected[end] or
                                       (end + 1 < len(selected) and selected[end + 1])):
            end += 1
        runs.append({"start_s": index / fps, "end_s": end / fps,
                     "max_purple_fraction": max(fractions[index:end])})
        index = end
    return runs, len(frames) / fps


def replay_pairs(runs: list[dict], max_duration_s: float) -> list[dict]:
    result, index = [], 0
    while index + 1 < len(runs):
        first, second = runs[index], runs[index + 1]
        gap = float(second["start_s"] - first["end_s"])
        short_bumpers = (first["end_s"] - first["start_s"] <= 5 and
                          second["end_s"] - second["start_s"] <= 5)
        if short_bumpers and 1 <= gap <= max_duration_s:
            result.append({"start_s": first["start_s"], "end_s": second["end_s"],
                           "duration_s": second["end_s"] - first["start_s"]})
            index += 2
        else:
            index += 1
    return result


def merge_intervals(intervals: list[dict]) -> list[dict]:
    merged = []
    for row in sorted(intervals, key=lambda item: item["start_s"]):
        if merged and row["start_s"] <= merged[-1]["end_s"]:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], row["end_s"])
            merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + row["reasons"]))
        else:
            merged.append(dict(row))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roi_queue", type=Path, nargs="?")
    parser.add_argument("--video")
    parser.add_argument("--source", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-replay-s", type=float, default=30)
    parser.add_argument("--padding-s", type=float, default=2)
    parser.add_argument("--frozen-evaluation", action="store_true")
    args = parser.parse_args()
    if args.video is not None:
        if args.source is None:
            parser.error("--source is required with --video")
        items = [{"source": args.source, "video": args.video}]
    elif args.roi_queue is not None:
        items = json.loads(args.roi_queue.read_text())["items"]
    else:
        parser.error("provide roi_queue or --video with --source")
    sources = {}
    for item in items:
        source, video = int(item["source"]), item["video"]
        runs, duration = purple_runs(video)
        pairs = replay_pairs(runs, args.max_replay_s)
        exclusions = [{"start_s": max(0, row["start_s"] - args.padding_s),
                       "end_s": min(duration, row["end_s"] + args.padding_s),
                       "reasons": ["purple_bumper"]} for row in runs]
        exclusions.extend({"start_s": max(0, row["start_s"] - args.padding_s),
                           "end_s": min(duration, row["end_s"] + args.padding_s),
                           "reasons": ["replay_between_bumpers"]} for row in pairs)
        sources[str(source)] = {"video": video, "duration_s": duration,
                                "purple_runs": runs, "replay_pairs": pairs,
                                "exclude_intervals": merge_intervals(exclusions)}
        print(f"source={source} purple={len(runs)} replay_pairs={len(pairs)}", flush=True)
    result = {"schema_version": 1, "kind": "volleytag.broadcast-replay-intervals",
              "frozen_tests_used": args.frozen_evaluation, "sources": sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
