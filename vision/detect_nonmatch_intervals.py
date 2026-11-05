#!/usr/bin/env python3
"""Detect broadcast replay/highlight blocks delimited by purple slates.

The Liga Chempionov Biznesa broadcast uses a nearly full-screen purple card
before and after replay packages.  This script samples the video sequentially,
finds those cards by colour coverage, and writes a JSON manifest containing
intervals that must not be used for training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def duration_seconds(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def purple_coverage(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Purple/magenta used by the broadcast graphics.  Requiring saturation
    # avoids counting the wooden floor and grey/black arena background.
    mask = cv2.inRange(hsv, np.array((118, 65, 45)), np.array((174, 255, 255)))
    return float(np.count_nonzero(mask)) / float(mask.size)


def merge_hits(hits: list[float], sample_period: float, max_gap: float = 1.25) -> list[list[float]]:
    if not hits:
        return []
    groups: list[list[float]] = [[hits[0], hits[0]]]
    for value in hits[1:]:
        if value - groups[-1][1] <= max_gap:
            groups[-1][1] = value
        else:
            groups.append([value, value])
    pad = sample_period * 0.75
    return [[max(0.0, start - pad), end + pad] for start, end in groups]


def pair_slates(slates: list[list[float]], duration: float, exclude_tail: bool = False,
                max_pair_gap: float = 60.0) -> list[dict]:
    excluded: list[dict] = []
    index = 0
    while index < len(slates):
        if index + 1 < len(slates) and slates[index + 1][0] - slates[index][1] <= max_pair_gap:
            excluded.append({
                "start_s": round(slates[index][0], 3),
                "end_s": round(slates[index + 1][1], 3),
                "reason": "replay_or_highlights",
            })
            index += 2
            continue
        # A single detected slate must not delete minutes of real play merely
        # because the matching transition was missed.  Exclude only the card.
        excluded.append({
            "start_s": round(slates[index][0], 3),
            "end_s": round(slates[index][1], 3),
            "reason": "unpaired_purple_slate",
        })
        index += 1
    if exclude_tail and slates and excluded:
        # The last purple transition introduces the post-match/highlights tail.
        excluded[-1]["end_s"] = round(duration, 3)
        excluded[-1]["reason"] = "replay_or_highlights_and_post_match"
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.16)
    parser.add_argument("--max-pair-gap", type=float, default=60.0,
                        help="Maximum seconds between replay boundary slates")
    parser.add_argument("--exclude-tail-after-last-slate", action="store_true")
    args = parser.parse_args()

    width, height = 320, 180
    frame_bytes = width * height * 3
    command = [
        "ffmpeg", "-v", "error", "-i", str(args.video),
        "-vf", f"fps={args.sample_fps},scale={width}:{height}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    hits: list[float] = []
    scores: list[dict] = []
    frame_index = 0
    assert process.stdout is not None
    while True:
        raw = process.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        time_s = frame_index / args.sample_fps
        coverage = purple_coverage(frame)
        if coverage >= args.threshold:
            hits.append(time_s)
            scores.append({"time_s": round(time_s, 3), "purple_coverage": round(coverage, 4)})
        frame_index += 1
    return_code = process.wait()
    if return_code:
        raise SystemExit(f"ffmpeg exited with code {return_code}")

    duration = duration_seconds(args.video)
    slates = merge_hits(hits, 1.0 / args.sample_fps)
    payload = {
        "schema": "volleytag.non-match-intervals.v1",
        "video": str(args.video),
        "duration_s": round(duration, 3),
        "method": {
            "name": "purple_slate_pairs",
            "sample_fps": args.sample_fps,
            "purple_threshold": args.threshold,
        },
        "purple_slates": [
            {"start_s": round(start, 3), "end_s": round(min(end, duration), 3)}
            for start, end in slates
        ],
        "excluded_intervals": pair_slates(slates, duration,
                                           args.exclude_tail_after_last_slate,
                                           args.max_pair_gap),
        "detections": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "slates": payload["purple_slates"],
        "excluded_intervals": payload["excluded_intervals"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
