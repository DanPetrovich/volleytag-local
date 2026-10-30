#!/usr/bin/env python3
"""Render the best active ReID candidates without pretending scores are probabilities."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from identify_player import Observation, read_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("similarity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--min-margin", type=float, default=.04)
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--min-height", type=int, default=90)
    parser.add_argument("--min-bottom", type=int, default=280)
    args = parser.parse_args()

    roi = tuple(int(value) for value in args.roi.split(","))
    grouped = read_tracks(args.tracks)
    scores = {}
    with args.similarity.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[int(row["track_id"])] = float(row["margin"])

    by_frame: dict[int, list[Observation]] = defaultdict(list)
    for track_id, items in grouped.items():
        for item in items:
            by_frame[item.frame].append(item)
        for left, right in zip(items, items[1:]):
            gap = right.frame - left.frame
            if gap <= 1 or gap > 6:
                continue
            for frame in range(left.frame + 1, right.frame):
                ratio = (frame - left.frame) / gap
                box = tuple(round(a + (b - a) * ratio) for a, b in zip(left.box, right.box))
                by_frame[frame].append(Observation(
                    time_s=left.time_s + (right.time_s - left.time_s) * ratio,
                    frame=frame, track_id=track_id,
                    confidence=min(left.confidence, right.confidence), box=box,
                ))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = roi[2] - roi[0], roi[3] - roi[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    colors = ((20, 210, 255), (255, 210, 40), (190, 90, 255))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        court = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        eligible = []
        for item in by_frame.get(frame_index, []):
            x1, y1, x2, y2 = item.box
            score = scores.get(item.track_id, -1.0)
            if score < args.min_margin or y2 - y1 < args.min_height or y2 < args.min_bottom:
                continue
            eligible.append((score, item))
        eligible.sort(key=lambda value: value[0], reverse=True)
        for rank, (score, item) in enumerate(eligible[:args.top], 1):
            x1, y1, x2, y2 = item.box
            color = colors[min(rank - 1, len(colors) - 1)]
            cv2.rectangle(court, (x1, y1), (x2, y2), color, 4 if rank == 1 else 2)
            label = f"ReID #{rank}  score {score:.3f}  track {item.track_id}"
            cv2.putText(court, label, (x1, max(28, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                        .66, color, 2, cv2.LINE_AA)
        cv2.putText(court, "score is similarity margin, NOT probability",
                    (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240),
                    2, cv2.LINE_AA)
        writer.write(court)
        frame_index += 1
    writer.release()
    cap.release()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
