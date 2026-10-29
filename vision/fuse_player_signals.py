#!/usr/bin/env python3
"""Fuse jersey-number anchors with track continuity for one player."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2

from identify_player import Observation, read_tracks


def center(item: Observation) -> tuple[float, float]:
    x1, y1, x2, y2 = item.box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def height(item: Observation) -> int:
    return item.box[3] - item.box[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("ocr", type=Path)
    parser.add_argument("--target", default="51")
    parser.add_argument("--output", type=Path, default=Path("vision/runs/fused-player"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--min-anchors", type=int, default=2)
    parser.add_argument("--max-gap", type=float, default=.5)
    parser.add_argument("--max-distance", type=float, default=1.25,
                        help="Maximum center distance as a multiple of player height")
    parser.add_argument("--post-anchor-seconds", type=float, default=.8,
                        help="Stop after this long without any later OCR confirmation")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    roi = tuple(int(value) for value in args.roi.split(","))
    grouped = read_tracks(args.tracks)

    anchor_counts: Counter[int] = Counter()
    target_times: list[float] = []
    with args.ocr.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ocr"] == args.target:
                anchor_counts[int(row["track_id"])] += 1
                target_times.append(float(row["time_s"]))
    selected = {track_id for track_id, count in anchor_counts.items() if count >= args.min_anchors}
    if not selected:
        raise SystemExit(f"No track has {args.min_anchors} OCR anchors for #{args.target}")
    trust_until = max(target_times) + args.post_anchor_seconds

    links: list[tuple[int, int, float, float]] = []
    frontier = deque(selected)
    while frontier:
        source_id = frontier.popleft()
        end = grouped[source_id][-1]
        best = None
        for candidate_id, items in grouped.items():
            if candidate_id in selected:
                continue
            start = items[0]
            if start.time_s > trust_until:
                continue
            gap = start.time_s - end.time_s
            if gap < -.05 or gap > args.max_gap:
                continue
            distance = math.dist(center(end), center(start))
            scale = max(height(end), height(start), 1)
            normalized = distance / scale
            if normalized > args.max_distance:
                continue
            # Prefer spatial continuity, then a smaller temporal gap.
            score = normalized + max(gap, 0) * .35
            if best is None or score < best[0]:
                best = (score, candidate_id, gap, distance)
        if best:
            _, candidate_id, gap, distance = best
            selected.add(candidate_id)
            links.append((source_id, candidate_id, gap, distance))
            frontier.append(candidate_id)

    with (args.output / "segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("track_id", "start_s", "end_s", "ocr_anchors"))
        for track_id in sorted(selected, key=lambda value: grouped[value][0].time_s):
            writer.writerow((track_id, f"{grouped[track_id][0].time_s:.3f}",
                             f"{grouped[track_id][-1].time_s:.3f}", anchor_counts[track_id]))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, frame_height = roi[2] - roi[0], roi[3] - roi[1]
    writer = cv2.VideoWriter(str(args.output / f"fused-{args.target}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, frame_height))
    by_frame: dict[int, list[Observation]] = defaultdict(list)
    for track_id in selected:
        items = grouped[track_id]
        items = [item for item in items if item.time_s <= trust_until]
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
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        court = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        for item in by_frame.get(frame_index, []):
            x1, y1, x2, y2 = item.box
            anchored = anchor_counts[item.track_id] >= args.min_anchors
            color = (20, 210, 255) if anchored else (60, 220, 80)
            label = f"Dan #{args.target}  {'OCR' if anchored else 'TRACK'}"
            cv2.rectangle(court, (x1, y1), (x2, y2), color, 4)
            cv2.putText(court, label, (x1, max(28, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                        .75, color, 2, cv2.LINE_AA)
        writer.write(court)
        frame_index += 1
    writer.release()
    cap.release()

    covered_frames = len(by_frame)
    total_frames = max(frame_index, 1)
    print("selected=" + ",".join(map(str, sorted(selected))))
    print("links=" + ", ".join(f"{a}->{b} gap={gap:.2f}s dist={dist:.0f}px"
                                for a, b, gap, dist in links))
    print(f"covered_frames={covered_frames}/{total_frames} ({covered_frames / total_frames:.1%})")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
