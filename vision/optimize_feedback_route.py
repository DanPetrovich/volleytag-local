#!/usr/bin/env python3
"""Find a globally consistent player path with feedback anchors as hard constraints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

from build_feedback_profile import split_chunks
from identify_player import Observation, read_tracks


def center(item: Observation):
    return ((item.box[0] + item.box[2]) / 2, (item.box[1] + item.box[3]) / 2)


def height(item: Observation):
    return max(1, item.box[3] - item.box[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("similarity", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision/runs/feedback-route"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    feedback = json.loads(args.feedback.read_text(encoding="utf-8"))
    grouped = read_tracks(args.tracks)
    chunks = split_chunks(grouped)
    item_key = {(item.frame, item.track_id): key for key, items in chunks.items() for item in items}
    scores = {}
    with args.similarity.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[row["key"]] = float(row["margin"])
    negative_keys = {entry["key"] for entry in feedback.get("negatives", [])}
    anchors = {min(chunks[entry["key"]], key=lambda item: abs(item.time_s - float(entry["time"]))).frame: entry["key"]
               for entry in feedback.get("anchors", []) if entry["key"] in chunks}

    by_frame = defaultdict(list)
    for items in grouped.values():
        for item in items:
            key = item_key[(item.frame, item.track_id)]
            if key not in negative_keys:
                by_frame[item.frame].append(item)
    frames = sorted(by_frame)
    costs, back, used_candidates = {}, {}, {}
    previous_frame = None
    for frame in frames:
        candidates = by_frame[frame]
        forced = anchors.get(frame)
        if forced:
            candidates = [item for item in candidates if item_key[(item.frame, item.track_id)] == forced]
        used_candidates[frame] = candidates
        current_costs, current_back = {}, {}
        for index, item in enumerate(candidates):
            key = item_key[(item.frame, item.track_id)]
            emission = -scores.get(key, -.25) * 5 - item.confidence * .08
            best = None
            if previous_frame is None:
                best = (emission, None)
            else:
                for previous_index, previous in enumerate(used_candidates[previous_frame]):
                    if previous_index not in costs:
                        continue
                    dt = max(item.time_s - previous.time_s, .04)
                    normalized = math.dist(center(previous), center(item)) / max(height(previous), height(item))
                    if normalized > .42 + dt * 1.8:
                        continue
                    previous_key = item_key[(previous.frame, previous.track_id)]
                    transition = normalized * 3.2 + abs(math.log(height(item) / height(previous))) * .35
                    if previous_key == key:
                        transition -= .75
                    value = costs[previous_index] + transition + emission
                    if best is None or value < best[0]:
                        best = (value, previous_index)
                if best is None and costs:
                    for previous_index, previous in enumerate(used_candidates[previous_frame]):
                        if previous_index not in costs:
                            continue
                        normalized = math.dist(center(previous), center(item)) / max(height(previous), height(item))
                        value = costs[previous_index] + normalized * 7 + emission + 2
                        if best is None or value < best[0]:
                            best = (value, previous_index)
            if best is not None:
                current_costs[index], current_back[index] = best
        if not current_costs:
            # Restart only when no geometrically reachable state remains.
            for index, item in enumerate(candidates):
                key = item_key[(item.frame, item.track_id)]
                current_costs[index] = -scores.get(key, -.25) * 5
                current_back[index] = None
        costs, back[frame], previous_frame = current_costs, current_back, frame

    last_frame = frames[-1]
    index = min(costs, key=costs.get)
    route = []
    for frame in reversed(frames):
        candidates = used_candidates[frame]
        if index is None or index >= len(candidates):
            break
        item = candidates[index]
        route.append((item, item_key[(item.frame, item.track_id)]))
        index = back[frame].get(index)
    route.reverse()

    with (args.output / "route.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "frame", "key", "track_id", "x1", "y1", "x2", "y2", "score"))
        for item, key in route:
            writer.writerow((f"{item.time_s:.3f}", item.frame, key, item.track_id, *item.box,
                             f"{scores.get(key, 0):.5f}"))

    roi = tuple(int(value) for value in args.roi.split(","))
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, out_height = roi[2] - roi[0], roi[3] - roi[1]
    writer = cv2.VideoWriter(str(args.output / "player51-global-route.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, out_height))
    route_by_frame = {item.frame: (item, key) for item, key in route}
    for (left, left_key), (right, right_key) in zip(route, route[1:]):
        gap = right.frame - left.frame
        if gap <= 1 or gap > 4:
            continue
        for frame in range(left.frame + 1, right.frame):
            ratio = (frame - left.frame) / gap
            box = tuple(round(a + (b - a) * ratio) for a, b in zip(left.box, right.box))
            route_by_frame[frame] = (Observation(
                time_s=left.time_s + (right.time_s - left.time_s) * ratio,
                frame=frame, track_id=right.track_id,
                confidence=min(left.confidence, right.confidence), box=box,
            ), right_key if ratio >= .5 else left_key)
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        court = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        selected = route_by_frame.get(frame_index)
        if selected:
            item, key = selected
            x1, y1, x2, y2 = item.box
            cv2.rectangle(court, (x1, y1), (x2, y2), (65, 235, 95), 4)
            cv2.putText(court, f"Dan #51 GLOBAL {key}", (x1, max(28, y1 - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (65, 235, 95), 2, cv2.LINE_AA)
        writer.write(court)
        frame_index += 1
    writer.release(); cap.release()

    switches = sum(left[1] != right[1] for left, right in zip(route, route[1:]))
    report = {"video_frames": frame_index, "tracked_frames": len(frames), "covered": len(route),
              "coverage": len(route) / max(len(frames), 1),
              "chunk_switches": switches, "anchors": len(anchors), "negative_chunks": len(negative_keys)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
