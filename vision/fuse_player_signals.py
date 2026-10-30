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
    parser.add_argument("--similarity", type=Path,
                        help="Optional ReID similarity.csv used to rank continuity candidates")
    parser.add_argument("--target", default="51")
    parser.add_argument("--output", type=Path, default=Path("vision/runs/fused-player"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--min-anchors", type=int, default=2)
    parser.add_argument(
        "--seed-segments", default="",
        help="Confirmed continuous track chunks as TRACK_ID:CHUNK, comma-separated",
    )
    parser.add_argument("--max-gap", type=float, default=.5)
    parser.add_argument("--max-distance", type=float, default=1.25,
                        help="Maximum center distance as a multiple of player height")
    parser.add_argument(
        "--min-continuation-margin", type=float, default=0.0,
        help="Reject automatic continuation below this ReID margin; manual/OCR seeds are exempt",
    )
    parser.add_argument("--segment-gap", type=float, default=.25,
                        help="Split a reused tracker ID after this observation gap")
    parser.add_argument("--bridge-gap", type=float, default=.8,
                        help="Interpolate short gaps between already selected segments")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    roi = tuple(int(value) for value in args.roi.split(","))
    grouped = read_tracks(args.tracks)

    reid_scores: dict[int, float] = {}
    if args.similarity:
        with args.similarity.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reid_scores[int(row["track_id"])] = float(row["margin"])

    # Tracker IDs can be reused after an occlusion. Split them into continuous
    # pieces first; an OCR hit validates a piece, never the global ID.
    chunks: dict[tuple[int, int], list[Observation]] = {}
    observation_chunk: dict[tuple[int, int], tuple[int, int]] = {}
    for track_id, items in grouped.items():
        chunk_index = 0
        current: list[Observation] = []
        for item in items:
            if current and item.time_s - current[-1].time_s > args.segment_gap:
                key = (track_id, chunk_index)
                chunks[key] = current
                for observation in current:
                    observation_chunk[(observation.frame, observation.track_id)] = key
                chunk_index += 1
                current = []
            current.append(item)
        if current:
            key = (track_id, chunk_index)
            chunks[key] = current
            for observation in current:
                observation_chunk[(observation.frame, observation.track_id)] = key

    with (args.output / "chunks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("track_id", "chunk", "start_s", "end_s", "observations"))
        for key, items in sorted(chunks.items(), key=lambda value: value[1][0].time_s):
            writer.writerow((key[0], key[1], f"{items[0].time_s:.3f}",
                             f"{items[-1].time_s:.3f}", len(items)))

    anchor_counts: Counter[tuple[int, int]] = Counter()
    with args.ocr.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ocr"] == args.target:
                lookup = (int(row["frame"]), int(row["track_id"]))
                key = observation_chunk.get(lookup)
                if key:
                    anchor_counts[key] += 1
    manual_seeds: set[tuple[int, int]] = set()
    for value in args.seed_segments.split(","):
        if not value:
            continue
        try:
            track_id, chunk_index = (int(part) for part in value.split(":"))
        except (ValueError, TypeError):
            raise SystemExit(f"Invalid --seed-segments value: {value!r}; expected TRACK_ID:CHUNK")
        key = (track_id, chunk_index)
        if key not in chunks:
            raise SystemExit(f"Seed segment is not present: {value}")
        manual_seeds.add(key)

    selected = {key for key, count in anchor_counts.items() if count >= args.min_anchors}
    selected.update(manual_seeds)
    if not selected:
        raise SystemExit(
            f"No continuous track segment has {args.min_anchors} OCR anchors for #{args.target}; "
            "provide --seed-segments after visually confirming a chunk"
        )

    sources: dict[tuple[int, int], str] = {
        key: ("MANUAL" if key in manual_seeds else "OCR") for key in selected
    }

    links: list[tuple[tuple[int, int], tuple[int, int], float, float]] = []
    frontier = deque(selected)
    while frontier:
        source_key = frontier.popleft()
        end = chunks[source_key][-1]
        best = None
        for candidate_key, items in chunks.items():
            if candidate_key in selected:
                continue
            if reid_scores and reid_scores.get(candidate_key[0], float("-inf")) < args.min_continuation_margin:
                continue
            start = items[0]
            gap = start.time_s - end.time_s
            if gap < -.05 or gap > args.max_gap:
                continue
            distance = math.dist(center(end), center(start))
            scale = max(height(end), height(start), 1)
            normalized = distance / scale
            if normalized > args.max_distance:
                continue
            # Prefer spatial continuity, then a smaller temporal gap.
            # Geometry is the main signal. ReID is only a tie-breaker: appearance
            # embeddings are useful, but volleyball uniforms make them unsafe as
            # a sole identity decision.
            appearance_bonus = max(-.10, min(.10, reid_scores.get(candidate_key[0], 0.0)))
            score = normalized + max(gap, 0) * .35 - appearance_bonus * 1.5
            if best is None or score < best[0]:
                best = (score, candidate_key, gap, distance)
        if best:
            _, candidate_key, gap, distance = best
            selected.add(candidate_key)
            sources[candidate_key] = "TRACK"
            links.append((source_key, candidate_key, gap, distance))
            frontier.append(candidate_key)

    with (args.output / "segments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("track_id", "chunk", "start_s", "end_s", "source", "ocr_anchors"))
        for key in sorted(selected, key=lambda value: chunks[value][0].time_s):
            items = chunks[key]
            writer.writerow((key[0], key[1], f"{items[0].time_s:.3f}",
                             f"{items[-1].time_s:.3f}", sources[key], anchor_counts[key]))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, frame_height = roi[2] - roi[0], roi[3] - roi[1]
    writer = cv2.VideoWriter(str(args.output / f"fused-{args.target}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, frame_height))
    by_frame: dict[int, list[tuple[Observation, str]]] = defaultdict(list)
    for key in selected:
        items = chunks[key]
        for item in items:
            by_frame[item.frame].append((item, sources[key]))
        for left, right in zip(items, items[1:]):
            gap = right.frame - left.frame
            if gap <= 1 or gap > 6:
                continue
            for frame in range(left.frame + 1, right.frame):
                ratio = (frame - left.frame) / gap
                box = tuple(round(a + (b - a) * ratio) for a, b in zip(left.box, right.box))
                by_frame[frame].append((Observation(
                    time_s=left.time_s + (right.time_s - left.time_s) * ratio,
                    frame=frame, track_id=key[0],
                    confidence=min(left.confidence, right.confidence), box=box,
                ), sources[key]))
    ordered_selected = sorted(selected, key=lambda key: chunks[key][0].time_s)
    bridged_frames = 0
    for left_key, right_key in zip(ordered_selected, ordered_selected[1:]):
        left, right = chunks[left_key][-1], chunks[right_key][0]
        time_gap = right.time_s - left.time_s
        frame_gap = right.frame - left.frame
        if frame_gap <= 1 or time_gap < 0 or time_gap > args.bridge_gap:
            continue
        normalized = math.dist(center(left), center(right)) / max(height(left), height(right), 1)
        if normalized > args.max_distance:
            continue
        for frame in range(left.frame + 1, right.frame):
            if frame in by_frame:
                continue
            ratio = (frame - left.frame) / frame_gap
            box = tuple(round(a + (b - a) * ratio) for a, b in zip(left.box, right.box))
            by_frame[frame].append((Observation(
                time_s=left.time_s + (right.time_s - left.time_s) * ratio,
                frame=frame, track_id=right_key[0],
                confidence=min(left.confidence, right.confidence), box=box,
            ), "BRIDGE"))
            bridged_frames += 1
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        court = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        for item, source in by_frame.get(frame_index, []):
            x1, y1, x2, y2 = item.box
            anchored = source in {"OCR", "MANUAL"}
            color = (20, 210, 255) if anchored else (60, 220, 80)
            label = f"Dan #{args.target}  {source}"
            cv2.rectangle(court, (x1, y1), (x2, y2), color, 4)
            cv2.putText(court, label, (x1, max(28, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                        .75, color, 2, cv2.LINE_AA)
        writer.write(court)
        frame_index += 1
    writer.release()
    cap.release()

    covered_frames = len(by_frame)
    total_frames = max(frame_index, 1)
    print("selected=" + ",".join(f"{track}:{chunk}" for track, chunk in sorted(selected)))
    print("links=" + ", ".join(f"{a[0]}:{a[1]}->{b[0]}:{b[1]} gap={gap:.2f}s dist={dist:.0f}px"
                                for a, b, gap, dist in links))
    print(f"covered_frames={covered_frames}/{total_frames} ({covered_frames / total_frames:.1%})")
    print(f"bridged_frames={bridged_frames}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
