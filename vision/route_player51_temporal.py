#!/usr/bin/env python3
"""Build one temporally coherent player-51 route from fused detections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def iou(first: list[float], second: list[float]) -> float:
    """Return box IoU without importing the heavyweight inference stack."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def normalized_speed(previous: dict, current: dict, dt: float) -> float:
    """Return centre speed in average person-box heights per second."""
    px, py = center(previous["box"])
    cx, cy = center(current["box"])
    height = max(40.0, ((previous["box"][3] - previous["box"][1]) +
                        (current["box"][3] - current["box"][1])) / 2.0)
    return math.hypot(cx - px, cy - py) / height / max(dt, 0.1)


def transition(previous: dict | None, current: dict | None, dt: float) -> float:
    if previous is None and current is None:
        return 0.0
    if previous is None or current is None:
        return -0.75
    speed = normalized_speed(previous, current, dt)
    value = -min(12.0, 0.55 * speed * speed)
    if previous.get("track_id") is not None and previous.get("track_id") == current.get("track_id"):
        value += 2.5
    value += 0.8 * iou(previous["box"], current["box"])
    return value


def unary(candidate: dict | None) -> float:
    if candidate is None:
        return -1.1
    direct = float(candidate.get("direct_confidence", 0.0))
    if direct > 0.0:
        return 3.5 * direct
    # Generic confidence says "person", not "Danya", so it remains a weak
    # fallback signal and cannot overpower a personal detection by itself.
    return 0.12 * float(candidate.get("confidence", 0.0))


def identity_score(candidate: dict) -> float:
    """Frozen identity score combining personal detector and reranker evidence."""
    direct = float(candidate.get("direct_confidence", 0.0))
    reranker = float(candidate.get("reranker_probability", 0.0))
    # Natural, untuned combination of two independent probabilities. The
    # reranker can recover a generic-only candidate while direct detections
    # retain a small tie-breaking advantage.
    return reranker + 0.15 * direct


def decode_anchor_first(candidate_sets: list[list[dict | None]], times: list[float],
                        max_speed: float) -> list[int]:
    """Causal route: personal detections anchor identity; tracks bridge short gaps."""
    selected: list[int] = []
    previous: dict | None = None
    last_anchor_time: float | None = None
    for frame_index, (timestamp, candidates) in enumerate(zip(times, candidate_sets)):
        direct_indexes = [index for index, row in enumerate(candidates)
                          if row is not None and float(row.get("direct_confidence", 0.0)) > 0.0]
        if direct_indexes:
            if previous is not None:
                dt = max(0.1, timestamp - times[max(0, frame_index - 1)])
                plausible = [index for index in direct_indexes
                             if normalized_speed(previous, candidates[index], dt) <= max_speed]
                # A personal detection on another player must not be allowed to
                # teleport the route. Emit an explicit gap and keep the last
                # trusted position for a later, physically plausible recovery.
                if not plausible:
                    selected.append(len(candidates) - 1)
                    # After a long detector outage, require one explicit
                    # missing sample before identity is reacquired elsewhere.
                    # This prevents a stale generic bridge from being drawn as
                    # a several-metre box teleport.
                    if last_anchor_time is None or timestamp - last_anchor_time > 1.6:
                        previous = None
                        last_anchor_time = None
                    continue
                direct_indexes = plausible
            def anchor_score(index: int) -> float:
                row = candidates[index]
                assert row is not None
                value = float(row["direct_confidence"])
                if previous is not None:
                    value += 0.04 * iou(previous["box"], row["box"])
                    if (previous.get("track_id") is not None and
                            previous.get("track_id") == row.get("track_id")):
                        value += 0.02
                return value
            choice = max(direct_indexes, key=anchor_score)
            previous = candidates[choice]
            last_anchor_time = timestamp
            selected.append(choice)
            continue

        visible_indexes = [index for index, row in enumerate(candidates) if row is not None]
        choice = len(candidates) - 1  # explicit missing state
        if previous is not None and visible_indexes and last_anchor_time is not None:
            # Only bridge a short personal-detector gap. Beyond that, continuing
            # a generic track is more likely to lock onto the wrong player.
            if timestamp - last_anchor_time <= 1.6:
                same_track = [index for index in visible_indexes
                              if previous.get("track_id") is not None and
                              candidates[index].get("track_id") == previous.get("track_id")]
                pool = same_track or visible_indexes
                dt = max(0.1, timestamp - times[max(0, frame_index - 1)])
                best = max(pool, key=lambda index: transition(previous, candidates[index],
                                                               dt))
                if normalized_speed(previous, candidates[best], dt) <= max_speed:
                    choice = best
        selected.append(choice)
        if candidates[choice] is not None:
            previous = candidates[choice]
    return selected


def decode_direct_bridge(candidate_sets: list[list[dict | None]], times: list[float],
                         max_speed: float, bridge_seconds: float = 1.6) -> list[int]:
    """Never override direct top-1; bridge only frames with no personal detection."""
    selected: list[int] = []
    previous: dict | None = None
    last_anchor_time: float | None = None
    for frame_index, (timestamp, candidates) in enumerate(zip(times, candidate_sets)):
        direct = [index for index, row in enumerate(candidates)
                  if row is not None and float(row.get("direct_confidence", 0.0)) > 0.0]
        if direct:
            choice = max(direct, key=lambda index:
                         float(candidates[index].get("direct_confidence", 0.0)))
            selected.append(choice)
            previous = candidates[choice]
            last_anchor_time = timestamp
            continue
        choice = len(candidates) - 1
        visible = [index for index, row in enumerate(candidates) if row is not None]
        if (previous is not None and last_anchor_time is not None and visible
                and timestamp - last_anchor_time <= bridge_seconds):
            same_track = [index for index in visible
                          if previous.get("track_id") is not None and
                          candidates[index].get("track_id") == previous.get("track_id")]
            pool = same_track or visible
            dt = max(.1, timestamp - times[max(0, frame_index - 1)])
            plausible = [index for index in pool
                         if normalized_speed(previous, candidates[index], dt) <= max_speed]
            if plausible:
                choice = max(plausible, key=lambda index:
                             transition(previous, candidates[index], dt))
        selected.append(choice)
        if candidates[choice] is not None:
            previous = candidates[choice]
        elif last_anchor_time is None or timestamp - last_anchor_time > bridge_seconds:
            previous = None
    return selected


def decode_direct_only(candidate_sets: list[list[dict | None]]) -> list[int]:
    """Direct-detector reference using the exact same temporal sampling cache."""
    selected = []
    for candidates in candidate_sets:
        direct = [index for index, row in enumerate(candidates)
                  if row is not None and float(row.get("direct_confidence", 0.0)) > 0.0]
        selected.append(max(direct, key=lambda index:
                            float(candidates[index].get("direct_confidence", 0.0)))
                        if direct else len(candidates) - 1)
    return selected


def decode_reranker(candidate_sets: list[list[dict | None]], times: list[float],
                    max_speed: float) -> list[int]:
    """Causal route driven by frozen reranker probabilities and a motion gate."""
    selected: list[int] = []
    previous: dict | None = None
    for frame_index, candidates in enumerate(candidate_sets):
        visible = [index for index, row in enumerate(candidates) if row is not None]
        if previous is not None:
            dt = max(.1, times[frame_index] - times[max(0, frame_index - 1)])
            visible = [index for index in visible
                       if normalized_speed(previous, candidates[index], dt) <= max_speed]
        credible = [index for index in visible
                    if (float(candidates[index].get("reranker_probability", 0.0)) >= .5 or
                        float(candidates[index].get("direct_confidence", 0.0)) > 0.0)]
        if not credible:
            selected.append(len(candidates) - 1)
            previous = None
            continue
        def score(index: int) -> float:
            row = candidates[index]
            assert row is not None
            value = identity_score(row)
            if previous is not None:
                value += .05 * iou(previous["box"], row["box"])
                if previous.get("track_id") is not None and previous.get("track_id") == row.get("track_id"):
                    value += .03
            return value
        choice = max(credible, key=score)
        selected.append(choice)
        previous = candidates[choice]
    return selected


def main() -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from evaluate_fused_detectors import load_tracks, merge_candidates

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-seconds", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--dedupe-iou", type=float, default=0.50)
    parser.add_argument("--max-time-delta", type=float, default=0.12)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--max-speed", type=float, default=6.0,
                        help="hard motion gate in average box-heights per second")
    parser.add_argument("--start-seconds", type=float, default=0.0,
                        help="optional absolute start time for a representative sanity run")
    parser.add_argument("--end-seconds", type=float,
                        help="optional absolute end time for a representative sanity run")
    parser.add_argument("--candidate-cache", type=Path,
                        help="optional pre-scored candidate cache")
    parser.add_argument("--decoder", choices=("anchor", "reranker", "direct_bridge", "direct_only"),
                        default="anchor")
    args = parser.parse_args()

    cache_path = args.candidate_cache or (args.output.parent / "candidates.json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        times = [float(value) for value in cached["times"]]
        candidate_sets = [[row for row in rows] + [None] for rows in cached["candidate_sets"]]
    else:
        candidate_sets = []

        track_times, by_time = load_tracks(args.tracks)
        cap = cv2.VideoCapture(str(args.video))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
        track_duration = float(track_times[-1]) if len(track_times) else 0.0
        if track_duration and abs(duration - track_duration) > max(5.0, .05 * track_duration):
            raise SystemExit(
                f"video/tracks duration mismatch: video={duration:.1f}s, "
                f"tracks={track_duration:.1f}s; choose the matching video"
            )
        start_seconds = max(0.0, args.start_seconds)
        end_seconds = min(duration, args.end_seconds) if args.end_seconds is not None else duration
        if end_seconds <= start_seconds:
            raise SystemExit("end-seconds must be greater than start-seconds")
        targets = np.arange(start_seconds, end_seconds + 1e-6, args.step_seconds)
        times = []
        batch_frames, batch_times = [], []
        target_index, first_pts = 0, None
        device = choose_device(args.device)
        model = YOLO(str(args.model))

        def process_batch() -> None:
            if not batch_frames:
                return
            results = model.predict(batch_frames, imgsz=args.imgsz, conf=args.conf,
                                    verbose=False, device=device, batch=len(batch_frames))
            for timestamp, result in zip(batch_times, results):
                nearest = float(track_times[np.argmin(np.abs(track_times - timestamp))])
                generic = by_time[nearest] if abs(nearest - timestamp) <= args.max_time_delta else []
                boxes = (result.boxes.xyxy.cpu().numpy() if result.boxes is not None
                         else np.empty((0, 4)))
                scores = (result.boxes.conf.cpu().numpy() if result.boxes is not None
                          else np.empty(0))
                direct = [{"box": [float(value) for value in box],
                           "confidence": float(score), "direct_confidence": float(score)}
                          for box, score in zip(boxes, scores)]
                merged = merge_candidates(generic, direct, args.dedupe_iou)
                candidate_sets.append([*merged, None])
                times.append(float(timestamp))
            batch_frames.clear()
            batch_times.clear()
            print(f"inference={len(times)}/{len(targets)}", flush=True)

        if start_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
        while target_index < len(targets):
            ok, frame = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if first_pts is None:
                first_pts = pts
            timestamp = start_seconds + max(0.0, pts - first_pts)
            if timestamp + 0.02 < targets[target_index]:
                continue
            batch_frames.append(frame)
            batch_times.append(float(timestamp))
            target_index += 1
            if len(batch_frames) >= args.batch:
                process_batch()
        cap.release()
        process_batch()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "times": times,
            "candidate_sets": [[row for row in rows if row is not None] for rows in candidate_sets],
        }) + "\n", encoding="utf-8")

    if args.decoder == "reranker":
        route_indexes = decode_reranker(candidate_sets, times, args.max_speed)
    elif args.decoder == "direct_bridge":
        route_indexes = decode_direct_bridge(candidate_sets, times, args.max_speed)
    elif args.decoder == "direct_only":
        route_indexes = decode_direct_only(candidate_sets)
    else:
        route_indexes = decode_anchor_first(candidate_sets, times, args.max_speed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = ("time_s", "state", "source", "track_id", "confidence",
                  "direct_confidence", "x1", "y1", "x2", "y2")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timestamp, candidates, selected in zip(times, candidate_sets, route_indexes):
            row = candidates[selected]
            if row is None:
                writer.writerow({"time_s": f"{timestamp:.3f}", "state": "missing"})
                continue
            writer.writerow({
                "time_s": f"{timestamp:.3f}", "state": "visible",
                "source": row["source"], "track_id": row.get("track_id"),
                "confidence": f"{float(row.get('confidence', 0.0)):.6f}",
                "direct_confidence": f"{float(row.get('direct_confidence', 0.0)):.6f}",
                **{key: f"{value:.2f}" for key, value in zip(("x1", "y1", "x2", "y2"), row["box"])},
            })
    print(f"cache={cache_path}")
    print(f"samples={len(times)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
