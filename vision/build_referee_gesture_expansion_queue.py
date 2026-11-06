#!/usr/bin/env python3
"""Build a visually diverse referee-gesture queue from one new training match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_referee_gesture_queue import expanded_crop, item, roi_motion, separated_peaks


def motion_descriptor(video: str, box: list[float], timestamp: float) -> np.ndarray:
    """Describe the referee's local motion while mostly cancelling the static court."""
    capture = cv2.VideoCapture(video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left, top, crop_width, crop_height = expanded_crop(box, width, height)
    frames = []
    for offset in (-1.5, -.75, 0, .75, 1.5):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp + offset) * 1000)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot decode {video} at {timestamp + offset:.3f}s")
        crop = frame[top:top + crop_height, left:left + crop_width]
        gray = cv2.cvtColor(cv2.resize(crop, (24, 36)), cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32) / 255)
    capture.release()
    deltas = [frames[index + 1] - frames[index] for index in range(len(frames) - 1)]
    vector = np.concatenate([delta.ravel() for delta in deltas])
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def diverse_selection(candidates: list[tuple[float, float]], descriptors: np.ndarray,
                      count: int) -> list[int]:
    """Greedily balance strong motion with distance from already selected motion."""
    if not candidates or count <= 0:
        return []
    scores = np.asarray([row[1] for row in candidates], dtype=np.float32)
    score_range = float(np.ptp(scores))
    strength = (scores - float(scores.min())) / max(score_range, 1e-8)
    selected = [int(np.argmax(scores))]
    remaining = set(range(len(candidates))) - set(selected)
    while remaining and len(selected) < count:
        indices = np.asarray(sorted(remaining))
        similarity = descriptors[indices] @ descriptors[np.asarray(selected)].T
        distance = 1 - similarity.max(axis=1)
        utility = .7 * distance + .3 * strength[indices]
        chosen = int(indices[int(np.argmax(utility))])
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--roi", type=float, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--replay-intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--candidate-pool", type=int, default=90)
    parser.add_argument("--separation-s", type=float, default=10)
    args = parser.parse_args()

    replay_payload = json.loads(args.replay_intervals.read_text())
    replay_source = replay_payload["sources"][str(args.source)]
    if replay_source["video"] != args.video:
        raise ValueError("replay metadata belongs to another video")
    excluded = replay_source["exclude_intervals"]
    scores, duration = roi_motion(args.video, args.roi)
    candidates = separated_peaks(scores, 4, args.candidate_pool, [], excluded,
                                  separation_s=args.separation_s)
    candidates = [row for row in candidates if row[0] <= duration - 1.5]
    descriptors = np.stack([motion_descriptor(args.video, args.roi, time)
                            for time, _ in candidates])
    selected = diverse_selection(candidates, descriptors, args.count)
    rows = []
    for index in selected:
        timestamp, score = candidates[index]
        row = item(args.source, args.video, timestamp, args.roi,
                   "diverse_roi_motion", score)
        before = [interval["end_s"] for interval in excluded
                  if row["window_start_s"] < interval["end_s"] < timestamp]
        after = [interval["start_s"] for interval in excluded
                 if timestamp < interval["start_s"] < row["window_end_s"]]
        if before:
            row["window_start_s"] = max(row["window_start_s"], max(before))
        if after:
            row["window_end_s"] = min(row["window_end_s"], min(after))
        row["diversity_rank"] = len(rows) + 1
        rows.append(row)
    rows.sort(key=lambda row: row["time_s"])
    result = {
        "schema_version": 1,
        "kind": "volleytag.referee-gesture-review-queue",
        "frozen_tests_used": False,
        "labels": ["serve_authorization", "rally_end_team_to_serve", "other_signal",
                   "no_signal", "uncertain"],
        "selection": {
            "method": "motion peaks followed by greedy motion-descriptor diversity",
            "candidate_pool": len(candidates),
            "minimum_time_separation_s": args.separation_s,
        },
        "counts": {"diverse_roi_motion": len(rows)},
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": len(rows), "duration_s": duration,
                      "candidate_pool": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
