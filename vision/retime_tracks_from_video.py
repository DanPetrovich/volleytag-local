#!/usr/bin/env python3
"""Replace synthetic frame/fps track timestamps with the video's actual frame PTS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.tracks.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("tracks CSV is empty")
    required_frames = {int(row["frame"]) for row in rows}
    max_frame = max(required_frames)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    timestamps = {}
    first_pts = None
    frame = 0
    while frame <= max_frame:
        ok, _ = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_pts is None:
            first_pts = pts
        if frame in required_frames:
            timestamps[frame] = max(0.0, pts - first_pts)
        frame += 1
    cap.release()
    missing = required_frames - timestamps.keys()
    if missing:
        raise SystemExit(f"missing PTS for {len(missing)} frames; first={min(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row["time_s"] = f"{timestamps[int(row['frame'])]:.3f}"
            writer.writerow(row)
    print(f"rows={len(rows)} frames={len(required_frames)} end_s={max(timestamps.values()):.3f}")


if __name__ == "__main__":
    main()
