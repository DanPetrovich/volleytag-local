#!/usr/bin/env python3
"""Render the real continuous video with the detector's sampled top-1 held between samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--speed", type=int, default=1,
                        help="write every Nth source frame while preserving source fps")
    args = parser.parse_args()

    cache = json.loads(args.candidates.read_text())
    times = np.asarray([float(value) for value in cache["times"]])
    candidate_sets = cache["candidate_sets"]
    if len(times) != len(candidate_sets) or not len(times):
        raise SystemExit("invalid or empty candidate cache")

    selected = []
    for candidates in candidate_sets:
        direct = [row for row in candidates if float(row.get("direct_confidence", 0.0)) > 0.0]
        selected.append(max(direct, key=lambda row: float(row["direct_confidence"])) if direct else None)

    cap = cv2.VideoCapture(str(args.video))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = round(height * args.width / width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
        (args.width, output_height),
    )

    frame_index = 0
    written = 0
    first_pts = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_pts is None:
            first_pts = pts
        timestamp = max(0.0, pts - first_pts)
        if timestamp > times[-1] + .4:
            break
        if frame_index % args.speed:
            frame_index += 1
            continue
        index = int(np.searchsorted(times, timestamp, side="right") - 1)
        index = max(0, min(index, len(times) - 1))
        row = selected[index]
        state = "MISSING"
        color = (45, 70, 245)
        if row is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in row["box"]]
            confidence = float(row["direct_confidence"])
            color = (255, 210, 30)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)
            cv2.rectangle(frame, (x1, max(0, y1 - 38)), (min(width, x1 + 320), y1), color, -1)
            cv2.putText(frame, f"V7 TOP-1 {confidence:.2f}", (x1 + 7, max(26, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .82, (5, 8, 12), 2, cv2.LINE_AA)
            state = f"TOP-1 confidence={confidence:.2f}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 90), (8, 10, 16), -1)
        cv2.addWeighted(overlay, .84, frame, .16, 0, frame)
        cv2.putText(frame, f"t={timestamp:7.2f}s   {state}", (26, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, .86, (245, 247, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, "REAL CONTINUOUS VIDEO | v7 inference every 0.4s; box held between samples",
                    (26, 73), cv2.FONT_HERSHEY_SIMPLEX, .67, (215, 222, 235), 2, cv2.LINE_AA)
        frame = cv2.resize(frame, (args.width, output_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
        frame_index += 1
        if written % 2500 == 0:
            print(f"written={written} t={timestamp:.1f}s", flush=True)

    cap.release()
    writer.release()
    if not written:
        raise SystemExit("no frames rendered")
    print(f"frames={written} duration={written/source_fps:.1f}s output={args.output}")


if __name__ == "__main__":
    main()
