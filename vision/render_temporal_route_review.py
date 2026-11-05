#!/usr/bin/env python3
"""Render a compact continuous review video for a saved temporal route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_fused_detectors import iou, keyframes


def box(frame, values, color, label, thickness):
    x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1 + 3, max(24, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    .63, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("route", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0,
                        help="10 fps makes a 0.4-second route play at 4x speed")
    args = parser.parse_args()

    with args.route.open() as handle:
        route = list(csv.DictReader(handle))
    cache = json.loads(args.candidates.read_text())
    samples = keyframes(json.loads(args.ground_truth.read_text()))
    gt_times = np.asarray([float(row["time_s"]) for row in samples])
    route_times = np.asarray([float(row["time_s"]) for row in route])
    assert len(route) == len(cache["times"])

    cap = cv2.VideoCapture(str(args.video))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target = 0
    output = None
    first_pts = None
    while target < len(route):
        ok, frame = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if first_pts is None:
            first_pts = pts
        timestamp = max(0.0, pts - first_pts)
        if timestamp + .02 < route_times[target]:
            continue

        candidates = cache["candidate_sets"][target]
        for candidate in candidates:
            if float(candidate.get("direct_confidence", 0.0)) > 0:
                box(frame, candidate["box"], (255, 130, 40), "v7", 2)
        selected = route[target]
        status = "MISSING"
        if selected["state"] == "visible":
            selected_box = [float(selected[key]) for key in ("x1", "y1", "x2", "y2")]
            box(frame, selected_box, (20, 230, 255), "TEMPORAL SELECTED", 5)
            status = f"SELECTED source={selected['source']} track={selected['track_id']}"

        gt_note = ""
        if len(gt_times):
            nearest = int(np.argmin(np.abs(gt_times - route_times[target])))
            if abs(gt_times[nearest] - route_times[target]) <= .22:
                gt = samples[nearest]["box"]
                box(frame, gt, (70, 230, 90), "HUMAN GT", 5)
                overlap = (iou(gt, selected_box) if selected["state"] == "visible" else 0.0)
                gt_note = f"  GT IoU={overlap:.2f} {'HIT' if overlap >= .3 else 'MISS'}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 82), (8, 10, 16), -1)
        cv2.addWeighted(overlay, .82, frame, .18, 0, frame)
        cv2.putText(frame, f"t={route_times[target]:.1f}s  {status}{gt_note}", (24, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .78, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, "yellow=temporal route  blue=all v7 proposals  green=human checkpoints",
                    (24, 68), cv2.FONT_HERSHEY_SIMPLEX, .67, (220, 225, 235), 2, cv2.LINE_AA)
        width = 1280
        height = round(frame.shape[0] * width / frame.shape[1])
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if output is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (width, height))
        output.write(frame)
        target += 1
    cap.release()
    if output is not None:
        output.release()
    if target != len(route):
        raise SystemExit(f"rendered only {target}/{len(route)} route points")
    print(f"frames={target} source_fps={source_fps:.3f} output={args.output}")


if __name__ == "__main__":
    main()
