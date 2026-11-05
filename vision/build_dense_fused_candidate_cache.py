#!/usr/bin/env python3
"""Build a dense cache from the fixed personal and generic person detectors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from evaluate_fused_detectors import merge_candidates


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def on_court(box: list[float], width: int, height: int) -> bool:
    x1, _, x2, y2 = box
    cx, foot_y = (x1 + x2) / (2 * width), y2 / height
    if foot_y < .23 or foot_y > 1.02:
        return False
    progress = min(1., max(0., (foot_y - .23) / .73))
    return .30 - .22 * progress <= cx <= .70 + .22 * progress


def excluded(timestamp: float, intervals: list[dict]) -> bool:
    return any(float(row["start_s"]) <= timestamp <= float(row["end_s"])
               for row in intervals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("direct_model", type=Path)
    parser.add_argument("--generic-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--step-seconds", type=float, default=.4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--dedupe-iou", type=float, default=.5)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--court-filter", action="store_true")
    args = parser.parse_args()

    intervals: list[dict] = []
    if args.exclude:
        payload = json.loads(args.exclude.read_text(encoding="utf-8"))
        intervals = payload.get("excluded_intervals", payload)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    end = min(duration, args.end_seconds) if args.end_seconds else duration
    targets = [float(value) for value in np.arange(args.start_seconds, end + 1e-6,
                                                    args.step_seconds)
               if not excluded(float(value), intervals)]
    device = choose_device(args.device)
    direct_model, generic_model = YOLO(str(args.direct_model)), YOLO(str(args.generic_model))
    times: list[float] = []
    candidate_sets: list[list[dict]] = []
    started = time.monotonic()

    for start in range(0, len(targets), args.batch):
        batch_times = targets[start:start + args.batch]
        frames, valid_times = [], []
        for timestamp in batch_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.)
            ok, frame = cap.read()
            if ok:
                frames.append(frame); valid_times.append(timestamp)
        if not frames:
            continue
        direct_results = direct_model.predict(frames, imgsz=args.imgsz, conf=args.conf,
                                              device=device, batch=len(frames), verbose=False)
        generic_results = generic_model.predict(frames, imgsz=args.imgsz, conf=args.conf,
                                                classes=[0], device=device, batch=len(frames),
                                                verbose=False)
        for timestamp, frame, direct_result, generic_result in zip(
                valid_times, frames, direct_results, generic_results):
            height, width = frame.shape[:2]
            direct = []
            if direct_result.boxes is not None:
                for box, confidence in zip(direct_result.boxes.xyxy.cpu().tolist(),
                                           direct_result.boxes.conf.cpu().tolist()):
                    direct.append({"box": [float(value) for value in box],
                                   "confidence": float(confidence),
                                   "direct_confidence": float(confidence)})
            generic = []
            if generic_result.boxes is not None:
                for box, confidence in zip(generic_result.boxes.xyxy.cpu().tolist(),
                                           generic_result.boxes.conf.cpu().tolist()):
                    values = [float(value) for value in box]
                    if not args.court_filter or on_court(values, width, height):
                        generic.append({"box": values, "confidence": float(confidence),
                                        "track_id": None})
            candidate_sets.append(merge_candidates(generic, direct, args.dedupe_iou))
            times.append(timestamp)
        print(json.dumps({"frames": len(times), "total": len(targets),
                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    cap.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "volleytag.dense-fused-candidates.v1",
        "video": str(args.video), "direct_model": str(args.direct_model),
        "generic_model": str(args.generic_model), "step_seconds": args.step_seconds,
        "imgsz": args.imgsz, "confidence": args.conf, "court_filter": args.court_filter,
        "excluded_intervals": len(intervals), "times": times,
        "candidate_sets": candidate_sets,
    }) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    print(json.dumps({"frames": len(times), "elapsed_s": round(elapsed, 1),
                      "fps": len(times) / elapsed if elapsed else 0.,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
