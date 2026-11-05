#!/usr/bin/env python3
"""Build fused candidate sequences on audited training frames for temporal validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from evaluate_fused_detectors import merge_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("generic_model", type=Path)
    parser.add_argument("personal_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sources", type=int, nargs="+")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    positives = [row for row in manifest if row["kind"] in {"positive", "manual_positive"}]
    if args.sources:
        positives = [row for row in positives if int(row["source"]) in set(args.sources)]
    # Prefer human boxes over tracker boxes when both describe the same instant.
    unique = {}
    for row in positives:
        key = (int(row["source"]), round(float(row["time_s"]), 3))
        if key not in unique or row["kind"] == "manual_positive":
            unique[key] = row
    rows = sorted(unique.values(), key=lambda row: (int(row["source"]), float(row["time_s"])))
    if args.limit and args.limit < len(rows):
        indexes = np.linspace(0, len(rows) - 1, args.limit, dtype=int)
        rows = [rows[index] for index in indexes]
    generic_model, personal_model = YOLO(str(args.generic_model)), YOLO(str(args.personal_model))
    output_rows = []
    for start in range(0, len(rows), args.batch):
        batch_rows = rows[start:start + args.batch]
        frames = [cv2.imread(str(args.dataset / row["image"])) for row in batch_rows]
        valid = [(row, frame) for row, frame in zip(batch_rows, frames) if frame is not None]
        if not valid:
            continue
        valid_rows, valid_frames = zip(*valid)
        generic_results = generic_model.predict(
            list(valid_frames), imgsz=960, conf=.08, classes=[0], device=args.device,
            batch=len(valid_frames), verbose=False)
        personal_results = personal_model.predict(
            list(valid_frames), imgsz=960, conf=.05, device=args.device,
            batch=len(valid_frames), verbose=False)
        for row, generic_result, personal_result in zip(valid_rows, generic_results, personal_results):
            generic = []
            if generic_result.boxes is not None:
                generic = [{"box": box, "confidence": float(conf), "source": "generic",
                            "track_id": None}
                           for box, conf in zip(generic_result.boxes.xyxy.cpu().tolist(),
                                                generic_result.boxes.conf.cpu().tolist())]
            direct = []
            if personal_result.boxes is not None:
                direct = [{"box": box, "confidence": float(conf),
                           "direct_confidence": float(conf), "source": "direct"}
                          for box, conf in zip(personal_result.boxes.xyxy.cpu().tolist(),
                                               personal_result.boxes.conf.cpu().tolist())]
            output_rows.append({
                "source": int(row["source"]), "time_s": float(row["time_s"]),
                "gt_box": [float(value) for value in row["box"]],
                "candidates": merge_candidates(generic, direct, .50),
            })
        print(f"processed={min(start + args.batch, len(rows))}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": "audited training frames only; generic yolo11n + personal v7",
        "frames": len(output_rows), "rows": output_rows,
    }) + "\n", encoding="utf-8")
    print(f"output={args.output} frames={len(output_rows)}")


if __name__ == "__main__":
    main()
