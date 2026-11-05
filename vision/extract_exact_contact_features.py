#!/usr/bin/env python3
"""Extract cached short-window features from reviewed exact-contact metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from build_touch_temporal_features import clip_feature, device_name, read_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path, nargs="+")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-clips", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = []
    for path in args.metadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("frozen_tests_used"):
            raise ValueError(f"refusing frozen metadata: {path}")
        rows.extend(payload["samples"])
    rows.sort(key=lambda row: (int(row["source"]), float(row["time_s"]), int(row["label"])))

    model, device = YOLO(str(args.model)), device_name(args.device)
    features, retained = [], []
    for start in range(0, len(rows), args.batch_clips):
        batch_rows = rows[start:start + args.batch_clips]
        valid, flat_frames = [], []
        for row in batch_rows:
            clip = read_clip(row["video"], float(row["time_s"]))
            if clip is not None:
                valid.append((row, clip))
                flat_frames.extend(clip)
        if valid:
            results = model.predict(flat_frames, imgsz=960, conf=.05, device=device,
                                    batch=min(20, len(flat_frames)), verbose=False)
            pointer = 0
            for row, clip in valid:
                count = len(clip)
                feature, motion, detected = clip_feature(clip, results[pointer:pointer + count])
                pointer += count
                features.append(feature)
                retained.append({**row, "motion": motion, "detected_frames": detected})
        print(f"processed={min(start + args.batch_clips, len(rows))}/{len(rows)}", flush=True)

    if not features:
        raise ValueError("no readable clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(features),
        labels=np.asarray([int(row["label"]) for row in retained]),
        sources=np.asarray([int(row["source"]) for row in retained]),
    )
    args.output.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1,
        "frozen_tests_used": False,
        "samples": retained,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "samples": len(retained),
        "positives": sum(int(row["label"]) for row in retained),
        "negatives": sum(not int(row["label"]) for row in retained),
        "sources": len({int(row["source"]) for row in retained}),
        "feature_dimensions": int(features[0].shape[0]),
        "device": device,
    }, indent=2))


if __name__ == "__main__":
    main()
