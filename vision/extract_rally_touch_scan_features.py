#!/usr/bin/env python3
"""Extract base features for an explicitly frozen rally-level diagnostic scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from build_touch_temporal_features import clip_feature, device_name, read_clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-clips", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    payload = json.loads(args.scan.read_text())
    if not payload.get("frozen_tests_used"):
        raise ValueError("this extractor is reserved for explicit frozen evaluation scans")
    rows = payload["samples"]
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
        raise ValueError("no readable scan clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(features),
                        labels=np.zeros(len(features), dtype=np.int8),
                        sources=np.asarray([int(row["source"]) for row in retained]))
    args.output.with_suffix(".json").write_text(json.dumps({
        "schema_version": 1, "kind": "volleytag.rally-touch-scan-features",
        "frozen_tests_used": True, "samples": retained,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"samples": len(features), "dimensions": int(features[0].shape[0]),
                      "device": device}, indent=2))


if __name__ == "__main__":
    main()
