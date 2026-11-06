#!/usr/bin/env python3
"""Score a continuous video interval for referee boundary gestures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

from extract_referee_gesture_embeddings import read_clip


def group_candidates(rows: list[dict], maximum_gap_s: float) -> list[dict]:
    groups: list[list[dict]] = []
    for row in rows:
        if groups and row["time_s"] - groups[-1][-1]["time_s"] <= maximum_gap_s:
            groups[-1].append(row)
        else:
            groups.append([row])
    result = []
    for group in groups:
        peak = max(group, key=lambda row: row["boundary_score"])
        result.append({
            **peak,
            "group_start_s": group[0]["time_s"],
            "group_end_s": group[-1]["time_s"],
            "raw_hits": len(group),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("video")
    parser.add_argument("--roi", type=float, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--stride-s", type=float, default=1)
    parser.add_argument("--candidate-queue", type=Path)
    parser.add_argument("--group-gap-s", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--replay-intervals", type=Path)
    parser.add_argument("--replay-source", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = joblib.load(args.model)
    threshold = float(package["boundary_threshold"])
    excluded = []
    if args.replay_intervals:
        if args.replay_source is None:
            parser.error("--replay-source is required with --replay-intervals")
        replay = json.loads(args.replay_intervals.read_text())["sources"][str(args.replay_source)]
        if replay["video"] != args.video:
            raise ValueError("replay metadata belongs to another video")
        excluded = replay["exclude_intervals"]

    if args.candidate_queue:
        candidate_payload = json.loads(args.candidate_queue.read_text())
        if candidate_payload["video"] != args.video:
            raise ValueError("candidate queue belongs to another video")
        timestamps = [float(row["time_s"]) for row in candidate_payload["items"]
                      if args.start_s <= float(row["time_s"]) <= args.end_s]
    else:
        timestamps = np.arange(args.start_s, args.end_s + 1e-6, args.stride_s)
        timestamps = [float(time) for time in timestamps]
    timestamps = [time for time in timestamps
                  if not any(row["start_s"] <= time <= row["end_s"] for row in excluded)]
    embedding_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    embedding_model.fc = torch.nn.Identity()
    embedding_model.eval()
    rows = []
    for start in range(0, len(timestamps), args.batch_size):
        times = timestamps[start:start + args.batch_size]
        items = [{
            "video": args.video, "time_s": time,
            "window_start_s": max(args.start_s, time - 2),
            "window_end_s": min(args.end_s, time + 2),
            "roi_box_norm": args.roi,
        } for time in times]
        tensors = [read_clip(item) for item in items]
        if any(tensor is None for tensor in tensors):
            raise RuntimeError("failed to decode a continuous referee clip")
        with torch.inference_mode():
            features = embedding_model(torch.stack(tensors)).numpy()
        probabilities = package["model"].predict_proba(features)
        columns = {label: index for index, label in enumerate(package["model"].classes_)}
        for time, probability in zip(times, probabilities):
            boundary_score = sum(float(probability[columns[label]])
                                 for label in package["boundary_labels"])
            rows.append({
                "time_s": time,
                "boundary_score": boundary_score,
                "predicted_label": str(package["model"].classes_[int(np.argmax(probability))]),
            })
        print(f"processed={min(start + args.batch_size, len(timestamps))}/{len(timestamps)}",
              flush=True)
    raw_candidates = [row for row in rows if row["boundary_score"] >= threshold]
    grouped = group_candidates(raw_candidates, args.group_gap_s)
    duration_minutes = (args.end_s - args.start_s) / 60
    result = {
        "schema_version": 1,
        "kind": "volleytag.referee-boundary-scores",
        "video": args.video,
        "start_s": args.start_s,
        "end_s": args.end_s,
        "stride_s": args.stride_s,
        "candidate_gate": (str(args.candidate_queue) if args.candidate_queue else None),
        "group_gap_s": args.group_gap_s,
        "boundary_threshold": threshold,
        "frozen_tests_used": False,
        "scored_windows": len(rows),
        "raw_candidates": raw_candidates,
        "grouped_candidates": grouped,
        "grouped_candidates_per_minute": len(grouped) / max(duration_minutes, 1e-8),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"scored": len(rows), "raw_candidates": len(raw_candidates),
                      "grouped_candidates": len(grouped),
                      "grouped_per_minute": result["grouped_candidates_per_minute"]},
                     indent=2))


if __name__ == "__main__":
    main()
