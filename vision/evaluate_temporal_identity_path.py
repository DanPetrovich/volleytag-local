#!/usr/bin/env python3
"""Evaluate a fixed global identity path on human-reviewed contiguous clips."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from benchmark_pairwise_reid import feature
from evaluate_candidate_head_frozen import padded_crop
from evaluate_frozen_benchmark_annotations import xyxy
from evaluate_fused_detectors import iou, merge_candidates


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def centre(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def speed(previous: dict, current: dict, dt: float) -> float:
    px, py = centre(previous["box"]); cx, cy = centre(current["box"])
    height = max(20.0, ((previous["box"][3] - previous["box"][1]) +
                        (current["box"][3] - current["box"][1])) / 2)
    return math.hypot(cx - px, cy - py) / height / max(dt, .05)


def near_domain(candidate: dict) -> bool:
    box = candidate["box"]
    return box[3] - box[1] >= 240 or box[3] >= 700


def switches(values: list[bool]) -> int:
    return sum(a != b for a, b in zip(values, values[1:]))


def decode(rows: list[dict], near_model, far_model) -> list[int]:
    states = [row["candidates"] + [None] for row in rows]
    scores = []
    for candidate in states[0]:
        scores.append(-1.0 if candidate is None else
                      3.5 * float(candidate.get("direct_confidence", 0)) +
                      .05 * float(candidate.get("confidence", 0)))
    backs = []
    for frame_index in range(1, len(rows)):
        dt = float(rows[frame_index]["time_s"] - rows[frame_index - 1]["time_s"])
        next_scores, pointers = [], []
        for current_index, current in enumerate(states[frame_index]):
            unary = (-1.0 if current is None else
                     3.5 * float(current.get("direct_confidence", 0)) +
                     .05 * float(current.get("confidence", 0)))
            best_score, best_pointer = -float("inf"), 0
            for previous_index, previous in enumerate(states[frame_index - 1]):
                if current is None and previous is None:
                    transition = 0.0
                elif current is None or previous is None:
                    transition = -.75
                else:
                    movement = speed(previous, current, dt)
                    if movement > 8.0:
                        transition = -100.0
                    else:
                        model = near_model if near_domain(current) else far_model
                        pair = feature(previous["embedding"], current["embedding"][None, :])
                        reid = float(np.clip(model.decision_function(pair)[0], -3.0, 3.0))
                        transition = (.8 * reid - .12 * movement * movement +
                                      .6 * iou(previous["box"], current["box"]))
                value = scores[previous_index] + transition + unary
                if value > best_score:
                    best_score, best_pointer = value, previous_index
            next_scores.append(best_score); pointers.append(best_pointer)
        scores = next_scores; backs.append(pointers)
    path = [int(np.argmax(scores))]
    for pointers in reversed(backs):
        path.append(pointers[path[-1]])
    return list(reversed(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("near_pairwise", type=Path)
    parser.add_argument("far_pairwise", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--max-generic", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--conservative-direct-priority", action="store_true",
                        help="Allow generic candidates only when no direct candidate exists")
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    decisions = json.loads(args.results.read_text(encoding="utf-8"))["decisions"]
    rows, crops, references = [], [], []
    for item in queue["items"]:
        decision = decisions[item["id"]]
        frame = cv2.imread(item["image"])
        if frame is None:
            raise ValueError(f"Cannot read {item['image']}")
        height, width = frame.shape[:2]
        gt_label = decision.get("correctedLabel") or item["labels"][0]
        generic = [{"box": xyxy(row["label"], width, height),
                    "confidence": float(row["confidence"]), "source": "generic",
                    "track_id": None}
                   for row in item.get("candidates", [])[:args.max_generic]]
        direct = [{"box": xyxy(row["label"], width, height),
                   "confidence": float(row["confidence"]),
                   "direct_confidence": float(row["confidence"]), "source": "direct"}
                  for row in item.get("personal_predictions", [])]
        candidates = merge_candidates(generic, direct, .5)
        row_index = len(rows)
        rows.append({"id": item["id"], "source": int(item["source"]),
                     "clip": item["action_event_id"], "time_s": float(item["time_s"]),
                     "gt_box": xyxy(gt_label, width, height), "candidates": candidates})
        for candidate_index, candidate in enumerate(candidates):
            crop = padded_crop(frame, candidate["box"])
            if crop is not None:
                crops.append(crop); references.append((row_index, candidate_index))

    device = device_name(args.device)
    if args.embedding_cache.exists():
        vectors = np.load(args.embedding_cache)
        if len(vectors) != len(crops):
            raise ValueError("Embedding cache/crop count mismatch")
    else:
        embedder = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
        ).eval().to(device)
        transform = create_transform(**resolve_model_data_config(embedder), is_training=False)
        batches = []
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(crops), args.batch_size):
                batch = torch.stack([
                    transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    for crop in crops[start:start + args.batch_size]
                ]).to(device)
                value = embedder(batch).float().cpu().numpy()
                value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
                batches.append(value)
                done = min(start + args.batch_size, len(crops))
                if done % 240 < args.batch_size:
                    elapsed = time.monotonic() - started
                    print(f"embeddings={done}/{len(crops)} elapsed={elapsed:.1f}s", flush=True)
        vectors = np.vstack(batches)
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embedding_cache, vectors)
    for (row_index, candidate_index), vector in zip(references, vectors):
        rows[row_index]["candidates"][candidate_index]["embedding"] = vector
    # A candidate without a valid crop cannot participate in appearance transitions.
    for row in rows:
        row["candidates"] = [candidate for candidate in row["candidates"] if "embedding" in candidate]
        if args.conservative_direct_priority:
            direct = [candidate for candidate in row["candidates"]
                      if float(candidate.get("direct_confidence", 0)) > 0]
            if direct:
                row["candidates"] = direct

    near_model, far_model = joblib.load(args.near_pairwise), joblib.load(args.far_pairwise)
    clips = defaultdict(list)
    for row in rows:
        clips[row["clip"]].append(row)
    details = []
    for clip, clip_rows in clips.items():
        clip_rows.sort(key=lambda row: row["time_s"])
        path = decode(clip_rows, near_model, far_model)
        baseline_correct, temporal_correct = [], []
        for row, selected in zip(clip_rows, path):
            direct = [candidate for candidate in row["candidates"]
                      if float(candidate.get("direct_confidence", 0)) > 0]
            baseline = max(direct, key=lambda candidate:
                           float(candidate["direct_confidence"])) if direct else None
            candidate = (None if selected == len(row["candidates"])
                         else row["candidates"][selected])
            base_hit = baseline is not None and iou(row["gt_box"], baseline["box"]) >= .3
            path_hit = candidate is not None and iou(row["gt_box"], candidate["box"]) >= .3
            baseline_correct.append(base_hit); temporal_correct.append(path_hit)
            details.append({"id": row["id"], "source": row["source"], "clip": clip,
                            "time_s": row["time_s"], "baseline_correct": base_hit,
                            "temporal_correct": path_hit, "temporal_missing": candidate is None,
                            "baseline_box": baseline["box"] if baseline is not None else None,
                            "temporal_box": candidate["box"] if candidate is not None else None})
        for row in details[-len(clip_rows):]:
            row["baseline_clip_switches"] = switches(baseline_correct)
            row["temporal_clip_switches"] = switches(temporal_correct)

    by_source = {}
    for source in sorted({row["source"] for row in details}):
        group = [row for row in details if row["source"] == source]
        by_source[str(source)] = {
            "frames": len(group),
            "baseline_top1": sum(row["baseline_correct"] for row in group) / len(group),
            "temporal_top1": sum(row["temporal_correct"] for row in group) / len(group),
        }
        by_source[str(source)]["gain"] = (by_source[str(source)]["temporal_top1"] -
                                           by_source[str(source)]["baseline_top1"])
    baseline = sum(row["baseline_correct"] for row in details) / len(details)
    temporal = sum(row["temporal_correct"] for row in details) / len(details)
    # Count each clip once.
    clip_first = {}
    for row in details:
        clip_first.setdefault(row["clip"], row)
    baseline_switches = sum(row["baseline_clip_switches"] for row in clip_first.values())
    temporal_switches = sum(row["temporal_clip_switches"] for row in clip_first.values())
    switch_reduction = ((baseline_switches - temporal_switches) / baseline_switches
                        if baseline_switches else 0.0)
    maximum_regression = max(-row["gain"] for row in by_source.values())
    report = {
        "protocol": "fixed global path on human-reviewed contiguous training clips",
        "frames": len(details), "clips": len(clips), "device": device,
        "baseline_top1": baseline, "temporal_top1": temporal,
        "top1_gain": temporal - baseline,
        "baseline_identity_switches": baseline_switches,
        "temporal_identity_switches": temporal_switches,
        "identity_switch_reduction": switch_reduction,
        "maximum_source_regression": maximum_regression,
        "by_source": by_source,
        "passed": (temporal - baseline >= .05 and switch_reduction >= .5 and
                   maximum_regression <= .02),
        "frozen_tests_used": False,
        "details": details,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"details", "by_source"}}, indent=2))
    print(json.dumps(report["by_source"], indent=2))


if __name__ == "__main__":
    main()
