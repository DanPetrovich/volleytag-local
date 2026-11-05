#!/usr/bin/env python3
"""Apply the fixed global identity path to a dense candidate cache.

The decoder and its weights deliberately match evaluate_temporal_identity_path.py.
This wrapper exists so that a path can be decoded on every sampled video frame
and then evaluated only on sparse, independently reviewed frozen keyframes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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


def decode(rows: list[dict], near_model, far_model) -> list[int]:
    """Viterbi decode with parameters frozen by the training-only sanity check."""
    states = [row["candidates"] + [None] for row in rows]
    scores = np.asarray([
        -1.0 if candidate is None else
        3.5 * float(candidate.get("direct_confidence", 0)) +
        .05 * float(candidate.get("confidence", 0))
        for candidate in states[0]
    ], dtype=np.float64)
    backs: list[list[int]] = []
    for frame_index in range(1, len(rows)):
        dt = float(rows[frame_index]["time_s"] - rows[frame_index - 1]["time_s"])
        previous_states = states[frame_index - 1]
        next_scores, pointers = [], []
        for current in states[frame_index]:
            unary = (-1.0 if current is None else
                     3.5 * float(current.get("direct_confidence", 0)) +
                     .05 * float(current.get("confidence", 0)))
            transitions = np.full(len(previous_states), -100.0, dtype=np.float64)
            if current is None:
                transitions[:] = -.75
                transitions[-1] = 0.0
            else:
                valid_indexes, valid_previous = [], []
                for index, previous in enumerate(previous_states[:-1]):
                    movement = speed(previous, current, dt)
                    if movement <= 8.0:
                        valid_indexes.append(index)
                        valid_previous.append(previous)
                        transitions[index] = (-.12 * movement * movement +
                                              .6 * iou(previous["box"], current["box"]))
                if valid_previous:
                    embeddings = np.vstack([row["embedding"] for row in valid_previous])
                    pair = feature(current["embedding"], embeddings)
                    model = near_model if near_domain(current) else far_model
                    values = np.clip(model.decision_function(pair), -3.0, 3.0)
                    for index, value in zip(valid_indexes, values):
                        transitions[index] += .8 * float(value)
                transitions[-1] = -.75
            values = scores + transitions + unary
            pointer = int(np.argmax(values))
            next_scores.append(float(values[pointer])); pointers.append(pointer)
        scores = np.asarray(next_scores, dtype=np.float64)
        backs.append(pointers)
    path = [int(np.argmax(scores))]
    for pointers in reversed(backs):
        path.append(pointers[path[-1]])
    return list(reversed(path))


def split_segments(rows: list[dict], max_gap: float = 2.0) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        if current and row["time_s"] - current[-1]["time_s"] > max_gap:
            segments.append(current); current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("near_pairwise", type=Path)
    parser.add_argument("far_pairwise", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--max-generic", type=int, default=8)
    parser.add_argument("--max-gap", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--conservative-direct-priority", action="store_true",
                        help="Use generic candidates only when no direct candidate exists")
    args = parser.parse_args()

    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    end = None if args.limit_frames is None else args.start_index + args.limit_frames
    times = payload["times"][args.start_index:end]
    raw_sets = payload["candidate_sets"][args.start_index:end]
    rows, references = [], []
    cap = cv2.VideoCapture(str(args.video))
    started = time.monotonic()
    use_embedding_cache = args.embedding_cache.exists()
    vector_batches: list[np.ndarray] = []
    crop_batch: list[np.ndarray] = []
    embedded = 0
    if not use_embedding_cache:
        device = device_name(args.device)
        embedder = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
        ).eval().to(device)
        transform = create_transform(**resolve_model_data_config(embedder), is_training=False)

        def flush_crop_batch() -> None:
            nonlocal crop_batch, embedded
            if not crop_batch:
                return
            with torch.inference_mode():
                batch = torch.stack([
                    transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    for crop in crop_batch
                ]).to(device)
                value = embedder(batch).float().cpu().numpy()
            value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
            vector_batches.append(value)
            embedded += len(value)
            crop_batch = []
            if embedded % 256 < args.batch_size:
                print(json.dumps({"stage": "embedding", "embeddings": embedded,
                                  "frames": len(rows),
                                  "elapsed_s": round(time.monotonic() - started, 1),
                                  "device": device}), flush=True)

    for timestamp, raw in zip(times, raw_sets):
        direct = [dict(row) for row in raw if float(row.get("direct_confidence", 0)) > 0]
        generic = sorted(
            [dict(row) for row in raw if float(row.get("direct_confidence", 0)) <= 0],
            key=lambda row: float(row.get("confidence", 0)), reverse=True
        )[:args.max_generic]
        candidates = merge_candidates(generic, direct, .5)
        # V12's rule is known before inference: when the personal detector has
        # any proposal, generic people are not legal states. Apply it before
        # crop extraction so we do not compute thousands of unused embeddings.
        if args.conservative_direct_priority:
            direct_candidates = [candidate for candidate in candidates
                                 if float(candidate.get("direct_confidence", 0)) > 0]
            if direct_candidates:
                candidates = direct_candidates
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read {args.video} at {timestamp:.3f}s")
        row_index = len(rows)
        rows.append({"time_s": float(timestamp), "candidates": candidates})
        for candidate_index, candidate in enumerate(candidates):
            value = padded_crop(frame, candidate["box"])
            if value is not None:
                references.append((row_index, candidate_index))
                if not use_embedding_cache:
                    crop_batch.append(value)
                    if len(crop_batch) >= args.batch_size:
                        flush_crop_batch()
        if len(rows) % 256 == 0:
            print(json.dumps({"stage": "frame_scan", "frames": len(rows),
                              "total_frames": len(times), "embeddings": embedded,
                              "elapsed_s": round(time.monotonic() - started, 1)}),
                  flush=True)
    cap.release()

    if use_embedding_cache:
        vectors = np.load(args.embedding_cache)
        if len(vectors) != len(references):
            raise ValueError(
                f"Embedding cache/crop mismatch: {len(vectors)} != {len(references)}")
    else:
        flush_crop_batch()
        vectors = (np.vstack(vector_batches) if vector_batches
                   else np.empty((0, 384), dtype=np.float32))
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embedding_cache, vectors)
    for (row_index, candidate_index), vector in zip(references, vectors):
        rows[row_index]["candidates"][candidate_index]["embedding"] = vector
    for row in rows:
        row["candidates"] = [candidate for candidate in row["candidates"]
                             if "embedding" in candidate]
        if args.conservative_direct_priority:
            direct = [candidate for candidate in row["candidates"]
                      if float(candidate.get("direct_confidence", 0)) > 0]
            if direct:
                row["candidates"] = direct

    near_model = joblib.load(args.near_pairwise)
    far_model = joblib.load(args.far_pairwise)
    selected: list[tuple[dict, int]] = []
    for segment in split_segments(rows, args.max_gap):
        path = decode(segment, near_model, far_model)
        selected.extend(zip(segment, path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = ("time_s", "state", "source", "track_id", "confidence",
                  "direct_confidence", "x1", "y1", "x2", "y2")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row, choice in selected:
            candidates = row["candidates"]
            if choice == len(candidates):
                writer.writerow({"time_s": f"{row['time_s']:.3f}", "state": "missing"})
                continue
            candidate = candidates[choice]
            writer.writerow({
                "time_s": f"{row['time_s']:.3f}", "state": "visible",
                "source": candidate.get("source"), "track_id": candidate.get("track_id"),
                "confidence": f"{float(candidate.get('confidence', 0)):.6f}",
                "direct_confidence": f"{float(candidate.get('direct_confidence', 0)):.6f}",
                **{key: f"{value:.2f}" for key, value in
                   zip(("x1", "y1", "x2", "y2"), candidate["box"])},
            })
    elapsed = time.monotonic() - started
    print(json.dumps({"frames": len(rows), "crops": len(references), "segments": len(split_segments(rows, args.max_gap)),
                      "elapsed_s": elapsed, "fps": len(rows) / elapsed if elapsed else 0,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
