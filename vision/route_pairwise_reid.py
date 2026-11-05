#!/usr/bin/env python3
"""Route identity with trusted personal-detector anchors and pairwise ReID gaps."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config


def feature(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    repeated = np.repeat(anchor[None, :], len(candidates), axis=0)
    cosine = np.sum(repeated * candidates, axis=1, keepdims=True)
    return np.hstack((np.abs(repeated - candidates), repeated * candidates, cosine))


def crop(frame, box, padding=.08):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box; bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, round(x1 - bw * padding)), min(width, round(x2 + bw * padding))
    y1, y2 = max(0, round(y1 - bh * padding)), min(height, round(y2 + bh * padding))
    value = frame[y1:y2, x1:x2]
    return value if value.shape[0] >= 25 and value.shape[1] >= 12 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("near_model", type=Path)
    parser.add_argument("far_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-confidence", type=float, default=.50)
    parser.add_argument("--bridge-seconds", type=float, default=1.60)
    parser.add_argument("--generic-candidates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    args = parser.parse_args()
    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    times = [float(value) for value in payload["times"]]
    candidate_sets = []
    for candidates in payload["candidate_sets"]:
        direct = [row for row in candidates if float(row.get("direct_confidence", 0)) > 0]
        generic = sorted(
            [row for row in candidates if float(row.get("direct_confidence", 0)) <= 0],
            key=lambda row: float(row.get("confidence", 0)), reverse=True)[:args.generic_candidates]
        candidate_sets.append(direct + generic)

    anchor_choice, ambiguous_anchor = {}, {}
    last_anchor = None
    for index, (timestamp, candidates) in enumerate(zip(times, candidate_sets)):
        direct = [i for i, row in enumerate(candidates)
                  if float(row.get("direct_confidence", 0)) >= args.anchor_confidence]
        if direct:
            anchor_choice[index] = max(direct, key=lambda i: candidates[i]["direct_confidence"])
            last_anchor = index
        elif last_anchor is not None and timestamp - times[last_anchor] <= args.bridge_seconds:
            ambiguous_anchor[index] = last_anchor

    needed = set(anchor_choice[index] for index in set(ambiguous_anchor.values()))
    cap = cv2.VideoCapture(str(args.video)); crops, refs = [], []
    for frame_index in sorted(set(ambiguous_anchor) | set(ambiguous_anchor.values())):
        cap.set(cv2.CAP_PROP_POS_MSEC, times[frame_index] * 1000.)
        ok, frame = cap.read()
        if not ok: continue
        indexes = ([anchor_choice[frame_index]] if frame_index in ambiguous_anchor.values()
                   else list(range(len(candidate_sets[frame_index]))))
        for candidate_index in indexes:
            value = crop(frame, candidate_sets[frame_index][candidate_index]["box"])
            if value is not None:
                refs.append((frame_index, candidate_index)); crops.append(value)
    cap.release()

    device = args.device
    if device == "auto": device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedder = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
    ).eval().to(device)
    transform = create_transform(**resolve_model_data_config(embedder), is_training=False)
    embeddings = {}; started = time.monotonic(); completed = 0
    with torch.inference_mode():
        for start in range(0, len(crops), args.batch_size):
            batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB)))
                                 for value in crops[start:start + args.batch_size]]).to(device)
            vectors = embedder(batch).float().cpu().numpy()
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            for offset, vector in enumerate(vectors): embeddings[refs[start + offset]] = vector
            completed += len(vectors)
            if completed >= 200 and completed - len(vectors) < 200:
                elapsed = time.monotonic() - started
                print(json.dumps({"sanity_embeddings": completed, "seconds": elapsed,
                                  "total_embeddings": len(crops),
                                  "estimated_total_seconds": len(crops) * elapsed / completed,
                                  "device": device}), flush=True)
    near, far = joblib.load(args.near_model), joblib.load(args.far_model)
    selected = []
    for frame_index, candidates in enumerate(candidate_sets):
        if frame_index in anchor_choice:
            selected.append(anchor_choice[frame_index]); continue
        if frame_index in ambiguous_anchor:
            anchor_index = ambiguous_anchor[frame_index]
            anchor_vector = embeddings.get((anchor_index, anchor_choice[anchor_index]))
            available = [(i, embeddings[(frame_index, i)]) for i in range(len(candidates))
                         if (frame_index, i) in embeddings]
            if anchor_vector is not None and available:
                values = np.vstack([vector for _, vector in available])
                pair_features = feature(anchor_vector, values)
                scores = []
                for offset, (candidate_index, _) in enumerate(available):
                    box = candidates[candidate_index]["box"]
                    model = near if box[3] - box[1] >= 240 or box[3] >= 700 else far
                    scores.append(float(model.decision_function(pair_features[offset:offset + 1])[0]))
                selected.append(available[int(np.argmax(scores))][0]); continue
        direct = [i for i, row in enumerate(candidates) if float(row.get("direct_confidence", 0)) > 0]
        selected.append(max(direct, key=lambda i: candidates[i]["direct_confidence"])
                        if direct else -1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = ("time_s", "state", "source", "track_id", "confidence",
                  "direct_confidence", "x1", "y1", "x2", "y2")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for timestamp, candidates, choice in zip(times, candidate_sets, selected):
            if choice < 0:
                writer.writerow({"time_s": f"{timestamp:.3f}", "state": "missing"}); continue
            row = candidates[choice]
            writer.writerow({"time_s": f"{timestamp:.3f}", "state": "visible",
                             "source": row.get("source"), "track_id": row.get("track_id"),
                             "confidence": f"{float(row.get('confidence', 0)):.6f}",
                             "direct_confidence": f"{float(row.get('direct_confidence', 0)):.6f}",
                             **{key: f"{value:.2f}" for key, value in
                                zip(("x1", "y1", "x2", "y2"), row["box"])}})
    print(json.dumps({"frames": len(times), "anchors": len(anchor_choice),
                      "ambiguous": len(ambiguous_anchor), "embeddings": len(crops),
                      "elapsed_s": time.monotonic() - started, "output": str(args.output)}))


if __name__ == "__main__":
    main()
