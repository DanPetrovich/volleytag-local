#!/usr/bin/env python3
"""Score tracked player chunks with the frozen DINOv2 personal classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

from build_feedback_profile import split_chunks
from identify_player import crop_observation, read_tracks


def choose(items, limit):
    selected = []
    for item in sorted(items, key=lambda value: value.quality, reverse=True):
        if all(abs(item.time_s - other.time_s) >= .25 for other in selected):
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def embed(crops, batch_size):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            images = [transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                      for crop in crops[start:start + batch_size]]
            values = model(torch.stack(images).to(device)).float().cpu().numpy()
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            vectors.append(values)
    return np.vstack(vectors)


def sheet(entries, output):
    tile_w, tile_h, cols = 180, 270, 8
    rows = min(10, (len(entries) + cols - 1) // cols)
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 242, dtype=np.uint8)
    for index, (key, probability, crop) in enumerate(entries[:rows * cols]):
        row, col = divmod(index, cols)
        scale = min((tile_w - 12) / crop.shape[1], (tile_h - 44) / crop.shape[0])
        resized = cv2.resize(crop, None, fx=scale, fy=scale)
        x = col * tile_w + (tile_w - resized.shape[1]) // 2
        y = row * tile_h + 36
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(canvas, f"{key}  {probability:.2f}", (col * tile_w + 5, row * tile_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("classifier", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision/runs/personal-scores"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roi = tuple(int(value) for value in args.roi.split(","))
    chunks = split_chunks(read_tracks(args.tracks))
    cap = cv2.VideoCapture(str(args.video))
    records, crops = [], []
    for key, items in chunks.items():
        for item in choose(items, args.samples):
            crop = crop_observation(cap, item, roi)
            if crop.size and crop.shape[0] >= 35 and crop.shape[1] >= 15:
                records.append((key, item, crop))
                crops.append(crop)
    cap.release()
    vectors = embed(crops, args.batch_size)
    classifier = joblib.load(args.classifier)
    probabilities = classifier.predict_proba(vectors)[:, 1]
    grouped = {}
    for (key, item, crop), probability in zip(records, probabilities):
        grouped.setdefault(key, []).append((float(probability), item, crop))
    ranked = []
    for key, values in grouped.items():
        # Median resists a single merged or blurry crop; max remains available for diagnostics.
        probability = float(np.median([value[0] for value in values]))
        best = max(values, key=lambda value: value[0])
        items = chunks[key]
        ranked.append((key, items[0].track_id, items[0].time_s, items[-1].time_s,
                       probability, max(value[0] for value in values), best[2]))
    ranked.sort(key=lambda value: value[4], reverse=True)
    with (args.output / "chunk_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "track_id", "start_s", "end_s", "probability", "max_probability", "margin"))
        for key, track_id, start, end, probability, maximum, _ in ranked:
            writer.writerow((key, track_id, f"{start:.3f}", f"{end:.3f}",
                             f"{probability:.5f}", f"{maximum:.5f}", f"{probability - .5:.5f}"))
    sheet([(key, probability, crop) for key, _, _, _, probability, _, crop in ranked],
          args.output / "top-candidates.jpg")
    print(f"chunks={len(ranked)} samples={len(records)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
