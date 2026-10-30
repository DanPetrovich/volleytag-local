#!/usr/bin/env python3
"""Build a multi-view ReID gallery from interactive positive/negative feedback."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from boxmot.reid import create_reid_encoder

from identify_player import Observation, crop_observation, encode_crops, read_tracks


def split_chunks(grouped: dict[int, list[Observation]], gap: float = .25):
    chunks: dict[str, list[Observation]] = {}
    for track_id, items in grouped.items():
        index, current = 0, []
        for item in sorted(items, key=lambda value: value.time_s):
            if current and item.time_s - current[-1].time_s > gap:
                chunks[f"{track_id}:{index}"] = current
                index, current = index + 1, []
            current.append(item)
        if current:
            chunks[f"{track_id}:{index}"] = current
    return chunks


def samples_around(items: list[Observation], time_s: float, limit: int = 5):
    nearby = sorted(items, key=lambda item: (abs(item.time_s - time_s), -item.quality))
    selected: list[Observation] = []
    for item in nearby:
        if abs(item.time_s - time_s) > .45:
            continue
        if all(abs(item.time_s - other.time_s) >= .09 for other in selected):
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected or nearby[:1]


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision/runs/feedback-profile"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--samples-per-anchor", type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output / "crops"
    crop_dir.mkdir(exist_ok=True)
    roi = tuple(int(value) for value in args.roi.split(","))
    feedback = json.loads(args.feedback.read_text(encoding="utf-8"))
    grouped = read_tracks(args.tracks)
    chunks = split_chunks(grouped)
    cap = cv2.VideoCapture(str(args.video))

    crops: dict[str, list[np.ndarray]] = {"positive": [], "negative": []}
    metadata = []
    seen = set()
    for kind, field in (("positive", "anchors"), ("negative", "negatives")):
        for entry in feedback.get(field, []):
            key, time_s = entry["key"], float(entry["time"])
            if key not in chunks:
                continue
            for item in samples_around(chunks[key], time_s, args.samples_per_anchor):
                identity = (kind, item.frame, item.track_id)
                if identity in seen:
                    continue
                seen.add(identity)
                crop = crop_observation(cap, item, roi)
                if not crop.size:
                    continue
                crops[kind].append(crop)
                filename = f"{kind}-{len(crops[kind]):03d}-{key.replace(':', '-')}-f{item.frame}.jpg"
                cv2.imwrite(str(crop_dir / filename), crop)
                metadata.append({"kind": kind, "key": key, "time": item.time_s,
                                 "frame": item.frame, "file": f"crops/{filename}"})
    if not crops["positive"]:
        raise SystemExit("Feedback contains no usable positive anchors")

    encoder = create_reid_encoder("osnet-x0-25-msmt17", device="cpu")
    positive = normalize_rows(encode_crops(encoder, crops["positive"]))
    negative = (normalize_rows(encode_crops(encoder, crops["negative"]))
                if crops["negative"] else np.empty((0, positive.shape[1]), dtype=np.float32))
    np.savez_compressed(args.output / "player51-feedback-profile.npz", positive=positive,
                        negative=negative, model="osnet-x0-25-msmt17")

    rows = []
    for key, items in chunks.items():
        chosen = sorted(items, key=lambda item: item.quality, reverse=True)[:5]
        vectors = normalize_rows(encode_crops(encoder, [crop_observation(cap, item, roi) for item in chosen]))
        positive_score = float(np.max(vectors @ positive.T))
        negative_score = float(np.max(vectors @ negative.T)) if len(negative) else 0.0
        # Without negatives, center the positive score around a conservative uniform-similarity baseline.
        margin = positive_score - negative_score if len(negative) else positive_score - .78
        rows.append((key, items[0].track_id, items[0].time_s, items[-1].time_s,
                     positive_score, negative_score, margin))
    rows.sort(key=lambda row: row[-1], reverse=True)
    with (args.output / "chunk_similarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "track_id", "start_s", "end_s", "positive_similarity",
                         "negative_similarity", "margin"))
        for row in rows:
            writer.writerow((row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}",
                             f"{row[4]:.5f}", f"{row[5]:.5f}", f"{row[6]:.5f}"))
    (args.output / "feedback-summary.json").write_text(json.dumps({
        "positive_crops": len(positive), "negative_crops": len(negative),
        "anchors": len(feedback.get("anchors", [])), "negatives": len(feedback.get("negatives", [])),
        "samples": metadata,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cap.release()
    print(f"positive_crops={len(positive)} negative_crops={len(negative)} chunks={len(rows)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
