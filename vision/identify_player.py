#!/usr/bin/env python3
"""Join fragmented tracks by similarity to confirmed player crops."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot.reid import create_reid_encoder
from boxmot.structures import Boxes, Detections, Frame


@dataclass(frozen=True)
class Observation:
    time_s: float
    frame: int
    track_id: int
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def quality(self) -> float:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1) * self.confidence


def read_tracks(path: Path) -> dict[int, list[Observation]]:
    grouped: dict[int, list[Observation]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = Observation(
                time_s=float(row["time_s"]), frame=int(row["frame"]),
                track_id=int(row["track_id"]), confidence=float(row["confidence"]),
                box=tuple(int(row[key]) for key in ("x1", "y1", "x2", "y2")),
            )
            grouped[item.track_id].append(item)
    return grouped


def choose_samples(items: list[Observation], limit: int) -> list[Observation]:
    selected: list[Observation] = []
    for item in sorted(items, key=lambda value: value.quality, reverse=True):
        if all(abs(item.time_s - other.time_s) >= 0.25 for other in selected):
            selected.append(item)
        if len(selected) == limit:
            break
    return selected


def crop_observation(cap: cv2.VideoCapture, item: Observation, roi: tuple[int, int, int, int]) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, item.frame)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {item.frame}")
    rx1, ry1, rx2, ry2 = roi
    court = frame[ry1:ry2, rx1:rx2]
    x1, y1, x2, y2 = item.box
    pad_x = max(2, int((x2 - x1) * 0.06))
    pad_y = max(2, int((y2 - y1) * 0.03))
    x1, x2 = max(0, x1 - pad_x), min(court.shape[1], x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(court.shape[0], y2 + pad_y)
    return court[y1:y2, x1:x2].copy()


def encode_crops(encoder, crops: list[np.ndarray]) -> np.ndarray:
    frames: list[Frame] = []
    detections: list[Detections] = []
    for index, crop in enumerate(crops):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        sample_id = f"crop-{index}"
        frames.append(Frame(image=image, sample_id=sample_id))
        h, w = crop.shape[:2]
        detections.append(Detections(
            geometry=Boxes(torch.tensor([[0, 0, w, h]], dtype=torch.float32)),
            scores=torch.ones(1, dtype=torch.float32),
            class_ids=torch.zeros(1, dtype=torch.int64), sample_id=sample_id,
        ))
    encoded = encoder.encode(frames, detections)
    return np.vstack([value.numpy()[0] for value in encoded])


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    value = vectors.mean(axis=0)
    return value / max(np.linalg.norm(value), 1e-12)


def make_sheet(entries, output: Path) -> None:
    tile_w, tile_h, cols = 220, 310, 6
    rows = (len(entries) + cols - 1) // cols
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 238, dtype=np.uint8)
    for index, (track_id, score, crop) in enumerate(entries):
        row, col = divmod(index, cols)
        scale = min((tile_w - 16) / crop.shape[1], (tile_h - 54) / crop.shape[0])
        resized = cv2.resize(crop, None, fx=scale, fy=scale)
        x = col * tile_w + (tile_w - resized.shape[1]) // 2
        y = row * tile_h + 40
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.putText(sheet, f"ID {track_id}  {score:.3f}", (col * tile_w + 7, row * tile_h + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(output), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--seed-ids", required=True, help="Confirmed temporary IDs, comma-separated")
    parser.add_argument("--negative-ids", default="", help="Confirmed other players, comma-separated")
    parser.add_argument("--output", type=Path, default=Path("vision/runs/identified"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.04,
                        help="Minimum positive-minus-negative similarity margin")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seed_ids = {int(value) for value in args.seed_ids.split(",")}
    negative_ids = {int(value) for value in args.negative_ids.split(",") if value}
    roi = tuple(int(value) for value in args.roi.split(","))
    grouped = read_tracks(args.tracks)
    missing = (seed_ids | negative_ids).difference(grouped)
    if missing:
        raise SystemExit(f"Seed IDs not present: {sorted(missing)}")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_crops: dict[int, list[np.ndarray]] = {}
    representatives: dict[int, np.ndarray] = {}
    for track_id, items in grouped.items():
        chosen = choose_samples(items, args.samples)
        crops = [crop_observation(cap, item, roi) for item in chosen]
        sample_crops[track_id] = crops
        representatives[track_id] = crops[0]

    encoder = create_reid_encoder("osnet-x0-25-msmt17", device="cpu")
    embeddings = {track_id: normalized_mean(encode_crops(encoder, crops))
                  for track_id, crops in sample_crops.items()}
    profile = normalized_mean(np.vstack([embeddings[track_id] for track_id in seed_ids]))
    positive_scores = {track_id: float(vector @ profile) for track_id, vector in embeddings.items()}
    if negative_ids:
        negative_vectors = np.vstack([embeddings[track_id] for track_id in negative_ids])
        negative_scores = {
            track_id: float(np.max(negative_vectors @ vector))
            for track_id, vector in embeddings.items()
        }
        scores = {
            track_id: positive_scores[track_id] - negative_scores[track_id]
            for track_id in embeddings
        }
    else:
        negative_scores = {track_id: 0.0 for track_id in embeddings}
        scores = positive_scores

    ranked = sorted(scores, key=scores.get, reverse=True)
    def is_match(track_id: int) -> bool:
        return track_id in seed_ids or scores[track_id] >= args.threshold

    with (args.output / "similarity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("track_id", "positive_similarity", "negative_similarity", "margin", "matched"))
        for track_id in ranked:
            writer.writerow((track_id, f"{positive_scores[track_id]:.5f}",
                             f"{negative_scores[track_id]:.5f}", f"{scores[track_id]:.5f}",
                             is_match(track_id)))
    make_sheet([(tid, scores[tid], representatives[tid]) for tid in ranked[:60]], args.output / "candidates.jpg")
    make_sheet([(tid, 1.0, representatives[tid]) for tid in sorted(seed_ids)],
               args.output / "confirmed-positive.jpg")
    if negative_ids:
        make_sheet([(tid, 0.0, representatives[tid]) for tid in sorted(negative_ids)],
                   args.output / "confirmed-negative.jpg")

    by_frame: dict[int, list[Observation]] = defaultdict(list)
    for items in grouped.values():
        for item in items:
            by_frame[item.frame].append(item)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    width, height = roi[2] - roi[0], roi[3] - roi[1]
    writer = cv2.VideoWriter(str(args.output / "player-51.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (width, height))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in by_frame:
            court = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
            for item in by_frame[frame_index]:
                score = scores[item.track_id]
                if not is_match(item.track_id):
                    continue
                x1, y1, x2, y2 = item.box
                cv2.rectangle(court, (x1, y1), (x2, y2), (20, 210, 255), 4)
                cv2.putText(court, f"Dan #51  {score:.2f}", (x1, max(26, y1 - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.76, (20, 210, 255), 2, cv2.LINE_AA)
            writer.write(court)
        frame_index += 1
    writer.release()
    cap.release()

    print(f"tracks={len(grouped)}")
    print(f"matched={sum(is_match(track_id) for track_id in scores)}")
    print("top=" + ", ".join(f"{tid}:{scores[tid]:.3f}" for tid in ranked[:12]))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
