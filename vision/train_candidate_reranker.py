#!/usr/bin/env python3
"""Train view-specific Dan/not-Dan candidate rerankers from reviewed videos.

Unlike the one-class detector, this model sees the actual decision set: the
confirmed player box is positive and other person boxes from the same instant
are hard negatives.  Frozen DINOv2 embeddings keep the experiment lightweight.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config

from evaluate_fused_detectors import iou
from export_danya_detector_dataset import read_tracks, split_chunks


def crop(frame: np.ndarray, box: list[float], padding: float = .08) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, round(x1 - bw * padding)), min(width, round(x2 + bw * padding))
    y1, y2 = max(0, round(y1 - bh * padding)), min(height, round(y2 + bh * padding))
    value = frame[y1:y2, x1:x2]
    if value.shape[0] < 25 or value.shape[1] < 12:
        return None
    return value


def sparse_rows(rows: list[dict], start: float, end: float, interval: float) -> list[dict]:
    rows = sorted((row for row in rows if start <= row["time_s"] <= end),
                  key=lambda row: row["time_s"])
    result, target = [], start
    while rows and target <= end:
        nearby = [row for row in rows if abs(row["time_s"] - target) <= .13]
        if nearby:
            chosen = max(nearby, key=lambda row: (row["box"][2] - row["box"][0])
                                                * (row["box"][3] - row["box"][1]))
            if not result or chosen["frame"] != result[-1]["frame"]:
                result.append(chosen)
        target += interval
    return result


def candidate_rows(all_rows: list[dict], timestamp: float, tolerance: float = .13) -> list[dict]:
    nearby = [row for row in all_rows if abs(row["time_s"] - timestamp) <= tolerance]
    if not nearby:
        return []
    nearest = min(abs(row["time_s"] - timestamp) for row in nearby)
    return [row for row in nearby if abs(abs(row["time_s"] - timestamp) - nearest) < 1e-4]


def mine(spec_path: Path, source_index: int, output: Path, interval: float,
         negatives_per_frame: int, resume_crops: bool) -> list[dict]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    video, tracks_path = Path(payload["video"]), Path(payload["tracks"])
    grouped = read_tracks(tracks_path)
    chunks = split_chunks(grouped)
    all_rows = sorted((row for values in grouped.values() for row in values),
                      key=lambda row: row["time_s"])
    excluded = [(float(value["start_s"]), float(value["end_s"]))
                for value in payload.get("excluded_intervals", [])]
    samples: list[tuple[dict, list[dict], str, str]] = []
    for segment_index, segment in enumerate(payload.get("segments", [])):
        if segment.get("label") != "positive":
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if any(left <= (start + end) / 2 <= right for left, right in excluded):
            continue
        route = [row for key in (segment.get("routeKeys") or [segment.get("key")])
                 for row in chunks.get(key, [])]
        domain = segment.get("domain") or "unknown"
        for gt in sparse_rows(route, start, end, interval):
            samples.append((gt, candidate_rows(all_rows, gt["time_s"]), domain,
                            f"segment-{segment_index}"))

    # Manual boxes cover the exact cases where the generic detector missed Dan.
    for manual_index, manual in enumerate(payload.get("manualBoxes", [])):
        timestamp = float(manual["time"])
        gt = {"time_s": timestamp, "frame": int(manual["frame"]), "track_id": -1,
              "box": [float(manual[key]) for key in ("x1", "y1", "x2", "y2")]}
        samples.append((gt, candidate_rows(all_rows, timestamp),
                        manual.get("domain") or "unknown", f"manual-{manual_index}"))

    wanted = defaultdict(list)
    for gt, candidates, domain, origin in samples:
        wanted[int(gt["frame"])].append((gt, candidates, domain, origin))
    if not wanted:
        return []
    cap = cv2.VideoCapture(str(video))
    records, frame_index = [], 0
    if resume_crops:
        pattern = re.compile(
            rf"^s{source_index}-f(\d+)-.*-r\d+-y([01])-(near_back|far_front|unknown)\.jpg$")
        completed_frames = []
        for path in (output / "crops").glob(f"s{source_index}-*.jpg"):
            match = pattern.match(path.name)
            if not match:
                continue
            frame = int(match.group(1)); label = int(match.group(2)); domain = match.group(3)
            timestamp = frame / (cap.get(cv2.CAP_PROP_FPS) or 50.0)
            records.append({"file": str(path), "label": label, "domain": domain,
                            "group": f"s{source_index}-b{int(timestamp // 120)}",
                            "source": source_index, "time_s": timestamp})
            completed_frames.append(frame)
        if completed_frames:
            frame_index = max(completed_frames) + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            print(f"source={source_index} resumed_at={frame_index}", flush=True)
    last = max(wanted)
    while frame_index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for gt, candidates, domain, origin in wanted.get(frame_index, []):
            # Add the confirmed box even if it is not among generic detections.
            ranked_negatives = sorted(
                (row for row in candidates if iou(gt["box"], row["box"]) <= .10),
                key=lambda row: (abs(((row["box"][0] + row["box"][2]) -
                                      (gt["box"][0] + gt["box"][2])) / 2),
                                 -((row["box"][2] - row["box"][0]) *
                                   (row["box"][3] - row["box"][1]))))
            examples = [(1, gt)] + [(0, row) for row in ranked_negatives[:negatives_per_frame]]
            for rank, (label, row) in enumerate(examples):
                image = crop(frame, row["box"])
                if image is None:
                    continue
                block = int(float(gt["time_s"]) // 120)
                name = (f"s{source_index}-f{frame_index}-{origin}-r{rank}-"
                        f"y{label}-{domain}.jpg")
                path = output / "crops" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    continue
                records.append({"file": str(path), "label": label, "domain": domain,
                                "group": f"s{source_index}-b{block}",
                                "source": source_index, "time_s": float(gt["time_s"])})
        frame_index += 1
        if frame_index % 2500 == 0:
            print(f"source={source_index} decoded={frame_index}/{last}", flush=True)
    cap.release()
    return records


def embed(paths: list[Path], batch_size: int) -> tuple[np.ndarray, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    values = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([transform(Image.open(path).convert("RGB"))
                                 for path in paths[start:start + batch_size]]).to(device)
            vector = model(batch).float().cpu().numpy()
            vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
            values.append(vector)
            print(f"embeddings={min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.vstack(values), device


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.25, class_weight="balanced", max_iter=3000, random_state=51))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--negatives-per-frame", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--resume-crops", action="store_true",
                        help="Resume extraction from already written source crop filenames")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, spec in enumerate(args.specs):
        records.extend(mine(spec, index, args.output, args.interval,
                            args.negatives_per_frame, args.resume_crops))
    if not records:
        raise SystemExit("No candidate crops were mined")
    with (args.output / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    vectors, device = embed([Path(row["file"]) for row in records], args.batch_size)
    np.save(args.output / "embeddings.npy", vectors)
    labels = np.asarray([int(row["label"]) for row in records])
    report = {"samples": len(records), "positives": int(labels.sum()),
              "negatives": int((1 - labels).sum()), "device": device, "domains": {}}
    for domain in ("near_back", "far_front"):
        indexes = np.asarray([i for i, row in enumerate(records) if row["domain"] == domain])
        if not len(indexes) or len(set(labels[indexes])) < 2:
            continue
        model = classifier().fit(vectors[indexes], labels[indexes])
        scores = model.predict_proba(vectors[indexes])[:, 1]
        joblib.dump(model, args.output / f"candidate-reranker-{domain}.joblib")
        report["domains"][domain] = {"samples": int(len(indexes)),
                                     "positives": int(labels[indexes].sum()),
                                     "training_auc": float(roc_auc_score(labels[indexes], scores))}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
