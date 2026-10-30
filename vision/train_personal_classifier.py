#!/usr/bin/env python3
"""Train a lightweight Dan/not-Dan classifier on frozen DINOv2 embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config

from build_feedback_profile import split_chunks
from identify_player import Observation, read_tracks


def sample_track(items: list[Observation], interval: float, limit: int) -> list[Observation]:
    selected = []
    for item in sorted(items, key=lambda value: (-value.quality, value.time_s)):
        if all(abs(item.time_s - other.time_s) >= interval for other in selected):
            selected.append(item)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda value: value.time_s)


def crop_at(cap, item: Observation, offset_seconds: float, roi) -> np.ndarray | None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(offset_seconds * fps) + item.frame)
    ok, frame = cap.read()
    if not ok:
        return None
    rx1, ry1, rx2, ry2 = roi
    court = frame[ry1:ry2, rx1:rx2]
    x1, y1, x2, y2 = item.box
    pad_x, pad_y = max(3, round((x2 - x1) * .08)), max(3, round((y2 - y1) * .04))
    crop = court[max(0, y1 - pad_y):min(court.shape[0], y2 + pad_y),
                 max(0, x1 - pad_x):min(court.shape[1], x2 + pad_x)]
    return crop.copy() if crop.size else None


def collect_id_clip(spec, full_video: Path, output: Path):
    grouped = read_tracks(Path(spec["tracks"]))
    positive_ids = set(spec["positive_ids"])
    positive_segments = spec.get("positive_segments")
    cap = cv2.VideoCapture(str(full_video))
    roi = tuple(spec.get("roi", [40, 70, 1900, 1020]))
    rows = []
    for track_id, items in grouped.items():
        if positive_segments:
            own_segments = [segment for segment in positive_segments
                            if segment["track_id"] == track_id]
            positives = [item for item in items if any(
                segment["start_s"] <= item.time_s <= segment["end_s"]
                for segment in own_segments)]
            # Never turn the unverified part of an ID-switching positive track into negatives.
            negatives = [] if own_segments else items
        else:
            positives = items if track_id in positive_ids else []
            negatives = [] if track_id in positive_ids else items
        labeled = ((1, positives, .35, 50), (0, negatives, .8, 8))
        for label, candidates, interval, limit in labeled:
            for item in sample_track(candidates, interval, limit):
                crop = crop_at(cap, item, spec["offset_seconds"], roi)
                if crop is None or crop.shape[0] < 35 or crop.shape[1] < 15:
                    continue
                name = f'{spec["name"]}-y{label}-id{track_id}-f{item.frame}.jpg'
                cv2.imwrite(str(output / name), crop)
                rows.append((name, label, spec["name"], item.time_s, track_id))
    cap.release()
    return rows


def collect_feedback_clip(spec, full_video: Path, output: Path):
    grouped = read_tracks(Path(spec["tracks"]))
    chunks = split_chunks(grouped)
    feedback = json.loads(Path(spec["feedback"]).read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(full_video))
    roi = tuple(spec.get("roi", [40, 70, 1900, 1020]))
    rows, seen = [], set()
    for label, field in ((1, "anchors"), (0, "negatives")):
        for entry in feedback.get(field, []):
            key = entry["key"]
            if key not in chunks:
                continue
            around = sorted(chunks[key], key=lambda item: abs(item.time_s - float(entry["time"])))
            selected = []
            for item in around:
                if abs(item.time_s - float(entry["time"])) > .6:
                    continue
                if all(abs(item.time_s - other.time_s) >= .12 for other in selected):
                    selected.append(item)
                if len(selected) >= 7:
                    break
            for item in selected:
                identity = (label, item.frame, item.track_id)
                if identity in seen:
                    continue
                seen.add(identity)
                crop = crop_at(cap, item, spec["offset_seconds"], roi)
                if crop is None or crop.shape[0] < 35 or crop.shape[1] < 15:
                    continue
                name = f'{spec["name"]}-y{label}-{key.replace(":", "-")}-f{item.frame}.jpg'
                cv2.imwrite(str(output / name), crop)
                rows.append((name, label, spec["name"], item.time_s, item.track_id))
    cap.release()
    return rows


def embed(crop_paths, batch_size: int):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crop_paths), batch_size):
            images = [transform(Image.open(path).convert("RGB"))
                      for path in crop_paths[start:start + batch_size]]
            batch = torch.stack(images).to(device)
            values = model(batch).float().cpu().numpy()
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            vectors.append(values)
    return np.vstack(vectors), device


def new_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.35, class_weight="balanced", max_iter=3000, random_state=51,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision/runs/personal-classifier"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output / "crops"
    crop_dir.mkdir(exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    full_video = Path(manifest["video"])
    rows = []
    for spec in manifest["training_clips"]:
        rows.extend((collect_feedback_clip if spec["type"] == "feedback" else collect_id_clip)(
            spec, full_video, crop_dir))
    if not rows or len({row[1] for row in rows}) != 2:
        raise SystemExit("Need both positive and negative crops")
    with (args.output / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "label", "group", "time_s", "track_id"))
        writer.writerows(rows)
    paths = [crop_dir / row[0] for row in rows]
    y = np.array([row[1] for row in rows])
    groups = np.array([row[2] for row in rows])
    x, device = embed(paths, args.batch_size)
    np.save(args.output / "embeddings.npy", x)

    predictions = np.zeros(len(y), dtype=float)
    folds = []
    for train, valid in LeaveOneGroupOut().split(x, y, groups):
        if len(set(y[train])) < 2 or len(set(y[valid])) < 2:
            continue
        classifier = new_classifier().fit(x[train], y[train])
        predictions[valid] = classifier.predict_proba(x[valid])[:, 1]
        folds.append({"held_out": str(groups[valid][0]), "samples": len(valid),
                      "auc": float(roc_auc_score(y[valid], predictions[valid]))})
    evaluated = predictions > 0
    threshold = .5
    report = {
        "model": "DINOv2 ViT-S/14 frozen + logistic regression",
        "device": device, "samples": len(y), "positives": int(y.sum()),
        "negatives": int((1 - y).sum()), "groups": sorted(set(groups)), "folds": folds,
    }
    if evaluated.any() and len(set(y[evaluated])) == 2:
        report["cross_clip_auc"] = float(roc_auc_score(y[evaluated], predictions[evaluated]))
        report["confusion_at_0.5"] = confusion_matrix(
            y[evaluated], predictions[evaluated] >= threshold).tolist()
        report["classification"] = classification_report(
            y[evaluated], predictions[evaluated] >= threshold, output_dict=True, zero_division=0)
    classifier = new_classifier().fit(x, y)
    joblib.dump(classifier, args.output / "player51-dinov2-logreg.joblib")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
