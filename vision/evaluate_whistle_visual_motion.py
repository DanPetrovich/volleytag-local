#!/usr/bin/env python3
"""Test whether court motion can reject whistles from neighboring courts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix


OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3)


def features(video: str, timestamp: float) -> np.ndarray | None:
    capture, frames = cv2.VideoCapture(video), []
    for offset in OFFSETS:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp + offset) * 1000)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    capture.release()
    differences = np.asarray([np.abs(frames[index + 1] - frames[index])
                              for index in range(len(frames) - 1)])
    result = []
    for difference in differences:
        result.extend([difference.mean(), difference.std(),
                       np.quantile(difference, .75), np.quantile(difference, .95)])
    for difference in differences:
        result.extend(part.mean() for part in np.array_split(difference, 3, axis=0))
    result.extend([differences[:3].mean(), differences[3:].mean(),
                   differences[-2:].mean() - differences[:2].mean()])
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"]
    if args.cache.exists():
        cached = np.load(args.cache)
        matrix, labels, sources = cached["features"], cached["labels"], cached["sources"]
    else:
        matrix, labels, sources = [], [], []
        for index, item in enumerate(queue, 1):
            decision = answers[item["id"]]["decision"]
            if decision == "uncertain":
                continue
            vector = features(item["video"], float(item["time_s"]))
            if vector is not None:
                matrix.append(vector)
                labels.append(int(decision == "serve_whistle"))
                sources.append(int(item["source"]))
            print(f"processed={index}/{len(queue)}", flush=True)
        matrix, labels, sources = np.stack(matrix), np.asarray(labels), np.asarray(sources)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, features=matrix, labels=labels, sources=sources)

    probabilities = np.zeros(len(labels), dtype=np.float32)
    for held in sorted(set(map(int, sources))):
        train, test = sources != held, sources == held
        model = ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=51, n_jobs=-1,
        )
        model.fit(matrix[train], labels[train])
        probabilities[test] = model.predict_proba(matrix[test])[:, 1]
    rows = []
    for threshold in (.15, .20, .25, .30, .35, .40, .50):
        tn, fp, fn, tp = confusion_matrix(labels, probabilities >= threshold, labels=[0, 1]).ravel()
        recall, precision = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
        rows.append({
            "threshold": threshold, "recall": recall, "precision": precision,
            "kept": int(tp + fp), "true_positive": int(tp), "false_positive": int(fp),
            "passed": bool(recall >= .90 and precision >= .40),
        })
    report = {
        "samples": int(len(labels)), "positives": int(labels.sum()),
        "baseline_all_candidate_recall": 1.0,
        "baseline_all_candidate_precision": float(labels.mean()),
        "acceptance_criterion": "recall >= 0.90 and precision >= 0.40",
        "thresholds": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
