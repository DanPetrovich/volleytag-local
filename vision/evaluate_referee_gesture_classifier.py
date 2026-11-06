#!/usr/bin/env python3
"""Evaluate referee boundary gestures without splitting a video source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSES = ("serve_authorization", "rally_end_team_to_serve", "other")


def classification_summary(labels: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predicted, labels=CLASSES)
    recalls = {label: float(matrix[index, index] / max(matrix[index].sum(), 1))
               for index, label in enumerate(CLASSES)}
    precisions = {label: float(matrix[index, index] / max(matrix[:, index].sum(), 1))
                  for index, label in enumerate(CLASSES)}
    return {
        "samples": len(labels),
        "recall": recalls,
        "precision": precisions,
        "boundary_min_recall": min(recalls[CLASSES[0]], recalls[CLASSES[1]]),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "confusion_matrix": matrix.tolist(),
    }


def candidates() -> dict[str, object]:
    return {
        "logistic-c01": make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=5000)),
        "logistic-c1": make_pipeline(StandardScaler(), LogisticRegression(
            C=1, class_weight="balanced", max_iter=5000)),
        "rbf-svm-c1": make_pipeline(StandardScaler(), SVC(C=1, class_weight="balanced")),
        "rbf-svm-c5": make_pipeline(StandardScaler(), SVC(C=5, class_weight="balanced")),
        "extra-trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=51, n_jobs=-1),
    }


def recall_first_operating_point(labels: np.ndarray, scores: np.ndarray) -> dict:
    truth = labels != "other"
    rows = []
    for threshold in np.linspace(0, 1, 1001):
        predicted = scores >= threshold
        true_positive = int(np.sum(predicted & truth))
        false_positive = int(np.sum(predicted & ~truth))
        false_negative = int(np.sum(~predicted & truth))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        rows.append((recall, precision, float(threshold), true_positive,
                     false_positive, false_negative))
    recall, precision, threshold, true_positive, false_positive, false_negative = max(
        (row for row in rows if row[0] == max(item[0] for item in rows)),
        key=lambda row: (row[1], row[2]),
    )
    return {
        "threshold": threshold, "recall": recall, "precision": precision,
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-item-id", action="append", default=[])
    args = parser.parse_args()
    payloads = [np.load(path) for path in args.features]
    raw_labels = np.concatenate([payload["labels"].astype(str) for payload in payloads])
    item_ids = np.concatenate([payload["item_ids"].astype(str) for payload in payloads])
    excluded = set(args.exclude_item_id)
    retained = ((raw_labels != "uncertain") &
                np.asarray([item_id not in excluded for item_id in item_ids]))
    all_features = np.concatenate([payload["features"] for payload in payloads])
    all_sources = np.concatenate([payload["sources"] for payload in payloads]).astype(int)
    x, sources = all_features[retained], all_sources[retained]
    labels = np.asarray([label if label in CLASSES[:2] else "other"
                         for label in raw_labels[retained]])
    reports = []
    for name, model in candidates().items():
        predicted = np.empty(len(labels), dtype="<U32")
        for source in sorted(set(sources)):
            train, test = sources != source, sources == source
            model.fit(x[train], labels[train])
            predicted[test] = model.predict(x[test])
        summary = classification_summary(labels, predicted)
        summary["per_source"] = {
            str(source): classification_summary(labels[sources == source],
                                                 predicted[sources == source])
            for source in sorted(set(sources))
        }
        reports.append({"model": name, **summary})
    reports.sort(key=lambda row: (row["boundary_min_recall"], row["macro_recall"]),
                 reverse=True)
    best = reports[0]
    boundary_scores = np.zeros(len(labels), dtype=np.float32)
    probability_model = candidates()[best["model"]]
    if not hasattr(probability_model, "predict_proba"):
        raise RuntimeError(f"best model {best['model']} cannot calibrate a boundary threshold")
    for source in sorted(set(sources)):
        train, test = sources != source, sources == source
        fold_model = clone(probability_model)
        fold_model.fit(x[train], labels[train])
        probabilities = fold_model.predict_proba(x[test])
        columns = {label: index for index, label in enumerate(fold_model.classes_)}
        boundary_scores[test] = (probabilities[:, columns[CLASSES[0]]] +
                                 probabilities[:, columns[CLASSES[1]]])
    boundary_candidate = recall_first_operating_point(labels, boundary_scores)
    boundary_candidate["status"] = "training calibration only; lock before frozen evaluation"
    passed = (best["recall"][CLASSES[0]] >= .70 and
              best["recall"][CLASSES[1]] >= .70 and best["macro_recall"] >= .70)
    report = {
        "protocol": "leave-one-whole-source-out; uncertain excluded; no/other merged",
        "frozen_tests_used": False, "samples": len(labels),
        "feature_files": len(payloads),
        "manually_excluded_samples": int(np.sum(~retained & (raw_labels != "uncertain"))),
        "class_counts": {label: int(np.sum(labels == label)) for label in CLASSES},
        "acceptance": "serve recall >= 0.70; end recall >= 0.70; macro recall >= 0.70",
        "candidates": reports, "best": best, "passed": passed,
        "boundary_candidate_operating_point": boundary_candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
