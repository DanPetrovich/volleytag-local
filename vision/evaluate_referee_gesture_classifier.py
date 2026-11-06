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


def binary_boundary_summary(labels: np.ndarray, predicted: np.ndarray,
                            sources: np.ndarray) -> dict:
    truth = labels != "other"
    output = {}
    for source in sorted(set(sources)):
        selected = sources == source
        tp = int(np.sum(predicted[selected] & truth[selected]))
        fp = int(np.sum(predicted[selected] & ~truth[selected]))
        fn = int(np.sum(~predicted[selected] & truth[selected]))
        tn = int(np.sum(~predicted[selected] & ~truth[selected]))
        output[str(source)] = {
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "specificity": tn / max(tn + fp, 1),
            "samples": int(np.sum(selected)),
        }
    return {
        "recall": float(np.sum(predicted & truth) / max(np.sum(truth), 1)),
        "precision": float(np.sum(predicted & truth) / max(np.sum(predicted), 1)),
        "specificity": float(np.sum(~predicted & ~truth) / max(np.sum(~truth), 1)),
        "minimum_source_recall": min(row["recall"] for row in output.values()),
        "per_source": output,
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
    binary_reports = []
    binary_labels = np.where(labels == "other", "other", "boundary")
    for name, model in candidates().items():
        binary_predicted = np.zeros(len(labels), dtype=bool)
        for source in sorted(set(sources)):
            train, test = sources != source, sources == source
            fold_model = clone(model)
            fold_model.fit(x[train], binary_labels[train])
            binary_predicted[test] = fold_model.predict(x[test]) == "boundary"
        binary_reports.append({"model": name,
                               **binary_boundary_summary(labels, binary_predicted, sources)})
    binary_reports.sort(key=lambda row: (row["minimum_source_recall"], row["recall"],
                                         row["precision"], row["specificity"]), reverse=True)
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
        "binary_boundary_candidates": binary_reports,
        "best_binary_boundary": binary_reports[0],
        "boundary_candidate_operating_point": boundary_candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
