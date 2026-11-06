#!/usr/bin/env python3
"""Evaluate referee boundary gestures without splitting a video source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSES = ("serve_authorization", "rally_end_team_to_serve", "other")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = np.load(args.features)
    raw_labels = payload["labels"].astype(str)
    retained = raw_labels != "uncertain"
    x, sources = payload["features"][retained], payload["sources"][retained].astype(int)
    labels = np.asarray([label if label in CLASSES[:2] else "other"
                         for label in raw_labels[retained]])
    reports = []
    for name, model in candidates().items():
        predicted = np.empty(len(labels), dtype="<U32")
        for source in sorted(set(sources)):
            train, test = sources != source, sources == source
            model.fit(x[train], labels[train])
            predicted[test] = model.predict(x[test])
        matrix = confusion_matrix(labels, predicted, labels=CLASSES)
        recalls = {label: float(matrix[index, index] / max(matrix[index].sum(), 1))
                   for index, label in enumerate(CLASSES)}
        precisions = {label: float(matrix[index, index] / max(matrix[:, index].sum(), 1))
                      for index, label in enumerate(CLASSES)}
        boundary_min = min(recalls[CLASSES[0]], recalls[CLASSES[1]])
        macro_recall = float(np.mean(list(recalls.values())))
        reports.append({"model": name, "recall": recalls, "precision": precisions,
                        "boundary_min_recall": boundary_min,
                        "macro_recall": macro_recall, "confusion_matrix": matrix.tolist()})
    reports.sort(key=lambda row: (row["boundary_min_recall"], row["macro_recall"]),
                 reverse=True)
    best = reports[0]
    passed = (best["recall"][CLASSES[0]] >= .70 and
              best["recall"][CLASSES[1]] >= .70 and best["macro_recall"] >= .70)
    report = {
        "protocol": "leave-one-whole-source-out; uncertain excluded; no/other merged",
        "frozen_tests_used": False, "samples": len(labels),
        "class_counts": {label: int(np.sum(labels == label)) for label in CLASSES},
        "acceptance": "serve recall >= 0.70; end recall >= 0.70; macro recall >= 0.70",
        "candidates": reports, "best": best, "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
