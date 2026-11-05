#!/usr/bin/env python3
"""Compare fixed nonlinear heads on cached touch-temporal features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def summarize(name: str, features: np.ndarray, labels: np.ndarray,
              sources: np.ndarray, factory) -> dict:
    folds = []
    all_scores = np.zeros(len(labels), dtype=np.float32)
    all_predictions = np.zeros(len(labels), dtype=bool)
    for held in sorted(set(map(int, sources))):
        train, test = sources != held, sources == held
        model = factory(int(train.sum()), features.shape[1])
        model.fit(features[train], labels[train])
        if hasattr(model, "decision_function"):
            score = model.decision_function(features[test])
            prediction = score >= 0.0
        else:
            score = model.predict_proba(features[test])[:, 1]
            prediction = score >= .5
        all_scores[test], all_predictions[test] = score, prediction
        tn, fp, fn, tp = confusion_matrix(labels[test], prediction, labels=[0, 1]).ravel()
        folds.append({
            "held_source": held, "samples": int(test.sum()),
            "roc_auc": float(roc_auc_score(labels[test], score)),
            "positive_recall": float(tp / max(tp + fn, 1)),
            "negative_specificity": float(tn / max(tn + fp, 1)),
        })
    tn, fp, fn, tp = confusion_matrix(labels, all_predictions, labels=[0, 1]).ravel()
    macro_recall = float(np.mean([row["positive_recall"] for row in folds]))
    macro_specificity = float(np.mean([row["negative_specificity"] for row in folds]))
    minimum_auc = min(row["roc_auc"] for row in folds)
    return {
        "name": name,
        "overall_roc_auc": float(roc_auc_score(labels, all_scores)),
        "overall_positive_recall": float(tp / max(tp + fn, 1)),
        "overall_negative_specificity": float(tn / max(tn + fp, 1)),
        "macro_positive_recall": macro_recall,
        "macro_negative_specificity": macro_specificity,
        "minimum_source_roc_auc": minimum_auc,
        "passed": bool(macro_recall >= .70 and macro_specificity >= .70 and
                       minimum_auc >= .55),
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.features)
    features, labels, sources = data["features"], data["labels"], data["sources"]

    def rbf(train_count: int, dimension: int):
        components = min(48, train_count - 2, dimension)
        return make_pipeline(StandardScaler(), PCA(n_components=components, whiten=True,
                                                    random_state=51),
                             SVC(C=1.0, gamma="scale", class_weight="balanced",
                                 random_state=51))

    def trees(_train_count: int, _dimension: int):
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                    max_features="sqrt", class_weight="balanced",
                                    n_jobs=-1, random_state=51)

    reports = [
        summarize("pca48_rbf_svm", features, labels, sources, rbf),
        summarize("extra_trees", features, labels, sources, trees),
    ]
    result = {"samples": int(len(labels)), "heads": reports,
              "passed_heads": [row["name"] for row in reports if row["passed"]]}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
