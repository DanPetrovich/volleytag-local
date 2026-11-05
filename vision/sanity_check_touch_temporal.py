#!/usr/bin/env python3
"""Leakage-safe leave-one-whole-source-out temporal touch sanity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.features)
    features, labels, sources = data["features"], data["labels"], data["sources"]
    folds, probabilities = [], np.zeros(len(labels), dtype=np.float32)
    for held in sorted(set(map(int, sources))):
        train, test = sources != held, sources == held
        components = min(48, int(train.sum()) - 2, features.shape[1])
        model = make_pipeline(StandardScaler(), PCA(n_components=components, whiten=True,
                                                     random_state=51),
                              LogisticRegression(C=.25, class_weight="balanced",
                                                 max_iter=3000, random_state=51))
        model.fit(features[train], labels[train])
        score = model.predict_proba(features[test])[:, 1]
        probabilities[test] = score
        prediction = score >= .5
        tn, fp, fn, tp = confusion_matrix(labels[test], prediction, labels=[0, 1]).ravel()
        folds.append({
            "held_source": held, "samples": int(test.sum()),
            "positives": int(labels[test].sum()),
            "roc_auc": float(roc_auc_score(labels[test], score)),
            "positive_recall": float(tp / max(tp + fn, 1)),
            "negative_specificity": float(tn / max(tn + fp, 1)),
        })
    prediction = probabilities >= .5
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    macro_recall = float(np.mean([row["positive_recall"] for row in folds]))
    macro_specificity = float(np.mean([row["negative_specificity"] for row in folds]))
    minimum_auc = min(row["roc_auc"] for row in folds)
    report = {
        "samples": int(len(labels)), "sources": len(folds),
        "overall_roc_auc": float(roc_auc_score(labels, probabilities)),
        "overall_positive_recall": float(tp / max(tp + fn, 1)),
        "overall_negative_specificity": float(tn / max(tn + fp, 1)),
        "macro_positive_recall": macro_recall,
        "macro_negative_specificity": macro_specificity,
        "minimum_source_roc_auc": minimum_auc,
        "passed": bool(macro_recall >= .70 and macro_specificity >= .70 and
                       minimum_auc >= .55),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
