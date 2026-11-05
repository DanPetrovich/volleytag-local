#!/usr/bin/env python3
"""Leakage-resistant temporal validation for saved personal-classifier embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def model(class_weight=None, c=1.0):
    return LogisticRegression(max_iter=2000, C=c, class_weight=class_weight,
                              random_state=51)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Directory containing dataset.csv and embeddings.npy")
    parser.add_argument("--block-seconds", type=float, default=45.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "temporal-validation.json"

    with (args.run / "dataset.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.load(args.run / "embeddings.npy")
    y = np.array([int(row["label"]) for row in rows])
    times = np.array([float(row["time_s"]) for row in rows])
    blocks = np.floor(times / args.block_seconds).astype(int)

    configs = {
        "logreg_current": (None, 1.0),
        "logreg_balanced_c0.1": ("balanced", .1),
        "logreg_balanced_c1": ("balanced", 1.0),
        "logreg_balanced_c10": ("balanced", 10.0),
    }
    result = {"samples": len(y), "positives": int(y.sum()), "negatives": int((1-y).sum()),
              "block_seconds": args.block_seconds, "temporal_blocks": int(len(set(blocks))),
              "models": {}}
    for name, (weight, c_value) in configs.items():
        probabilities = np.full(len(y), np.nan)
        used_blocks = []
        for block in sorted(set(blocks)):
            test = blocks == block
            train = ~test
            if len(set(y[test])) < 2 or len(set(y[train])) < 2:
                continue
            classifier = model(weight, c_value).fit(x[train], y[train])
            probabilities[test] = classifier.predict_proba(x[test])[:, 1]
            used_blocks.append(int(block))
        valid = np.isfinite(probabilities)
        predicted = probabilities[valid] >= .5
        result["models"][name] = {
            "evaluated_samples": int(valid.sum()), "held_out_blocks": used_blocks,
            "roc_auc": float(roc_auc_score(y[valid], probabilities[valid])),
            "average_precision": float(average_precision_score(y[valid], probabilities[valid])),
            "precision_at_0.5": float(precision_score(y[valid], predicted, zero_division=0)),
            "recall_at_0.5": float(recall_score(y[valid], predicted, zero_division=0)),
            "f1_at_0.5": float(f1_score(y[valid], predicted, zero_division=0)),
            "confusion_at_0.5": confusion_matrix(y[valid], predicted).tolist(),
        }

    # Label-noise diagnostic: agreement with the five nearest samples outside a 2 s window.
    similarity = x @ x.T
    agreement = []
    for index in range(len(y)):
        allowed = np.abs(times - times[index]) > 2.0
        allowed[index] = False
        candidates = np.where(allowed)[0]
        nearest = candidates[np.argsort(similarity[index, candidates])[-5:]]
        agreement.append(float(np.mean(y[nearest] == y[index])))
    agreement = np.array(agreement)
    result["noise_diagnostic"] = {
        "mean_5nn_label_agreement": float(agreement.mean()),
        "suspicious_below_40pct": int((agreement < .4).sum()),
        "suspicious_positive_below_40pct": int(((agreement < .4) & (y == 1)).sum()),
        "suspicious_negative_below_40pct": int(((agreement < .4) & (y == 0)).sum()),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
