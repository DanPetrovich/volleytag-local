#!/usr/bin/env python3
"""Evaluate the fixed Danya/ball proximity rule on cached training clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BASE_DIMENSIONS = 3535
POSE_DERIVED_DIMENSIONS = 493


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.75)
    args = parser.parse_args()
    data = np.load(args.features)
    ball = data["features"][:, BASE_DIMENSIONS + POSE_DERIVED_DIMENSIONS:
                              BASE_DIMENSIONS + POSE_DERIVED_DIMENSIONS + 25].reshape(-1, 5, 5)
    labels, sources = data["labels"].astype(int), data["sources"].astype(int)
    predicted = ((ball[:, :, 0] > 0) & (ball[:, :, 4] <= args.distance)).any(axis=1).astype(int)

    def metrics(indexes):
        actual, guess = labels[indexes], predicted[indexes]
        positives, negatives = actual == 1, actual == 0
        return {
            "samples": int(len(actual)),
            "positive_recall": float((guess[positives] == 1).mean()),
            "negative_specificity": float((guess[negatives] == 0).mean()),
            "predicted_positive_rate": float(guess.mean()),
        }

    overall = metrics(np.arange(len(labels)))
    per_source = {str(source): metrics(np.where(sources == source)[0])
                  for source in sorted(set(sources))}
    report = {
        "rule": {"normalized_distance_max": args.distance, "frames": 5},
        "overall": overall, "per_source": per_source,
        "passed": overall["positive_recall"] >= 0.70 and overall["negative_specificity"] >= 0.70,
        "frozen_tests_used": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
