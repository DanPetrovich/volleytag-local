#!/usr/bin/env python3
"""Fit the selected pre-serve classifier on all reviewed training sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("r3d_features", type=Path)
    parser.add_argument("motion_features", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r3d = np.load(args.r3d_features)
    motion = np.load(args.motion_features)
    if not np.array_equal(r3d["item_ids"], motion["item_ids"]):
        raise ValueError("feature item order differs")
    report = json.loads(args.evaluation.read_text())
    best = report["best"]
    if best["model"] != "logistic-c01" or report["feature_set"] != "r3d18+motion":
        raise ValueError("evaluation did not select the expected candidate")
    x = np.concatenate((r3d["features"], motion["features"]), axis=1)
    labels = r3d["labels"].astype(str)
    y = (labels == "pre_serve").astype(np.int8)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight="balanced", max_iter=5000))
    model.fit(x, y)
    artifact = {
        "model": model,
        "threshold": best["recall_first_point"]["threshold"],
        "feature_set": report["feature_set"],
        "feature_dimensions": int(x.shape[1]),
        "samples": int(len(y)),
        "positives": int(y.sum()),
        "training_sources": sorted(set(map(int, r3d["sources"]))),
        "frozen_tests_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    print(json.dumps({key: value for key, value in artifact.items() if key != "model"},
                     indent=2))


if __name__ == "__main__":
    main()
