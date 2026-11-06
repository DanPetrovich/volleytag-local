#!/usr/bin/env python3
"""Train on one reviewed source and evaluate exact segments on another source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_dense_rally_segmentation import exact_boundary_matches, intervals_from_predictions


def load_features(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.load(path) for path in paths]
    item_ids = arrays[0]["item_ids"].astype(str)
    labels = arrays[0]["labels"].astype(str)
    if any(not np.array_equal(item_ids, row["item_ids"].astype(str)) for row in arrays[1:]):
        raise ValueError("feature item order differs")
    if any(not np.array_equal(labels, row["labels"].astype(str)) for row in arrays[1:]):
        raise ValueError("feature labels differ")
    return np.concatenate([row["features"] for row in arrays], axis=1), labels, item_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_queue", type=Path)
    parser.add_argument("evaluation_queue", type=Path)
    parser.add_argument("--training-feature", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-feature", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--tolerance-s", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.training_feature) != len(args.evaluation_feature):
        raise ValueError("training/evaluation feature sets differ")
    training_queue = json.loads(args.training_queue.read_text())
    evaluation_queue = json.loads(args.evaluation_queue.read_text())
    train_x, train_labels, _ = load_features(args.training_feature)
    eval_x, eval_labels, eval_ids = load_features(args.evaluation_feature)
    eval_lookup = {row["id"]: row for row in evaluation_queue["items"]}
    eval_rows = [eval_lookup[item_id] for item_id in eval_ids]
    train_y = train_labels == "active_play"
    eval_y = eval_labels == "active_play"
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight="balanced", max_iter=5000, random_state=51))
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(eval_x)[:, 1]
    predicted = probabilities >= args.threshold
    tn, fp, fn, tp = confusion_matrix(eval_y, predicted, labels=[False, True]).ravel()
    times = np.asarray([float(row["time_s"]) for row in eval_rows])
    predicted_intervals = intervals_from_predictions(
        times, predicted, float(evaluation_queue["step_s"]))
    gt = evaluation_queue["rallies"]
    matches = exact_boundary_matches(gt, predicted_intervals, args.tolerance_s)
    errors = [value for row in matches for value in
              (row["start_error_s"], row["end_error_s"])]
    recall = len(matches) / max(len(gt), 1)
    precision = len(matches) / max(len(predicted_intervals), 1)
    report = {
        "schema_version": 1,
        "kind": "volleytag.cross-source-rally-segmentation",
        "frozen_tests_used": False,
        "protocol": "train on source A; fixed threshold; evaluate once on calibration source B",
        "threshold": args.threshold,
        "training_samples": len(train_y),
        "evaluation_samples": len(eval_y),
        "point_metrics": {
            "active_recall": tp / max(tp + fn, 1),
            "dead_specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
        },
        "segment_metrics": {
            "gt_rallies": len(gt), "predicted_rallies": len(predicted_intervals),
            "exact_matches": len(matches), "recall": recall, "precision": precision,
            "boundary_error_p90_s": float(np.quantile(errors, .9)) if errors else None,
            "tolerance_s": args.tolerance_s,
        },
        "matches": matches,
        "predicted_intervals": predicted_intervals,
        "acceptance": "recall >= 0.95, precision >= 0.80, boundary p90 <= 2.0 s",
        "passed": (recall >= .95 and precision >= .80 and bool(errors)
                   and float(np.quantile(errors, .9)) <= 2.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
