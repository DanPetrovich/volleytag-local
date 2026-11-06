#!/usr/bin/env python3
"""Evaluate a fixed dense active-play classifier as exact rally segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def intervals_from_predictions(times: np.ndarray, predicted: np.ndarray,
                               step_s: float) -> list[dict]:
    groups: list[list[float]] = []
    for time, active in zip(times, predicted):
        if not active:
            continue
        if groups and time - groups[-1][-1] <= step_s * 1.5:
            groups[-1].append(float(time))
        else:
            groups.append([float(time)])
    return [{"start_s": row[0] - step_s / 2, "end_s": row[-1] + step_s / 2}
            for row in groups]


def exact_boundary_matches(gt: list[dict], predicted: list[dict], tolerance_s: float) -> list[dict]:
    candidates = []
    for gt_index, truth in enumerate(gt):
        for pred_index, candidate in enumerate(predicted):
            start_error = abs(float(candidate["start_s"]) - float(truth["start_s"]))
            end_error = abs(float(candidate["end_s"]) - float(truth["end_s"]))
            if start_error <= tolerance_s and end_error <= tolerance_s:
                candidates.append((start_error + end_error, gt_index, pred_index,
                                   start_error, end_error))
    matches, used_gt, used_pred = [], set(), set()
    for _, gt_index, pred_index, start_error, end_error in sorted(candidates):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index); used_pred.add(pred_index)
        matches.append({"gt_id": gt[gt_index]["id"], "prediction_index": pred_index,
                        "start_error_s": start_error, "end_error_s": end_error})
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("features", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=2.0)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    arrays = [np.load(path) for path in args.features]
    item_ids = arrays[0]["item_ids"].astype(str)
    if any(not np.array_equal(item_ids, row["item_ids"].astype(str)) for row in arrays[1:]):
        raise ValueError("feature item order differs")
    lookup = {row["id"]: row for row in queue["items"]}
    rows = [lookup[item_id] for item_id in item_ids]
    x = np.concatenate([row["features"] for row in arrays], axis=1)
    labels = arrays[0]["labels"].astype(str)
    y = labels == "active_play"
    development = np.asarray([row["partition"] == "development" for row in rows])
    evaluation = ~development
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight="balanced", max_iter=5000, random_state=51))
    model.fit(x[development], y[development])
    probabilities = model.predict_proba(x[evaluation])[:, 1]
    predicted = probabilities >= .5
    tn, fp, fn, tp = confusion_matrix(y[evaluation], predicted, labels=[False, True]).ravel()
    times = np.asarray([float(row["time_s"]) for row in rows])[evaluation]
    predicted_intervals = intervals_from_predictions(times, predicted, float(queue["step_s"]))
    gt = [row for row in queue["rallies"] if float(row["start_s"]) >= float(queue["split_s"])]
    matches = exact_boundary_matches(gt, predicted_intervals, args.tolerance_s)
    errors = [value for row in matches for value in
              (row["start_error_s"], row["end_error_s"])]
    report = {
        "schema_version": 1,
        "kind": "volleytag.dense-rally-segmentation-sanity",
        "frozen_tests_used": False,
        "protocol": "blocked same-source development/evaluation sanity; fixed threshold 0.5",
        "feature_files": [str(path) for path in args.features],
        "development_samples": int(development.sum()),
        "evaluation_samples": int(evaluation.sum()),
        "point_metrics": {
            "active_recall": tp / max(tp + fn, 1),
            "dead_specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
        },
        "segment_metrics": {
            "gt_rallies": len(gt),
            "predicted_rallies": len(predicted_intervals),
            "exact_matches": len(matches),
            "recall": len(matches) / max(len(gt), 1),
            "precision": len(matches) / max(len(predicted_intervals), 1),
            "boundary_error_p90_s": float(np.quantile(errors, .9)) if errors else None,
            "tolerance_s": args.tolerance_s,
        },
        "matches": matches,
        "predicted_intervals": predicted_intervals,
        "passed_sanity": (len(matches) / max(len(gt), 1) >= .8
                          and len(matches) / max(len(predicted_intervals), 1) >= .6),
        "note": "This is method sanity only; promotion still requires a held-source check.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
