#!/usr/bin/env python3
"""Evaluate a pre-serve visual classifier without splitting a source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def candidates() -> dict[str, object]:
    return {
        "logistic-c01": make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=5000)),
        "logistic-c1": make_pipeline(StandardScaler(), LogisticRegression(
            C=1, class_weight="balanced", max_iter=5000)),
        "rbf-svm-c1": make_pipeline(StandardScaler(), SVC(
            C=1, class_weight="balanced")),
        "rbf-svm-c5": make_pipeline(StandardScaler(), SVC(
            C=5, class_weight="balanced")),
        "extra-trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=51, n_jobs=-1),
    }


def score(model: object, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    return np.asarray(model.predict_proba(features))[:, 1]


def operating_point(y: np.ndarray, scores: np.ndarray,
                    min_recall: float) -> dict[str, float | int] | None:
    best = None
    for threshold in np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores)]:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        row = {"threshold": float(threshold), "recall": recall,
               "precision": precision, "tp": tp, "fp": fp, "fn": fn}
        if recall >= min_recall and (best is None or
                                     (precision, threshold) >
                                     (best["precision"], best["threshold"])):
            best = row
    return best


def recall_first_point(y: np.ndarray, scores: np.ndarray,
                       min_precision: float) -> dict[str, float | int] | None:
    best = None
    for threshold in np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores)]:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        row = {"threshold": float(threshold), "recall": recall,
               "precision": precision, "tp": tp, "fp": fp, "fn": fn}
        if precision >= min_precision and (best is None or
                                           (recall, precision, threshold) >
                                           (best["recall"], best["precision"],
                                            best["threshold"])):
            best = row
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--motion-features", type=Path)
    parser.add_argument("--formation-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-recall", type=float, default=.90)
    parser.add_argument("--min-precision", type=float, default=.60)
    args = parser.parse_args()
    payload = np.load(args.features)
    x = payload["features"]
    labels = payload["labels"].astype(str)
    y = (labels == "pre_serve").astype(np.int8)
    groups = payload["sources"].astype(int)
    item_ids = payload["item_ids"].astype(str)
    feature_set = "r3d18"
    if args.motion_features:
        motion = np.load(args.motion_features)
        if not np.array_equal(payload["item_ids"], motion["item_ids"]):
            raise ValueError("motion and R3D item order differs")
        x = np.concatenate((x, motion["features"]), axis=1)
        feature_set = "r3d18+motion"
    if args.formation_features:
        formation = np.load(args.formation_features)
        formation_ids = formation["item_ids"].astype(str)
        lookup = {item_id: index for index, item_id in enumerate(formation_ids)}
        retained = np.asarray([index for index, item_id in enumerate(item_ids)
                               if item_id in lookup])
        formation_rows = np.asarray([lookup[item_ids[index]] for index in retained])
        x = np.concatenate((x[retained], formation["features"][formation_rows]), axis=1)
        labels, y, groups, item_ids = (labels[retained], y[retained],
                                       groups[retained], item_ids[retained])
        feature_set += "+formation"
    baseline = {"recall": 1.0, "precision": float(y.mean())}
    reports = []
    for name, model in candidates().items():
        scores = np.zeros(len(y), dtype=np.float64)
        for source in sorted(set(groups)):
            train, test = groups != source, groups == source
            model.fit(x[train], y[train])
            scores[test] = score(model, x[test])
        point = operating_point(y, scores, args.min_recall)
        recall_point = recall_first_point(y, scores, args.min_precision)
        assert point is not None
        predicted = scores >= point["threshold"]
        recall_first_predicted = scores >= recall_point["threshold"]
        source_recall, recall_first_source_recall = {}, {}
        for source in sorted(set(groups)):
            positives = (groups == source) & (y == 1)
            source_recall[str(source)] = float(np.mean(predicted[positives]))
            recall_first_source_recall[str(source)] = float(
                np.mean(recall_first_predicted[positives]))
        reports.append({"model": name, "operating_point": point,
                        "recall_first_point": recall_point,
                        "source_recall": source_recall,
                        "recall_first_source_recall": recall_first_source_recall})
    reports.sort(key=lambda row: (
        row["recall_first_point"]["recall"],
        row["recall_first_point"]["precision"],
        row["operating_point"]["precision"]), reverse=True)
    best = reports[0]
    passed = (best["recall_first_point"]["recall"] >= args.min_recall and
              best["recall_first_point"]["precision"] >= args.min_precision)
    report = {
        "protocol": "leave-one-whole-source-out; binary pre_serve vs rest",
        "feature_set": feature_set,
        "frozen_tests_used": False,
        "samples": len(y), "positives": int(y.sum()),
        "baseline": baseline,
        "acceptance": {"min_recall": args.min_recall,
                       "min_precision": args.min_precision},
        "candidates": reports, "best": best, "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
