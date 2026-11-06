#!/usr/bin/env python3
"""Evaluate late fusion of generic rally state and ball-trajectory evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_long_context_rally_state import evaluate, load_source


def fit_scores(training: list[dict], held: dict, c_value: float) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000,
                           random_state=51),
    )
    model.fit(np.concatenate([row["features"] for row in training]),
              np.concatenate([row["labels"] for row in training]))
    return model.predict_proba(held["features"])[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    radii = tuple(map(int, config["context_radii_s"]))
    base_sources, ball_sources = [], []
    for spec in config["sources"]:
        base_sources.append(load_source(
            {**spec, "feature_files": spec["base_feature_files"]}, radii))
        ball_sources.append(load_source(
            {**spec, "feature_files": [spec["ball_feature_file"]]}, radii))
    scores = {}
    for c_value in config["logistic_c"]:
        for held_index in range(len(base_sources)):
            base_training = [row for index, row in enumerate(base_sources)
                             if index != held_index]
            ball_training = [row for index, row in enumerate(ball_sources)
                             if index != held_index]
            scores[(float(c_value), held_index, "base")] = fit_scores(
                base_training, base_sources[held_index], float(c_value))
            scores[(float(c_value), held_index, "ball")] = fit_scores(
                ball_training, ball_sources[held_index], float(c_value))

    best = None
    for c_value in config["logistic_c"]:
        for ball_weight in config["ball_fusion_weights"]:
            for threshold in config["thresholds"]:
                for smoothing_radius in config["smoothing_radius_s"]:
                    for bridge_s in config["bridge_s"]:
                        for minimum_active_s in config["minimum_active_s"]:
                            for padding_s in config["padding_s"]:
                                parameters = {
                                    "threshold": float(threshold),
                                    "smoothing_radius": int(smoothing_radius),
                                    "bridge_s": float(bridge_s),
                                    "minimum_active_s": float(minimum_active_s),
                                    "padding_s": float(padding_s),
                                }
                                folds = []
                                for index, source in enumerate(base_sources):
                                    base = scores[(float(c_value), index, "base")]
                                    ball = scores[(float(c_value), index, "ball")]
                                    fused = ((1 - float(ball_weight)) * base
                                             + float(ball_weight) * ball)
                                    folds.append(evaluate(
                                        source, fused, parameters,
                                        float(config["maximum_padding_s"])))
                                recalls = [row["recall"] for row in folds]
                                precisions = [row["precision"] for row in folds]
                                key = (min(recalls), float(np.mean(recalls)),
                                       float(np.mean(precisions)),
                                       -sum(row["candidates"] for row in folds))
                                if best is None or key > best[0]:
                                    best = (key, {
                                        "logistic_c": float(c_value),
                                        "ball_weight": float(ball_weight),
                                        **parameters,
                                        "minimum_fold_recall": min(recalls),
                                        "macro_recall": float(np.mean(recalls)),
                                        "macro_precision": float(np.mean(precisions)),
                                        "folds": folds,
                                    })
    assert best is not None
    selected = best[1]
    sanity = config["ball_sanity_criterion"]
    product = config["product_criterion"]
    report = {
        "schema_version": 1,
        "kind": "volleytag.ball-state-late-fusion-loso",
        "frozen_tests_used": False,
        "protocol": "separate base/ball heads; three-source leave-one-whole-source-out",
        "baseline": config["baseline"],
        "best": selected,
        "sanity_criterion": sanity,
        "product_criterion": product,
        "sanity_passed": (
            selected["macro_recall"] >= sanity["minimum_macro_recall"]
            and selected["macro_precision"] >= sanity["minimum_macro_precision"]
        ),
        "product_passed": (
            selected["macro_recall"] >= product["minimum_macro_recall"]
            and selected["macro_precision"] >= product["minimum_macro_precision"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "best": {key: value for key, value in selected.items() if key != "folds"},
        "folds": [{key: row[key] for key in (
            "evaluation", "truth", "candidates", "matches", "recall", "precision")}
                  for row in selected["folds"]],
        "sanity_passed": report["sanity_passed"],
        "product_passed": report["product_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
