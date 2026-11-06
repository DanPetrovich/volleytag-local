#!/usr/bin/env python3
"""Measure whole-source rally-boundary fusion with and without referee features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_rally_boundary_fusion import build_source, match_events, select_peaks


REFEREE_FEATURES = np.arange(36, 45)


def choose_threshold(sources: list[dict], scores: list[np.ndarray], kind: str) -> float:
    values = np.concatenate(scores)
    best = None
    for threshold in np.unique(np.quantile(values, np.linspace(0, 1, 101))):
        candidates = [select_peaks(source["times"], source_scores, float(threshold))
                      for source, source_scores in zip(sources, scores)]
        matches = sum(len(match_events(source["labels"][f"{kind}_times"], rows))
                      for source, rows in zip(sources, candidates))
        truth = sum(len(source["labels"][f"{kind}_times"]) for source in sources)
        count = sum(len(rows) for rows in candidates)
        minutes = sum(source["duration_minutes"] for source in sources)
        recall = matches / max(truth, 1)
        precision = matches / max(count, 1)
        density = count / max(minutes, 1e-8)
        key = (density <= 8.0, recall, precision, -density, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1]


def evaluate_mode(sources: list[dict], columns: np.ndarray, c: float) -> dict:
    folds = []
    for held in sources:
        training = [source for source in sources if source is not held]
        output = {"evaluation": held["name"]}
        for kind in ("start", "end"):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=51))
            model.fit(np.concatenate([row["features"][:, columns] for row in training]),
                      np.concatenate([row["labels"][kind] for row in training]))
            training_scores = [model.predict_proba(row["features"][:, columns])[:, 1]
                               for row in training]
            threshold = choose_threshold(training, training_scores, kind)
            scores = model.predict_proba(held["features"][:, columns])[:, 1]
            candidates = select_peaks(held["times"], scores, threshold)
            truth = held["labels"][f"{kind}_times"]
            matches = match_events(truth, candidates)
            output[kind] = {
                "threshold": threshold, "truth": len(truth),
                "candidates": len(candidates), "matches": len(matches),
                "recall": len(matches) / max(len(truth), 1),
                "precision": len(matches) / max(len(candidates), 1),
            }
        folds.append(output)
    return {"folds": folds,
            "macro_start_recall": float(np.mean([row["start"]["recall"] for row in folds])),
            "macro_end_recall": float(np.mean([row["end"]["recall"] for row in folds]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logistic-c", type=float, default=0.01)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sources = [build_source(row) for row in config["sources"]]
    width = sources[0]["features"].shape[1]
    if width != 54:
        raise ValueError(f"unexpected fusion feature width: {width}")
    all_columns = np.arange(width)
    modes = {
        "all_signals": all_columns,
        "without_referee": np.setdiff1d(all_columns, REFEREE_FEATURES),
        "referee_only": REFEREE_FEATURES,
    }
    report = {
        "schema_version": 1,
        "kind": "volleytag.rally-boundary-signal-ablation",
        "frozen_tests_used": False,
        "protocol": "three-source leave-one-whole-source-out; thresholds calibrated on training sources",
        "referee_feature_columns": REFEREE_FEATURES.tolist(),
        "modes": {name: evaluate_mode(sources, columns, args.logistic_c)
                  for name, columns in modes.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
