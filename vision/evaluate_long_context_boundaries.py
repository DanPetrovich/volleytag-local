#!/usr/bin/env python3
"""Spot rally starts/ends from long-context embeddings and pair them into intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_long_context_rally_state import load_source, safe_matches


def boundary_labels(source: dict, key: str) -> np.ndarray:
    labels = np.zeros(len(source["times"]), dtype=bool)
    for rally in source["rallies"]:
        index = int(np.argmin(np.abs(source["times"] - float(rally[key]))))
        labels[index] = True
    return labels


def select_peaks(times: np.ndarray, scores: np.ndarray, threshold: float,
                 separation_s: float = 4.0) -> list[dict]:
    indexes = [
        index for index, score in enumerate(scores)
        if score >= threshold
        and score >= scores[max(0, index - 2):min(len(scores), index + 3)].max()
    ]
    retained = []
    for index in sorted(indexes, key=lambda value: scores[value], reverse=True):
        if all(abs(times[index] - times[old]) >= separation_s for old in retained):
            retained.append(index)
    return [
        {"time_s": float(times[index]), "score": float(scores[index])}
        for index in sorted(retained)
    ]


def pair_intervals(starts: list[dict], ends: list[dict], padding_s: float) -> list[dict]:
    result, pending = [], []
    events = sorted(
        [(row["time_s"], "start", row) for row in starts]
        + [(row["time_s"], "end", row) for row in ends]
    )
    for time, kind, row in events:
        if kind == "start":
            pending.append(row)
            pending = [value for value in pending if time - value["time_s"] <= 45]
            continue
        eligible = [value for value in pending if 1 <= time - value["time_s"] <= 45]
        if not eligible:
            continue
        start = max(eligible, key=lambda value: value["time_s"])
        result.append({
            "start_s": start["time_s"] - padding_s,
            "end_s": time + padding_s,
            "start_score": start["score"],
            "end_score": row["score"],
        })
        pending = []
    return result


def evaluate(source: dict, start_scores: np.ndarray, end_scores: np.ndarray,
             start_threshold: float, end_threshold: float, padding_s: float,
             maximum_padding_s: float) -> dict:
    starts = select_peaks(source["times"], start_scores, start_threshold)
    ends = select_peaks(source["times"], end_scores, end_threshold)
    predictions = pair_intervals(starts, ends, padding_s)
    matches = safe_matches(source["rallies"], predictions, maximum_padding_s)
    return {
        "evaluation": source["name"],
        "truth": len(source["rallies"]),
        "start_candidates": len(starts),
        "end_candidates": len(ends),
        "candidates": len(predictions),
        "matches": len(matches),
        "recall": len(matches) / max(len(source["rallies"]), 1),
        "precision": len(matches) / max(len(predictions), 1),
        "predictions": predictions,
        "details": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    radii = tuple(map(int, config["context_radii_s"]))
    sources = [load_source(row, radii) for row in config["sources"]]
    scores = {}
    for c_value in config["boundary_logistic_c"]:
        for held_index, held in enumerate(sources):
            training = [row for index, row in enumerate(sources) if index != held_index]
            x = np.concatenate([row["features"] for row in training])
            for kind, key in (("start", "start_s"), ("end", "end_s")):
                y = np.concatenate([boundary_labels(row, key) for row in training])
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=float(c_value), class_weight="balanced",
                                       max_iter=3000, random_state=51),
                )
                model.fit(x, y)
                scores[(float(c_value), held_index, kind)] = model.predict_proba(
                    held["features"])[:, 1]

    best = None
    for c_value in config["boundary_logistic_c"]:
        for start_threshold in config["boundary_thresholds"]:
            for end_threshold in config["boundary_thresholds"]:
                for padding_s in config["boundary_padding_s"]:
                    folds = [
                        evaluate(
                            source,
                            scores[(float(c_value), index, "start")],
                            scores[(float(c_value), index, "end")],
                            float(start_threshold), float(end_threshold), float(padding_s),
                            float(config["maximum_padding_s"]),
                        )
                        for index, source in enumerate(sources)
                    ]
                    recalls = [row["recall"] for row in folds]
                    precisions = [row["precision"] for row in folds]
                    key = (min(recalls), float(np.mean(recalls)),
                           float(np.mean(precisions)),
                           -sum(row["candidates"] for row in folds))
                    if best is None or key > best[0]:
                        best = (key, {
                            "logistic_c": float(c_value),
                            "start_threshold": float(start_threshold),
                            "end_threshold": float(end_threshold),
                            "padding_s": float(padding_s),
                            "minimum_fold_recall": min(recalls),
                            "macro_recall": float(np.mean(recalls)),
                            "macro_precision": float(np.mean(precisions)),
                            "folds": folds,
                        })
    assert best is not None
    selected = best[1]
    sanity = config["sanity_criterion"]
    product = config["product_criterion"]
    report = {
        "schema_version": 1,
        "kind": "volleytag.long-context-boundary-loso",
        "frozen_tests_used": False,
        "protocol": "direct start/end spotting; three-source leave-one-whole-source-out",
        "context_radii_s": list(radii),
        "best": selected,
        "sanity_criterion": sanity,
        "product_criterion": product,
        "sanity_passed": (
            selected["macro_recall"] >= sanity["minimum_macro_recall"]
            and selected["minimum_fold_recall"] >= sanity["minimum_fold_recall"]
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
