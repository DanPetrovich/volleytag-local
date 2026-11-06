#!/usr/bin/env python3
"""Evaluate long-context active-play segmentation on reviewed training sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def robust_rows(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    return (values - center) / np.maximum(scale, 1e-4)


def window_summary(values: np.ndarray, radii: tuple[int, ...]) -> np.ndarray:
    """Add bidirectional multi-scale context without crossing source boundaries."""
    rows = [values]
    for radius in radii:
        means, stds = [], []
        for index in range(len(values)):
            window = values[max(0, index - radius):min(len(values), index + radius + 1)]
            means.append(window.mean(axis=0))
            stds.append(window.std(axis=0))
        rows.extend((np.stack(means), np.stack(stds)))
    return np.concatenate(rows, axis=1)


def load_source(spec: dict, radii: tuple[int, ...]) -> dict:
    queue = json.loads(Path(spec["queue"]).read_text())
    lookup = {row["id"]: row for row in queue["items"]}
    feature_paths = spec.get("feature_files") or [
        spec["r3d_features"], spec["motion_features"]
    ]
    arrays = [np.load(path) for path in feature_paths]
    item_ids = arrays[0]["item_ids"].astype(str)
    labels = arrays[0]["labels"].astype(str)
    for row in arrays[1:]:
        if not np.array_equal(item_ids, row["item_ids"].astype(str)):
            raise ValueError(f"feature item order differs for {spec['name']}")
        if not np.array_equal(labels, row["labels"].astype(str)):
            raise ValueError(f"feature labels differ for {spec['name']}")
    items = [lookup[item_id] for item_id in item_ids]
    times = np.asarray([float(row["time_s"]) for row in items])
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be strictly increasing")
    feature_groups = []
    for array in arrays:
        normalized = robust_rows(array["features"].astype(np.float64))
        feature_groups.append(window_summary(normalized, radii))
    return {
        "name": spec["name"],
        "times": times,
        "features": np.concatenate(feature_groups, axis=1),
        "labels": labels == "active_play",
        "rallies": queue["rallies"],
        "step_s": float(queue["step_s"]),
    }


def smooth(scores: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return scores.copy()
    return np.asarray([
        scores[max(0, index - radius):min(len(scores), index + radius + 1)].mean()
        for index in range(len(scores))
    ])


def bridge_gaps(active: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = active.copy()
    if maximum_gap <= 0:
        return result
    false_indexes = np.flatnonzero(~result)
    for index in false_indexes:
        left = max(0, index - maximum_gap)
        right = min(len(result), index + maximum_gap + 1)
        if result[left:index].any() and result[index + 1:right].any():
            result[index] = True
    return result


def decode(times: np.ndarray, scores: np.ndarray, step_s: float, threshold: float,
           smoothing_radius: int, bridge_s: float, minimum_active_s: float,
           padding_s: float) -> list[dict]:
    active = smooth(scores, smoothing_radius) >= threshold
    active = bridge_gaps(active, int(round(bridge_s / step_s)))
    groups: list[list[int]] = []
    for index, value in enumerate(active):
        if not value:
            continue
        if groups and index - groups[-1][-1] == 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    result = []
    for group in groups:
        duration = (len(group) - 1) * step_s + step_s
        if duration < minimum_active_s:
            continue
        result.append({
            "start_s": float(times[group[0]] - step_s / 2 - padding_s),
            "end_s": float(times[group[-1]] + step_s / 2 + padding_s),
            "mean_score": float(scores[group].mean()),
            "max_score": float(scores[group].max()),
        })
    return result


def safe_matches(truth: list[dict], predicted: list[dict], maximum_padding_s: float) -> list[dict]:
    pairs = []
    for truth_index, gt in enumerate(truth):
        for prediction_index, row in enumerate(predicted):
            start_lead = float(gt["start_s"]) - row["start_s"]
            end_lag = row["end_s"] - float(gt["end_s"])
            if 0 <= start_lead <= maximum_padding_s and 0 <= end_lag <= maximum_padding_s:
                pairs.append((start_lead + end_lag, truth_index, prediction_index,
                              start_lead, end_lag))
    matches, used_truth, used_predictions = [], set(), set()
    for _, truth_index, prediction_index, start_lead, end_lag in sorted(pairs):
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index); used_predictions.add(prediction_index)
        matches.append({
            "gt_id": truth[truth_index]["id"],
            "prediction_index": prediction_index,
            "start_lead_s": start_lead,
            "end_lag_s": end_lag,
        })
    return matches


def evaluate(source: dict, scores: np.ndarray, parameters: dict,
             maximum_padding_s: float) -> dict:
    predictions = decode(source["times"], scores, source["step_s"], **parameters)
    matches = safe_matches(source["rallies"], predictions, maximum_padding_s)
    return {
        "evaluation": source["name"],
        "truth": len(source["rallies"]),
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
    probability_cache = {}
    for c_value in config["logistic_c"]:
        for held_index, held in enumerate(sources):
            training = [row for index, row in enumerate(sources) if index != held_index]
            x = np.concatenate([row["features"] for row in training])
            y = np.concatenate([row["labels"] for row in training])
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=float(c_value), class_weight="balanced",
                                   max_iter=3000, random_state=51),
            )
            model.fit(x, y)
            probability_cache[(float(c_value), held_index)] = model.predict_proba(
                held["features"])[:, 1]

    best = None
    for c_value in config["logistic_c"]:
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
                            folds = [
                                evaluate(source, probability_cache[(float(c_value), index)],
                                         parameters, float(config["maximum_padding_s"]))
                                for index, source in enumerate(sources)
                            ]
                            recalls = [row["recall"] for row in folds]
                            precisions = [row["precision"] for row in folds]
                            macro_recall = float(np.mean(recalls))
                            macro_precision = float(np.mean(precisions))
                            key = (min(recalls), macro_recall, macro_precision,
                                   -sum(row["candidates"] for row in folds))
                            if best is None or key > best[0]:
                                best = (key, {
                                    "logistic_c": float(c_value),
                                    **parameters,
                                    "minimum_fold_recall": min(recalls),
                                    "macro_recall": macro_recall,
                                    "macro_precision": macro_precision,
                                    "folds": folds,
                                })
    assert best is not None
    selected = best[1]
    sanity = config["sanity_criterion"]
    product = config["product_criterion"]
    report = {
        "schema_version": 1,
        "kind": "volleytag.long-context-rally-state-loso",
        "frozen_tests_used": False,
        "protocol": "three-source leave-one-whole-source-out; per-source robust normalization; bidirectional long context",
        "context_radii_s": list(radii),
        "search_space": {key: config[key] for key in (
            "logistic_c", "thresholds", "smoothing_radius_s", "bridge_s",
            "minimum_active_s", "padding_s")},
        "baseline": config["baseline"],
        "sanity_criterion": sanity,
        "product_criterion": product,
        "best": selected,
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
