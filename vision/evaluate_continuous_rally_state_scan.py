#!/usr/bin/env python3
"""Run a nested held-source visual filter over continuous audio candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_rally_state_classifier import recall_first_point


def model(name: str, c: float) -> object:
    if name == "extra-trees":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=51, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, class_weight="balanced", max_iter=5000))


def scores(model_instance: object, features: np.ndarray) -> np.ndarray:
    if hasattr(model_instance, "decision_function"):
        return np.asarray(model_instance.decision_function(features))
    return np.asarray(model_instance.predict_proba(features))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_r3d", type=Path)
    parser.add_argument("training_motion", type=Path)
    parser.add_argument("scan_queue", type=Path)
    parser.add_argument("scan_r3d", type=Path)
    parser.add_argument("scan_motion", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("--training-formation", type=Path)
    parser.add_argument("--scan-formation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-precision", type=float, default=.60)
    parser.add_argument("--c", type=float, default=.1)
    parser.add_argument("--model", choices=("logistic", "extra-trees"), default="logistic")
    parser.add_argument("--before-s", type=float, default=8.0)
    parser.add_argument("--after-s", type=float, default=1.0)
    args = parser.parse_args()
    r3d, motion = np.load(args.training_r3d), np.load(args.training_motion)
    if not np.array_equal(r3d["item_ids"], motion["item_ids"]):
        raise ValueError("training feature order differs")
    x = np.concatenate((r3d["features"], motion["features"]), axis=1)
    y = (r3d["labels"].astype(str) == "pre_serve").astype(np.int8)
    groups = r3d["sources"].astype(int)
    item_ids = r3d["item_ids"].astype(str)
    if args.training_formation:
        formation = np.load(args.training_formation)
        formation_ids = formation["item_ids"].astype(str)
        lookup = {item_id: index for index, item_id in enumerate(formation_ids)}
        retained = np.asarray([index for index, item_id in enumerate(item_ids)
                               if item_id in lookup])
        formation_rows = np.asarray([lookup[item_ids[index]] for index in retained])
        x = np.concatenate((x[retained], formation["features"][formation_rows]), axis=1)
        y, groups, item_ids = y[retained], groups[retained], item_ids[retained]
    queue = json.loads(args.scan_queue.read_text())
    held = int(queue["source"])
    retained = groups != held
    inner_scores = np.zeros(int(retained.sum()))
    inner_x, inner_y, inner_groups = x[retained], y[retained], groups[retained]
    for source in sorted(set(inner_groups)):
        train, test = inner_groups != source, inner_groups == source
        candidate = model(args.model, args.c)
        candidate.fit(inner_x[train], inner_y[train])
        inner_scores[test] = scores(candidate, inner_x[test])
    point = recall_first_point(inner_y, inner_scores, args.min_precision)
    if point is None:
        raise RuntimeError("no inner-CV threshold satisfies precision floor")
    final_model = model(args.model, args.c)
    final_model.fit(inner_x, inner_y)
    scan_r3d, scan_motion = np.load(args.scan_r3d), np.load(args.scan_motion)
    if not np.array_equal(scan_r3d["item_ids"], scan_motion["item_ids"]):
        raise ValueError("scan feature order differs")
    scan_x = np.concatenate((scan_r3d["features"], scan_motion["features"]), axis=1)
    if args.scan_formation:
        scan_formation = np.load(args.scan_formation)
        if not np.array_equal(scan_r3d["item_ids"], scan_formation["item_ids"]):
            raise ValueError("scan formation item order differs")
        scan_x = np.concatenate((scan_x, scan_formation["features"]), axis=1)
    scan_scores = scores(final_model, scan_x)
    kept = scan_scores >= point["threshold"]
    times = [float(item["time_s"]) for item, retain in zip(queue["items"], kept) if retain]
    events = json.loads(args.events.read_text())["events"]
    serves = sorted(float(row["time_s"]) for row in events
                    if row.get("action") == "serve" and int(row["source"]) == held)

    def matches(candidate_times: list[float]) -> list[dict]:
        result = []
        for serve in serves:
            eligible = [time for time in candidate_times
                        if serve - args.before_s <= time <= serve + args.after_s]
            nearest = min(eligible, key=lambda time: abs(time - serve)) if eligible else None
            result.append({"serve_time_s": serve, "matched": nearest is not None,
                           "candidate_time_s": nearest})
        return result

    baseline_times = [float(item["time_s"]) for item in queue["items"]]
    baseline_matches, filtered_matches = matches(baseline_times), matches(times)
    duration_minutes = float(queue["duration_s"]) / 60
    baseline_hits = sum(row["matched"] for row in baseline_matches)
    filtered_hits = sum(row["matched"] for row in filtered_matches)
    reduction = 1 - len(times) / len(baseline_times)
    passed = filtered_hits == baseline_hits == len(serves) and reduction >= .25
    report = {
        "protocol": "nested held-source continuous candidate diagnostic",
        "frozen_tests_used": False,
        "held_source": held,
        "threshold_calibration_sources": sorted(set(map(int, inner_groups))),
        "threshold": point,
        "logistic_c": args.c,
        "model": args.model,
        "baseline": {"candidates": len(baseline_times),
                     "candidates_per_minute": len(baseline_times) / duration_minutes,
                     "serve_hits": baseline_hits, "serves": len(serves),
                     "details": baseline_matches},
        "visual_filter": {"candidates": len(times),
                          "candidates_per_minute": len(times) / duration_minutes,
                          "candidate_reduction": reduction,
                          "serve_hits": filtered_hits, "serves": len(serves),
                          "details": filtered_matches},
        "acceptance": "retain all baseline serve anchors and reduce candidates >= 25%",
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
