#!/usr/bin/env python3
"""Evaluate a locked rally-boundary fusion model on an unseen source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from train_rally_boundary_fusion import (
    build_source,
    interval_matches,
    match_events,
    pair_intervals,
    select_peaks,
)


def directional_event_matches(
    truth: list[float], candidates: list[dict], kind: str, padding_s: float
) -> list[dict]:
    """Match starts before truth or ends after truth, within allowed padding."""
    if kind not in {"start", "end"}:
        raise ValueError("kind must be start or end")
    pairs = []
    for truth_index, truth_s in enumerate(truth):
        for candidate_index, row in enumerate(candidates):
            candidate_s = float(row["time_s"])
            padding = truth_s - candidate_s if kind == "start" else candidate_s - truth_s
            if 0 <= padding <= padding_s:
                pairs.append((padding, truth_index, candidate_index))
    used_truth, used_candidates, matches = set(), set(), []
    for padding, truth_index, candidate_index in sorted(pairs):
        if truth_index in used_truth or candidate_index in used_candidates:
            continue
        used_truth.add(truth_index)
        used_candidates.add(candidate_index)
        matches.append({
            "truth_s": float(truth[truth_index]),
            "candidate_s": float(candidates[candidate_index]["time_s"]),
            "padding_s": float(padding),
        })
    return matches


def safe_interval_matches(
    truth: list[dict], predicted: list[dict], padding_s: float
) -> list[dict]:
    """Match predictions that contain the whole truth interval without excess padding."""
    pairs = []
    for truth_index, gt in enumerate(truth):
        for prediction_index, row in enumerate(predicted):
            lead = float(gt["start_s"]) - float(row["start_s"])
            lag = float(row["end_s"]) - float(gt["end_s"])
            if 0 <= lead <= padding_s and 0 <= lag <= padding_s:
                pairs.append((lead + lag, truth_index, prediction_index, lead, lag))
    used_truth, used_predictions, matches = set(), set(), []
    for _, truth_index, prediction_index, lead, lag in sorted(pairs):
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append({
            "gt_id": truth[truth_index]["id"],
            "prediction_index": prediction_index,
            "start_lead_s": float(lead),
            "end_lag_s": float(lag),
        })
    return matches


def metric(matches: list[dict], truth_count: int, candidate_count: int) -> dict:
    return {
        "truth": truth_count,
        "candidates": candidate_count,
        "matches": len(matches),
        "recall": len(matches) / max(truth_count, 1),
        "precision": len(matches) / max(candidate_count, 1),
        "details": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--padding-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = joblib.load(args.model)
    spec = json.loads(args.source.read_text())
    source = build_source(spec)
    if source["name"] in package["training_sources"]:
        raise ValueError("evaluation source is present in model training_sources")
    if package.get("proposal_mode") != "global":
        raise ValueError("only locked global-proposal packages are supported")

    candidates = {}
    legacy_events = {}
    safe_events = {}
    for kind in ("start", "end"):
        scores = package["models"][kind].predict_proba(source["features"])[:, 1]
        rows = select_peaks(source["times"], scores, package["thresholds"][kind])
        truth = source["labels"][f"{kind}_times"]
        candidates[kind] = rows
        legacy_events[kind] = metric(match_events(truth, rows), len(truth), len(rows))
        safe_events[kind] = metric(
            directional_event_matches(truth, rows, kind, args.padding_s),
            len(truth), len(rows),
        )

    intervals = pair_intervals(candidates["start"], candidates["end"])
    legacy = interval_matches(source["rallies"], intervals)
    safe = safe_interval_matches(source["rallies"], intervals, args.padding_s)
    safe_metric = metric(safe, len(source["rallies"]), len(intervals))
    report = {
        "schema_version": 1,
        "kind": "volleytag.rally-boundary-fusion-held-evaluation",
        "frozen_tests_used": False,
        "held_source": True,
        "parameters_changed_after_review": False,
        "model": str(args.model),
        "training_sources": package["training_sources"],
        "evaluation_source": source["name"],
        "thresholds": package["thresholds"],
        "product_criterion": {
            "description": "prediction must contain the complete reviewed rally",
            "maximum_extra_padding_each_side_s": args.padding_s,
            "clipped_boundaries_allowed": False,
        },
        "safe_events": safe_events,
        "safe_intervals": {**safe_metric, "predictions": intervals},
        "legacy_symmetric_2s": {
            "start": legacy_events["start"],
            "end": legacy_events["end"],
            "intervals": metric(legacy, len(source["rallies"]), len(intervals)),
        },
        "acceptance": {
            "required_safe_interval_recall": 1.0,
            "required_safe_interval_precision": 0.7,
            "passed": safe_metric["recall"] >= 1.0 and safe_metric["precision"] >= 0.7,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
