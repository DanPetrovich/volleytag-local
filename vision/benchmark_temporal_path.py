#!/usr/bin/env python3
"""Nested leave-one-source-out benchmark for a motion-constrained identity path."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from evaluate_fused_detectors import iou


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def speed(previous, current, dt):
    px, py = center(previous["box"]); cx, cy = center(current["box"])
    height = max(20., ((previous["box"][3] - previous["box"][1]) +
                       (current["box"][3] - current["box"][1])) / 2)
    return math.hypot(cx - px, cy - py) / height / max(dt, .1)


def segments(rows, max_gap=2.0):
    result, current = [], []
    for row in sorted(rows, key=lambda value: value["time_s"]):
        if current and row["time_s"] - current[-1]["time_s"] > max_gap:
            result.append(current); current = []
        current.append(row)
    if current: result.append(current)
    return result


def decode(rows, params):
    chosen = []
    for segment in segments(rows):
        states = [row["candidates"] + [None] for row in segment]
        scores = []
        for candidate in states[0]:
            scores.append(params["missing"] if candidate is None else
                          params["direct"] * float(candidate.get("direct_confidence", 0)))
        backs = []
        for index in range(1, len(segment)):
            dt = segment[index]["time_s"] - segment[index - 1]["time_s"]
            next_scores, pointers = [], []
            for current in states[index]:
                best_value, best_pointer = -float("inf"), 0
                unary = params["missing"] if current is None else (
                    params["direct"] * float(current.get("direct_confidence", 0)))
                for pointer, previous in enumerate(states[index - 1]):
                    if current is None or previous is None:
                        transition = params["missing_transition"] if current is not previous else 0.
                    else:
                        value = speed(previous, current, dt)
                        if value > params["max_speed"]:
                            transition = -100.
                        else:
                            transition = (-params["motion"] * value * value +
                                          params["iou"] * iou(previous["box"], current["box"]))
                    total = scores[pointer] + transition + unary
                    if total > best_value: best_value, best_pointer = total, pointer
                next_scores.append(best_value); pointers.append(best_pointer)
            scores = next_scores; backs.append(pointers)
        path = [max(range(len(scores)), key=scores.__getitem__)]
        for pointers in reversed(backs): path.append(pointers[path[-1]])
        path.reverse()
        chosen.extend(state[position] for state, position in zip(states, path))
    return chosen


def accuracy(rows, params):
    predicted = decode(rows, params)
    hits = sum(candidate is not None and iou(row["gt_box"], candidate["box"]) >= .30
               for row, candidate in zip(sorted(rows, key=lambda value: value["time_s"]), predicted))
    return hits, len(rows)


def direct_accuracy(rows):
    hits = candidates = 0
    for row in rows:
        ranked = sorted(row["candidates"], key=lambda candidate:
                        float(candidate.get("direct_confidence", 0)), reverse=True)
        hits += bool(ranked and iou(row["gt_box"], ranked[0]["box"]) >= .30)
        candidates += any(iou(row["gt_box"], candidate["box"]) >= .30
                          for candidate in row["candidates"])
    return hits, candidates, len(rows)


def grid():
    # Keep the search deliberately small; missing-state and physical-speed
    # constants are fixed before evaluation rather than tuned per match.
    for direct, motion, overlap in itertools.product(
            (2., 4., 6.), (.05, .15, .35), (0., .5, 1.5)):
        yield {"direct": direct, "motion": motion, "iou": overlap,
               "missing": -1., "missing_transition": -.5, "max_speed": 6.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_source = defaultdict(list)
    rows = []
    for path in args.cache:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    for row in rows:
        direct = [candidate for candidate in row["candidates"]
                  if float(candidate.get("direct_confidence", 0)) > 0]
        generic = sorted(
            [candidate for candidate in row["candidates"]
             if float(candidate.get("direct_confidence", 0)) <= 0],
            key=lambda candidate: float(candidate.get("confidence", 0)), reverse=True)[:4]
        by_source[int(row["source"])].append({**row, "candidates": direct + generic})
    sources = sorted(by_source)
    baselines, folds = {}, []
    for source in sources:
        direct_hits, candidate_hits, count = direct_accuracy(by_source[source])
        baselines[source] = {"frames": count, "direct_top1": direct_hits / count,
                             "candidate_recall": candidate_hits / count}
    for held in sources:
        training_sources = [source for source in sources if source != held]
        best_params, best_score = None, -1.
        for params in grid():
            hits = count = 0
            for source in training_sources:
                value_hits, value_count = accuracy(by_source[source], params)
                hits += value_hits; count += value_count
            score = hits / count
            if score > best_score: best_score, best_params = score, params
        held_hits, held_count = accuracy(by_source[held], best_params)
        folds.append({"held_source": held, "selected_on_other_sources": best_score,
                      "parameters": best_params, "frames": held_count,
                      "baseline": baselines[held]["direct_top1"],
                      "temporal": held_hits / held_count,
                      "gain": held_hits / held_count - baselines[held]["direct_top1"]})
    total = sum(row["frames"] for row in folds)
    baseline = sum(row["baseline"] * row["frames"] for row in folds) / total
    temporal = sum(row["temporal"] * row["frames"] for row in folds) / total
    report = {"protocol": "nested leave-one-whole-source-out",
              "success_criterion": {"minimum_gain": .08, "maximum_source_regression": .02},
              "baselines": baselines, "folds": folds,
              "direct_top1": baseline, "temporal_top1": temporal,
              "gain": temporal - baseline,
              "maximum_source_regression": max(row["baseline"] - row["temporal"] for row in folds)}
    report["passed"] = (report["gain"] >= .08 and report["maximum_source_regression"] <= .02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
