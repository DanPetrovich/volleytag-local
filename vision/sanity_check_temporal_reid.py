#!/usr/bin/env python3
"""Sanity-check temporal appearance continuity on leave-one-match-out folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_candidate_reranker import classifier


def logit(value: float) -> float:
    value = min(max(value, 1e-5), 1 - 1e-5)
    return math.log(value / (1 - value))


def decode(groups: list[tuple[float, list[int]]], probabilities: np.ndarray,
           vectors: np.ndarray, similarity_weight: float, max_gap: float) -> list[int]:
    """Offline Viterbi path over candidate appearances, resetting across long gaps."""
    chosen: list[int] = []
    segment: list[tuple[float, list[int]]] = []

    def flush(values: list[tuple[float, list[int]]]) -> None:
        if not values:
            return
        scores = np.asarray([logit(float(probabilities[i])) for i in values[0][1]])
        backpointers: list[np.ndarray] = []
        for (_, previous), (_, current) in zip(values, values[1:]):
            similarity = vectors[previous] @ vectors[current].T
            totals = scores[:, None] + similarity_weight * similarity
            pointers = np.argmax(totals, axis=0)
            scores = totals[pointers, np.arange(len(current))] + np.asarray(
                [logit(float(probabilities[i])) for i in current])
            backpointers.append(pointers)
        path = [int(np.argmax(scores))]
        for pointers in reversed(backpointers):
            path.append(int(pointers[path[-1]]))
        path.reverse()
        chosen.extend(indices[position] for (_, indices), position in zip(values, path))

    for value in groups:
        if segment and value[0] - segment[-1][0] > max_gap:
            flush(segment); segment = []
        segment.append(value)
    flush(segment)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--similarity-weight", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=2.0)
    args = parser.parse_args()

    with args.dataset.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    vectors = np.load(args.embeddings)
    labels = np.asarray([int(row["label"]) for row in rows])
    sources = np.asarray([int(row["source"]) for row in rows])
    domains = np.asarray([row["domain"] for row in rows])
    folds = []
    for held_source in sorted(set(sources)):
        for domain in ("near_back", "far_front"):
            train = np.flatnonzero((sources != held_source) & (domains == domain))
            test = np.flatnonzero((sources == held_source) & (domains == domain))
            if not len(train) or not len(test) or len(set(labels[train])) < 2:
                continue
            model = classifier().fit(vectors[train], labels[train])
            all_probabilities = np.zeros(len(rows), dtype=float)
            all_probabilities[test] = model.predict_proba(vectors[test])[:, 1]
            grouped: dict[float, list[int]] = defaultdict(list)
            for index in test:
                grouped[round(float(rows[index]["time_s"]), 3)].append(int(index))
            groups = sorted(grouped.items())
            frame_top1 = [max(indices, key=lambda i: all_probabilities[i]) for _, indices in groups]
            temporal_top1 = decode(groups, all_probabilities, vectors,
                                   args.similarity_weight, args.max_gap)
            folds.append({
                "held_source": int(held_source), "domain": domain, "groups": len(groups),
                "frame_top1": float(np.mean(labels[frame_top1])) if frame_top1 else 0.0,
                "temporal_top1": float(np.mean(labels[temporal_top1])) if temporal_top1 else 0.0,
            })
    baseline = float(np.mean([row["frame_top1"] for row in folds]))
    temporal = float(np.mean([row["temporal_top1"] for row in folds]))
    report = {
        "protocol": "leave-one-whole-match-out",
        "fixed_similarity_weight": args.similarity_weight,
        "fixed_max_gap_s": args.max_gap,
        "success_criterion_absolute_gain": .10,
        "frame_macro_top1": baseline,
        "temporal_macro_top1": temporal,
        "absolute_gain": temporal - baseline,
        "passed": temporal - baseline >= .10,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
