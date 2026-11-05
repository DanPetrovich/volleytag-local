#!/usr/bin/env python3
"""Cheap leave-one-match-out sanity check for cached reranker embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from train_candidate_reranker import classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.dataset.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    vectors = np.load(args.embeddings)
    if len(rows) != len(vectors):
        raise SystemExit(f"row/vector mismatch: {len(rows)} != {len(vectors)}")
    labels = np.asarray([int(row["label"]) for row in rows])
    sources = np.asarray([int(row["source"]) for row in rows])
    domains = np.asarray([row["domain"] for row in rows])

    report = {"protocol": "leave-one-whole-match-out", "folds": []}
    for held_source in sorted(set(sources)):
        for domain in ("near_back", "far_front"):
            train = np.flatnonzero((sources != held_source) & (domains == domain))
            test = np.flatnonzero((sources == held_source) & (domains == domain))
            if not len(train) or not len(test) or len(set(labels[train])) < 2 or len(set(labels[test])) < 2:
                continue
            model = classifier().fit(vectors[train], labels[train])
            probabilities = model.predict_proba(vectors[test])[:, 1]
            groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for index, probability in zip(test, probabilities):
                # All candidates originating from the same reviewed instant form
                # the real ranking decision; millisecond rounding avoids float text noise.
                key = f"{rows[index]['source']}:{float(rows[index]['time_s']):.3f}"
                groups[key].append((float(probability), int(labels[index])))
            top1 = sum(max(values)[1] == 1 for values in groups.values())
            report["folds"].append({
                "held_source": int(held_source), "domain": domain,
                "train_samples": int(len(train)), "test_samples": int(len(test)),
                "test_groups": len(groups),
                "roc_auc": float(roc_auc_score(labels[test], probabilities)),
                "top1_accuracy": top1 / len(groups) if groups else 0.0,
            })
    folds = report["folds"]
    report["macro_roc_auc"] = float(np.mean([row["roc_auc"] for row in folds])) if folds else 0.0
    report["macro_top1_accuracy"] = float(np.mean([row["top1_accuracy"] for row in folds])) if folds else 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
