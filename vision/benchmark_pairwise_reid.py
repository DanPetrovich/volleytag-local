#!/usr/bin/env python3
"""Leave-one-source-out benchmark for adjacent-frame pairwise ReID."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DOMAINS = ("near_back", "far_front")


def feature(anchor: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    repeated = np.repeat(anchor[None, :], len(candidates), axis=0)
    cosine = np.sum(repeated * candidates, axis=1, keepdims=True)
    return np.hstack((np.abs(repeated - candidates), repeated * candidates, cosine))


def transition_groups(records, source, domain, negatives_per_group=5, max_gap=2.0):
    positives = sorted(
        [i for i, row in enumerate(records)
         if int(row["source"]) == source and row["domain"] == domain and int(row["label"]) == 1],
        key=lambda i: float(records[i]["time_s"]))
    negatives = [i for i, row in enumerate(records)
                 if int(row["source"]) == source and row["domain"] == domain and int(row["label"]) == 0]
    groups = []
    for previous, current in zip(positives, positives[1:]):
        if float(records[current]["time_s"]) - float(records[previous]["time_s"]) > max_gap:
            continue
        ranked = sorted(negatives, key=lambda i: (
            abs(float(records[i]["time_s"]) - float(records[current]["time_s"])), i
        ))[:negatives_per_group]
        if ranked:
            groups.append((previous, current, ranked))
    return groups


def estimators():
    return {
        "logistic-c01": make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=3000, random_state=51)),
        "logistic-c1": make_pipeline(StandardScaler(), LogisticRegression(
            C=1., class_weight="balanced", max_iter=3000, random_state=51)),
        "rbf-c1": make_pipeline(StandardScaler(), SVC(
            C=1., gamma="scale", class_weight="balanced", random_state=51)),
        "rbf-c5": make_pipeline(StandardScaler(), SVC(
            C=5., gamma="scale", class_weight="balanced", random_state=51)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    vectors = np.load(args.embeddings)
    sources = sorted({int(row["source"]) for row in records})
    groups = {(source, domain): transition_groups(records, source, domain)
              for source in sources for domain in DOMAINS}
    totals = {name: defaultdict(int) for name in estimators()}
    baseline = defaultdict(int); folds = []
    for held in sources:
        fold = {"held_source": held, "domains": {}}
        for domain in DOMAINS:
            train_groups = [group for source in sources if source != held
                            for group in groups[source, domain]]
            test_groups = groups[held, domain]
            if not train_groups or not test_groups:
                continue
            train_x, train_y = [], []
            for previous, current, negatives in train_groups:
                indexes = [current] + negatives
                train_x.append(feature(vectors[previous], vectors[indexes]))
                train_y.extend([1] + [0] * len(negatives))
            train_x = np.vstack(train_x); train_y = np.asarray(train_y)
            domain_result = {"groups": len(test_groups), "models": {}}
            cosine_hits = 0
            for previous, current, negatives in test_groups:
                indexes = [current] + negatives
                similarities = vectors[indexes] @ vectors[previous]
                cosine_hits += int(int(np.argmax(similarities)) == 0)
            baseline[f"{domain}_hits"] += cosine_hits
            baseline[f"{domain}_groups"] += len(test_groups)
            for name, estimator in estimators().items():
                estimator.fit(train_x, train_y)
                hits = 0
                for previous, current, negatives in test_groups:
                    indexes = [current] + negatives
                    values = feature(vectors[previous], vectors[indexes])
                    scores = (estimator.predict_proba(values)[:, 1]
                              if hasattr(estimator, "predict_proba") else estimator.decision_function(values))
                    hits += int(int(np.argmax(scores)) == 0)
                domain_result["models"][name] = hits / len(test_groups)
                totals[name][f"{domain}_hits"] += hits
                totals[name][f"{domain}_groups"] += len(test_groups)
            fold["domains"][domain] = domain_result
        folds.append(fold)
    base_hits = sum(baseline[f"{domain}_hits"] for domain in DOMAINS)
    base_groups = sum(baseline[f"{domain}_groups"] for domain in DOMAINS)
    base_score = base_hits / base_groups
    summary = {}
    for name, values in totals.items():
        hits = sum(values[f"{domain}_hits"] for domain in DOMAINS)
        count = sum(values[f"{domain}_groups"] for domain in DOMAINS)
        domain_scores = {domain: values[f"{domain}_hits"] / values[f"{domain}_groups"]
                         if values[f"{domain}_groups"] else None for domain in DOMAINS}
        domain_baselines = {domain: baseline[f"{domain}_hits"] / baseline[f"{domain}_groups"]
                            if baseline[f"{domain}_groups"] else None for domain in DOMAINS}
        regressions = [domain_baselines[d] - domain_scores[d] for d in DOMAINS
                       if domain_scores[d] is not None]
        score = hits / count
        summary[name] = {"groups": count, "top1": score, "gain_vs_cosine": score - base_score,
                         "domains": domain_scores, "passed": score - base_score >= .08 and
                         max(regressions, default=0.) <= .02}
    report = {"protocol": "leave-one-whole-source-out adjacent positive transitions",
              "cosine_baseline": base_score, "minimum_gain": .08,
              "summary": dict(sorted(summary.items(), key=lambda item: item[1]["top1"], reverse=True)),
              "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.model_output:
        args.model_output.mkdir(parents=True, exist_ok=True)
        for domain in DOMAINS:
            domain_groups = [group for source in sources for group in groups[source, domain]]
            train_x, train_y = [], []
            for previous, current, negatives in domain_groups:
                indexes = [current] + negatives
                train_x.append(feature(vectors[previous], vectors[indexes]))
                train_y.extend([1] + [0] * len(negatives))
            final_model = make_pipeline(StandardScaler(), SVC(
                C=5., gamma="scale", class_weight="balanced", random_state=51))
            final_model.fit(np.vstack(train_x), np.asarray(train_y))
            joblib.dump(final_model, args.model_output / f"pairwise-reid-{domain}.joblib")
    print(json.dumps({"cosine_baseline": base_score, "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
