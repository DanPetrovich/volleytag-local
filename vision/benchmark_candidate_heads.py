#!/usr/bin/env python3
"""Compare lightweight identity heads on frozen cached candidate embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DOMAINS = ("near_back", "far_front")
REFERENCE_TOP1 = 0.7489361702127659
MINIMUM_GAIN = 0.08
MAXIMUM_DOMAIN_REGRESSION = 0.02
REFERENCE_DOMAINS = {"near_back": 0.7749603803486529, "far_front": 0.7278562259306803}


def models() -> dict:
    return {
        "logistic-c025": make_pipeline(StandardScaler(), LogisticRegression(
            C=.25, class_weight="balanced", max_iter=3000, random_state=51)),
        "logistic-c01": make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=3000, random_state=51)),
        "logistic-c1": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000, random_state=51)),
        "lda-shrinkage": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto")),
        "knn-cosine-7": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine"),
        "knn-cosine-15": KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine"),
        "rbf-svm-c05": make_pipeline(StandardScaler(), SVC(
            C=.5, gamma="scale", class_weight="balanced", random_state=51)),
        "rbf-svm-c1": make_pipeline(StandardScaler(), SVC(
            C=1.0, gamma="scale", class_weight="balanced", random_state=51)),
        "rbf-svm-c2": make_pipeline(StandardScaler(), SVC(
            C=2.0, gamma="scale", class_weight="balanced", random_state=51)),
        "rbf-svm-c5": make_pipeline(StandardScaler(), SVC(
            C=5.0, gamma="scale", class_weight="balanced", random_state=51)),
        "rbf-svm-c1-gamma001": make_pipeline(StandardScaler(), SVC(
            C=1.0, gamma=.001, class_weight="balanced", random_state=51)),
        "rbf-svm-c1-gamma005": make_pipeline(StandardScaler(), SVC(
            C=1.0, gamma=.005, class_weight="balanced", random_state=51)),
        "extra-trees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=51),
    }


def groups(records: list[dict], indexes: np.ndarray, negatives_per_group: int = 5):
    by_domain: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indexes:
        by_domain[records[index]["domain"]][int(records[index]["label"])].append(int(index))
    result = []
    for domain in DOMAINS:
        negatives = by_domain[domain][0]
        for positive in by_domain[domain][1]:
            ranked = sorted(negatives, key=lambda i: (
                abs(float(records[i]["time_s"]) - float(records[positive]["time_s"])), i
            ))[:negatives_per_group]
            if ranked:
                result.append((domain, positive, ranked))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()

    with args.dataset.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    vectors = np.load(args.embeddings)
    labels = np.asarray([int(row["label"]) for row in records])
    sources = np.asarray([int(row["source"]) for row in records])
    candidates = models()
    totals = {name: defaultdict(int) for name in candidates}
    folds = []
    for held_source in sorted(set(sources)):
        train_all = np.flatnonzero(sources != held_source)
        test = np.flatnonzero(sources == held_source)
        fold = {"held_source": int(held_source), "models": {}}
        for name, estimator in candidates.items():
            hits, counts = defaultdict(int), defaultdict(int)
            for domain in DOMAINS:
                train = train_all[np.asarray([records[i]["domain"] == domain for i in train_all])]
                domain_groups = [row for row in groups(records, test) if row[0] == domain]
                if len(train) < 20 or len(set(labels[train])) < 2:
                    continue
                estimator.fit(vectors[train], labels[train])
                for _, positive, negatives in domain_groups:
                    indexes = [positive] + negatives
                    if hasattr(estimator, "predict_proba"):
                        scores = estimator.predict_proba(vectors[indexes])[:, 1]
                    else:
                        scores = estimator.decision_function(vectors[indexes])
                    hits[domain] += int(int(np.argmax(scores)) == 0)
                    counts[domain] += 1
            fold["models"][name] = {
                domain: {"groups": counts[domain],
                         "top1": hits[domain] / counts[domain] if counts[domain] else None}
                for domain in DOMAINS
            }
            for domain in DOMAINS:
                totals[name][f"{domain}_hits"] += hits[domain]
                totals[name][f"{domain}_groups"] += counts[domain]
        folds.append(fold)

    summary = {}
    for name, values in totals.items():
        domain_scores = {}
        total_hits = total_groups = 0
        for domain in DOMAINS:
            hits, count = values[f"{domain}_hits"], values[f"{domain}_groups"]
            domain_scores[domain] = hits / count if count else None
            total_hits += hits; total_groups += count
        top1 = total_hits / total_groups
        gain = top1 - REFERENCE_TOP1
        regressions = {domain: REFERENCE_DOMAINS[domain] - domain_scores[domain]
                       for domain in DOMAINS}
        summary[name] = {
            "groups": total_groups, "micro_top1": top1,
            "gain_vs_reranker_v2": gain, "domains": domain_scores,
            "maximum_domain_regression": max(regressions.values()),
            "passed": gain >= MINIMUM_GAIN and max(regressions.values()) <= MAXIMUM_DOMAIN_REGRESSION,
        }
    report = {
        "protocol": "leave-one-whole-source-out; cached DINOv2; five nearest-time negatives",
        "reference_reranker_v2": REFERENCE_TOP1,
        "minimum_gain": MINIMUM_GAIN,
        "maximum_domain_regression": MAXIMUM_DOMAIN_REGRESSION,
        "summary": dict(sorted(summary.items(), key=lambda item: item[1]["micro_top1"], reverse=True)),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.model_output:
        args.model_output.mkdir(parents=True, exist_ok=True)
        for domain in DOMAINS:
            indexes = np.asarray([i for i, row in enumerate(records) if row["domain"] == domain])
            final_model = make_pipeline(StandardScaler(), SVC(
                C=5.0, gamma="scale", class_weight="balanced", random_state=51))
            final_model.fit(vectors[indexes], labels[indexes])
            joblib.dump(final_model, args.model_output / f"candidate-reranker-v3-{domain}.joblib")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
