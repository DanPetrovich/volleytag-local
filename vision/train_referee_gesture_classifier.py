#!/usr/bin/env python3
"""Train the compact referee-gesture head on cached video embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BOUNDARY_LABELS = ("serve_authorization", "rally_end_team_to_serve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-source", type=int, action="append", default=[])
    parser.add_argument("--boundary-threshold", type=float, default=.215)
    parser.add_argument("--model", choices=("logistic", "extra-trees"), default="logistic")
    args = parser.parse_args()

    payloads = [np.load(path) for path in args.features]
    features = np.concatenate([payload["features"] for payload in payloads])
    raw_labels = np.concatenate([payload["labels"].astype(str) for payload in payloads])
    sources = np.concatenate([payload["sources"].astype(int) for payload in payloads])
    retained = raw_labels != "uncertain"
    if args.exclude_source:
        retained &= ~np.isin(sources, args.exclude_source)
    labels = np.asarray([label if label in BOUNDARY_LABELS else "other"
                         for label in raw_labels[retained]])
    if args.model == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, class_weight="balanced", max_iter=5000, random_state=51))
    else:
        model = ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=51, n_jobs=-1)
    model.fit(features[retained], labels)
    package = {
        "schema_version": 1,
        "kind": "volleytag.referee-gesture-classifier",
        "model": model,
        "boundary_labels": BOUNDARY_LABELS,
        "boundary_threshold": args.boundary_threshold,
        "embedding": "torchvision-r3d18-default-fc-identity",
        "head": args.model,
        "excluded_sources": sorted(set(args.exclude_source)),
        "frozen_tests_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.output)
    print(json.dumps({
        "samples": len(labels),
        "class_counts": {label: int(np.sum(labels == label))
                         for label in sorted(set(labels))},
        "excluded_sources": package["excluded_sources"],
        "boundary_threshold": args.boundary_threshold,
        "head": args.model,
    }, indent=2))


if __name__ == "__main__":
    main()
