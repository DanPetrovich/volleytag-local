#!/usr/bin/env python3
"""Aggregate a fixed exact-contact head into a rally-level diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_features", type=Path)
    parser.add_argument("scan_features", type=Path)
    parser.add_argument("scan_metadata", type=Path)
    parser.add_argument("touch_results", type=Path)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--minimum-negative-rejection", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = np.load(args.training_features)
    scan = np.load(args.scan_features)
    metadata = json.loads(args.scan_metadata.read_text())
    results = json.loads(args.touch_results.read_text())
    if not metadata.get("frozen_tests_used"):
        raise ValueError("scan metadata is not marked frozen")
    if len(metadata["samples"]) != len(scan["features"]):
        raise ValueError("scan feature/metadata length mismatch")
    components = min(48, len(training["labels"]) - 2, training["features"].shape[1])
    model = make_pipeline(StandardScaler(), PCA(n_components=components, whiten=True,
                                                 random_state=51),
                          LogisticRegression(C=.25, class_weight="balanced", max_iter=3000,
                                             random_state=51))
    model.fit(training["features"], training["labels"])
    scores = model.predict_proba(scan["features"])[:, 1]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(metadata["samples"], scores):
        grouped[row["item_id"]].append(float(score))
    rows = []
    for item_id, answer in results["answers"].items():
        decision = answer["decision"]
        maximum = max(grouped[item_id])
        rows.append({"item_id": item_id, "decision": decision,
                     "windows": len(grouped[item_id]), "maximum_score": maximum,
                     "predicted_touch": maximum >= args.threshold})
    certain = [row for row in rows if row["decision"] != "uncertain"]
    positives = [row for row in certain if row["decision"] == "player51_touch"]
    negatives = [row for row in certain if row["decision"] == "no_player51_touch"]
    tp = sum(row["predicted_touch"] for row in positives)
    tn = sum(not row["predicted_touch"] for row in negatives)
    predicted = sum(row["predicted_touch"] for row in certain)
    recall = tp / max(len(positives), 1)
    rejection = tn / max(len(negatives), 1)
    precision = tp / max(predicted, 1)
    metric_passed = recall == 1 and rejection >= args.minimum_negative_rejection
    report = {
        "schema_version": 1, "kind": "volleytag.rally-touch-scan-diagnostic",
        "frozen_tests_used": True, "post_frozen_diagnostic": True,
        "training_samples": int(len(training["labels"])),
        "scan_windows": int(len(scores)), "certain_rallies": len(certain),
        "positive_rallies": len(positives), "negative_rallies": len(negatives),
        "fixed_threshold": args.threshold,
        "baseline_all_positive": {"recall": 1.0,
                                  "precision": len(positives) / max(len(certain), 1),
                                  "negative_rejection": 0.0},
        "candidate": {"recall": recall, "precision": precision,
                      "negative_rejection": rejection, "true_positive": tp,
                      "true_negative": tn, "predicted_positive": predicted},
        "acceptance": {"recall": 1.0,
                       "minimum_negative_rejection": args.minimum_negative_rejection},
        "metric_passed": metric_passed, "promotable": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
