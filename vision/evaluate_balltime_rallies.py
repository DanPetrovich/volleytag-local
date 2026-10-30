#!/usr/bin/env python3
"""Evaluate ReID ranking on manually checked Balltime rally candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path,
                        default=Path("vision/evaluations/balltime-rallies.json"), nargs="?")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))

    rows = []
    for rally in data["rallies"]:
        ranking_path = ROOT / rally["run"]
        ranking = list(csv.DictReader(ranking_path.open(encoding="utf-8")))[:5]
        labels = rally["top5_labels"]
        observed = [(row["track_id"], labels.get(row["track_id"])) for row in ranking]
        if any(label is None for _, label in observed):
            raise SystemExit(f"Rally {rally['id']}: top five changed; audit labels again")
        first_positive_rank = next((index for index, (_, label) in enumerate(observed, 1) if label), None)
        rows.append({
            "rally": rally["id"],
            "top1_correct": bool(observed[0][1]),
            "top2_hit": any(label for _, label in observed[:2]),
            "top5_correct": sum(bool(label) for _, label in observed),
            "first_positive_rank": first_positive_rank,
        })

    total = len(rows)
    confirmed = sum(row["top5_correct"] for row in rows)
    report = {
        "rallies": rows,
        "top1_accuracy": sum(row["top1_correct"] for row in rows) / total,
        "top2_rally_recall": sum(row["top2_hit"] for row in rows) / total,
        "top5_candidate_precision": confirmed / (total * 5),
        "note": "Candidate-level metric on manually audited top-five results; not frame-level recall.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
