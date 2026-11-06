#!/usr/bin/env python3
"""Evaluate a continuous referee boundary sanity interval against reviewed clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BOUNDARY_LABELS = {"serve_authorization", "rally_end_team_to_serve"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--boundary-ground-truth", type=Path)
    parser.add_argument("--tolerance-s", type=float, default=3)
    parser.add_argument("--before-score-change-s", type=float, default=10)
    parser.add_argument("--after-score-change-s", type=float, default=4)
    parser.add_argument("--maximum-candidates-per-minute", type=float, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text())
    items = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"]
    if args.boundary_ground_truth:
        ground_truth = json.loads(args.boundary_ground_truth.read_text())
        boundaries = [{"id": f"score-change-{index}", "time_s": row["time_s"]}
                      for index, row in enumerate(ground_truth["events"], 1)]
    else:
        boundaries = [item for item in items
                      if scores["start_s"] <= item["time_s"] <= scores["end_s"] and
                      answers[item["id"]]["decision"] in BOUNDARY_LABELS]
    candidates = scores["grouped_candidates"]
    hits = [{"item_id": item["id"], "time_s": item["time_s"],
             "matched_candidate_s": min(
                 (row["time_s"] for row in candidates),
                 key=lambda time: abs(time - item["time_s"]), default=None)}
            for item in boundaries]
    for row in hits:
        if args.boundary_ground_truth:
            row["hit"] = (row["matched_candidate_s"] is not None and
                          row["time_s"] - args.before_score_change_s <=
                          row["matched_candidate_s"] <=
                          row["time_s"] + args.after_score_change_s)
        else:
            row["hit"] = (row["matched_candidate_s"] is not None and
                          abs(row["matched_candidate_s"] - row["time_s"]) <= args.tolerance_s)
    recall = sum(row["hit"] for row in hits) / max(len(hits), 1)
    density = float(scores["grouped_candidates_per_minute"])
    passed = recall == 1 and density <= args.maximum_candidates_per_minute
    report = {
        "protocol": "known reviewed boundaries in continuous training sanity interval",
        "frozen_tests_used": False,
        "known_boundaries": len(boundaries),
        "hits": hits,
        "recall": recall,
        "grouped_candidates": len(candidates),
        "grouped_candidates_per_minute": density,
        "acceptance": ("recall == 1.0 and grouped candidates per minute <= "
                       f"{args.maximum_candidates_per_minute}"),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
