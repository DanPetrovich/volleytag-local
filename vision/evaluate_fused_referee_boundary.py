#!/usr/bin/env python3
"""Evaluate frozen scoreboard plus unmatched referee-motion boundary candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BOUNDARY_LABELS = {"serve_authorization", "rally_end_team_to_serve"}


def evaluate(route: dict, queue: dict, results: dict,
             maximum_candidates_per_minute: float) -> dict:
    answers = results["answers"]
    missing = [item["id"] for item in queue["items"] if item["id"] not in answers]
    if missing:
        raise ValueError(f"missing {len(missing)} frozen fallback answers")
    decisions = [answers[item["id"]]["decision"] for item in queue["items"]]
    recovered_ends = decisions.count("rally_end_team_to_serve")
    recovered_starts = decisions.count("serve_authorization")
    expected_ends = int(route["expected_score_changes"])
    found_ends = min(expected_ends, int(route["scoreboard_events"]) + recovered_ends)
    end_recall = found_ends / max(expected_ends, 1)
    useful_candidates = int(route["scoreboard_events"]) + sum(
        decision in BOUNDARY_LABELS for decision in decisions)
    total_candidates = len(route["events"])
    precision = useful_candidates / max(total_candidates, 1)
    density = float(route["candidate_density_per_minute"])
    passed = end_recall == 1 and density <= maximum_candidates_per_minute
    return {
        "schema_version": 1,
        "kind": "volleytag.frozen-fused-boundary-evaluation",
        "frozen_tests_used": True,
        "parameters_changed_after_frozen_review": False,
        "expected_rally_ends": expected_ends,
        "scoreboard_rally_ends": int(route["scoreboard_events"]),
        "recovered_referee_rally_ends": recovered_ends,
        "referee_serve_authorizations": recovered_starts,
        "false_motion_fallback": len(decisions) - recovered_ends - recovered_starts,
        "rally_end_recall": end_recall,
        "useful_boundary_candidates": useful_candidates,
        "total_candidates": total_candidates,
        "boundary_candidate_precision": precision,
        "candidate_density_per_minute": density,
        "acceptance": ("rally end recall == 1.0 and candidate density per minute <= "
                       f"{maximum_candidates_per_minute}"),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--maximum-candidates-per-minute", type=float, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.route.read_text()),
                      json.loads(args.queue.read_text()),
                      json.loads(args.results.read_text()),
                      args.maximum_candidates_per_minute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
