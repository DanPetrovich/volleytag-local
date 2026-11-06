#!/usr/bin/env python3
"""Compare automatic rally-end timestamps with reviewed live-play endings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from build_rally_intervals import accepted_boundaries
except ModuleNotFoundError:  # Imported from the repository root in unit tests.
    from vision.build_rally_intervals import accepted_boundaries


def ordered_matches(predicted: list[float], expected: list[float], tolerance: float) -> list[tuple[float, float]]:
    predicted, expected = sorted(predicted), sorted(expected)
    table: list[list[tuple[int, float, list[tuple[float, float]]]]] = [
        [(0, 0.0, []) for _ in range(len(expected) + 1)]
        for _ in range(len(predicted) + 1)
    ]
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(expected) + 1):
            choices = [table[i - 1][j], table[i][j - 1]]
            error = abs(predicted[i - 1] - expected[j - 1])
            if error <= tolerance:
                count, total_error, pairs = table[i - 1][j - 1]
                choices.append((count + 1, total_error + error,
                                pairs + [(predicted[i - 1], expected[j - 1])]))
            table[i][j] = max(choices, key=lambda row: (row[0], -row[1]))
    return table[-1][-1][2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("route", type=Path)
    parser.add_argument("fallback_queue", type=Path)
    parser.add_argument("fallback_results", type=Path)
    parser.add_argument("timeline_results", type=Path)
    parser.add_argument("--tolerance-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = json.loads(args.route.read_text())
    queue = json.loads(args.fallback_queue.read_text())
    fallback = json.loads(args.fallback_results.read_text())
    timeline = json.loads(args.timeline_results.read_text())
    automatic, _ = accepted_boundaries(route, queue, fallback)
    predicted = [float(row["time_s"]) for row in automatic]
    expected = list(map(float, timeline["rally_end_times_s"]))
    pairs = ordered_matches(predicted, expected, args.tolerance_s)
    errors = [abs(left - right) for left, right in pairs]
    report = {
        "schema_version": 1,
        "kind": "volleytag.rally-end-timestamp-diagnostic",
        "frozen_tests_used": True,
        "post_frozen_diagnostic": True,
        "tolerance_s": args.tolerance_s,
        "automatic_events": len(predicted),
        "reviewed_events": len(expected),
        "matched_events": len(pairs),
        "timestamp_recall": len(pairs) / max(len(expected), 1),
        "timestamp_precision": len(pairs) / max(len(predicted), 1),
        "mean_absolute_error_s": sum(errors) / len(errors) if errors else None,
        "matches": [{"automatic_s": left, "reviewed_s": right,
                     "absolute_error_s": abs(left - right)} for left, right in pairs],
        "passed": False,
        "conclusion": "diagnostic only; exact timestamp route requires redesign",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
