#!/usr/bin/env python3
"""Evaluate temporal grouping of whistle candidates against serve anchors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def group_latest(rows: list[dict], max_gap_s: float) -> list[dict]:
    groups: list[list[dict]] = []
    for row in sorted(rows, key=lambda item: item["time_s"]):
        if not groups or float(row["time_s"]) - float(groups[-1][-1]["time_s"]) > max_gap_s:
            groups.append([row])
        else:
            groups[-1].append(row)
    return [group[-1] for group in groups]


def anchor_hits(serves: list[float], rows: list[dict],
                before_s: float, after_s: float) -> int:
    times = [float(row["time_s"]) for row in rows]
    return sum(any(serve - before_s <= time <= serve + after_s for time in times)
               for serve in serves)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=6.0)
    parser.add_argument("--max-gap-s", type=float, default=2.0)
    parser.add_argument("--before-s", type=float, default=8.0)
    parser.add_argument("--after-s", type=float, default=1.0)
    parser.add_argument("--exclude-source", type=int, action="append", default=[5, 6, 8])
    args = parser.parse_args()
    events = json.loads(args.events.read_text())["events"]
    by_source: dict[int, list[float]] = defaultdict(list)
    for row in events:
        source = int(row["source"])
        if row.get("action") == "serve" and source not in args.exclude_source:
            by_source[source].append(float(row["time_s"]))
    sources, baseline_count, grouped_count = [], 0, 0
    baseline_hits, grouped_hits = 0, 0
    for source, serves in sorted(by_source.items()):
        payload = json.loads((args.candidate_dir / f"source-{source}-whistles.json").read_text())
        keep = round(float(payload["duration_s"]) / 60 * args.rate)
        baseline = sorted(payload["candidates"], key=lambda row: row["score"], reverse=True)[:keep]
        grouped = group_latest(baseline, args.max_gap_s)
        source_baseline_hits = anchor_hits(serves, baseline, args.before_s, args.after_s)
        source_grouped_hits = anchor_hits(serves, grouped, args.before_s, args.after_s)
        baseline_count += len(baseline)
        grouped_count += len(grouped)
        baseline_hits += source_baseline_hits
        grouped_hits += source_grouped_hits
        sources.append({
            "source": source, "serves": len(serves),
            "baseline_candidates": len(baseline), "grouped_candidates": len(grouped),
            "baseline_hits": source_baseline_hits, "grouped_hits": source_grouped_hits,
        })
    reduction = 1 - grouped_count / baseline_count
    report = {
        "schema_version": 1,
        "kind": "volleytag.whistle-candidate-grouping-evaluation",
        "frozen_tests_used": False,
        "rate": args.rate, "max_gap_s": args.max_gap_s,
        "baseline": {"candidates": baseline_count, "serve_hits": baseline_hits},
        "grouped": {"candidates": grouped_count, "serve_hits": grouped_hits,
                    "candidate_reduction": reduction},
        "acceptance": "preserve baseline serve hits and reduce candidates >= 25%",
        "passed": grouped_hits == baseline_hits and reduction >= .25,
        "sources": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
