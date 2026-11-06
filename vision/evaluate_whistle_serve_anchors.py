#!/usr/bin/env python3
"""Evaluate whistle candidates against reviewed training serve contacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=(1, 2, 3, 4, 5, 6, 8, 10))
    parser.add_argument("--before-s", type=float, default=8.0)
    parser.add_argument("--after-s", type=float, default=1.0)
    parser.add_argument("--exclude-source", type=int, action="append", default=[5, 6, 8])
    args = parser.parse_args()

    payload = json.loads(args.events.read_text(encoding="utf-8"))
    by_source: dict[int, list[float]] = defaultdict(list)
    for row in payload["events"]:
        source = int(row["source"])
        if row.get("action") == "serve" and source not in args.exclude_source:
            by_source[source].append(float(row["time_s"]))

    reports = []
    for rate in args.rates:
        source_reports, total, hits = [], 0, 0
        for source, serves in sorted(by_source.items()):
            path = args.candidate_dir / f"source-{source}-whistles.json"
            detected = json.loads(path.read_text(encoding="utf-8"))
            keep = round(float(detected["duration_s"]) / 60 * rate)
            candidates = sorted(detected["candidates"], key=lambda row: row["score"], reverse=True)[:keep]
            times = [float(row["time_s"]) for row in candidates]
            details = []
            for serve in sorted(serves):
                eligible = [time for time in times
                            if serve - args.before_s <= time <= serve + args.after_s]
                nearest = min(eligible, key=lambda time: abs(time - serve)) if eligible else None
                details.append({
                    "serve_time_s": serve,
                    "matched": nearest is not None,
                    "whistle_time_s": nearest,
                    "offset_s": round(nearest - serve, 3) if nearest is not None else None,
                })
            source_hits = sum(row["matched"] for row in details)
            hits += source_hits
            total += len(serves)
            source_reports.append({
                "source": source, "serves": len(serves), "hits": source_hits,
                "recall": source_hits / len(serves), "details": details,
            })
        reports.append({
            "candidates_per_minute": rate,
            "serves": total,
            "hits": hits,
            "recall": hits / total,
            "passed_80pct_recall": hits / total >= .80,
            "sources": source_reports,
        })

    result = {
        "schema_version": 1,
        "kind": "volleytag.whistle-serve-anchor-evaluation",
        "frozen_tests_used": False,
        "window_before_s": args.before_s,
        "window_after_s": args.after_s,
        "acceptance_criterion": "serve-anchor recall >= 0.80",
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in (
        "candidates_per_minute", "serves", "hits", "recall", "passed_80pct_recall"
    )} for row in reports], indent=2))


if __name__ == "__main__":
    main()
