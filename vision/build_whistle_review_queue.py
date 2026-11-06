#!/usr/bin/env python3
"""Build a source-balanced audit queue for whistle-like audio peaks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operating-rate", type=float, default=6.0)
    parser.add_argument("--per-source", type=int, default=8)
    args = parser.parse_args()

    events = json.loads(args.events.read_text(encoding="utf-8"))
    videos = {index: row["video"] for index, row in enumerate(events["playlist"])}
    sources = sorted(int(path.stem.split("-")[1])
                     for path in args.candidate_dir.glob("source-*-whistles.json"))
    items = []
    for source in sources:
        payload = json.loads((args.candidate_dir / f"source-{source}-whistles.json").read_text())
        keep = round(float(payload["duration_s"]) / 60 * args.operating_rate)
        pool = sorted(payload["candidates"], key=lambda row: row["score"], reverse=True)[:keep]
        count = min(args.per_source, len(pool))
        indexes = np.linspace(0, len(pool) - 1, count, dtype=int)
        for rank_index in indexes:
            row = pool[int(rank_index)]
            timestamp = float(row["time_s"])
            items.append({
                "id": f"whistle-s{source}-t{timestamp:.3f}",
                "source": source,
                "video": videos[source],
                "time_s": timestamp,
                "window_start_s": round(max(0.0, timestamp - 5.0), 3),
                "window_end_s": round(min(float(payload["duration_s"]), timestamp + 5.0), 3),
                "score": row["score"],
                "dominant_hz": row["dominant_hz"],
                "operating_rank": int(rank_index + 1),
                "operating_pool": len(pool),
            })
    items.sort(key=lambda row: (row["source"], row["time_s"]))
    result = {
        "schema_version": 1,
        "kind": "volleytag.whistle-review-queue",
        "frozen_tests_used": False,
        "operating_rate_per_minute": args.operating_rate,
        "counts": {"items": len(items), "sources": len(sources), "per_source": args.per_source},
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
