#!/usr/bin/env python3
"""Build a compact human audit queue for deterministic replay intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_queue(payloads: list[dict]) -> dict:
    items = []
    for payload in payloads:
        if payload.get("frozen_tests_used"):
            raise ValueError("frozen replay metadata cannot enter a training audit")
        for source, row in payload.get("sources", {}).items():
            duration = float(row["duration_s"])
            for index, interval in enumerate(row.get("replay_pairs", []), 1):
                start, end = float(interval["start_s"]), float(interval["end_s"])
                items.append({
                    "id": f"replay-s{source}-{index:03d}-{start:.3f}",
                    "source": int(source),
                    "video": row["video"],
                    "start_s": start,
                    "end_s": end,
                    "window_start_s": max(0.0, start - 2.0),
                    "window_end_s": min(duration, end + 2.0),
                })
    items.sort(key=lambda item: (item["source"], item["start_s"]))
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate replay audit ids")
    return {
        "schema_version": 1,
        "kind": "volleytag.replay-interval-review-queue",
        "frozen_tests_used": False,
        "counts": {"items": len(items), "sources": len({i["source"] for i in items})},
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = build_queue([json.loads(path.read_text()) for path in args.reports])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(queue["counts"], indent=2))


if __name__ == "__main__":
    main()
