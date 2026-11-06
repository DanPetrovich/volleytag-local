#!/usr/bin/env python3
"""Build one referee-ROI calibration item per training source."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TRAINING_SOURCES = (0, 1, 2, 3, 4, 7, 9)


def build_queue(payload: dict) -> dict:
    videos = {index: row["video"] for index, row in enumerate(payload["playlist"])}
    serves: dict[int, list[float]] = defaultdict(list)
    for row in payload["events"]:
        source = int(row["source"])
        if source in TRAINING_SOURCES and row.get("action") == "serve":
            serves[source].append(float(row["time_s"]))
    missing = [source for source in TRAINING_SOURCES if not serves[source]]
    if missing:
        raise ValueError(f"training sources without reviewed serves: {missing}")
    items = [{
        "id": f"referee-roi-s{source}",
        "source": source,
        "video": videos[source],
        "frame_time_s": max(1.0, min(serves[source]) - 4.0),
        "selection_reason": "before_first_reviewed_serve",
    } for source in TRAINING_SOURCES]
    return {
        "schema_version": 1,
        "kind": "volleytag.referee-roi-queue",
        "frozen_tests_used": False,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_queue(json.loads(args.events.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": len(result["items"]),
                      "sources": [row["source"] for row in result["items"]]}, indent=2))


if __name__ == "__main__":
    main()
