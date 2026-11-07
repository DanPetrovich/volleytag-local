#!/usr/bin/env python3
"""Reuse cached feature vectors with a newly reviewed dense rally queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def relabel(queue: dict, results: dict, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if queue.get("frozen_tests_used"):
        raise ValueError("frozen evaluation features cannot be relabelled for training")
    items = queue.get("items", [])
    answers = results.get("answers", {})
    wanted = [row["id"] for row in items]
    cached_ids = cache["item_ids"].astype(str).tolist()
    if len(cached_ids) != len(set(cached_ids)):
        raise ValueError("cached feature ids are not unique")
    lookup = {item_id: index for index, item_id in enumerate(cached_ids)}
    missing = [item_id for item_id in wanted if item_id not in lookup]
    if missing:
        raise ValueError(f"cache misses {len(missing)} reviewed items")
    missing_labels = [item_id for item_id in wanted if item_id not in answers]
    if missing_labels:
        raise ValueError(f"results miss {len(missing_labels)} reviewed items")
    indices = np.asarray([lookup[item_id] for item_id in wanted])
    return {
        "features": cache["features"][indices],
        "labels": np.asarray([answers[item_id]["decision"] for item_id in wanted]),
        "sources": np.asarray([int(row["source"]) for row in items]),
        "item_ids": np.asarray(wanted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    results = json.loads(args.results.read_text())
    with np.load(args.cache) as source:
        payload = relabel(queue, results, {key: source[key] for key in source.files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(json.dumps({"samples": len(payload["item_ids"]),
                      "dimensions": int(payload["features"].shape[1])}, indent=2))


if __name__ == "__main__":
    main()
