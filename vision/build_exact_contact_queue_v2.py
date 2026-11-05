#!/usr/bin/env python3
"""Build a balanced second exact-contact queue from cached temporal samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def key(source: int, timestamp: float) -> tuple[int, int]:
    return source, round(timestamp * 1000)


def spread(rows: list[dict], count: int) -> list[dict]:
    """Prefer diverse times while retaining high-motion hard negatives."""
    ranked = sorted(rows, key=lambda row: (-float(row.get("motion", 0)), float(row["time_s"])))
    chosen: list[dict] = []
    for minimum_gap in (8.0, 4.0, 2.0, 0.0):
        for row in ranked:
            if row in chosen:
                continue
            if all(abs(float(row["time_s"]) - float(other["time_s"])) >= minimum_gap
                   for other in chosen):
                chosen.append(row)
                if len(chosen) == count:
                    return sorted(chosen, key=lambda value: float(value["time_s"]))
    return sorted(chosen, key=lambda value: float(value["time_s"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("reviewed_queue", type=Path, nargs="+")
    parser.add_argument("--per-label-source", type=int, default=6)
    parser.add_argument("--sources", type=int, nargs="+")
    parser.add_argument("--labels", type=int, nargs="+", choices=(0, 1), default=(1, 0))
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = json.loads(args.features.read_text(encoding="utf-8"))["samples"]
    reviewed = []
    for path in args.reviewed_queue:
        reviewed.extend(json.loads(path.read_text(encoding="utf-8"))["items"])
    excluded = {key(int(row["source"]), float(row["approx_time_s"])) for row in reviewed}

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in samples:
        source, label, timestamp = int(row["source"]), int(row["label"]), float(row["time_s"])
        if key(source, timestamp) not in excluded:
            groups[source, label].append(row)

    selected: list[dict] = []
    sources = sorted(args.sources or {source for source, _label in groups})
    for source in sources:
        for label in args.labels:
            available = groups[source, label]
            if len(available) < args.per_label_source and not args.allow_short:
                raise ValueError(
                    f"source {source}, label {label}: need {args.per_label_source}, "
                    f"only {len(available)} unreviewed samples"
                )
            selected.extend(spread(available, min(len(available), args.per_label_source)))

    selected.sort(key=lambda row: (int(row["source"]), -int(row["label"]), float(row["time_s"])))
    items = []
    for index, row in enumerate(selected):
        timestamp = float(row["time_s"])
        items.append({
            "id": f"contact-v2-{index:03d}-s{int(row['source'])}-t{timestamp:.3f}",
            "source": int(row["source"]),
            "video": row["video"],
            "approx_time_s": round(timestamp, 3),
            "window_start_s": round(max(0.0, timestamp - 1.5), 3),
            "window_end_s": round(timestamp + 1.5, 3),
            "prior_label": int(row["label"]),
            "action": row.get("action"),
            "event_id": row.get("id"),
            "selection_reason": "unreviewed_contact" if int(row["label"]) else "high_motion_hard_negative",
            "motion": round(float(row.get("motion", 0)), 6),
        })

    counts = {
        "items": len(items),
        "positives": sum(item["prior_label"] == 1 for item in items),
        "negatives": sum(item["prior_label"] == 0 for item in items),
        "sources": len(sources),
        "per_label_source": args.per_label_source,
        "excluded_reviewed": len(excluded),
    }
    payload = {
        "schema_version": 1,
        "kind": "volleytag.exact-contact-queue",
        "frozen_tests_used": False,
        "instructions": "Mark the exact physical ball-contact frame, no contact, or uncertain.",
        "counts": counts,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
