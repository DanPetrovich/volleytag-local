#!/usr/bin/env python3
"""Build dense active/dead samples from exact reviewed rally intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_rallies(rallies: list[dict], start_s: float, end_s: float) -> None:
    previous_end = start_s
    for row in rallies:
        start, end = float(row["start_s"]), float(row["end_s"])
        if start < previous_end or end <= start or end > end_s:
            raise ValueError("rallies must be ordered, non-overlapping, and inside the queue")
        if row.get("start_confidence") not in {"exact", "inferred"}:
            raise ValueError("missing start confidence")
        if row.get("end_confidence") not in {"exact", "inferred"}:
            raise ValueError("missing end confidence")
        previous_end = end


def build_samples(queue: dict, results: dict, source: int, step_s: float,
                  development_rallies: int) -> tuple[dict, dict]:
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    if results.get("pending_start") is not None:
        raise ValueError("review has an unfinished rally")
    rallies = results.get("rallies", [])
    if not 0 < development_rallies < len(rallies):
        raise ValueError("development_rallies must leave at least one evaluation rally")
    validate_rallies(rallies, float(queue["start_s"]), float(queue["end_s"]))
    split_s = (float(rallies[development_rallies - 1]["end_s"])
               + float(rallies[development_rallies]["start_s"])) / 2
    items, answers = [], {}
    timestamp = float(queue["start_s"]) + 3.0
    last_usable = float(queue["end_s"]) - 2.0
    while timestamp <= last_usable + 1e-6:
        active = next((row for row in rallies
                       if float(row["start_s"]) <= timestamp <= float(row["end_s"])), None)
        item_id = f"dense-rally-s{source}-t{timestamp:.3f}"
        partition = "development" if timestamp < split_s else "evaluation"
        items.append({
            "id": item_id,
            "source": source,
            "video": queue["video"],
            "time_s": round(timestamp, 3),
            "partition": partition,
            "rally_id": active["id"] if active else None,
        })
        answers[item_id] = {"decision": "active_play" if active else "dead_ball"}
        timestamp += step_s
    queue_payload = {
        "schema_version": 1,
        "kind": "volleytag.dense-rally-state-queue",
        "frozen_tests_used": False,
        "source": source,
        "video": queue["video"],
        "start_s": queue["start_s"],
        "end_s": queue["end_s"],
        "step_s": step_s,
        "split_s": split_s,
        "development_rallies": development_rallies,
        "evaluation_rallies": len(rallies) - development_rallies,
        "rallies": rallies,
        "items": items,
    }
    result_payload = {
        "schema_version": 1,
        "kind": "volleytag.dense-rally-state-results",
        "frozen_tests_used": False,
        "answers": answers,
    }
    return queue_payload, result_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--development-rallies", type=int, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    args = parser.parse_args()
    queue, results = build_samples(json.loads(args.queue.read_text()),
                                   json.loads(args.results.read_text()), args.source,
                                   args.step_s, args.development_rallies)
    for path, payload in ((args.queue_output, queue), (args.results_output, results)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    counts = {label: sum(row["decision"] == label for row in results["answers"].values())
              for label in ("active_play", "dead_ball")}
    print(json.dumps({"samples": len(queue["items"]), "split_s": queue["split_s"],
                      "labels": counts}, indent=2))


if __name__ == "__main__":
    main()
