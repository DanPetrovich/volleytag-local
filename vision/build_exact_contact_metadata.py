#!/usr/bin/env python3
"""Convert exact-contact review answers into trajectory evaluation metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    answers = json.loads(args.results.read_text(encoding="utf-8"))["answers"]
    samples = []
    omitted = []
    for item in queue["items"]:
        answer = answers.get(item["id"])
        if not answer:
            raise ValueError(f"Missing answer: {item['id']}")
        decision = answer["decision"]
        if decision in {"offscreen_contact", "uncertain"}:
            omitted.append({"id": item["id"], "decision": decision})
            continue
        exact = answer.get("exact_contact_time_s")
        samples.append({
            "id": item["id"],
            "source": item["source"],
            "video": item["video"],
            "time_s": float(exact if exact is not None else item["approx_time_s"]),
            "label": 1 if decision == "contact" else 0,
            "decision": decision,
            "action": item.get("action"),
            "prior_label": item.get("prior_label"),
            "approx_time_s": item["approx_time_s"],
        })

    payload = {
        "schema_version": 1,
        "kind": "volleytag.exact-contact-evaluation-metadata",
        "frozen_tests_used": False,
        "samples": samples,
        "omitted": omitted,
        "counts": {
            "samples": len(samples),
            "positive_game_contacts": sum(row["label"] == 1 for row in samples),
            "negative_or_nongame": sum(row["label"] == 0 for row in samples),
            "nongame_contacts": sum(row["decision"] == "nongame_contact" for row in samples),
            "omitted_offscreen_or_uncertain": len(omitted),
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
