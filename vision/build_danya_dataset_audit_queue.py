#!/usr/bin/env python3
"""Merge structural and model-disagreement signals into a review queue."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-items", type=Path, required=True)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--personal", type=Path, required=True)
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_payload = load(args.all_items)
    all_items = {item["id"]: item for item in all_payload["items"]}
    structural = {item["id"]: item for item in load(args.structural)["items"]}
    personal = {item["id"]: item for item in load(args.personal)["items"]}
    person = {item["id"]: item for item in load(args.person)["items"]}
    reviewed_payload = load(args.reviewed)
    reviewed = set(reviewed_payload.get("decisions", {}))

    combined = []
    for item_id, original in all_items.items():
        if item_id in reviewed:
            continue
        reasons = list(structural.get(item_id, {}).get("risk_reasons", []))
        score = float(structural.get(item_id, {}).get("risk_score", 0))
        personal_item = personal[item_id]
        person_item = person[item_id]
        labels = original.get("labels", [])

        if labels and personal_item["audit_model_best_iou"] < .25:
            reasons.append("personal_model_disagrees_with_positive")
            score += 8 + 4 * personal_item["audit_model_max_conf"]
        if not labels and personal_item["audit_model_max_conf"] >= .25:
            reasons.append("personal_model_finds_danya_in_negative")
            score += 6 + 4 * personal_item["audit_model_max_conf"]
        if labels and person_item["audit_model_best_iou"] < .25:
            reasons.append("person_detector_disagrees_with_positive")
            score += 8 + 4 * person_item["audit_model_max_conf"]

        if not reasons:
            continue
        # Preserve order while removing duplicate signals.
        reasons = list(dict.fromkeys(reasons))
        combined.append({
            **original,
            "risk_score": round(score, 3),
            "risk_reasons": reasons,
            "personal_model_max_conf": personal_item["audit_model_max_conf"],
            "personal_model_best_iou": personal_item["audit_model_best_iou"],
            "person_model_max_conf": person_item["audit_model_max_conf"],
            "person_model_best_iou": person_item["audit_model_best_iou"],
        })

    combined.sort(key=lambda item: (-item["risk_score"], item["source"], item["time_s"]))
    output = {
        "schema_version": 1,
        "dataset": all_payload.get("dataset"),
        "total": len(combined),
        "reviewed_excluded": len(reviewed),
        "reason_counts": dict(Counter(reason for item in combined for reason in item["risk_reasons"])),
        "items": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "items"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
