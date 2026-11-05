#!/usr/bin/env python3
"""Build a balanced exact-contact review queue from trajectory sanity samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sample_key(row: dict) -> tuple:
    return (
        int(row["source"]),
        int(row["label"]),
        round(float(row["time_s"]), 3),
        str(row.get("id") or ""),
    )


def select_rows(rows: list[dict], per_label_source: int) -> list[dict]:
    selected: list[dict] = []
    for source in sorted({int(row["source"]) for row in rows}):
        for label in (0, 1):
            group = sorted(
                (row for row in rows if int(row["source"]) == source and int(row["label"]) == label),
                key=lambda row: float(row["time_s"]),
            )
            count = min(per_label_source, len(group))
            if count == 1:
                indexes = [0]
            else:
                indexes = [round(index * (len(group) - 1) / (count - 1)) for index in range(count)]
            selected.extend(group[index] for index in indexes)
    return selected


def attach_results(rows: list[dict], report: dict, pass_name: str) -> list[dict]:
    if len(rows) != len(report["details"]):
        raise ValueError(f"{pass_name}: rows/report mismatch: {len(rows)} != {len(report['details'])}")
    merged = []
    for row, detail in zip(rows, report["details"]):
        item = dict(row)
        item["trajectory_pass"] = pass_name
        item["trajectory_predicted"] = bool(detail["predicted"])
        item["trajectory_signature"] = detail.get("signature")
        item["trajectory_error"] = (
            "false_negative" if int(row["label"]) == 1 and not detail["predicted"]
            else "false_positive" if int(row["label"]) == 0 and detail["predicted"]
            else "control"
        )
        merged.append(item)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("v1_report", type=Path)
    parser.add_argument("v2_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_rows = json.loads(args.metadata.read_text(encoding="utf-8"))["samples"]
    first = select_rows(all_rows, 4)
    first_keys = {sample_key(row) for row in first}
    second = select_rows([row for row in all_rows if sample_key(row) not in first_keys], 4)
    candidates = attach_results(
        first, json.loads(args.v1_report.read_text(encoding="utf-8")), "v1"
    ) + attach_results(
        second, json.loads(args.v2_report.read_text(encoding="utf-8")), "v2"
    )

    queue: list[dict] = []
    for source in sorted({int(row["source"]) for row in candidates}):
        source_rows = [row for row in candidates if int(row["source"]) == source]
        positives = [row for row in source_rows if int(row["label"]) == 1]
        negatives = [row for row in source_rows if int(row["label"]) == 0]
        positives.sort(key=lambda row: (row["trajectory_error"] != "false_negative", float(row["time_s"])))
        negatives.sort(key=lambda row: (row["trajectory_error"] != "false_positive", float(row["time_s"])))
        queue.extend(positives[:6] + negatives[:2])

    queue.sort(key=lambda row: (int(row["source"]), int(row["label"]) == 0, float(row["time_s"])))
    items = []
    for index, row in enumerate(queue):
        items.append({
            "id": f"contact-{index:03d}-s{int(row['source'])}-t{float(row['time_s']):.3f}",
            "source": int(row["source"]),
            "video": row["video"],
            "approx_time_s": round(float(row["time_s"]), 3),
            "window_start_s": round(max(0.0, float(row["time_s"]) - 1.5), 3),
            "window_end_s": round(float(row["time_s"]) + 1.5, 3),
            "prior_label": int(row["label"]),
            "action": row.get("action"),
            "event_id": row.get("id"),
            "selection_reason": row["trajectory_error"],
            "trajectory_pass": row["trajectory_pass"],
        })

    counts = {
        "items": len(items),
        "positives": sum(item["prior_label"] == 1 for item in items),
        "negatives": sum(item["prior_label"] == 0 for item in items),
        "false_negatives": sum(item["selection_reason"] == "false_negative" for item in items),
        "false_positives": sum(item["selection_reason"] == "false_positive" for item in items),
        "sources": len({item["source"] for item in items}),
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
