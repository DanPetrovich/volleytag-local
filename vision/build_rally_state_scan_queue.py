#!/usr/bin/env python3
"""Build a time-ordered visual scan queue from top audio candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--rate", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.candidates.read_text())
    keep = round(float(payload["duration_s"]) / 60 * args.rate)
    rows = sorted(payload["candidates"], key=lambda row: row["score"], reverse=True)[:keep]
    items = []
    for row in sorted(rows, key=lambda row: row["time_s"]):
        timestamp = float(row["time_s"])
        items.append({
            "id": f"rally-scan-s{args.source}-t{timestamp:.3f}",
            "source": args.source,
            "video": payload["video"],
            "time_s": timestamp,
            "audio_score": float(row["score"]),
        })
    result = {
        "schema_version": 1,
        "kind": "volleytag.rally-state-scan-queue",
        "frozen_tests_used": False,
        "source": args.source,
        "duration_s": float(payload["duration_s"]),
        "baseline_candidates_per_minute": args.rate,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source": args.source, "items": len(items),
                      "duration_minutes": result["duration_s"] / 60}, indent=2))


if __name__ == "__main__":
    main()
