#!/usr/bin/env python3
"""Build dense referee-event windows around reviewed live rallies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_replays(path: Path, video: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if "sources" in payload:
        matches = [row for row in payload["sources"].values() if row["video"] == video]
        if len(matches) != 1:
            raise ValueError("could not resolve replay source")
        return matches[0]["exclude_intervals"]
    if payload["video"] != video:
        raise ValueError("replay metadata belongs to another video")
    return payload.get("excluded_intervals", [])


def clamp_to_replays(start_s: float, end_s: float, target_s: float,
                     replays: list[dict]) -> tuple[float, float] | None:
    if any(row["start_s"] <= target_s <= row["end_s"] for row in replays):
        return None
    before = [float(row["end_s"]) for row in replays
              if start_s < float(row["end_s"]) < target_s]
    after = [float(row["start_s"]) for row in replays
             if target_s < float(row["start_s"]) < end_s]
    if before:
        start_s = max(start_s, max(before))
    if after:
        end_s = min(end_s, min(after))
    return start_s, end_s


def make_item(source: dict, rally: dict, kind: str, ordinal: int,
              replays: list[dict]) -> dict | None:
    start, end = float(rally["start_s"]), float(rally["end_s"])
    if kind == "serve_search":
        target, window_start, window_end = start - 3, start - 8, start + 1
    elif kind == "end_search":
        target, window_start, window_end = end + 2, end - 1, end + 8
    elif kind == "active_negative":
        target = (start + end) / 2
        window_start, window_end = target - 1.5, target + 1.5
    else:
        raise ValueError(f"unknown kind: {kind}")
    clipped = clamp_to_replays(max(0.0, window_start), window_end, target, replays)
    if clipped is None:
        return None
    window_start, window_end = clipped
    return {
        "id": f"referee-dense-{source['name']}-{kind}-{ordinal:03d}",
        "source": int(source["source"]),
        "source_name": source["name"],
        "video": source["video"],
        "time_s": target,
        "window_start_s": window_start,
        "window_end_s": window_end,
        "roi_box_norm": source["roi_box_norm"],
        "selection_reason": kind,
        "rally_id": rally["id"],
        "requires_exact_time": kind != "active_negative",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    items = []
    for source in config["sources"]:
        results = json.loads(Path(source["results"]).read_text())
        if results.get("frozen_tests_used") or results.get("split") == "frozen_eval":
            raise ValueError(f"frozen source is forbidden: {source['name']}")
        replays = load_replays(Path(source["replays"]), source["video"])
        for index, rally in enumerate(results["rallies"], 1):
            for kind in ("serve_search", "end_search"):
                row = make_item(source, rally, kind, index, replays)
                if row:
                    items.append(row)
            if index % 2:
                row = make_item(source, rally, "active_negative", index, replays)
                if row:
                    items.append(row)
    items.sort(key=lambda row: (row["source_name"], row["window_start_s"]))
    payload = {
        "schema_version": 1,
        "kind": "volleytag.referee-gesture-review-queue",
        "frozen_tests_used": False,
        "labels": ["serve_authorization", "rally_end_team_to_serve", "other_signal",
                   "no_signal", "uncertain"],
        "selection": {
            "method": "reviewed-rally temporal windows; no motion preselection",
            "exact_event_time_required": True,
            "negative_sampling": "middle of every other live rally",
        },
        "counts": {kind: sum(row["selection_reason"] == kind for row in items)
                   for kind in ("serve_search", "end_search", "active_negative")},
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"items": len(items), **payload["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
