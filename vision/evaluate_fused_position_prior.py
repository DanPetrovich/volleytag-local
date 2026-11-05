#!/usr/bin/env python3
"""Evaluate a soft court-zone prior on fused generic/personal candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from evaluate_fused_detectors import iou


ADJACENT = {
    1: {2, 6}, 2: {1, 3}, 3: {2, 4, 6},
    4: {3, 5}, 5: {4, 6}, 6: {1, 3, 5},
}


def expected_zones(interval: dict) -> set[int]:
    result: set[int] = set()
    for key in ("pre_serve", "playing_base"):
        if isinstance(interval.get(key), list):
            result.update(map(int, interval[key]))
    return result


def zone_for_box(box: list[float], polygons: dict[int, np.ndarray]) -> int | None:
    point = ((box[0] + box[2]) / 2.0, box[3])
    for zone, polygon in polygons.items():
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return zone
    return None


def logit(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return math.log(score / (1.0 - score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("rotation", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--expected-bonus", type=float, default=0.20)
    parser.add_argument("--adjacent-bonus", type=float, default=0.08)
    args = parser.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    rotation = json.loads(args.rotation.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    cycle_end = float(rotation["first_full_cycle_at_s"])
    intervals = [row for row in rotation["intervals"]
                 if float(row["start_s"]) < cycle_end]
    polygons = {int(zone): np.asarray(points, dtype=np.float32)
                for zone, points in calibration["zones"].items()}
    rows = [row for row in cache["rows"]
            if int(row["source"]) == args.source and float(row["time_s"]) < cycle_end]
    details = []
    for row in rows:
        time_s = float(row["time_s"])
        interval = next(item for item in reversed(intervals)
                        if float(item["start_s"]) <= time_s)
        expected = expected_zones(interval)
        adjacent = set().union(*(ADJACENT[zone] for zone in expected)) - expected
        candidates = row["candidates"]
        direct_indexes = [index for index, candidate in enumerate(candidates)
                          if float(candidate.get("direct_confidence", 0.0)) > 0.0]
        pool = direct_indexes or list(range(len(candidates)))
        enriched = []
        for index, candidate in enumerate(candidates):
            zone = zone_for_box(candidate["box"], polygons)
            bonus = (args.expected_bonus if zone in expected else
                     args.adjacent_bonus if zone in adjacent else 0.0)
            score = (float(candidate.get("direct_confidence", 0.0))
                     if direct_indexes else float(candidate.get("confidence", 0.0)))
            enriched.append({**candidate, "zone": zone, "bonus": bonus,
                             "identity_score": score,
                             "adjusted_logit": logit(score) + bonus,
                             "overlap": iou(row["gt_box"], candidate["box"])})
        baseline_index = max(pool, key=lambda index: enriched[index]["identity_score"],
                             default=-1)
        prior_index = max(pool, key=lambda index: enriched[index]["adjusted_logit"],
                          default=-1)
        candidate_hit = any(item["overlap"] >= args.iou for item in enriched)
        baseline_hit = baseline_index >= 0 and enriched[baseline_index]["overlap"] >= args.iou
        prior_hit = prior_index >= 0 and enriched[prior_index]["overlap"] >= args.iou
        details.append({
            "time_s": time_s, "rotation": interval["rotation"],
            "serve_owner": interval["serve_owner"],
            "expected_zones": sorted(expected), "adjacent_zones": sorted(adjacent),
            "candidate_count": len(candidates), "direct_candidate_count": len(direct_indexes),
            "candidate_hit": candidate_hit,
            "baseline_top1_hit": bool(baseline_hit),
            "prior_top1_hit": bool(prior_hit),
            "changed_selection": baseline_index != prior_index,
            "recovered": bool(prior_hit and not baseline_hit),
            "lost": bool(baseline_hit and not prior_hit),
            "baseline_index": baseline_index, "prior_index": prior_index,
            "candidates": enriched,
        })

    total = len(details)
    baseline_hits = sum(row["baseline_top1_hit"] for row in details)
    prior_hits = sum(row["prior_top1_hit"] for row in details)
    recovered = sum(row["recovered"] for row in details)
    lost = sum(row["lost"] for row in details)
    metrics = {
        "source": args.source,
        "samples": total,
        "candidate_recall": sum(row["candidate_hit"] for row in details) / total,
        "baseline_top1_recall": baseline_hits / total,
        "position_prior_top1_recall": prior_hits / total,
        "absolute_gain": (prior_hits - baseline_hits) / total,
        "changed_selections": sum(row["changed_selection"] for row in details),
        "recovered_baseline_errors": recovered,
        "introduced_errors": lost,
        "frames_without_personal_candidate": sum(
            row["direct_candidate_count"] == 0 for row in details),
        "passed": prior_hits >= baseline_hits and recovered >= 2,
        "protocol": {"iou": args.iou,
                     "expected_zone_logit_bonus": args.expected_bonus,
                     "adjacent_zone_logit_bonus": args.adjacent_bonus,
                     "hard_filter": False},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fused-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "fused-details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
