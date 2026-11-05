#!/usr/bin/env python3
"""Evaluate cached candidate ranking on sparse frozen ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_fused_detectors import iou, keyframes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=.30)
    parser.add_argument("--max-time-delta", type=float, default=.25)
    args = parser.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    times = np.asarray(cache["times"], dtype=float)
    candidate_sets = cache["candidate_sets"]
    samples = keyframes(json.loads(args.ground_truth.read_text(encoding="utf-8")))
    details = []
    for sample in samples:
        time_s = float(sample["time_s"])
        index = int(np.argmin(np.abs(times - time_s)))
        candidates = candidate_sets[index] if abs(times[index] - time_s) <= args.max_time_delta else []
        ranked = sorted(candidates, key=lambda row: float(row.get("reranker_probability", -1.0)),
                        reverse=True)
        gt = [float(value) for value in sample["box"]]
        candidate_hit = any(iou(gt, row["box"]) >= args.iou for row in candidates)
        top1_hit = bool(ranked and iou(gt, ranked[0]["box"]) >= args.iou)
        details.append({"time_s": time_s, "cache_time_s": float(times[index]),
                        "candidate_count": len(candidates), "candidate_hit": candidate_hit,
                        "top1_hit": top1_hit,
                        "top1_probability": float(ranked[0].get("reranker_probability", 0.0)) if ranked else 0.0,
                        "top1_source": ranked[0].get("source") if ranked else None})
    total = len(details)
    detected = sum(row["candidate_hit"] for row in details)
    top1 = sum(row["top1_hit"] for row in details)
    metrics = {"keyframes": total,
               "candidate_recall": detected / total if total else 0.0,
               "reranker_top1_recall": top1 / total if total else 0.0,
               "reranker_top1_given_candidate": top1 / detected if detected else 0.0,
               "iou_threshold": args.iou, "max_time_delta_s": args.max_time_delta}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
