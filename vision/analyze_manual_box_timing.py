#!/usr/bin/env python3
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path)
    args = parser.parse_args()
    for spec in args.specs:
        payload = json.loads(spec.read_text())
        manual = payload.get("manualBoxes", [])
        if not manual:
            continue
        rx, ry = payload.get("roi", [0, 0])
        with Path(payload["tracks"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        matches = []
        same_time_matches = []
        for box in manual:
            target = [box["x1"] + rx, box["y1"] + ry, box["x2"] + rx, box["y2"] + ry]
            best = (0.0, None)
            same_time_best = 0.0
            for row in rows:
                delta = float(row["time_s"]) - float(box["time"])
                if abs(delta) > 3:
                    continue
                candidate = [float(row["x1"]) + rx, float(row["y1"]) + ry,
                             float(row["x2"]) + rx, float(row["y2"]) + ry]
                overlap = iou(target, candidate)
                if abs(delta) <= .16:
                    same_time_best = max(same_time_best, overlap)
                if overlap > best[0]:
                    best = (overlap, delta)
            matches.append(best)
            same_time_matches.append(same_time_best)
        print(spec)
        print("count", len(matches), "iou>=.3", sum(score >= .3 for score, _ in matches),
              "median_iou", round(statistics.median(score for score, _ in matches), 3))
        print("same_time_iou>=.3", sum(score >= .3 for score in same_time_matches),
              "same_time_median", round(statistics.median(same_time_matches), 3))
        print("common_deltas", Counter(round(delta, 1) if delta is not None else None
                                       for _, delta in matches).most_common(12))


if __name__ == "__main__":
    main()
