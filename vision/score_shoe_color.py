#!/usr/bin/env python3
"""Measure a positive-only light-orange-shoe cue for tracked chunks."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from build_feedback_profile import split_chunks
from identify_player import Observation, read_tracks


def orange_shoe_fraction(frame: np.ndarray, item: Observation,
                         roi: tuple[int, int, int, int]) -> tuple[float, int]:
    left, top = roi[:2]
    x1, y1, x2, y2 = item.box
    width, height = x2 - x1, y2 - y1
    ax1 = max(0, round(left + x1 - width * .08))
    ax2 = min(frame.shape[1], round(left + x2 + width * .08))
    ay1 = max(0, round(top + y1 + height * .64))
    ay2 = min(frame.shape[0], round(top + y2 + height * .04))
    crop = frame[ay1:ay2, ax1:ax2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 3:
        return 0.0, 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # In this hall the wooden floor is a dull yellow-orange (usually H>=24 or
    # low saturation), while Dan's shoes are a lighter salmon-orange. Requiring
    # both the narrower hue band and saturation prevents the floor from making
    # every player look like a match.
    orange = (hue >= 14) & (hue <= 23) & (saturation >= 120) & (value >= 130)
    return float(np.mean(orange)), int(np.count_nonzero(orange))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vision/runs/shoe-scores.csv"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    roi = tuple(int(value) for value in args.roi.split(","))
    chunks = split_chunks(read_tracks(args.tracks))

    wanted: dict[int, list[tuple[str, Observation]]] = defaultdict(list)
    for key, items in chunks.items():
        chosen: list[Observation] = []
        for item in sorted(items, key=lambda value: value.quality, reverse=True):
            if all(abs(item.time_s - other.time_s) >= .12 for other in chosen):
                chosen.append(item)
            if len(chosen) >= args.samples:
                break
        for item in chosen:
            wanted[item.frame].append((key, item))

    values: dict[str, list[tuple[float, int]]] = defaultdict(list)
    cap = cv2.VideoCapture(str(args.video))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for key, item in wanted.get(frame_index, []):
            values[key].append(orange_shoe_fraction(frame, item, roi))
        frame_index += 1
    cap.release()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, items in chunks.items():
        samples = values.get(key, [])
        fractions = [fraction for fraction, _ in samples]
        counts = [count for _, count in samples]
        score = float(np.quantile(fractions, .75)) if fractions else 0.0
        peak = max(fractions, default=0.0)
        pixel_peak = max(counts, default=0)
        # A few orange pixels are common in the wooden floor. Require the cue to
        # occupy a meaningful share across more than a single lucky sample.
        reliable = int(pixel_peak >= 12 and score >= .035 and peak >= .045)
        rows.append((key, items[0].track_id, items[0].time_s, items[-1].time_s,
                     score, peak, pixel_peak, reliable, len(samples)))
    rows.sort(key=lambda row: row[4], reverse=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "track_id", "start_s", "end_s", "orange_score",
                         "orange_peak", "orange_pixel_peak", "reliable", "samples"))
        for row in rows:
            writer.writerow((row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}",
                             f"{row[4]:.6f}", f"{row[5]:.6f}", *row[6:]))
    print(f"chunks={len(rows)} reliable_orange={sum(row[7] for row in rows)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
