#!/usr/bin/env python3
"""Extract cached person-formation features around rally-state moments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


OFFSETS = (-3, -2, -1, 0, 1, 2)


def frame_vector(rows: list[tuple[float, float, float, float]],
                 width: float, height: float) -> np.ndarray:
    if not rows:
        return np.zeros(23, dtype=np.float32)
    boxes = np.asarray(rows, dtype=np.float32)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
    centers_y = (boxes[:, 1] + boxes[:, 3]) / (2 * height)
    widths = (boxes[:, 2] - boxes[:, 0]) / width
    heights = (boxes[:, 3] - boxes[:, 1]) / height
    histogram, _, _ = np.histogram2d(
        centers_y, centers_x, bins=(3, 4), range=((0, 1), (0, 1)))
    histogram = histogram.ravel() / max(len(rows), 1)
    stats = np.asarray([
        min(len(rows), 20) / 20,
        centers_x.mean(), centers_x.std(),
        centers_y.mean(), centers_y.std(),
        *np.quantile(centers_x, (.25, .5, .75)),
        *np.quantile(centers_y, (.25, .5, .75)),
    ], dtype=np.float32)
    return np.concatenate((histogram.astype(np.float32), stats))


def source_features(items: list[dict], tracks_path: Path) -> dict[str, np.ndarray]:
    capture = cv2.VideoCapture(items[0]["video"])
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    requested = {}
    for item in items:
        frames = [max(0, round((float(item["time_s"]) + offset) * fps))
                  for offset in OFFSETS]
        requested[item["id"]] = frames
    wanted = {frame + delta for frames in requested.values()
              for frame in frames for delta in (-1, 0, 1)}
    detections: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with tracks_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame in wanted and float(row["confidence"]) >= .25:
                detections[frame].append(tuple(float(row[key]) for key in
                                               ("x1", "y1", "x2", "y2")))
    result = {}
    for item_id, frames in requested.items():
        vectors = []
        for frame in frames:
            available = min((candidate for candidate in (frame, frame - 1, frame + 1)
                             if detections.get(candidate)),
                            key=lambda candidate: abs(candidate - frame), default=None)
            vectors.append(frame_vector(detections[available] if available is not None else [],
                                        width, height))
        differences = []
        for left, right in zip(vectors, vectors[1:]):
            differences.extend([
                np.abs(right[:12] - left[:12]).sum(),
                abs(float(right[12] - left[12])),
                abs(float(right[13] - left[13])),
                abs(float(right[15] - left[15])),
            ])
        result[item_id] = np.concatenate((*vectors, np.asarray(differences, dtype=np.float32)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path, nargs="?")
    parser.add_argument("--tracks", action="append", required=True,
                        help="SOURCE=relative/path/to/tracks.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"] if args.results else None
    mapping = {int(value.split("=", 1)[0]): Path(value.split("=", 1)[1])
               for value in args.tracks}
    by_source: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        if int(item["source"]) in mapping:
            by_source[int(item["source"])].append(item)
    vectors = {}
    for source, source_items in sorted(by_source.items()):
        vectors.update(source_features(source_items, mapping[source]))
        print(f"source={source} processed={len(source_items)}", flush=True)
    retained = [item for item in items if item["id"] in vectors]
    if not retained:
        raise RuntimeError("no items covered by tracks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack([vectors[item["id"]] for item in retained]),
        labels=np.asarray([answers[item["id"]]["decision"] if answers else ""
                           for item in retained]),
        sources=np.asarray([int(item["source"]) for item in retained]),
        item_ids=np.asarray([item["id"] for item in retained]),
    )
    print(json.dumps({"samples": len(retained), "sources": sorted(by_source),
                      "dimensions": len(vectors[retained[0]["id"]])}, indent=2))


if __name__ == "__main__":
    main()
