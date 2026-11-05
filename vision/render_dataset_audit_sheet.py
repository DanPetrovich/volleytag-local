#!/usr/bin/env python3
"""Render reviewed dataset examples with their YOLO labels for QA."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions = list(json.loads(args.results.read_text())["decisions"].values())
    tiles = []
    for decision in decisions:
        item = decision["item"]
        image = cv2.imread(str(args.root / item["image"]))
        if image is None:
            continue
        h, w = image.shape[:2]
        for label in item.get("labels", []):
            _, cx, cy, bw, bh = label
            x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (70, 255, 60), 4)
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        caption = f"{decision['verdict']} | s{item['source']} {item['time_s']:.1f}s | {item['kind']}"
        cv2.rectangle(image, (0, 0), (480, 32), (0, 0, 0), -1)
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
        tiles.append(image)
    while len(tiles) % 3:
        tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)


if __name__ == "__main__":
    main()
