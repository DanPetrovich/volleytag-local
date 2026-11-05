#!/usr/bin/env python3
"""Render manual boxes at their true video timestamps for visual validation."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source, spec in enumerate(args.specs):
        payload = json.loads(spec.read_text(encoding="utf-8"))
        video = Path(payload["video"])
        with Path(payload["tracks"]).open(newline="", encoding="utf-8") as handle:
            track_rows = list(csv.DictReader(handle))
        roi_x, roi_y = payload.get("roi", [0, 0])
        cap = cv2.VideoCapture(str(video))
        tiles = []
        for index, box in enumerate(payload.get("manualBoxes", [])):
            reference = min(track_rows, key=lambda row: abs(float(row["time_s"]) - float(box["time"])))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(reference["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = [int(round(float(box[key]))) for key in ("x1", "y1", "x2", "y2")]
            x1 += int(roi_x); x2 += int(roi_x); y1 += int(roi_y); y2 += int(roi_y)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 255, 60), 5)
            frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (480, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"manual {index+1} | {box['time']:.2f}s", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
            tiles.append(frame)
        cap.release()
        while len(tiles) % 4:
            tiles.append(np.zeros((270, 480, 3), dtype=np.uint8))
        if tiles:
            sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(str(args.output / f"source-{source}-manual-boxes.jpg"), sheet)
        print(source, video, len(tiles))


if __name__ == "__main__":
    main()
