#!/usr/bin/env python3
"""Find a jersey number inside tracked person boxes using local Tesseract OCR."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from identify_player import Observation, read_tracks


def number_crop(frame, item: Observation, roi: tuple[int, int, int, int]):
    rx1, ry1, rx2, ry2 = roi
    court = frame[ry1:ry2, rx1:rx2]
    x1, y1, x2, y2 = item.box
    person = court[y1:y2, x1:x2]
    if person.size == 0:
        return None
    height, width = person.shape[:2]
    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array((28, 45, 35)), np.array((95, 255, 255)))
    dark = cv2.inRange(hsv, np.array((0, 0, 0)), np.array((180, 255, 105)))
    if cv2.countNonZero(green) / green.size < .012 or cv2.countNonZero(dark) / dark.size < .08:
        return None
    # Back numbers sit in the middle of the upper body. A tight crop is much
    # more reliable than feeding the whole player and court background to OCR.
    crop = person[int(height * .19):int(height * .42),
                  int(width * .12):int(width * .88)]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


def recognize(image, temp_path: Path) -> str:
    cv2.imwrite(str(temp_path), image)
    result = subprocess.run(
        ["tesseract", str(temp_path), "stdout", "--psm", "7",
         "-c", "tessedit_char_whitelist=0123456789"],
        capture_output=True, text=True, check=False,
    )
    return "".join(character for character in result.stdout if character.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--target", default="51")
    parser.add_argument("--output", type=Path, default=Path("vision/runs/number-ocr"))
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--sample-seconds", type=float, default=.20)
    parser.add_argument("--smooth-seconds", type=float, default=.35)
    parser.add_argument("--min-height", type=int, default=135)
    parser.add_argument("--reuse-ocr", type=Path, help="Reuse a previous ocr.csv and only render")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    roi = tuple(int(value) for value in args.roi.split(","))
    grouped = read_tracks(args.tracks)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_frames = max(1, round(args.sample_seconds * fps))

    candidates: dict[int, list[Observation]] = defaultdict(list)
    observation_index = {}
    for items in grouped.values():
        for item in items:
            observation_index[(item.frame, item.track_id)] = item
            x1, y1, x2, y2 = item.box
            if y2 - y1 < args.min_height or y2 < 300:
                continue
            if item.frame % sample_frames <= 1:
                candidates[item.frame].append(item)

    evidence: list[tuple[Observation, str, str]] = []
    if args.reuse_ocr:
        with args.reuse_ocr.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item = observation_index.get((int(row["frame"]), int(row["track_id"])))
                if item:
                    evidence.append((item, row["ocr"], row["kind"]))
    else:
        with tempfile.TemporaryDirectory(prefix="volleytag-ocr-") as temp_dir:
            temp_path = Path(temp_dir) / "crop.png"
            for frame_index in sorted(candidates):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                for item in candidates[frame_index]:
                    crop = number_crop(frame, item, roi)
                    if crop is None:
                        continue
                    text = recognize(crop, temp_path)
                    if text:
                        evidence.append((item, text, "exact" if text == args.target else "other"))

    positives = [item for item, text, _ in evidence if text == args.target]
    hit_counts: dict[int, int] = defaultdict(int)
    for item in positives:
        hit_counts[item.track_id] += 1
    reliable_positives = [item for item in positives if hit_counts[item.track_id] >= 2]
    smooth_frames = round(args.smooth_seconds * fps)

    def confirmed(item: Observation) -> bool:
        return any(abs(item.frame - hit.frame) <= smooth_frames and item.track_id == hit.track_id
                   for hit in reliable_positives)

    with (args.output / "ocr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "frame", "track_id", "ocr", "kind"))
        for item, text, kind in evidence:
            writer.writerow((f"{item.time_s:.3f}", item.frame, item.track_id, text, kind))

    by_frame: dict[int, list[Observation]] = defaultdict(list)
    for items in grouped.values():
        for item in items:
            by_frame[item.frame].append(item)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    width, height = roi[2] - roi[0], roi[3] - roi[1]
    writer = cv2.VideoWriter(str(args.output / f"number-{args.target}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        court = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        for item in by_frame.get(frame_index, []):
            if not confirmed(item):
                continue
            x1, y1, x2, y2 = item.box
            cv2.rectangle(court, (x1, y1), (x2, y2), (20, 210, 255), 4)
            cv2.putText(court, f"jersey #{args.target}", (x1, max(28, y1 - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 210, 255), 2, cv2.LINE_AA)
        writer.write(court)
        frame_index += 1
    writer.release()
    cap.release()

    print(f"ocr_results={len(evidence)}")
    print(f"target_hits={len(positives)}")
    print(f"reliable_hits={len(reliable_positives)}")
    print("hits=" + ", ".join(f"{hit.time_s:.2f}s/id{hit.track_id}" for hit in positives))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
