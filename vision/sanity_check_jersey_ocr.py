#!/usr/bin/env python3
"""Small OCR sanity check for jersey number 51 on cached candidate crops."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def recognize(path: Path) -> str:
    image = cv2.imread(str(path))
    if image is None:
        return ""
    # Jersey number normally occupies the torso; excluding legs/background
    # gives generic OCR substantially less clutter.
    height = image.shape[0]
    torso = image[max(0, int(.08 * height)):max(1, int(.72 * height))]
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        cv2.imwrite(handle.name, gray)
        result = subprocess.run([
            "tesseract", handle.name, "stdout", "--psm", "11",
            "-c", "tessedit_char_whitelist=0123456789",
        ], capture_output=True, text=True, check=False)
    return "".join(character for character in result.stdout if character.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class-domain", type=int, default=40)
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(51)
    selected = []
    for domain in ("near_back", "far_front"):
        for label in (0, 1):
            pool = [row for row in rows if row["domain"] == domain and int(row["label"]) == label]
            count = min(args.per_class_domain, len(pool))
            selected.extend(pool[index] for index in rng.choice(len(pool), count, replace=False))
    details = []
    for index, row in enumerate(selected, 1):
        text = recognize(Path(row["file"]))
        details.append({"file": row["file"], "label": int(row["label"]),
                        "domain": row["domain"], "digits": text, "has_51": "51" in text})
        if index % 20 == 0:
            print(f"ocr={index}/{len(selected)}", flush=True)
    positives = [row for row in details if row["label"] == 1]
    negatives = [row for row in details if row["label"] == 0]
    tp = sum(row["has_51"] for row in positives)
    fp = sum(row["has_51"] for row in negatives)
    report = {
        "samples": len(details), "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "number_51_recall": tp / len(positives) if positives else 0.0,
        "number_51_false_positive_rate": fp / len(negatives) if negatives else 0.0,
        "passed": (tp / len(positives) if positives else 0.0) >= .30 and fp == 0,
        "success_criterion": "recall>=0.30 and zero false positives on the sample",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
