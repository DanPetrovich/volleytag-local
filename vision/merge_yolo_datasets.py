#!/usr/bin/env python3
"""Merge YOLO datasets with prefixed names and space-efficient hard links."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def normalized_labels(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    output = []
    epsilon = 1e-8
    for line in source.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 5 or int(values[0]) != 0:
            raise SystemExit(f"Invalid YOLO label: {source}: {line}")
        _, cx, cy, width, height = values
        x1, x2 = max(epsilon, cx - width / 2), min(1 - epsilon, cx + width / 2)
        y1, y2 = max(epsilon, cy - height / 2), min(1 - epsilon, cy + height / 2)
        if x2 <= x1 or y2 <= y1:
            raise SystemExit(f"Empty YOLO box: {source}: {line}")
        clean = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        output.append(" ".join(["0", *(f"{value:.10f}" for value in clean)]))
    target.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if len(args.dataset) != len(args.name):
        raise SystemExit("Each --dataset needs one --name")
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    manifest = []
    for root, name in zip(args.dataset, args.name):
        rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            split = row["split"]
            source_image = root / row["image"]
            source_label = root / "labels" / split / f"{source_image.stem}.txt"
            if not source_image.exists() or not source_label.exists():
                raise SystemExit(f"Missing pair: {source_image} / {source_label}")
            stem = f"{name}-{index:05d}-{source_image.stem}"
            target_image = args.output / "images" / split / f"{stem}.jpg"
            target_label = args.output / "labels" / split / f"{stem}.txt"
            link_or_copy(source_image, target_image)
            normalized_labels(source_label, target_label)
            manifest.append({
                **row, "image": str(target_image.relative_to(args.output)),
                "origin_dataset": name, "origin_image": row["image"],
            })

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: danya\n",
        encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "examples": len(manifest),
        "train": sum(row["split"] == "train" for row in manifest),
        "val": sum(row["split"] == "val" for row in manifest),
        "origins": dict(sorted(Counter(row["origin_dataset"] for row in manifest).items())),
        "origin_splits": dict(sorted(Counter(
            f"{row['origin_dataset']}_{row['split']}" for row in manifest).items())),
        "kinds": dict(sorted(Counter(row["kind"] for row in manifest).items())),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
