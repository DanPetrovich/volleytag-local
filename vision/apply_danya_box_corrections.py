#!/usr/bin/env python3
"""Restore explicitly corrected positive boxes into a cleaned YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads((args.source_dataset / "manifest.json").read_text())
    target_manifest = json.loads((args.target_dataset / "manifest.json").read_text())
    source_by_image = {row["image"]: row for row in source_manifest}
    target_by_image = {row["image"]: row for row in target_manifest}
    decisions = json.loads(args.results.read_text())["decisions"]
    restored = []

    for decision in decisions.values():
        if decision.get("verdict") != "corrected_box":
            continue
        item = decision["item"]
        marker = "/dataset/"
        relative = item["image"].split(marker, 1)[-1]
        row = dict(source_by_image[relative])
        label = [float(value) for value in decision["correctedLabel"]]
        _, cx, cy, bw, bh = label
        if not (0 <= cx - bw / 2 < cx + bw / 2 <= 1 and
                0 <= cy - bh / 2 < cy + bh / 2 <= 1):
            raise SystemExit(f"Corrected box outside image: {relative}: {label}")

        old_label = item["labels"][0]
        old_box = row["box"]
        frame_width = (old_box[2] - old_box[0]) / float(old_label[3])
        frame_height = (old_box[3] - old_box[1]) / float(old_label[4])
        row["box"] = [
            (cx - bw / 2) * frame_width,
            (cy - bh / 2) * frame_height,
            (cx + bw / 2) * frame_width,
            (cy + bh / 2) * frame_height,
        ]

        source_image = args.source_dataset / relative
        target_image = args.target_dataset / relative
        target_label = args.target_dataset / "labels" / row["split"] / f"{target_image.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
        target_label.write_text(" ".join(["0", *(f"{v:.8f}" for v in label[1:])]) + "\n")
        target_by_image[relative] = row
        restored.append(relative)

    cleaned = sorted(target_by_image.values(), key=lambda row: (row["source"], row["frame"], row["image"]))
    (args.target_dataset / "manifest.json").write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n")
    counts = Counter(f"source_{r['source']}_{r['split']}_{r['kind']}" for r in cleaned)
    summary = {
        "examples": len(cleaned),
        "counts": dict(sorted(counts.items())),
        "kinds": dict(sorted(Counter(r["kind"] for r in cleaned).items())),
        "domains": dict(sorted(Counter(r.get("domain") or "unknown" for r in cleaned).items())),
        "box_corrections": {"results": str(args.results), "restored": len(restored)},
    }
    (args.target_dataset / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = {"restored": len(restored), "images": restored, "examples_after": len(cleaned)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"restored": len(restored), "examples_after": len(cleaned)}, indent=2))


if __name__ == "__main__":
    main()
