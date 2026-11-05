#!/usr/bin/env python3
"""Export reviewed ball crops as a leakage-safe one-class YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--correction-queue", type=Path, required=True)
    parser.add_argument("--correction-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_items = read(args.queue)["items"]
    base_decisions = read(args.results)["decisions"]
    correction_items = read(args.correction_queue)["items"]
    correction_decisions = read(args.correction_results)["decisions"]
    superseded = {item["original_item_id"] for item in correction_items}
    rows = [(item, base_decisions[item["id"]]) for item in base_items if item["id"] not in superseded]
    rows += [(item, correction_decisions[item["id"]]) for item in correction_items]
    if len(rows) != len(base_items):
        raise SystemExit(f"merged row mismatch: {len(rows)} != {len(base_items)}")

    by_source = defaultdict(set)
    for item, _ in rows:
        by_source[int(item["source"])].add(item["action_event_id"])
    validation_events, test_events = set(), set()
    for source, events in sorted(by_source.items()):
        ordered = sorted(events, key=lambda event: hashlib.sha1(f"ball-v1:{event}".encode()).hexdigest())
        count = max(2, round(len(ordered) * 0.15))
        test_events.update(ordered[:count])
        validation_events.update(ordered[count:2 * count])

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val", "test"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)
    manifest = []
    for item, decision in rows:
        split = ("test" if item["action_event_id"] in test_events else
                 "val" if item["action_event_id"] in validation_events else "train")
        source_image = Path(item["image"])
        image = cv2.imread(str(source_image))
        if image is None:
            raise SystemExit(f"cannot read {source_image}")
        height, width = image.shape[:2]
        name = f"s{item['source']}-{item['id']}.jpg"
        target_image = args.output / "images" / split / name
        target_label = args.output / "labels" / split / name.replace(".jpg", ".txt")
        shutil.copy2(source_image, target_image)
        label = decision.get("correctedLabel")
        if label:
            if len(label) != 5 or any(not 0 <= float(value) <= 1 for value in label[1:]):
                raise SystemExit(f"invalid label for {item['id']}: {label}")
            _, cx, cy, box_width, box_height = map(float, label)
            if box_width * width < 2 or box_height * height < 2:
                raise SystemExit(f"tiny label for {item['id']}: {label}")
            target_label.write_text(f"0 {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}\n")
        else:
            target_label.write_text("")
        manifest.append({
            "id": item["id"], "original_item_id": item.get("original_item_id", item["id"]),
            "event_id": item["action_event_id"], "source": int(item["source"]),
            "time_s": float(item["time_s"]), "offset_s": float(item["sample_offset_s"]),
            "action": item["action"], "split": split, "positive": bool(label),
            "image": str(target_image), "label": str(target_label),
            "image_size": [width, height], "verdict": decision.get("verdict"),
            "corrected_person_crop": bool(item.get("original_item_id")),
        })

    (args.output / "dataset.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: volleyball\n",
        encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "items": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary = {
        "images": len(manifest), "labels": len(manifest),
        "positive_images": sum(row["positive"] for row in manifest),
        "negative_images": sum(not row["positive"] for row in manifest),
        "train_images": sum(row["split"] == "train" for row in manifest),
        "val_images": sum(row["split"] == "val" for row in manifest),
        "test_images": sum(row["split"] == "test" for row in manifest),
        "train_events": len({row["event_id"] for row in manifest if row["split"] == "train"}),
        "val_events": len({row["event_id"] for row in manifest if row["split"] == "val"}),
        "test_events": len({row["event_id"] for row in manifest if row["split"] == "test"}),
        "event_split_overlap": sum(len(left & right) for left, right in (
            ({row["event_id"] for row in manifest if row["split"] == "train"},
             {row["event_id"] for row in manifest if row["split"] == "val"}),
            ({row["event_id"] for row in manifest if row["split"] == "train"},
             {row["event_id"] for row in manifest if row["split"] == "test"}),
            ({row["event_id"] for row in manifest if row["split"] == "val"},
             {row["event_id"] for row in manifest if row["split"] == "test"}),
        )),
        "by_split_positive": dict(Counter(f"{row['split']}:{row['positive']}" for row in manifest)),
        "by_source": dict(Counter(str(row["source"]) for row in manifest)),
        "corrected_person_crops": sum(row["corrected_person_crop"] for row in manifest),
        "frozen_tests_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
