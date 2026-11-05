#!/usr/bin/env python3
"""Build evenly distributed frames with generic-person candidate boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO


def overlap_metrics(first: list[float], second: list[float]) -> tuple[float, float]:
    """Return IoU and intersection over the smaller box for YOLO labels."""
    def xyxy(label: list[float]) -> tuple[float, float, float, float]:
        _, cx, cy, width, height = label
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
    ax1, ay1, ax2, ay2 = xyxy(first)
    bx1, by1, bx2, by2 = xyxy(second)
    intersection = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    area_a, area_b = (ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return (intersection / union if union else 0.,
            intersection / min(area_a, area_b) if min(area_a, area_b) else 0.)


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """Keep one hypothesis per person, including strongly nested detections."""
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda value: value["confidence"], reverse=True):
        if any((lambda metrics: metrics[0] >= .45 or metrics[1] >= .78)(
                overlap_metrics(candidate["label"], previous["label"])) for previous in kept):
            continue
        kept.append(candidate)
    return kept


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["time_s"])
    if len(rows) <= count:
        return rows
    return [rows[round(index * (len(rows) - 1) / (count - 1))]
            for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, required=True)
    parser.add_argument("--exclude-results", type=Path, action="append", default=[],
                        help="Audit result files whose source images must not be selected again")
    parser.add_argument("--exclude-queue", type=Path, action="append", default=[],
                        help="Queue files whose source images must not be selected again")
    parser.add_argument("--conf", type=float, default=.10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    excluded_images: set[str] = set()
    for result_path in args.exclude_results:
        decisions = json.loads(result_path.read_text()).get("decisions", {}).values()
        for decision in decisions:
            image = str(decision.get("item", {}).get("image", ""))
            marker = "/dataset/"
            excluded_images.add(image.split(marker, 1)[-1] if marker in image else image)
    for queue_path in args.exclude_queue:
        for item in json.loads(queue_path.read_text()).get("items", []):
            image = str(item.get("image", ""))
            marker = "/dataset/"
            excluded_images.add(image.split(marker, 1)[-1] if marker in image else image)
    # Use only full-frame examples. Hard negatives are deliberately cropped.
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in manifest:
        if (row["kind"] in {"positive", "manual_positive"} and
                row["image"] not in excluded_images):
            grouped[int(row["source"])].append(row)
    selected = [row for source in sorted(grouped)
                for row in evenly_spaced(grouped[source], args.per_source)]

    model = YOLO(str(args.model))
    paths = [str(args.dataset / row["image"]) for row in selected]
    started = time.monotonic()
    results = model.predict(paths, imgsz=960, conf=args.conf, device=args.device,
                            batch=8, verbose=False, stream=True)
    items = []
    for row, result in zip(selected, results):
        height, width = result.orig_shape
        candidates = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                                  result.boxes.conf.cpu().tolist(),
                                                  result.boxes.cls.cpu().tolist()):
                if int(class_id) != 0:
                    continue
                x1, y1, x2, y2 = box
                candidates.append({
                    "label": [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                              (x2 - x1) / width, (y2 - y1) / height],
                    "confidence": round(float(confidence), 4),
                })
        candidates = deduplicate_candidates(candidates)
        items.append({
            "id": hashlib.sha1(row["image"].encode()).hexdigest()[:16],
            "image": str(args.dataset / row["image"]),
            "source": row["source"], "split": row["split"],
            "kind": "person_selection", "domain": row.get("domain") or "unknown",
            "time_s": row["time_s"], "frame": row["frame"],
            "track_id": row["track_id"], "labels": [], "candidates": candidates,
            "risk_reasons": ["new_dataset_anchor_frame"],
        })
    elapsed = time.monotonic() - started
    output = {
        "schema_version": 1, "dataset": str(args.dataset), "total": len(items),
        "mode": "person_selection", "model": str(args.model), "conf": args.conf,
        "elapsed_s": elapsed,
        "candidate_counts": dict(sorted(Counter(len(item["candidates"]) for item in items).items())),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"frames": len(items), "elapsed_s": round(elapsed, 2),
                      "fps": round(len(items) / elapsed, 2),
                      "with_candidates": sum(bool(item["candidates"]) for item in items),
                      "median_candidates": sorted(len(item["candidates"]) for item in items)[len(items)//2]}, indent=2))


if __name__ == "__main__":
    main()
