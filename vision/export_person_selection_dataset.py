#!/usr/bin/env python3
"""Export human-selected person candidates as a clean one-class YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--val-tail-fraction", type=float, default=0.0,
        help=("Put the final temporal fraction of every source into validation. "
              "This creates a deterministic contiguous holdout and avoids adjacent "
              "random-frame leakage."),
    )
    args = parser.parse_args()
    if not 0 <= args.val_tail_fraction < 1:
        raise SystemExit("--val-tail-fraction must be in [0, 1)")
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    decisions = []
    for result_path in args.results:
        decisions.extend(json.loads(result_path.read_text())["decisions"].values())
    eligible = [decision for decision in decisions if decision["verdict"] not in {
        "unusable", "occluded", "missing_box", "uncertain"
    }]
    validation_keys: set[tuple[int, int]] = set()
    if args.val_tail_fraction:
        by_source: dict[int, list[dict]] = {}
        for decision in eligible:
            item = decision["item"]
            by_source.setdefault(int(item["source"]), []).append(decision)
        for source, source_decisions in by_source.items():
            source_decisions.sort(key=lambda row: (
                float(row["item"]["time_s"]), int(row["item"]["frame"])))
            count = max(1, round(len(source_decisions) * args.val_tail_fraction))
            for decision in source_decisions[-count:]:
                validation_keys.add((source, int(decision["item"]["frame"])))
    manifest = []
    for decision in decisions:
        verdict = decision["verdict"]
        # Occluded/unusable frames and visible frames without a human box are
        # ambiguous for one-class detection, so keep them in the audit trail
        # but never silently turn them into negatives.
        if verdict in {"unusable", "occluded", "missing_box", "uncertain"}:
            continue
        if verdict not in {"selected_candidate", "corrected_box", "in_box",
                           "not_in_box", "not_visible"}:
            raise SystemExit(f"Unsupported verdict: {verdict}")
        item = decision["item"]
        source_image = Path(item["image"])
        key = (int(item["source"]), int(item["frame"]))
        split = "val" if key in validation_keys else item["split"]
        kind = ("background_negative"
                if verdict in {"not_in_box", "not_visible"} else "positive")
        stem = f"s{item['source']}-f{item['frame']}-human-{kind}"
        image = args.output / "images" / split / f"{stem}.jpg"
        label_path = args.output / "labels" / split / f"{stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image)

        box = None
        if kind == "positive":
            if verdict == "in_box":
                labels = item.get("labels") or []
                if not labels:
                    raise SystemExit(f"in_box decision has no proposed label: {source_image}")
                label = [float(value) for value in labels[0]]
            else:
                label = [float(value) for value in decision["correctedLabel"]]
            _, cx, cy, width, height = label
            epsilon = 1e-7
            x1, x2 = max(epsilon, cx - width / 2), min(1.0 - epsilon, cx + width / 2)
            y1, y2 = max(epsilon, cy - height / 2), min(1.0 - epsilon, cy + height / 2)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise SystemExit(f"Box outside image for {source_image}: {label}")
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            label = [0.0, cx, cy, width, height]
            label_path.write_text(" ".join(["0", *(f"{v:.8f}" for v in label[1:])]) + "\n")
            frame = cv2.imread(str(source_image))
            image_height, image_width = frame.shape[:2]
            box = [(cx - width / 2) * image_width, (cy - height / 2) * image_height,
                   (cx + width / 2) * image_width, (cy + height / 2) * image_height]
        else:
            label_path.write_text("")

        manifest.append({
            "source": item["source"], "split": split, "kind": kind,
            "domain": item.get("domain"), "video": None,
            "time_s": item["time_s"], "frame": item["frame"], "track_id": -1,
            "box": box, "image": str(image.relative_to(args.output)),
            "annotation": verdict,
        })

    (args.output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: danya\n")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "examples": len(manifest),
        "counts": dict(sorted(Counter(
            f"source_{r['source']}_{r['split']}_{r['kind']}" for r in manifest).items())),
        "kinds": dict(sorted(Counter(r["kind"] for r in manifest).items())),
        "domains": dict(sorted(Counter(r.get("domain") or "unknown" for r in manifest).items())),
        "annotations": dict(sorted(Counter(r["annotation"] for r in manifest).items())),
        "validation": {
            "strategy": ("per-source-contiguous-tail" if args.val_tail_fraction else "input"),
            "tail_fraction": args.val_tail_fraction,
            "examples": sum(row["split"] == "val" for row in manifest),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
