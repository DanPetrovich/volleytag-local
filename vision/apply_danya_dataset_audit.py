#!/usr/bin/env python3
"""Apply explicit human audit decisions to a cloned YOLO dataset.

Only examples explicitly judged unusable are removed.  The source dataset and
the original review JSON are never modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


KEEP = {"in_box"}
REMOVE = {"not_in_box", "bad_box", "occluded", "unusable"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", {})

    by_image = {row["image"]: row for row in manifest}
    verdict_counts: Counter[str] = Counter()
    removed_by_source: Counter[str] = Counter()
    remove_images: set[str] = set()
    unmatched: list[str] = []

    dataset_marker = "/dataset/"
    for decision in decisions.values():
        verdict = decision.get("verdict")
        verdict_counts[verdict] += 1
        item = decision.get("item", {})
        image = str(item.get("image", ""))
        if dataset_marker in image:
            relative = image.split(dataset_marker, 1)[-1]
        elif "/images/" in image:
            relative = "images/" + image.split("/images/", 1)[1]
        else:
            relative = image
        if relative not in by_image:
            unmatched.append(image)
            continue
        if verdict in REMOVE:
            remove_images.add(relative)
            row = by_image[relative]
            removed_by_source[f"source_{row['source']}_{row['split']}_{row['kind']}"] += 1
        elif verdict not in KEEP:
            raise SystemExit(f"Unsupported verdict {verdict!r} for {image}")

    if unmatched:
        raise SystemExit(f"{len(unmatched)} audited images are absent from manifest; first: {unmatched[0]}")

    for relative in sorted(remove_images):
        image_path = args.dataset / relative
        label_path = args.dataset / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        image_path.unlink(missing_ok=False)
        label_path.unlink(missing_ok=False)

    cleaned = [row for row in manifest if row["image"] not in remove_images]
    manifest_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(f"source_{r['source']}_{r['split']}_{r['kind']}" for r in cleaned)
    summary = {
        "examples": len(cleaned),
        "counts": dict(sorted(counts.items())),
        "kinds": dict(sorted(Counter(r["kind"] for r in cleaned).items())),
        "domains": dict(sorted(Counter(r.get("domain") or "unknown" for r in cleaned).items())),
        "audit": {
            "results": str(args.results),
            "decisions": len(decisions),
            "verdicts": dict(sorted(verdict_counts.items())),
            "removed": len(remove_images),
        },
    }
    (args.dataset / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = {
        "source_dataset": str(args.dataset),
        "results": str(args.results),
        "before": len(manifest),
        "after": len(cleaned),
        "removed": len(remove_images),
        "verdicts": dict(sorted(verdict_counts.items())),
        "removed_by_source_split_kind": dict(sorted(removed_by_source.items())),
        "removed_images": sorted(remove_images),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("before", "after", "removed", "verdicts", "removed_by_source_split_kind")}, indent=2))


if __name__ == "__main__":
    main()
