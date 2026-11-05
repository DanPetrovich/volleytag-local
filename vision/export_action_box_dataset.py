#!/usr/bin/env python3
"""Export reviewed action-contact boxes as a leakage-safe YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


VISIBLE = {"in_box", "selected_candidate", "corrected_box"}
NEGATIVE = {"not_visible", "not_in_box"}
EXCLUDED = {"occluded", "uncertain", "unusable"}


def checked_label(decision: dict, item: dict) -> list[float] | None:
    verdict = decision["verdict"]
    if verdict in {"selected_candidate", "corrected_box"}:
        return [float(value) for value in decision["correctedLabel"]]
    if verdict == "in_box":
        labels = item.get("labels", [])
        if len(labels) != 1:
            raise SystemExit(f"Confirmed item must contain one label: {item['id']}")
        return [float(value) for value in labels[0]]
    return None


def valid_label(label: list[float]) -> bool:
    if len(label) != 5 or int(label[0]) != 0:
        return False
    _, cx, cy, width, height = label
    return (width > 0 and height > 0 and
            0 <= cx - width / 2 < cx + width / 2 <= 1 and
            0 <= cy - height / 2 < cy + height / 2 <= 1)


def clamp_label(label: list[float]) -> list[float]:
    """Clamp a human box to the image after fixed-precision serialization."""
    _, cx, cy, width, height = label
    # Leave a tiny serialization margin: rounding cx/size separately can make
    # an exact 1.0 edge become 1.00000000005 when parsed again.
    epsilon = 1e-8
    x1, x2 = max(epsilon, cx - width / 2), min(1.0 - epsilon, cx + width / 2)
    y1, y2 = max(epsilon, cy - height / 2), min(1.0 - epsilon, cy + height / 2)
    return [0.0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, action="append", required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if len(args.queue) != len(args.results):
        raise SystemExit("Each --queue needs a matching --results")
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    reviewed = []
    seen_frames: set[tuple[str, int]] = set()
    for queue_path, result_path in zip(args.queue, args.results):
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in queue.get("items", [])}
        decisions = json.loads(result_path.read_text(encoding="utf-8")).get("decisions", {})
        for item_id, decision in decisions.items():
            item = decision.get("item") or by_id.get(item_id)
            if item is None:
                raise SystemExit(f"Decision is not present in its queue: {item_id}")
            verdict = decision.get("verdict")
            if verdict not in VISIBLE | NEGATIVE | EXCLUDED:
                raise SystemExit(f"Unsupported verdict {verdict}: {item_id}")
            # The frame index was not stored by the first action queue. Time at
            # millisecond precision is stable and sufficient for deduplication.
            frame_key = (str(item.get("video", "")), round(float(item["time_s"]) * 1000))
            if frame_key in seen_frames:
                continue
            seen_frames.add(frame_key)
            reviewed.append((item, decision))

    # Split whole action events, not their -0.3/+0.3 neighbouring frames.
    events_by_source: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for item, decision in reviewed:
        if decision["verdict"] in EXCLUDED:
            continue
        event_id = str(item.get("action_event_id") or item["id"])
        pair = (float(item.get("action_time_s", item["time_s"])), event_id)
        if pair not in events_by_source[int(item["source"])]:
            events_by_source[int(item["source"])].append(pair)
    validation_events = set()
    for source, values in events_by_source.items():
        ordered = sorted(set(values))
        # Deterministic 20% validation spread across each source timeline.
        validation_events.update(event_id for index, (_time, event_id) in enumerate(ordered)
                                 if index % 5 == 4)

    manifest = []
    excluded = Counter()
    for item, decision in reviewed:
        verdict = decision["verdict"]
        if verdict in EXCLUDED:
            excluded[verdict] += 1
            continue
        event_id = str(item.get("action_event_id") or item["id"])
        split = "val" if event_id in validation_events else "train"
        kind = "positive" if verdict in VISIBLE else "background_negative"
        source_image = Path(item["image"])
        if not source_image.exists() or source_image.stat().st_size == 0:
            raise SystemExit(f"Missing source image: {source_image}")
        digest = hashlib.sha1(
            f"{item.get('video')}:{float(item['time_s']):.3f}:{event_id}".encode()
        ).hexdigest()[:12]
        stem = f"s{int(item['source'])}-t{float(item['time_s']):.3f}-{digest}"
        image_path = args.output / "images" / split / f"{stem}.jpg"
        label_path = args.output / "labels" / split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_path)

        label = checked_label(decision, item)
        if label is not None:
            label = clamp_label(label)
            if not valid_label(label):
                raise SystemExit(f"Invalid box for {item['id']}: {label}")
            label_path.write_text(
                " ".join(["0", *(f"{value:.10f}" for value in label[1:])]) + "\n",
                encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")
        manifest.append({
            "source": int(item["source"]), "split": split, "kind": kind,
            "domain": item.get("domain") or "unknown", "video": item.get("video"),
            "time_s": float(item["time_s"]), "frame": None, "track_id": -1,
            "box_yolo": label, "image": str(image_path.relative_to(args.output)),
            "annotation": verdict, "action": item.get("action"),
            "action_event_id": event_id,
            "sample_offset_s": item.get("sample_offset_s"),
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
        "kinds": dict(sorted(Counter(row["kind"] for row in manifest).items())),
        "annotations": dict(sorted(Counter(row["annotation"] for row in manifest).items())),
        "actions": dict(sorted(Counter(row.get("action") or "unknown" for row in manifest).items())),
        "sources": dict(sorted(Counter(str(row["source"]) for row in manifest).items())),
        "excluded": dict(sorted(excluded.items())),
        "event_split_overlap": len(
            {row["action_event_id"] for row in manifest if row["split"] == "train"} &
            {row["action_event_id"] for row in manifest if row["split"] == "val"}),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
