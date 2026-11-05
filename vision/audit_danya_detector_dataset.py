#!/usr/bin/env python3
"""Audit a YOLO identity dataset and build reproducible manual-review queues."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from PIL import Image


POSITIVE_KINDS = {"positive", "manual_positive"}


def read_label(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([float(value) for value in line.split()])
    return rows


def robust_z(value: float, values: list[float]) -> float:
    if len(values) < 5:
        return 0.0
    median = statistics.median(values)
    mad = statistics.median(abs(item - median) for item in values)
    return abs(value - median) / max(1e-9, 1.4826 * mad)


def stable_id(row: dict) -> str:
    raw = f"{row['source']}|{row['frame']}|{row['kind']}|{row['image']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for row in rows:
        strata[(row["source"], row.get("domain") or "unknown", row["kind"], row["split"])].append(row)
    for values in strata.values():
        rng.shuffle(values)

    selected = []
    keys = sorted(strata, key=str)
    while len(selected) < min(count, len(rows)):
        changed = False
        rng.shuffle(keys)
        for key in keys:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
                changed = True
        if not changed:
            break
    rng.shuffle(selected)
    return selected


def queue_entry(row: dict, dataset: Path) -> dict:
    label_path = dataset / row["image"].replace("images/", "labels/").replace(".jpg", ".txt")
    labels = read_label(label_path)
    return {
        "id": stable_id(row),
        "image": str((dataset / row["image"]).relative_to(dataset.parent.parent.parent.parent)),
        "source": row["source"],
        "split": row["split"],
        "kind": row["kind"],
        "domain": row.get("domain") or "unknown",
        "time_s": row["time_s"],
        "frame": row["frame"],
        "track_id": row["track_id"],
        "labels": labels,
        "risk_score": round(row.get("risk_score", 0.0), 3),
        "risk_reasons": row.get("risk_reasons", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    positive_features = []

    for row in manifest:
        image_path = args.dataset / row["image"]
        label_path = args.dataset / row["image"].replace("images/", "labels/").replace(".jpg", ".txt")
        if not image_path.exists():
            errors.append({"image": row["image"], "error": "missing_image"})
            continue
        labels = read_label(label_path)
        expected = 1 if row["kind"] in POSITIVE_KINDS else 0
        if len(labels) != expected:
            errors.append({"image": row["image"], "error": "label_count", "count": len(labels), "expected": expected})
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            errors.append({"image": row["image"], "error": "invalid_image_size"})
        for label in labels:
            if len(label) != 5 or int(label[0]) != 0 or any(not math.isfinite(v) for v in label):
                errors.append({"image": row["image"], "error": "invalid_label", "label": label})
                continue
            _, cx, cy, bw, bh = label
            epsilon = 1e-7  # YOLO text rounding can move an edge by a few nanometres.
            if not (0 < bw <= 1 and 0 < bh <= 1 and
                    bw / 2 - epsilon <= cx <= 1 - bw / 2 + epsilon and
                    bh / 2 - epsilon <= cy <= 1 - bh / 2 + epsilon):
                errors.append({"image": row["image"], "error": "out_of_bounds", "label": label})
            positive_features.append((row, bw * bh, bw / max(bh, 1e-9), bw, bh))

    groups = defaultdict(lambda: defaultdict(list))
    for row, area, aspect, bw, bh in positive_features:
        key = (row["source"], row.get("domain") or "unknown")
        groups[key]["area"].append(area)
        groups[key]["aspect"].append(aspect)
        groups[key]["bw"].append(bw)
        groups[key]["bh"].append(bh)

    for row in manifest:
        row["risk_score"] = 0.0
        row["risk_reasons"] = []
    for row, area, aspect, bw, bh in positive_features:
        key = (row["source"], row.get("domain") or "unknown")
        scores = {
            "unusual_box_area": robust_z(area, groups[key]["area"]),
            "unusual_box_aspect": robust_z(aspect, groups[key]["aspect"]),
            "unusual_box_width": robust_z(bw, groups[key]["bw"]),
            "unusual_box_height": robust_z(bh, groups[key]["bh"]),
        }
        for reason, score in scores.items():
            if score >= 3.5:
                row["risk_reasons"].append(reason)
                row["risk_score"] += min(score, 10)

    by_track = defaultdict(list)
    for row in manifest:
        if row["kind"] in POSITIVE_KINDS and row["track_id"] >= 0:
            by_track[(row["source"], row["track_id"])].append(row)
    for rows in by_track.values():
        rows.sort(key=lambda value: value["time_s"])
        for previous, current in zip(rows, rows[1:]):
            dt = current["time_s"] - previous["time_s"]
            if not (0 < dt <= 1.5):
                continue
            px1, py1, px2, py2 = previous["box"]
            cx1, cy1, cx2, cy2 = current["box"]
            prev_h = max(py2 - py1, 1.0)
            displacement = math.hypot((cx1 + cx2 - px1 - px2) / 2, (cy1 + cy2 - py1 - py2) / 2)
            speed = displacement / prev_h / dt
            if speed > 6:
                current["risk_reasons"].append("abrupt_track_jump")
                current["risk_score"] += min(speed, 20)

    duplicate_states = defaultdict(set)
    for row in manifest:
        # Hard negatives are crops around a confusing *other* player. Danya may
        # legitimately be present elsewhere in the full source frame.
        if row["kind"] == "hard_negative":
            continue
        state = "positive" if row["kind"] in POSITIVE_KINDS else "negative"
        duplicate_states[(row["source"], row["frame"])].add(state)
    contradictions = [list(key) for key, states in duplicate_states.items() if len(states) > 1]

    clean_rows = [row for row in manifest if not any(error["image"] == row["image"] for error in errors)]
    manual_sample = stratified_sample(clean_rows, args.sample_size, args.seed)
    suspicious = sorted(
        (row for row in clean_rows if row["risk_score"] > 0),
        key=lambda row: (-row["risk_score"], row["source"], row["time_s"]),
    )
    queues = {
        "all-items.json": clean_rows,
        "manual-sample.json": manual_sample,
        "suspicious.json": suspicious,
    }
    for filename, rows in queues.items():
        payload = {
            "schema_version": 1,
            "dataset": str(args.dataset),
            "seed": args.seed,
            "total": len(rows),
            "items": [queue_entry(row, args.dataset) for row in rows],
        }
        (args.output / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "examples": len(manifest),
        "automatic_errors": errors,
        "contradictory_source_frames": contradictions,
        "manual_sample_size": len(manual_sample),
        "suspicious_size": len(suspicious),
        "suspicious_reasons": dict(Counter(reason for row in suspicious for reason in row["risk_reasons"])),
        "sample_strata": {
            "sources": dict(Counter(str(row["source"]) for row in manual_sample)),
            "domains": dict(Counter(row.get("domain") or "unknown" for row in manual_sample)),
            "kinds": dict(Counter(row["kind"] for row in manual_sample)),
            "splits": dict(Counter(row["split"] for row in manual_sample)),
        },
    }
    (args.output / "automatic-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
