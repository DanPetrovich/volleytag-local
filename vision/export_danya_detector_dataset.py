#!/usr/bin/env python3
"""Export human-confirmed player #51 tracks as a one-class YOLO dataset.

The review UI stores positive time intervals plus the tracker chunks that were
confirmed inside each interval.  This exporter samples those chunks sparsely,
so adjacent video frames do not dominate the dataset, and splits whole temporal
blocks between train and validation to reduce leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2


def read_tracks(path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = {
                "time_s": float(row["time_s"]),
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "box": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            }
            grouped[item["track_id"]].append(item)
    return grouped


def split_chunks(grouped: dict[int, list[dict]], gap_s: float = .25) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for track_id, rows in grouped.items():
        chunk, index = [], 0
        for row in sorted(rows, key=lambda value: value["time_s"]):
            if chunk and row["time_s"] - chunk[-1]["time_s"] > gap_s:
                result[f"{track_id}:{index}"] = chunk
                chunk, index = [], index + 1
            chunk.append(row)
        if chunk:
            result[f"{track_id}:{index}"] = chunk
    return result


def choose_sparse(rows: list[dict], start: float, end: float, interval_s: float) -> list[dict]:
    eligible = sorted((row for row in rows if start <= row["time_s"] <= end),
                      key=lambda row: row["time_s"])
    if not eligible:
        return []
    chosen, target = [], start
    while target <= end:
        nearby = [row for row in eligible if abs(row["time_s"] - target) <= .12]
        if nearby:
            # Duplicate detector boxes are common. Prefer the fullest person box
            # at a sampling instant rather than teaching two targets for one body.
            best = max(nearby, key=lambda row: ((row["box"][2] - row["box"][0])
                                                * (row["box"][3] - row["box"][1])))
            if not chosen or best["frame"] != chosen[-1]["frame"]:
                chosen.append(best)
        target += interval_s
    return chosen


def write_example(frame, row: dict, image_path: Path, label_path: Path) -> bool:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = row["box"]
    x1, x2 = max(0., min(width - 1., x1)), max(0., min(width, x2))
    y1, y2 = max(0., min(height - 1., y1)), max(0., min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return False
    cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return False
    label_path.write_text(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n", encoding="utf-8")
    return True


def write_negative_crop(frame, row: dict, image_path: Path, label_path: Path,
                        padding: float) -> bool:
    """Write a confirmed wrong-player crop as an empty-label YOLO example."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = row["box"]
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(round(x1 - padding * box_w)))
    x2 = min(width, int(round(x2 + padding * box_w)))
    y1 = max(0, int(round(y1 - padding * box_h)))
    y2 = min(height, int(round(y2 + padding * box_h)))
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < 32 or crop.shape[1] < 16:
        return False
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return False
    label_path.write_text("", encoding="utf-8")
    return True


def write_background(frame, image_path: Path, label_path: Path) -> bool:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return False
    label_path.write_text("", encoding="utf-8")
    return True


def nearest_row(rows: list[dict], time_s: float, tolerance: float = .20) -> dict | None:
    if not rows:
        return None
    row = min(rows, key=lambda value: abs(value["time_s"] - time_s))
    return row if abs(row["time_s"] - time_s) <= tolerance else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", type=Path,
                        help="Training-data JSON files exported by tracking-review")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("--validation-block", type=float, default=180.)
    parser.add_argument("--negative-padding", type=float, default=.25)
    parser.add_argument("--max-width", type=int, default=960,
                        help="Downscale full frames before writing; 0 keeps source resolution")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = []
    per_source = defaultdict(int)
    excluded_by_source = {}
    for source_index, spec_path in enumerate(args.specs):
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        roi_x, roi_y = [float(value) for value in payload.get("roi", [0, 0])]
        excluded_by_source[source_index] = [
            (float(interval["start_s"]), float(interval["end_s"]))
            for interval in payload.get("excluded_intervals", [])
        ]
        video = Path(payload["video"])
        tracks = Path(payload["tracks"])
        grouped = read_tracks(tracks)
        chunks = split_chunks(grouped)
        all_rows = [row for rows in grouped.values() for row in rows]

        def full_frame_row(source_row):
            """Convert an ROI-relative tracker row to full-frame coordinates."""
            row = dict(source_row)
            x1, y1, x2, y2 = source_row["box"]
            row["box"] = [x1 + roi_x, y1 + roi_y, x2 + roi_x, y2 + roi_y]
            return row

        for segment in payload.get("segments", []):
            if segment.get("label") != "positive":
                continue
            start, end = float(segment["start"]), float(segment["end"])
            segment_rows = [row for key in (segment.get("routeKeys") or [segment.get("key")])
                            for row in chunks.get(key, [])]
            for row in choose_sparse(segment_rows, start, end, args.interval):
                row = full_frame_row(row)
                identity = (source_index, row["frame"])
                candidates.append((identity, source_index, video, row, "positive", segment.get("domain")))

        # Manually drawn boxes are the most valuable positives: they cover cases
        # where the generic person detector failed completely. Prefer them over a
        # tracked box on the same source frame.
        for manual in payload.get("manualBoxes", []):
            manual_time = float(manual["time"])
            # Older browser exports assumed every video was 50 FPS. Several
            # training videos are 25 FPS, so their saved manual.frame points to
            # a completely different moment. Tracks already contain the true
            # decoded frame/time mapping; use the nearest observation only for
            # its frame index while preserving the human box and timestamp.
            temporal_reference = nearest_row(all_rows, manual_time, tolerance=.30)
            if temporal_reference is None:
                continue
            row = {"time_s": manual_time, "frame": int(temporal_reference["frame"]),
                   "track_id": -1, "box": [float(manual[key])
                                             for key in ("x1", "y1", "x2", "y2")]}
            row["box"] = [row["box"][0] + roi_x, row["box"][1] + roi_y,
                          row["box"][2] + roi_x, row["box"][3] + roi_y]
            identity = (source_index, row["frame"])
            candidates.append((identity, source_index, video, row, "manual_positive",
                               manual.get("domain")))

        # A manual "not me" click identifies a particularly confusing player.
        # Use a padded crop rather than the full frame, because Danya can still be
        # visible elsewhere in that same frame.
        for negative_index, negative in enumerate(payload.get("negatives", [])):
            key = negative.get("key")
            row = nearest_row(chunks.get(key, []), float(negative["time"]))
            if row is None:
                track_rows = grouped.get(int(negative.get("trackId", -999999)), [])
                row = nearest_row(track_rows, float(negative["time"]))
            if row is None:
                continue
            row = full_frame_row(row)
            identity = (source_index, f"negative-{negative_index}-{row['frame']}")
            candidates.append((identity, source_index, video, row, "hard_negative",
                               negative.get("domain")))

        for lost_index, event in enumerate(payload.get("lostEvents", [])):
            if event.get("reason") != "not-visible":
                continue
            start = float(event.get("start", event["time"]))
            end = max(start, float(event.get("end", start)))
            # Sample explicit absence intervals sparsely. Empty one-class labels
            # teach the detector that the other players are not Danya.
            sample_times = [start]
            target = start + 2.0
            while target <= end:
                sample_times.append(target)
                target += 2.0
            for sample_index, sample_time in enumerate(sample_times):
                row = nearest_row(all_rows, sample_time, tolerance=.30)
                if row is None:
                    continue
                identity = (source_index,
                            f"background-{lost_index}-{sample_index}-{row['frame']}")
                candidates.append((identity, source_index, video, row,
                                   "background_negative", None))

    # Deduplicate chunks that occur in overlapping confirmed segments.
    unique = {identity: (source_index, video, row, kind, domain)
              for identity, source_index, video, row, kind, domain in candidates
              if not any(start <= float(row["time_s"]) <= end
                         for start, end in excluded_by_source.get(source_index, []))}
    # Never emit the same source frame as both a positive and an empty
    # background label. Manual positives can close an absence interval between
    # two sampled timestamps, and contradictory duplicates poison one-class
    # detector training disproportionately.
    positive_frames = {
        (source_index, row["frame"])
        for source_index, _video, row, kind, _domain in unique.values()
        if kind in {"positive", "manual_positive"}
    }
    deduplicated = {}
    for value in unique.values():
        source_index, _video, row, kind, _domain = value
        if kind == "background_negative" and (source_index, row["frame"]) in positive_frames:
            continue
        signature = (source_index, row["frame"], row["track_id"], kind)
        deduplicated[signature] = value
    by_source_block: dict[tuple[int, int], list[tuple[Path, dict, str, str | None]]] = defaultdict(list)
    for source_index, video, row, kind, domain in deduplicated.values():
        block = int(row["time_s"] // args.validation_block)
        by_source_block[(source_index, block)].append((video, row, kind, domain))

    manifest = []
    tasks: dict[Path, list[tuple[int, str, dict, str, str | None]]] = defaultdict(list)
    for (source_index, block), rows in sorted(by_source_block.items()):
        # Every fifth whole time block is validation. This keeps both camera
        # orientations represented without leaking adjacent frames.
        split = "val" if block % 5 == 4 else "train"
        for video, row, kind, domain in rows:
            tasks[video].append((source_index, split, row, kind, domain))

    # Decode each source once in frame order. Repeated random seeking in long
    # H.264 videos is dramatically slower because every seek starts at a keyframe.
    for video, video_tasks in tasks.items():
        cap = cv2.VideoCapture(str(video))
        wanted: dict[int, list[tuple[int, str, dict, str, str | None]]] = defaultdict(list)
        for source_index, split, row, kind, domain in video_tasks:
            wanted[row["frame"]].append((source_index, split, row, kind, domain))
        last_frame = max(wanted)
        frame_index = 0
        while frame_index <= last_frame:
            ok, frame = cap.read()
            if not ok:
                break
            for source_index, split, row, kind, domain in wanted.get(frame_index, []):
                example_frame = frame
                example_row = row
                if args.max_width and frame.shape[1] > args.max_width:
                    scale = args.max_width / frame.shape[1]
                    example_frame = cv2.resize(
                        frame, (args.max_width, int(round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
                    example_row = {**row, "box": [value * scale for value in row["box"]]}
                stem = f"s{source_index}-f{row['frame']}-id{row['track_id']}-{kind}"
                image = args.output / "images" / split / f"{stem}.jpg"
                label = args.output / "labels" / split / f"{stem}.txt"
                if kind == "hard_negative":
                    written = write_negative_crop(example_frame, example_row, image, label,
                                                   args.negative_padding)
                elif kind == "background_negative":
                    written = write_background(example_frame, image, label)
                else:
                    written = write_example(example_frame, example_row, image, label)
                if written:
                    per_source[(source_index, split, kind)] += 1
                    manifest.append({"source": source_index, "split": split,
                                     "kind": kind, "domain": domain,
                                     "video": str(video), **row,
                                     "image": str(image.relative_to(args.output))})
            frame_index += 1
        cap.release()

    # A relative root keeps the archive portable between local, Colab and Kaggle.
    yaml = ("path: .\n"
            "train: images/train\nval: images/val\n"
            "names:\n  0: danya\n")
    (args.output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"examples": len(manifest), "counts": {
        f"source_{source}_{split}_{kind}": count
        for (source, split, kind), count in per_source.items()},
        "kinds": dict(sorted(Counter(row["kind"] for row in manifest).items())),
        "domains": dict(sorted(Counter(
            row.get("domain") or "unknown" for row in manifest).items()))}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
