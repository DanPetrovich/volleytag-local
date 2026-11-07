#!/usr/bin/env python3
"""Validate VNL-STES data and export VolleyTag rally boundaries for transfer.

The generated data is deliberately written below ``vision/runs`` by default and
must stay private: it may contain local video paths and frame-level annotations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


EXPECTED_VNL_CLASSES = {"serve", "receive", "set", "spike", "block", "score"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _normalise_label(value: object) -> str:
    return str(value).strip().lower()


def validate_split(
    rows: object,
    *,
    split: str,
    allowed_classes: set[str],
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{split}: expected a non-empty JSON list")

    videos: set[str] = set()
    counts: Counter[str] = Counter()
    frames = 0
    declared_event_count_mismatches = 0
    duplicate_event_instances = 0
    conflicting_event_frames = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{split}[{index}]: expected an object")
        video = str(row.get("video", "")).strip()
        if not video or video in videos:
            raise ValueError(f"{split}[{index}]: missing or duplicate video id")
        videos.add(video)

        num_frames = row.get("num_frames")
        if not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError(f"{split}/{video}: invalid num_frames")
        frames += num_frames

        events = row.get("events")
        if not isinstance(events, list):
            raise ValueError(f"{split}/{video}: events must be a list")
        if row.get("num_events") != len(events):
            declared_event_count_mismatches += 1
        seen: set[tuple[int, str]] = set()
        labels_by_frame: dict[int, set[str]] = {}
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(
                    f"{split}/{video}/events[{event_index}]: expected an object"
                )
            frame = event.get("frame")
            label = _normalise_label(event.get("label", ""))
            if not isinstance(frame, int) or not 0 <= frame < num_frames:
                raise ValueError(
                    f"{split}/{video}/events[{event_index}]: frame out of bounds"
                )
            if label not in allowed_classes:
                raise ValueError(
                    f"{split}/{video}/events[{event_index}]: unknown label {label!r}"
                )
            key = (frame, label)
            if key in seen:
                duplicate_event_instances += 1
            seen.add(key)
            labels_by_frame.setdefault(frame, set()).add(label)
            counts[label] += 1

            xy = event.get("xy")
            if xy is None and "x" in event and "y" in event:
                xy = [event["x"], event["y"]]
            if xy is not None:
                if (
                    not isinstance(xy, list)
                    or len(xy) != 2
                    or any(not isinstance(value, (int, float)) for value in xy)
                    or any(not 0 <= float(value) <= 1 for value in xy)
                ):
                    raise ValueError(
                        f"{split}/{video}/events[{event_index}]: invalid xy"
                    )
        conflicting_event_frames += sum(
            len(frame_labels) > 1 for frame_labels in labels_by_frame.values()
        )

    return {
        "videos": videos,
        "video_count": len(videos),
        "frame_count": frames,
        "event_count": sum(counts.values()),
        "events_by_class": dict(sorted(counts.items())),
        "declared_event_count_mismatches": declared_event_count_mismatches,
        "duplicate_event_instances": duplicate_event_instances,
        "conflicting_event_frames": conflicting_event_frames,
    }


def validate_frames(dataset_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame_root = dataset_dir / "frames_224p"
    if not frame_root.is_dir():
        raise ValueError(f"missing {frame_root}")

    expected_videos = {str(row["video"]) for row in rows}
    actual_videos = {
        str(path.relative_to(frame_root))
        for path in frame_root.glob("*/*")
        if path.is_dir()
    }
    missing_directories = sorted(expected_videos - actual_videos)
    if missing_directories:
        raise ValueError(f"missing frame directories: {missing_directories[:3]}")

    missing_frames = 0
    extra_frames = 0
    videos_with_extra_frames = 0
    image_count = 0
    for row in rows:
        directory = frame_root / str(row["video"])
        frame_numbers = {
            int(path.stem) for path in directory.glob("*.jpg") if path.stem.isdigit()
        }
        image_count += len(frame_numbers)
        expected = set(range(int(row["num_frames"])))
        missing_frames += len(expected - frame_numbers)
        extras = len(frame_numbers - expected)
        extra_frames += extras
        videos_with_extra_frames += int(extras > 0)
    if missing_frames:
        raise ValueError(f"dataset is missing {missing_frames} expected frames")

    orphan_videos = sorted(actual_videos - expected_videos)
    orphan_frames = sum(
        len(list((frame_root / video).glob("*.jpg"))) for video in orphan_videos
    )
    return {
        "expected_video_directories": len(expected_videos),
        "image_count_for_expected_videos": image_count,
        "missing_frames": missing_frames,
        "extra_terminal_frames": extra_frames,
        "videos_with_extra_terminal_frames": videos_with_extra_frames,
        "orphan_video_directories": orphan_videos,
        "orphan_frames": orphan_frames,
    }


def validate_vnl_dataset(dataset_dir: Path) -> dict[str, Any]:
    class_file = dataset_dir / "class.txt"
    if not class_file.is_file():
        raise ValueError(f"missing {class_file}")
    class_file_classes = [
        _normalise_label(line)
        for line in class_file.read_text().splitlines()
        if line.strip()
    ]
    if len(class_file_classes) != len(set(class_file_classes)):
        raise ValueError("class.txt contains duplicate classes")
    classes = set(class_file_classes) - {"background"}
    if classes != EXPECTED_VNL_CLASSES:
        raise ValueError(
            f"unexpected classes: got {sorted(classes)}, "
            f"expected {sorted(EXPECTED_VNL_CLASSES)}"
        )

    reports: dict[str, dict[str, Any]] = {}
    all_videos: set[str] = set()
    split_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.json"
        if not path.is_file():
            raise ValueError(f"missing {path}")
        report = validate_split(read_json(path), split=split, allowed_classes=classes)
        overlap = all_videos & report["videos"]
        if overlap:
            raise ValueError(f"{split}: videos leak across splits: {sorted(overlap)[:3]}")
        all_videos.update(report["videos"])
        split_rows.extend(read_json(path))
        reports[split] = report

    all_path = dataset_dir / "all.json"
    all_annotations = None
    if all_path.is_file():
        all_rows = read_json(all_path)
        all_report = validate_split(
            all_rows, split="all", allowed_classes=classes
        )
        split_by_video = {str(row["video"]): row for row in split_rows}
        all_by_video = {str(row["video"]): row for row in all_rows}
        if set(all_by_video) != set(split_by_video):
            raise ValueError("all.json video ids differ from train/val/test union")
        differing = sum(
            all_by_video[video]["events"] != split_by_video[video]["events"]
            for video in all_by_video
        )
        all_annotations = {
            key: value for key, value in all_report.items() if key != "videos"
        }
        all_annotations["videos_differing_from_split_annotations"] = differing

    frame_report = validate_frames(dataset_dir, split_rows)

    return {
        "schema_version": 1,
        "kind": "volleytag.stes-dataset-validation",
        "dataset_dir": str(dataset_dir),
        "class_file_classes": class_file_classes,
        "event_classes": sorted(classes),
        "splits": {
            split: {key: value for key, value in report.items() if key != "videos"}
            for split, report in reports.items()
        },
        "total_videos": len(all_videos),
        "total_frames": sum(report["frame_count"] for report in reports.values()),
        "total_events": sum(report["event_count"] for report in reports.values()),
        "all_annotations": all_annotations,
        "frames": frame_report,
    }


def probe_video(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,width,height,nb_frames,duration",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {video}")
    stream = streams[0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    if fps <= 0:
        raise ValueError(f"invalid frame rate in {video}")
    if stream.get("nb_frames") not in (None, "N/A"):
        num_frames = int(stream["nb_frames"])
    else:
        num_frames = round(float(stream["duration"]) * fps)
    return {
        "fps": fps,
        "num_frames": num_frames,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def export_queue(queue: dict[str, Any], *, video_id: str, meta: dict[str, Any]) -> dict:
    if queue.get("frozen_tests_used") is not False:
        raise ValueError("refusing a queue that is frozen or does not declare frozen_tests_used=false")
    rallies = queue.get("rallies")
    if not isinstance(rallies, list) or not rallies:
        raise ValueError("queue must contain at least one rally")

    fps = float(meta["fps"])
    num_frames = int(meta["num_frames"])
    events: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, rally in enumerate(rallies, 1):
        start = float(rally["start_s"])
        end = float(rally["end_s"])
        if start < previous_end or end <= start:
            raise ValueError(f"rally {index}: overlapping or invalid interval")
        start_frame = round(start * fps)
        end_frame = round(end * fps)
        if not 0 <= start_frame < end_frame < num_frames:
            raise ValueError(f"rally {index}: event frame outside video")
        events.extend(
            [
                {
                    "frame": start_frame,
                    "label": "serve",
                    "comment": f"boundary:{rally.get('start_confidence', 'unknown')}",
                },
                {
                    "frame": end_frame,
                    "label": "score",
                    "comment": f"boundary:{rally.get('end_confidence', 'unknown')}",
                },
            ]
        )
        previous_end = end

    return {
        "video": video_id,
        "num_frames": num_frames,
        "num_events": len(events),
        "events": events,
        "fps": fps,
        "width": int(meta["width"]),
        "height": int(meta["height"]),
    }


def command_validate(args: argparse.Namespace) -> None:
    report = validate_vnl_dataset(args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_export(args: argparse.Namespace) -> None:
    rows = []
    sources = []
    for spec in args.source:
        try:
            video_id, queue_text = spec.split("=", 1)
        except ValueError as error:
            raise ValueError("--source must be VIDEO_ID=QUEUE.json") from error
        if not video_id or "/" in video_id or "\\" in video_id:
            raise ValueError(f"invalid video id {video_id!r}")
        queue_path = Path(queue_text)
        queue = read_json(queue_path)
        video = Path(queue["video"])
        meta = probe_video(video)
        rows.append(export_queue(queue, video_id=video_id, meta=meta))
        sources.append({"video": video_id, "queue": str(queue_path), "source_video": str(video)})

    payload = {
        "schema_version": 1,
        "kind": "volleytag.stes-temporal-transfer",
        "frozen_tests_used": False,
        "classes": ["serve", "score"],
        "sources": sources,
        "videos": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "videos": len(rows),
        "rallies": sum(row["num_events"] for row in rows) // 2,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    validate = subparsers.add_parser("validate-vnl")
    validate.add_argument("dataset_dir", type=Path)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=command_validate)

    export = subparsers.add_parser("export-volleytag")
    export.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeatable VIDEO_ID=path/to/dense-state-queue.json",
    )
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=command_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
