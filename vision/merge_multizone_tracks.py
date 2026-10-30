#!/usr/bin/env python3
"""Merge full-court tracks with a higher-resolution far-court pass."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


FIELDS = ("time_s", "frame", "track_id", "confidence", "x1", "y1", "x2", "y2")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(FIELDS) - set(rows[0] if rows else FIELDS)
    if missing:
        raise SystemExit(f"{path} misses columns: {sorted(missing)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("secondary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--id-offset", type=int, default=100_000)
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    primary = read_rows(args.primary)
    secondary = read_rows(args.secondary)
    occupied = {int(row["track_id"]) for row in primary}
    if any(int(row["track_id"]) + args.id_offset in occupied for row in secondary):
        raise SystemExit("Secondary ID namespace collides with primary IDs")
    merged = list(primary)
    for row in secondary:
        copied = {field: row[field] for field in FIELDS}
        copied["track_id"] = str(int(copied["track_id"]) + args.id_offset)
        merged.append(copied)
    merged.sort(key=lambda row: (int(row["frame"]), int(row["track_id"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    if args.backup and args.output.resolve() == args.primary.resolve():
        backup = args.primary.with_name(args.primary.stem + ".before-multizone.csv")
        if not backup.exists():
            shutil.copy2(args.primary, backup)
    temporary.replace(args.output)
    print(f"primary_rows={len(primary)} secondary_rows={len(secondary)} merged_rows={len(merged)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
