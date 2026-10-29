#!/usr/bin/env python3
"""Run the local #51 discovery pipeline end to end."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track people, read #51, and fuse identity signals")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "models/player51-multiview.npz")
    parser.add_argument("--target", default="51")
    parser.add_argument("--roi", default="40,70,1900,1020")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--ocr-sample-seconds", type=float, default=.25)
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse completed stage files in the output directory")
    args = parser.parse_args()

    python = sys.executable
    output = args.output.resolve()
    tracking = output / "tracking"
    ocr = output / "ocr"
    reid = output / "reid"
    fused = output / "fused"
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if not (args.reuse and (tracking / "tracks.csv").exists()):
        run([python, "vision/track_players.py", str(args.video), "--output", str(tracking),
             "--roi", args.roi, "--stride", str(args.stride)])
    if not (args.reuse and (ocr / "ocr.csv").exists()):
        run([python, "vision/detect_jersey_number.py", str(args.video), str(tracking / "tracks.csv"),
             "--output", str(ocr), "--roi", args.roi, "--target", args.target,
             "--sample-seconds", str(args.ocr_sample_seconds)])
    if not (args.reuse and (reid / "similarity.csv").exists()):
        run([python, "vision/identify_player.py", str(args.video), str(tracking / "tracks.csv"),
             "--profile", str(args.profile), "--output", str(reid), "--roi", args.roi,
             "--stride", str(args.stride)])

    fusion_ok = True
    try:
        run([python, "vision/fuse_player_signals.py", str(args.video), str(tracking / "tracks.csv"),
             str(ocr / "ocr.csv"), "--similarity", str(reid / "similarity.csv"),
             "--output", str(fused), "--roi", args.roi, "--target", args.target])
    except subprocess.CalledProcessError:
        fusion_ok = False
        print("No reliable OCR anchor: review reid/candidates.jpg; no automatic final box was emitted.")

    ocr_rows = list(csv.DictReader((ocr / "ocr.csv").open(encoding="utf-8")))
    target_hits = [row for row in ocr_rows if row["ocr"] == args.target]
    anchor_counts: dict[str, int] = {}
    for row in target_hits:
        anchor_counts[row["track_id"]] = anchor_counts.get(row["track_id"], 0) + 1
    reid_rows = list(csv.DictReader((reid / "similarity.csv").open(encoding="utf-8")))
    selected_tracks = []
    if fusion_ok:
        selected_tracks = [row["track_id"] for row in
                           csv.DictReader((fused / "segments.csv").open(encoding="utf-8"))]

    report = {
        "video": str(args.video.resolve()),
        "target": args.target,
        "profile": str(args.profile.resolve()),
        "fusion_status": "complete" if fusion_ok else "needs_review",
        "ocr_target_hits": len(target_hits),
        "reliable_ocr_tracks": [track_id for track_id, count in anchor_counts.items() if count >= 2],
        "selected_tracks": selected_tracks,
        "top_reid_candidates": [
            {"track_id": row["track_id"], "margin": float(row["margin"])}
            for row in reid_rows[:5]
        ],
        "elapsed_seconds": round(time.time() - started, 1),
        "final_video": str((fused / f"fused-{args.target}.mp4").resolve()) if fusion_ok else None,
        "review_sheet": str((reid / "candidates.jpg").resolve()),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
