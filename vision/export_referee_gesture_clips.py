#!/usr/bin/env python3
"""Export compact uint8 referee clips for private remote fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extract_referee_gesture_embeddings import MEAN, STD, read_clip


def to_uint8(tensor) -> np.ndarray:
    restored = (tensor * STD + MEAN).clamp(0, 1)
    return np.rint(restored.numpy() * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"]
    missing = [row["id"] for row in items if row["id"] not in answers]
    if missing:
        raise ValueError(f"missing {len(missing)} answers")
    clips, labels, sources, item_ids = [], [], [], []
    for index, item in enumerate(items, 1):
        tensor = read_clip(item)
        if tensor is None:
            raise RuntimeError(f"cannot decode {item['id']}")
        clips.append(to_uint8(tensor))
        labels.append(answers[item["id"]]["decision"])
        sources.append(int(item["source"]))
        item_ids.append(item["id"])
        print(f"processed={index}/{len(items)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, clips=np.stack(clips), labels=np.asarray(labels),
                        sources=np.asarray(sources), item_ids=np.asarray(item_ids))
    print(json.dumps({"samples": len(items), "shape": list(clips[0].shape),
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
