#!/usr/bin/env python3
"""Extract R3D-18 embeddings for reviewed rally-state clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


OFFSETS = np.linspace(-3.0, 2.0, 16)
MEAN = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
STD = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)


def read_clip(video: str, timestamp: float) -> torch.Tensor | None:
    capture, frames = cv2.VideoCapture(video), []
    for offset in OFFSETS:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp + float(offset)) * 1000)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return None
        height, width = frame.shape[:2]
        scale = 128 / min(height, width)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)),
                             interpolation=cv2.INTER_AREA)
        top = (resized.shape[0] - 112) // 2
        left = (resized.shape[1] - 112) // 2
        crop = resized[top:top + 112, left:left + 112]
        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    capture.release()
    tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255
    return (tensor - MEAN) / STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    items = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"]
    missing = [row["id"] for row in items if row["id"] not in answers]
    if missing:
        raise ValueError(f"missing {len(missing)} review answers")

    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    vectors, labels, sources, item_ids = [], [], [], []
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        tensors, retained = [], []
        for item in batch:
            tensor = read_clip(item["video"], float(item["time_s"]))
            if tensor is not None:
                tensors.append(tensor)
                retained.append(item)
        if tensors:
            with torch.inference_mode():
                embeddings = model(torch.stack(tensors)).numpy()
            vectors.extend(embeddings.astype(np.float32))
            labels.extend(answers[item["id"]]["decision"] for item in retained)
            sources.extend(int(item["source"]) for item in retained)
            item_ids.extend(item["id"] for item in retained)
        print(f"processed={min(start + args.batch_size, len(items))}/{len(items)}", flush=True)
    if len(vectors) != len(items):
        raise RuntimeError(f"decoded {len(vectors)} of {len(items)} clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=np.stack(vectors),
                        labels=np.asarray(labels), sources=np.asarray(sources),
                        item_ids=np.asarray(item_ids))
    print(json.dumps({"samples": len(labels), "dimensions": len(vectors[0]),
                      "labels": {label: labels.count(label) for label in sorted(set(labels))}},
                     indent=2))


if __name__ == "__main__":
    main()
