#!/usr/bin/env python3
"""Extract full-scene video embeddings around reviewed whistle candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


OFFSETS = np.linspace(-4.0, 1.0, 16)
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
        top, left = (resized.shape[0] - 112) // 2, (resized.shape[1] - 112) // 2
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    items = json.loads(args.queue.read_text())["items"]
    answers = json.loads(args.results.read_text())["answers"]
    rows = [(item, answers[item["id"]]["decision"]) for item in items
            if answers[item["id"]]["decision"] != "uncertain"]

    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(args.device)
    vectors, labels, sources = [], [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        tensors, retained = [], []
        for item, decision in batch:
            tensor = read_clip(item["video"], float(item["time_s"]))
            if tensor is not None:
                tensors.append(tensor)
                retained.append((item, decision))
        if tensors:
            with torch.inference_mode():
                embeddings = model(torch.stack(tensors).to(args.device)).cpu().numpy()
            vectors.extend(embeddings.astype(np.float32))
            labels.extend(int(decision == "serve_whistle") for _item, decision in retained)
            sources.extend(int(item["source"]) for item, _decision in retained)
        print(f"processed={min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    vectors, labels, sources = np.stack(vectors), np.asarray(labels), np.asarray(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=vectors, labels=labels, sources=sources)
    print(json.dumps({"samples": len(labels), "positives": int(labels.sum()),
                      "dimensions": int(vectors.shape[1]), "device": args.device}, indent=2))


if __name__ == "__main__":
    main()
