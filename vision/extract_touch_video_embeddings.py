#!/usr/bin/env python3
"""Extract leakage-free pretrained video embeddings for touch sanity clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18
from ultralytics import YOLO


OFFSETS = np.linspace(-0.8, 0.8, 9)
VIDEO_FRAMES = 16
MEAN = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1, 1)
STD = torch.tensor((0.22803, 0.22145, 0.216989)).view(3, 1, 1, 1)


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def read_clip(video: str, timestamp: float) -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(video)
    frames = []
    for offset in OFFSETS:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp + float(offset)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        frames.append(frame)
    cap.release()
    return frames


def top_box(result) -> list[float] | None:
    if result.boxes is None or not len(result.boxes):
        return None
    scores = result.boxes.conf.detach().cpu().numpy()
    return result.boxes.xyxy[int(np.argmax(scores))].detach().cpu().tolist()


def person_context_crop(frame: np.ndarray, box: list[float] | None) -> np.ndarray:
    height, width = frame.shape[:2]
    if box is None:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(64.0, (y2 - y1) * 1.8, (x2 - x1) * 2.4)
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(width, int(cx + side / 2)), min(height, int(cy + side / 2))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def tensor_from_crops(crops: list[np.ndarray]) -> torch.Tensor:
    indexes = np.linspace(0, len(crops) - 1, VIDEO_FRAMES).round().astype(int)
    rgb = np.stack([cv2.cvtColor(crops[index], cv2.COLOR_BGR2RGB) for index in indexes])
    tensor = torch.from_numpy(rgb).permute(3, 0, 1, 2).float() / 255.0
    return (tensor - MEAN) / STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_features", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("personal_model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path)
    parser.add_argument("--batch-clips", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cached = np.load(args.base_features)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))["samples"]
    if len(metadata) != len(cached["features"]):
        raise SystemExit("feature/metadata length mismatch")

    device = device_name(args.device)
    detector = YOLO(str(args.personal_model))
    backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    embeddings: list[np.ndarray] = []

    for start in range(0, len(metadata), args.batch_clips):
        rows = metadata[start:start + args.batch_clips]
        clips = [read_clip(row["video"], float(row["time_s"])) for row in rows]
        flat_frames = [frame for clip in clips if clip is not None for frame in clip]
        results = detector.predict(flat_frames, imgsz=960, conf=.05, device=device,
                                   batch=min(18, len(flat_frames)), verbose=False)
        pointer, batch_tensors = 0, []
        for clip in clips:
            if clip is None:
                batch_tensors.append(torch.zeros(3, VIDEO_FRAMES, 112, 112))
                continue
            clip_results = results[pointer:pointer + len(clip)]
            pointer += len(clip)
            crops = [person_context_crop(frame, top_box(result))
                     for frame, result in zip(clip, clip_results)]
            batch_tensors.append(tensor_from_crops(crops))
        with torch.inference_mode():
            vectors = backbone(torch.stack(batch_tensors).to(device)).detach().cpu().numpy()
        embeddings.extend(vectors.astype(np.float32))
        print(f"processed={min(start + args.batch_clips, len(metadata))}/{len(metadata)}", flush=True)

    vectors = np.stack(embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=vectors, labels=cached["labels"],
                        sources=cached["sources"])
    if args.combined_output:
        combined = np.concatenate([cached["features"], vectors], axis=1)
        np.savez_compressed(args.combined_output, features=combined, labels=cached["labels"],
                            sources=cached["sources"])
    print(json.dumps({"samples": len(vectors), "dimensions": int(vectors.shape[1]),
                      "device": device, "output": str(args.output),
                      "combined_output": str(args.combined_output) if args.combined_output else None}, indent=2))


if __name__ == "__main__":
    main()
