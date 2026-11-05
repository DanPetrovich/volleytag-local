#!/usr/bin/env python3
"""Score every cached fused candidate with the frozen view-specific rerankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("near_classifier", type=Path)
    parser.add_argument("far_classifier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-generic-candidates", type=int, default=12,
                        help="keep this many strongest generic proposals per frame, plus every direct proposal")
    args = parser.parse_args()

    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    times = [float(value) for value in payload["times"]]
    candidate_sets = payload["candidate_sets"]
    if args.max_generic_candidates > 0:
        pruned_sets = []
        for candidates in candidate_sets:
            direct = [row for row in candidates if float(row.get("direct_confidence", 0.0)) > 0.0]
            generic = [row for row in candidates if float(row.get("direct_confidence", 0.0)) <= 0.0]
            generic.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
            pruned_sets.append(direct + generic[:args.max_generic_candidates])
        candidate_sets = pruned_sets
        payload["candidate_sets"] = candidate_sets
    cap = cv2.VideoCapture(str(args.video))
    crops: list[np.ndarray] = []
    refs: list[tuple[int, int]] = []
    for frame_index, (time_s, candidates) in enumerate(zip(times, candidate_sets)):
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        for candidate_index, row in enumerate(candidates):
            x1, y1, x2, y2 = [int(round(value)) for value in row["box"]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 < 12 or y2 - y1 < 25:
                continue
            crops.append(frame[y1:y2, x1:x2])
            refs.append((frame_index, candidate_index))
    cap.release()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    near_model = joblib.load(args.near_classifier)
    far_model = joblib.load(args.far_classifier)

    with torch.inference_mode():
        for start in range(0, len(crops), args.batch_size):
            batch_crops = crops[start:start + args.batch_size]
            batch = [transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                     for crop in batch_crops]
            vectors = model(torch.stack(batch).to(device)).float().cpu().numpy()
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            near = near_model.predict_proba(vectors)[:, 1]
            far = far_model.predict_proba(vectors)[:, 1]
            for offset, (near_p, far_p) in enumerate(zip(near, far)):
                frame_index, candidate_index = refs[start + offset]
                row = candidate_sets[frame_index][candidate_index]
                box = row["box"]
                use_near = box[3] - box[1] >= 240 or box[3] >= 700
                row["reranker_domain"] = "near_back" if use_near else "far_front"
                row["reranker_probability"] = float(near_p if use_near else far_p)
            print(f"scored={min(start + args.batch_size, len(crops))}/{len(crops)}", flush=True)

    payload["reranker"] = {
        "near_classifier": str(args.near_classifier),
        "far_classifier": str(args.far_classifier),
        "device": device,
        "candidates_scored": len(refs),
        "router": "near if height>=240 or bottom>=700; otherwise far",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
