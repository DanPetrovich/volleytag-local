from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import subprocess
import sys
import time

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.22", "scikit-learn", "joblib"],
    check=True,
)

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import timm
import torch
from timm.data import create_transform, resolve_model_data_config


SEED = 51
DOMAINS = ("near_back", "far_front")
OLD_SOURCES = {0, 1, 2}
MIN_IMPROVEMENT = 0.08
MAX_DOMAIN_REGRESSION = 0.02


def classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25, class_weight="balanced", max_iter=3000, random_state=SEED
        ),
    )


def padded_crop(image: Image.Image, box: list[float], padding: float = 0.08) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    bounds = (
        max(0, round(x1 - bw * padding)),
        max(0, round(y1 - bh * padding)),
        min(width, round(x2 + bw * padding)),
        min(height, round(y2 + bh * padding)),
    )
    return image.crop(bounds)


def load_records(root: Path) -> tuple[list[dict], list[Image.Image]]:
    manifest = json.loads((root / "manifest.json").read_text())
    records, images = [], []
    for row in manifest:
        if row["kind"] == "background_negative":
            continue
        image = Image.open(root / row["image"]).convert("RGB")
        label = int(row["kind"] == "positive")
        if label:
            image = padded_crop(image, row["box"])
        if image.width < 12 or image.height < 25:
            continue
        records.append(
            {
                "source": int(row["source"]),
                "domain": row["domain"],
                "time_s": float(row["time_s"]),
                "label": label,
                "kind": row["kind"],
                "image": row["image"],
            }
        )
        images.append(image.copy())
        image.close()
    return records, images


def embed(images: list[Image.Image], batch_size: int = 96) -> tuple[np.ndarray, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
    ).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)

    # Mandatory representative sanity check before committing to the full pass.
    sanity_count = min(200, len(images))
    started = time.monotonic()
    with torch.inference_mode():
        sanity_batch = torch.stack([transform(image) for image in images[:sanity_count]])
        sanity_vectors = []
        for start in range(0, sanity_count, batch_size):
            sanity_vectors.append(model(sanity_batch[start:start + batch_size].to(device)).cpu())
    sanity_seconds = time.monotonic() - started
    sanity = {
        "samples": sanity_count,
        "seconds": sanity_seconds,
        "samples_per_second": sanity_count / max(sanity_seconds, 1e-9),
        "estimated_full_seconds": len(images) * sanity_seconds / max(sanity_count, 1),
        "device": device,
        "embedding_dim": int(sanity_vectors[0].shape[1]),
    }
    assert sanity["embedding_dim"] >= 300
    assert np.isfinite(torch.cat(sanity_vectors).numpy()).all()
    print("sanity", json.dumps(sanity), flush=True)

    values = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [transform(image) for image in images[start:start + batch_size]]
            ).to(device)
            vector = model(batch).float().cpu().numpy()
            vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
            values.append(vector)
            print(f"embeddings={min(start + batch_size, len(images))}/{len(images)}", flush=True)
    return np.vstack(values), sanity


def ranking_groups(records: list[dict], indexes: np.ndarray, negatives_per_group: int = 5):
    by_domain: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in indexes:
        by_domain[records[index]["domain"]][records[index]["label"]].append(int(index))
    groups = []
    for domain in DOMAINS:
        negatives = by_domain[domain][0]
        for positive in by_domain[domain][1]:
            ranked = sorted(
                negatives,
                key=lambda i: (abs(records[i]["time_s"] - records[positive]["time_s"]), i),
            )[:negatives_per_group]
            if ranked:
                groups.append((domain, positive, ranked))
    return groups


def evaluate_fold(records, vectors, held_source: int, train_sources: set[int], name: str):
    labels = np.asarray([row["label"] for row in records])
    train = np.asarray(
        [i for i, row in enumerate(records) if row["source"] in train_sources], dtype=int
    )
    test = np.asarray(
        [i for i, row in enumerate(records) if row["source"] == held_source], dtype=int
    )
    result = {"name": name, "held_source": held_source, "train_sources": sorted(train_sources)}
    domain_scores, domain_counts = defaultdict(int), defaultdict(int)
    for domain, positive, negatives in ranking_groups(records, test):
        domain_train = train[np.asarray([records[i]["domain"] == domain for i in train])]
        if len(domain_train) < 20 or len(set(labels[domain_train])) < 2:
            continue
        model = classifier().fit(vectors[domain_train], labels[domain_train])
        candidates = [positive] + negatives
        probabilities = model.predict_proba(vectors[candidates])[:, 1]
        domain_counts[domain] += 1
        domain_scores[domain] += int(int(np.argmax(probabilities)) == 0)
    result["domains"] = {
        domain: {
            "groups": domain_counts[domain],
            "top1_accuracy": domain_scores[domain] / domain_counts[domain]
            if domain_counts[domain]
            else None,
        }
        for domain in DOMAINS
    }
    total = sum(domain_counts.values())
    result["groups"] = total
    result["top1_accuracy"] = sum(domain_scores.values()) / total if total else None
    return result


def aggregate(folds: list[dict]) -> dict:
    total_groups = sum(row["groups"] for row in folds)
    total_hits = sum(row["top1_accuracy"] * row["groups"] for row in folds if row["groups"])
    result = {"groups": total_groups, "micro_top1_accuracy": total_hits / total_groups}
    result["domains"] = {}
    for domain in DOMAINS:
        count = sum(row["domains"][domain]["groups"] for row in folds)
        hits = sum(
            row["domains"][domain]["top1_accuracy"] * row["domains"][domain]["groups"]
            for row in folds
            if row["domains"][domain]["groups"]
        )
        result["domains"][domain] = {
            "groups": count,
            "top1_accuracy": hits / count if count else None,
        }
    return result


def main():
    input_root = Path("/kaggle/input")
    dataset_root = next(input_root.rglob("manifest.json")).parent
    output = Path("/kaggle/working")
    records, images = load_records(dataset_root)
    assert len(records) >= 2000, len(records)
    assert {row["source"] for row in records} == {0, 1, 2, 3, 4}
    assert {row["domain"] for row in records} == set(DOMAINS)
    vectors, sanity = embed(images)
    np.save(output / "embeddings.npy", vectors)
    with (output / "dataset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    baseline_folds, candidate_folds = [], []
    all_sources = {0, 1, 2, 3, 4}
    for held_source in sorted(all_sources):
        baseline_sources = OLD_SOURCES - {held_source}
        candidate_sources = all_sources - {held_source}
        baseline_folds.append(
            evaluate_fold(records, vectors, held_source, baseline_sources, "baseline-old-three")
        )
        candidate_folds.append(
            evaluate_fold(records, vectors, held_source, candidate_sources, "candidate-five-source")
        )
    baseline = aggregate(baseline_folds)
    candidate = aggregate(candidate_folds)
    improvement = candidate["micro_top1_accuracy"] - baseline["micro_top1_accuracy"]
    domain_regressions = {
        domain: baseline["domains"][domain]["top1_accuracy"]
        - candidate["domains"][domain]["top1_accuracy"]
        for domain in DOMAINS
    }
    passed = improvement >= MIN_IMPROVEMENT and max(domain_regressions.values()) <= MAX_DOMAIN_REGRESSION
    report = {
        "protocol": "leave-one-whole-source-out; one positive plus five nearest-in-time audited hard negatives",
        "samples": len(records),
        "positives": sum(row["label"] for row in records),
        "negatives": sum(1 - row["label"] for row in records),
        "sanity": sanity,
        "baseline_folds": baseline_folds,
        "candidate_folds": candidate_folds,
        "baseline": baseline,
        "candidate": candidate,
        "absolute_improvement": improvement,
        "domain_regressions": domain_regressions,
        "success_criterion": {
            "minimum_absolute_improvement": MIN_IMPROVEMENT,
            "maximum_domain_regression": MAX_DOMAIN_REGRESSION,
            "passed": passed,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    labels = np.asarray([row["label"] for row in records])
    for domain in DOMAINS:
        indexes = np.asarray([i for i, row in enumerate(records) if row["domain"] == domain])
        model = classifier().fit(vectors[indexes], labels[indexes])
        joblib.dump(model, output / f"candidate-reranker-{domain}.joblib")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
