#!/usr/bin/env python3
"""Representative VNL event-learning sanity before full STES pretraining."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


WORK = Path("/kaggle/working")
OUTPUT = WORK / "stes-vnl-event-sanity"
FRAMES_PER_EPOCH = 5000
EPOCHS = 5
MINIMUM_VAL_MAP = 0.01
MINIMUM_LOSS_REDUCTION = 0.10
MINIMUM_SHUFFLE_LIFT = 0.25
FOREGROUND_CLIP_PROBABILITY = 0.50
SHUFFLE_REPEATS = 12


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_one(pattern: str) -> Path:
    matches = list(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern}, found {matches}")
    return matches[0]


def ensure_dependencies() -> None:
    required = {
        "timm": "timm",
        "tabulate": "tabulate",
        "easydict": "easydict",
        "wandb": "wandb",
    }
    missing = [
        package
        for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        run([sys.executable, "-m", "pip", "install", "-q", *missing])


def temporal_map_against_shuffle(
    repository: Path, truth_path: Path, predictions_path: Path
) -> tuple[float, float, float]:
    sys.path.insert(0, str(repository))
    from util.eval import non_maximum_supression
    from util.io import load_gz_json, load_json
    from util.score import compute_mAPs

    truth = load_json(str(truth_path))
    predictions = load_gz_json(str(predictions_path))
    video_lengths = {item["video"]: item["num_frames"] for item in truth}

    def temporal_map(items: list[dict]) -> float:
        maps, _ = compute_mAPs(
            truth,
            non_maximum_supression(copy.deepcopy(items), 1),
            tolerances=[0, 1, 2, 4],
        )
        return float(statistics.mean(maps))

    actual = temporal_map(predictions)
    rng = random.Random(20261107)
    shuffled_maps = []
    for _ in range(SHUFFLE_REPEATS):
        shuffled = copy.deepcopy(predictions)
        for video in shuffled:
            num_frames = video_lengths[video["video"]]
            for event in video["events"]:
                event["frame"] = rng.randrange(num_frames)
        shuffled_maps.append(temporal_map(shuffled))
    shuffle_mean = float(statistics.mean(shuffled_maps))
    relative_lift = (actual - shuffle_mean) / max(shuffle_mean, 1e-12)
    return actual, shuffle_mean, relative_lift


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ensure_dependencies()

    source_repository = find_one("**/spot/train_e2e_spatial.py").parent
    repository = WORK / "spot-event-sanity"
    shutil.copytree(source_repository, repository, ignore=shutil.ignore_patterns(".git"))
    training_file = repository / "train_e2e_spatial.py"
    source = training_file.read_text()
    old = "EPOCH_NUM_FRAMES = 500000"
    if old not in source:
        raise RuntimeError("upstream epoch-size constant changed")
    source = source.replace(old, f"EPOCH_NUM_FRAMES = {FRAMES_PER_EPOCH}", 1)
    # Upstream initializes the disabled location loss as a Python float, but
    # unconditionally calls ``detach`` on it in the progress display.
    old_location_loss = "                    loss_loc = 0.0"
    if source.count(old_location_loss) != 1:
        raise RuntimeError("upstream location-loss initialization changed")
    source = source.replace(
        old_location_loss,
        "                    loss_loc = torch.tensor(0.0).to(self.device)",
        1,
    )
    training_file.write_text(source)

    vnl = find_one("**/vnl_1.5/frames_224p").parent
    data_dir = repository / "data" / "vnl_1.5"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.json", "val.json", "test.json", "all.json"):
        shutil.copy2(vnl / name, data_dir / name)
    classes = [
        line.strip()
        for line in (vnl / "class.txt").read_text().splitlines()
        if line.strip() and line.strip().lower() != "background"
    ]
    (data_dir / "class.txt").write_text("\n".join(classes) + "\n")

    experiment = OUTPUT / "experiment"
    run(
        [
            sys.executable,
            "train_e2e_spatial.py",
            "vnl_1.5",
            str(vnl / "frames_224p"),
            "-m", "rny008_gsm",
            "-t", "gru",
            "--clip_len", "64",
            "--batch_size", "8",
            "--num_epochs", str(EPOCHS),
            "--start_val_epoch", str(EPOCHS - 1),
            "--fg_upsample", str(FOREGROUND_CLIP_PROBABILITY),
            "--num_workers", "2",
            "--eval_split", "challenge",
            "-s", str(experiment),
            "--wandb_project", "volleytag-stes-private-event-sanity",
        ],
        cwd=repository,
    )

    history = json.loads((experiment / "loss.json").read_text())
    train_first = float(history[0]["train"])
    train_last = float(history[-1]["train"])
    val_map = float(history[-1]["val_mAP"])
    loss_reduction = (train_first - train_last) / train_first
    finite = all(
        math.isfinite(float(row[key])) for row in history for key in ("train", "val")
    ) and math.isfinite(val_map)
    temporal_map, shuffle_map, shuffle_lift = temporal_map_against_shuffle(
        repository,
        data_dir / "val.json",
        experiment / f"pred-val.{EPOCHS - 1}.recall.json.gz",
    )
    accepted = (
        len(history) == EPOCHS
        and finite
        and loss_reduction >= MINIMUM_LOSS_REDUCTION
        and temporal_map >= MINIMUM_VAL_MAP
        and shuffle_lift >= MINIMUM_SHUFFLE_LIFT
    )
    report = {
        "schema_version": 1,
        "kind": "volleytag.stes-vnl-event-sanity",
        "model": "ImageNet RegNet-Y 800MF + GSM + GRU temporal head",
        "epochs": EPOCHS,
        "sampled_training_frames_per_epoch": FRAMES_PER_EPOCH,
        "foreground_clip_probability": FOREGROUND_CLIP_PROBABILITY,
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "train_loss_reduction": loss_reduction,
        "val_loss_last": history[-1]["val"],
        "upstream_spatiotemporal_harmonic_map": val_map,
        "temporal_val_map": temporal_map,
        "shuffle_temporal_map": shuffle_map,
        "temporal_map_relative_lift_over_shuffle": shuffle_lift,
        "minimum_loss_reduction": MINIMUM_LOSS_REDUCTION,
        "minimum_val_map": MINIMUM_VAL_MAP,
        "minimum_shuffle_relative_lift": MINIMUM_SHUFFLE_LIFT,
        "finite_losses": finite,
        "accepted": accepted,
        "full_pretraining_authorized": accepted,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    checkpoints = sorted(experiment.glob("checkpoint_*.pt"))
    for checkpoint in checkpoints[:-1]:
        checkpoint.unlink()
    for optimizer in experiment.glob("optim_*.pt"):
        optimizer.unlink()
    shutil.rmtree(repository)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
