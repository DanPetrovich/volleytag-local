#!/usr/bin/env python3
"""Representative VNL event-learning sanity before full STES pretraining."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


WORK = Path("/kaggle/working")
OUTPUT = WORK / "stes-vnl-event-sanity"
FRAMES_PER_EPOCH = 5000
EPOCHS = 5
MINIMUM_VAL_MAP = 0.01
MINIMUM_LOSS_REDUCTION = 0.10


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
    training_file.write_text(source.replace(old, f"EPOCH_NUM_FRAMES = {FRAMES_PER_EPOCH}", 1))

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
            "--predict_location",
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
    accepted = (
        len(history) == EPOCHS
        and finite
        and loss_reduction >= MINIMUM_LOSS_REDUCTION
        and val_map >= MINIMUM_VAL_MAP
    )
    report = {
        "schema_version": 1,
        "kind": "volleytag.stes-vnl-event-sanity",
        "model": "ImageNet RegNet-Y 800MF + GSM + GRU + spatial head",
        "epochs": EPOCHS,
        "sampled_training_frames_per_epoch": FRAMES_PER_EPOCH,
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "train_loss_reduction": loss_reduction,
        "val_loss_last": history[-1]["val"],
        "val_map_last": val_map,
        "minimum_loss_reduction": MINIMUM_LOSS_REDUCTION,
        "minimum_val_map": MINIMUM_VAL_MAP,
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
    if not accepted:
        raise SystemExit("event-learning sanity acceptance failed")


if __name__ == "__main__":
    main()
