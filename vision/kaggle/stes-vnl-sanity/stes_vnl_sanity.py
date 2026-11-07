#!/usr/bin/env python3
"""Short, private VNL-STES pipeline smoke on Kaggle GPU.

This is intentionally not the full VNL pretraining run. It checks that the
published model, downloaded archive, ImageNet initialization, spatial labels,
and evaluation path work together before spending the full GPU budget.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


UPSTREAM_REPOSITORY = "https://github.com/hoangqnguyen/spot.git"
UPSTREAM_COMMIT = "bbaa249c285c6bc133b091e50e6da2bf4ed42a39"
WORK = Path("/kaggle/working")
OUTPUT = WORK / "stes-vnl-pipeline-smoke"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_vnl_data() -> Path:
    # Kaggle expands a top-level ZIP while creating a dataset. Retain archive
    # support as a fallback for a future version that preserves the upload.
    direct = list(Path("/kaggle/input").glob("**/vnl_1.5/train.json"))
    if len(direct) == 1:
        return direct[0].parent
    archives = list(Path("/kaggle/input").glob("**/vnl_1.0.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            f"expected one VNL dataset or archive, found direct={direct}, zip={archives}"
        )
    extracted = WORK / "vnl_1.5"
    with zipfile.ZipFile(archives[0]) as source:
        bad = source.testzip()
        if bad:
            raise RuntimeError(f"corrupt member: {bad}")
        source.extractall(WORK)
    return extracted


def ensure_dependencies() -> None:
    missing = []
    for module, package in (
        ("timm", "timm"),
        ("tabulate", "tabulate"),
        ("easydict", "easydict"),
        ("wandb", "wandb"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        run([sys.executable, "-m", "pip", "install", "-q", *missing])


def prepare_upstream() -> tuple[Path, Path]:
    extracted = find_vnl_data()
    if not (extracted / "train.json").is_file():
        raise RuntimeError("VNL train.json missing after extraction")

    repository = WORK / "spot"
    run(["git", "clone", "--filter=blob:none", UPSTREAM_REPOSITORY, str(repository)])
    run(["git", "checkout", UPSTREAM_COMMIT], cwd=repository)

    data_dir = repository / "data" / "vnl_1.5"
    data_dir.mkdir(parents=True)
    for name in ("train.json", "val.json", "test.json", "all.json"):
        shutil.copy2(extracted / name, data_dir / name)
    # The downloaded archive lists `background` in class.txt, while upstream
    # already reserves class index zero for background and adds one to every
    # class-file entry. Keeping it would create a duplicate foreground class.
    classes = [
        line.strip()
        for line in (extracted / "class.txt").read_text().splitlines()
        if line.strip() and line.strip().lower() != "background"
    ]
    (data_dir / "class.txt").write_text("\n".join(classes) + "\n")
    return repository, extracted


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ensure_dependencies()
    repository, extracted = prepare_upstream()
    experiment = OUTPUT / "experiment"
    command = [
        sys.executable,
        "train_e2e_spatial.py",
        "vnl_1.5",
        str(extracted / "frames_224p"),
        "-m",
        "rny008_gsm",
        "-t",
        "gru",
        "--clip_len",
        "64",
        "--batch_size",
        "4",
        "--num_epochs",
        "5",
        "--start_val_epoch",
        "4",
        "--predict_location",
        "--num_workers",
        "2",
        "--debug_only",
        "--eval_split",
        "challenge",
        "-s",
        str(experiment),
        "--wandb_project",
        "volleytag-stes-private-smoke",
    ]
    run(command, cwd=repository)

    history_path = experiment / "loss.json"
    history = json.loads(history_path.read_text())
    checkpoints = sorted(experiment.glob("checkpoint_*.pt"))
    prediction_files = sorted(experiment.glob("pred-val.*.json"))
    finite_losses = all(
        math.isfinite(float(row[key])) for row in history for key in ("train", "val")
    )
    accepted = (
        len(history) == 5
        and len(checkpoints) == 5
        and bool(prediction_files)
        and finite_losses
    )
    report = {
        "schema_version": 1,
        "kind": "volleytag.stes-vnl-pipeline-smoke",
        "upstream_commit": UPSTREAM_COMMIT,
        "model": "ImageNet RegNet-Y 800MF + GSM + GRU + spatial head",
        "epochs": len(history),
        "sampled_training_frames_per_epoch": 500,
        "train_loss_first": history[0]["train"],
        "train_loss_last": history[-1]["train"],
        "val_loss_last": history[-1]["val"],
        "val_map_last": history[-1]["val_mAP"],
        "checkpoints": len(checkpoints),
        "prediction_files": len(prediction_files),
        "finite_losses": finite_losses,
        "accepted": accepted,
        "full_pretraining_authorized": False,
        "note": "Pipeline smoke only; transfer acceptance remains separate.",
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not accepted:
        raise SystemExit("pipeline smoke acceptance failed")


if __name__ == "__main__":
    main()
