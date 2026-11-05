from pathlib import Path
import json
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.198"], check=True)

from ultralytics import YOLO

input_root = Path("/kaggle/input")
dataset_root = next(path.parent for path in input_root.rglob("dataset.yaml")
                    if path.parent.name == "dataset")
model_path = next(input_root.rglob("yolo11n.pt"))
train_count = len(list((dataset_root / "images/train").glob("*.jpg")))
val_count = len(list((dataset_root / "images/val").glob("*.jpg")))
test_count = len(list((dataset_root / "images/test").glob("*.jpg")))
assert (train_count, val_count, test_count) == (297, 66, 66), (train_count, val_count, test_count)
print("dataset:", dataset_root)
print("warm start:", model_path)
print("train/model-selection-val/untouched-test:", train_count, val_count, test_count)

# The archived YAML intentionally contains the canonical local path.  Kaggle
# mounts the same dataset elsewhere, so write a runtime-only YAML with explicit
# mount paths.  The test directory is not referenced by training.
runtime_yaml = Path("/kaggle/working/dataset.yaml")
runtime_yaml.write_text(
    f"train: {dataset_root / 'images/train'}\n"
    f"val: {dataset_root / 'images/val'}\n"
    "names:\n  0: volleyball\n"
)

model = YOLO(str(model_path))
results = model.train(
    data=str(runtime_yaml),
    epochs=70,
    imgsz=960,
    batch=16,
    device=0,
    workers=4,
    patience=14,
    close_mosaic=8,
    cache="disk",
    seed=51,
    project="/kaggle/working/runs",
    name="ball-detector-v1",
    plots=True,
)

run = Path("/kaggle/working/runs/ball-detector-v1")
best = run / "weights/best.pt"
shutil.copy2(best, "/kaggle/working/ball-detector-v1-best.pt")
metrics = {
    "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
    "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
    "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    "train_images": train_count,
    "model_selection_val_images": val_count,
    "untouched_test_images": test_count,
}
Path("/kaggle/working/validation-metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
