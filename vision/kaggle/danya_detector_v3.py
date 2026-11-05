from pathlib import Path
import json
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.192"], check=True)

from ultralytics import YOLO

input_root = Path("/kaggle/input")
model_path = next(input_root.rglob("danya-detector-v2-best.pt"))
source_yaml = next(input_root.rglob("dataset.yaml"))
dataset_root = source_yaml.parent
data_yaml = Path("/kaggle/working/dataset.yaml")
data_yaml.write_text(
    f"path: {dataset_root}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n  0: danya\n"
)
train_count = len(list((dataset_root / "images/train").glob("*.jpg")))
val_count = len(list((dataset_root / "images/val").glob("*.jpg")))
print("dataset:", dataset_root)
print("warm start:", model_path)
print("train/val:", train_count, val_count)
assert train_count == 5341 and val_count == 778, (train_count, val_count)

model = YOLO(str(model_path))
results = model.train(
    data=str(data_yaml),
    epochs=60,
    imgsz=960,
    batch=16,
    device=0,
    workers=4,
    patience=12,
    close_mosaic=8,
    cache="disk",
    seed=51,
    project="/kaggle/working/runs",
    name="danya-detector-v3",
    plots=True,
)

run = Path("/kaggle/working/runs/danya-detector-v3")
best = run / "weights/best.pt"
shutil.copy2(best, "/kaggle/working/danya-detector-v3-best.pt")
metrics = {
    "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
    "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
    "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    "train_images": train_count,
    "val_images": val_count,
}
Path("/kaggle/working/validation-metrics.json").write_text(json.dumps(metrics, indent=2))
print(metrics)
