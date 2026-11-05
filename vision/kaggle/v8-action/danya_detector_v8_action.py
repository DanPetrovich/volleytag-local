from pathlib import Path
import json
import shutil
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.198"], check=True)

from ultralytics import YOLO

input_root = Path("/kaggle/input")
model_path = next(input_root.rglob("danya-detector-v7-best.pt"))
action_root = next(path.parent for path in input_root.rglob("dataset.yaml")
                   if path.parent.name == "action-dataset-v1")
base_root = next(path.parent for path in input_root.rglob("dataset.yaml")
                 if path.parent.name == "dataset")
data_yaml = Path("/kaggle/working/dataset.yaml")
data_yaml.write_text(
    "train:\n"
    f"  - {base_root / 'images/train'}\n"
    f"  - {action_root / 'images/train'}\n"
    "val:\n"
    f"  - {base_root / 'images/val'}\n"
    f"  - {action_root / 'images/val'}\n"
    "names:\n  0: danya\n"
)
train_count = sum(len(list((root / "images/train").glob("*.jpg")))
                  for root in (base_root, action_root))
val_count = sum(len(list((root / "images/val").glob("*.jpg")))
                for root in (base_root, action_root))
print("base dataset:", base_root)
print("action dataset:", action_root)
print("warm start:", model_path)
print("train/val:", train_count, val_count)
assert train_count == 2242 and val_count == 292, (train_count, val_count)

model = YOLO(str(model_path))
results = model.train(
    data=str(data_yaml),
    epochs=45,
    imgsz=960,
    batch=16,
    device=0,
    workers=4,
    patience=10,
    close_mosaic=6,
    cache="disk",
    seed=51,
    project="/kaggle/working/runs",
    name="danya-detector-v8-action",
    plots=True,
)

run = Path("/kaggle/working/runs/danya-detector-v8-action")
best = run / "weights/best.pt"
shutil.copy2(best, "/kaggle/working/danya-detector-v8-action-best.pt")
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
