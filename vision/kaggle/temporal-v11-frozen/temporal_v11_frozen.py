from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.22", "joblib", "scikit-learn", "ultralytics"],
    check=True,
)

work = next(Path("/kaggle/input").rglob("acceptance-criterion.json")).parent

near = work / "models/pairwise-reid-near_back.joblib"
far = work / "models/pairwise-reid-far_front.joblib"
runtime_scripts = Path("/kaggle/working/runtime-scripts")
shutil.copytree(work / "scripts", runtime_scripts, dirs_exist_ok=True)
script = runtime_scripts / "evaluate_dense_temporal_identity_path.py"
source = script.read_text()
source = source.replace(
    "from evaluate_candidate_head_frozen import padded_crop",
    """def padded_crop(frame, box, padding=.08):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, round(x1 - bw * padding)), min(width, round(x2 + bw * padding))
    y1, y2 = max(0, round(y1 - bh * padding)), min(height, round(y2 + bh * padding))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.shape[0] >= 25 and crop.shape[1] >= 12 else None""",
)
script.write_text(source)
evaluate = runtime_scripts / "evaluate_temporal_route.py"

benchmarks = {
    "set3": {
        "video": work / "set3/source.mp4",
        "cache": work / "set3/candidates.json",
        "ground_truth": work / "set3/ground-truth.json",
        "baseline": work / "set3/baseline-route-metrics.json",
    },
    "rasu": {
        "video": work / "rasu/test-8min.mp4",
        "cache": work / "rasu/candidates.json",
        "ground_truth": work / "rasu/ground-truth.json",
        "baseline": work / "rasu/baseline-route-metrics.json",
    },
}

output = Path("/kaggle/working/output")
output.mkdir(parents=True, exist_ok=True)
reports = {}
for name, paths in benchmarks.items():
    target = output / name
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(script), str(paths["video"]), str(paths["cache"]),
        str(near), str(far), "--output", str(target / "route.csv"),
        "--embedding-cache", str(target / "embeddings.npy"),
        "--device", "cuda", "--batch-size", "64",
    ], check=True)
    subprocess.run([
        sys.executable, str(evaluate), str(target / "route.csv"),
        str(paths["ground_truth"]), "--output", str(target / "evaluation"),
    ], check=True)
    candidate = json.loads((target / "evaluation/metrics.json").read_text())
    baseline = json.loads(paths["baseline"].read_text())
    reports[name] = {"baseline": baseline, "candidate": candidate}

baseline_errors = sum(
    row["baseline"]["keyframes"] - row["baseline"]["hits"] for row in reports.values()
)
candidate_errors = sum(
    row["candidate"]["keyframes"] - row["candidate"]["hits"] for row in reports.values()
)
error_reduction = ((baseline_errors - candidate_errors) / baseline_errors
                   if baseline_errors else 0.0)
regressions = {
    name: row["baseline"]["top1_recall"] - row["candidate"]["top1_recall"]
    for name, row in reports.items()
}
visibility_deltas = {
    name: row["candidate"]["visible_route_recall"] - row["baseline"]["visible_route_recall"]
    for name, row in reports.items()
}
criterion = json.loads((work / "acceptance-criterion.json").read_text())
limits = criterion["success_criterion"]
passed = (
    error_reduction >= limits["minimum_combined_error_reduction"]
    and max(regressions.values()) <= limits["maximum_absolute_regression_per_benchmark"]
    and min(visibility_deltas.values()) >= limits["minimum_visible_route_recall_delta"]
)
summary = {
    "protocol": "fixed global path evaluated once on sparse frozen human keyframes",
    "criterion": criterion,
    "benchmarks": reports,
    "baseline_errors": baseline_errors,
    "candidate_errors": candidate_errors,
    "combined_error_reduction": error_reduction,
    "top1_regression_by_benchmark": regressions,
    "visible_route_recall_delta_by_benchmark": visibility_deltas,
    "passed": passed,
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
