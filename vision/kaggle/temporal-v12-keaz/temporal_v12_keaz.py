from __future__ import annotations

import json
import gzip
from pathlib import Path
import shutil
import subprocess
import sys
import time


output = Path("/kaggle/working/output")
output.mkdir(parents=True, exist_ok=True)
events = output / "events.jsonl"


def emit(stage: str, status: str, **details) -> None:
    event = {"time": time.time(), "stage": stage, "status": status, **details}
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    print("VOLLEYTAG_EVENT " + json.dumps(event, ensure_ascii=False, default=str), flush=True)


def run_stage(name: str, command: list[str]) -> None:
    started = time.monotonic()
    emit(name, "started", command=command)
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        emit(name, "failed", elapsed_s=time.monotonic() - started,
             error=f"{type(exc).__name__}: {exc}")
        raise
    emit(name, "passed", elapsed_s=time.monotonic() - started)


emit("runtime_install", "started")
install_started = time.monotonic()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.22",
     "joblib", "scikit-learn", "ultralytics"],
    check=True,
)
emit("runtime_install", "passed", elapsed_s=time.monotonic() - install_started)

root = next(Path("/kaggle/input").rglob("acceptance-criterion.json")).parent
scripts = root / "scripts"
runtime = Path("/kaggle/working/runtime-scripts")
shutil.copytree(scripts, runtime, dirs_exist_ok=True)
sys.path.insert(0, str(runtime))
dense_script = runtime / "evaluate_dense_temporal_identity_path.py"
dense_source = dense_script.read_text()
dense_source = dense_source.replace(
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
dense_script.write_text(dense_source)

video = root / "benchmark/source.mp4"
ground_truth = root / "benchmark/ground-truth.json"
exclude = root / "benchmark/non-match-intervals.json"
direct_model = root / "models/danya-detector-v9-keaz-best.pt"
generic_model = root / "models/yolo11n.pt"
near = root / "models/pairwise-reid-near_back.joblib"
far = root / "models/pairwise-reid-far_front.joblib"
criterion = json.loads((root / "acceptance-criterion.json").read_text())

cache = output / "candidates.json"
direct_route = output / "direct-only-route.csv"
candidate_route = output / "v12-route.csv"

available_caches = list(Path("/kaggle/input").rglob("candidates.json"))
available_caches += list(Path("/kaggle/input").rglob("candidates.json.gz"))
print(json.dumps({"available_candidate_caches": [str(path) for path in available_caches]}),
      flush=True)
if not available_caches:
    raise FileNotFoundError("Fresh kernel input snapshot has no candidate cache")
cached_input = next((path for path in available_caches if path.parent == root),
                    available_caches[0])
emit("candidate_cache_copy", "started", source=str(cached_input),
     bytes=cached_input.stat().st_size)
if cached_input.suffix == ".gz":
    with gzip.open(cached_input, "rb") as source, cache.open("wb") as target:
        shutil.copyfileobj(source, target)
else:
    shutil.copy2(cached_input, cache)
payload = json.loads(cache.read_text())
times = payload.get("times")
candidate_sets = payload.get("candidate_sets")
if not isinstance(times, list) or not isinstance(candidate_sets, list) or len(times) != len(candidate_sets):
    raise ValueError("Invalid dense candidate cache schema")
emit("candidate_cache_copy", "passed", frames=len(times), first_time_s=times[0],
     last_time_s=times[-1])

dummy_tracks = output / "unused-tracks.csv"
dummy_tracks.write_text("time_s,track_id,x1,y1,x2,y2,confidence\n")
run_stage("direct_only_route", [
    sys.executable, str(runtime / "route_player51_temporal.py"),
    str(direct_model), str(video), str(dummy_tracks),
    "--output", str(direct_route), "--candidate-cache", str(cache),
    "--decoder", "direct_only",
])

run_stage("v12_embedding_and_decode", [
    sys.executable, str(dense_script),
    str(video), str(cache), str(near), str(far),
    "--output", str(candidate_route),
    "--embedding-cache", str(output / "embeddings.npy"),
    "--device", "cuda", "--batch-size", "64",
    "--conservative-direct-priority",
])

reports = {}
for name, route in (("baseline", direct_route), ("candidate", candidate_route)):
    target = output / f"evaluation-{name}"
    run_stage(f"evaluation_{name}", [
        sys.executable, str(runtime / "evaluate_temporal_route.py"),
        str(route), str(ground_truth), "--output", str(target),
        "--iou", "0.30", "--max-time-delta", "0.25",
    ])
    reports[name] = json.loads((target / "metrics.json").read_text())

baseline_errors = reports["baseline"]["keyframes"] - reports["baseline"]["hits"]
candidate_errors = reports["candidate"]["keyframes"] - reports["candidate"]["hits"]
error_reduction = ((baseline_errors - candidate_errors) / baseline_errors
                   if baseline_errors else 0.0)
top1_delta = reports["candidate"]["top1_recall"] - reports["baseline"]["top1_recall"]
visibility_delta = (reports["candidate"]["visible_route_recall"] -
                    reports["baseline"]["visible_route_recall"])
limits = criterion["pass_if_all"]
passed = (
    error_reduction >= limits["minimum_relative_error_reduction"]
    and top1_delta >= -limits["maximum_absolute_top1_regression"]
    and visibility_delta >= limits["minimum_visible_route_recall_delta"]
)
summary = {
    "protocol": "one untouched dense temporal evaluation on KEAZ 2026-04-25",
    "criterion": criterion,
    "baseline": reports["baseline"],
    "candidate": reports["candidate"],
    "baseline_errors": baseline_errors,
    "candidate_errors": candidate_errors,
    "relative_error_reduction": error_reduction,
    "top1_delta": top1_delta,
    "visible_route_recall_delta": visibility_delta,
    "passed": passed,
    "frozen_set3_or_rasu_used": False,
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
emit("final_summary", "passed", passed=passed,
     baseline_top1=reports["baseline"]["top1_recall"],
     candidate_top1=reports["candidate"]["top1_recall"],
     relative_error_reduction=error_reduction)
print(json.dumps(summary, indent=2))
