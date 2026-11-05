from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


OUTPUT = Path("/kaggle/working/smoke")
OUTPUT.mkdir(parents=True, exist_ok=True)
EVENTS = OUTPUT / "events.jsonl"
REPORT = {"status": "running", "stages": [], "wandb": {"enabled": False}}


def emit(stage: str, status: str, **details) -> None:
    event = {"time": time.time(), "stage": stage, "status": status, **details}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    print("VOLLEYTAG_EVENT " + json.dumps(event, ensure_ascii=False, default=str), flush=True)


def stage(name: str, fn):
    started = time.monotonic()
    emit(name, "started")
    try:
        details = fn() or {}
    except Exception as exc:
        elapsed = time.monotonic() - started
        REPORT["stages"].append({"name": name, "status": "failed", "elapsed_s": elapsed,
                                 "error": f"{type(exc).__name__}: {exc}"})
        emit(name, "failed", elapsed_s=elapsed, error=f"{type(exc).__name__}: {exc}")
        raise
    elapsed = time.monotonic() - started
    row = {"name": name, "status": "passed", "elapsed_s": elapsed, **details}
    REPORT["stages"].append(row)
    emit(name, "passed", elapsed_s=elapsed, **details)
    return details


def locate_inputs():
    roots = [path.parent for path in Path("/kaggle/input").rglob("acceptance-criterion.json")]
    if len(roots) != 1:
        raise RuntimeError(f"expected one input root, found {roots}")
    root = roots[0]
    required = [
        root / "benchmark/source.mp4",
        root / "benchmark/ground-truth.json",
        root / "benchmark/non-match-intervals.json",
        root / "models/danya-detector-v9-keaz-best.pt",
        root / "models/yolo11n.pt",
        root / "models/pairwise-reid-near_back.joblib",
        root / "models/pairwise-reid-far_front.joblib",
        root / "scripts/evaluate_dense_temporal_identity_path.py",
        root / "scripts/evaluate_temporal_route.py",
        root / "scripts/route_player51_temporal.py",
    ]
    caches = [root / "candidates.json", root / "candidates.json.gz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    cache = next((path for path in caches if path.is_file()), None)
    if cache is None:
        raise FileNotFoundError(f"candidate cache absent; checked {caches}")
    REPORT["input_root"] = str(root)
    REPORT["cache"] = str(cache)
    return {"root": root, "cache": cache, "required_files": len(required) + 1,
            "cache_bytes": cache.stat().st_size}


def setup_wandb():
    try:
        from kaggle_secrets import UserSecretsClient
        import wandb
    except Exception as exc:
        return {"wandb_enabled": False, "reason": f"unavailable: {exc}"}
    token = None
    secret_name = None
    for name in ("WANDB_API_KEY", "wandb_api_key", "WANDB_KEY"):
        try:
            token = UserSecretsClient().get_secret(name)
        except Exception:
            continue
        if token:
            secret_name = name
            break
    if not token:
        return {"wandb_enabled": False, "reason": "Kaggle W&B secret not found"}
    wandb.login(key=token, relogin=True)
    run = wandb.init(project="volleytag", job_type="remote-smoke",
                     name="temporal-v12-keaz-smoke", config={"frames": 256})
    REPORT["wandb"] = {"enabled": True, "secret_name": secret_name, "url": run.url}
    return {"wandb_enabled": True, "wandb_url": run.url}


def install_runtime():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.22", "joblib", "scikit-learn", "ultralytics"], check=True)
    import cv2
    import joblib
    import timm
    import torch
    import ultralytics
    return {"python": sys.version.split()[0], "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cv2": cv2.__version__, "timm": timm.__version__,
            "ultralytics": ultralytics.__version__, "joblib": joblib.__version__}


def prepare_runtime(root: Path):
    runtime = Path("/kaggle/working/runtime-scripts")
    shutil.copytree(root / "scripts", runtime, dirs_exist_ok=True)
    dense_script = runtime / "evaluate_dense_temporal_identity_path.py"
    source = dense_script.read_text()
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
    dense_script.write_text(source)
    subprocess.run([sys.executable, "-m", "py_compile", *map(str, runtime.glob("*.py"))],
                   check=True)
    return {"runtime": runtime, "dense_script": dense_script,
            "compiled_scripts": len(list(runtime.glob("*.py")))}


def validate_data(root: Path, cache_source: Path):
    import cv2
    cache = OUTPUT / "candidates.json"
    if cache_source.suffix == ".gz":
        with gzip.open(cache_source, "rb") as source, cache.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        shutil.copy2(cache_source, cache)
    payload = json.loads(cache.read_text())
    sets = payload.get("candidate_sets")
    if not isinstance(sets, list) or len(sets) < 100:
        raise ValueError(f"bad candidate_sets: {type(sets)} / {len(sets) if sets else 0}")
    times = payload.get("times")
    if not isinstance(times, list) or len(times) != len(sets):
        raise ValueError(
            f"times/candidate_sets mismatch: {len(times) if isinstance(times, list) else type(times)} "
            f"vs {len(sets)}")
    sample = sets[:256]
    if any(not isinstance(row, list) for row in sample):
        raise ValueError("candidate_sets must contain candidate lists")
    capture = cv2.VideoCapture(str(root / "benchmark/source.mp4"))
    ok, frame = capture.read()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("video first frame unreadable")
    gt = json.loads((root / "benchmark/ground-truth.json").read_text())
    gt_samples = gt.get("samples") if isinstance(gt, dict) else gt
    if not isinstance(gt_samples, list) or not gt_samples:
        raise ValueError("ground truth is empty")
    return {"cache": cache, "candidate_sets": len(sets), "sample_sets": len(sample),
            "first_time_s": float(times[0]), "last_time_s": float(times[-1]),
            "video_frames": frames, "video_fps": fps,
            "ground_truth_items": len(gt_samples)}


def model_load(root: Path):
    import joblib
    from ultralytics import YOLO
    YOLO(str(root / "models/danya-detector-v9-keaz-best.pt"))
    YOLO(str(root / "models/yolo11n.pt"))
    joblib.load(root / "models/pairwise-reid-near_back.joblib")
    joblib.load(root / "models/pairwise-reid-far_front.joblib")
    return {"models_loaded": 4}


def mini_pipeline(root: Path, runtime: Path, dense_script: Path, cache: Path):
    route = OUTPUT / "mini-v12-route.csv"
    embeddings = OUTPUT / "mini-embeddings.npy"
    command = [
        sys.executable, str(dense_script), str(root / "benchmark/source.mp4"), str(cache),
        str(root / "models/pairwise-reid-near_back.joblib"),
        str(root / "models/pairwise-reid-far_front.joblib"),
        "--output", str(route), "--embedding-cache", str(embeddings),
        "--device", "cuda", "--batch-size", "64", "--limit-frames", "256",
        "--conservative-direct-priority",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    (OUTPUT / "mini-pipeline.stdout.log").write_text(result.stdout)
    (OUTPUT / "mini-pipeline.stderr.log").write_text(result.stderr)
    rows = max(0, sum(1 for _ in route.open()) - 1)
    if rows != 256 or not embeddings.is_file():
        raise RuntimeError(f"mini pipeline incomplete: rows={rows}, embeddings={embeddings.exists()}")
    return {"route_rows": rows, "embedding_bytes": embeddings.stat().st_size,
            "stdout_tail": result.stdout[-500:]}


try:
    located = stage("input_mount_and_manifest", locate_inputs)
    stage("wandb_connection", setup_wandb)
    stage("runtime_install_and_imports", install_runtime)
    prepared = stage("script_packaging_and_compile", lambda: prepare_runtime(located["root"]))
    data = stage("cache_video_and_ground_truth", lambda: validate_data(located["root"], located["cache"]))
    stage("model_deserialization", lambda: model_load(located["root"]))
    stage("exact_mini_pipeline", lambda: mini_pipeline(
        located["root"], prepared["runtime"], prepared["dense_script"], data["cache"]))
    REPORT["status"] = "passed"
except Exception as exc:
    REPORT["status"] = "failed"
    REPORT["error"] = f"{type(exc).__name__}: {exc}"
    REPORT["traceback"] = traceback.format_exc()
finally:
    (OUTPUT / "smoke-report.json").write_text(
        json.dumps(REPORT, ensure_ascii=False, indent=2, default=str) + "\n")
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"smoke_passed": int(REPORT["status"] == "passed"),
                       "stages_passed": sum(row["status"] == "passed" for row in REPORT["stages"])})
            wandb.save(str(OUTPUT / "smoke-report.json"), base_path=str(OUTPUT))
            wandb.finish(exit_code=0 if REPORT["status"] == "passed" else 1)
    except Exception:
        pass
    print(json.dumps(REPORT, ensure_ascii=False, indent=2, default=str), flush=True)

if REPORT["status"] != "passed":
    raise SystemExit(1)
