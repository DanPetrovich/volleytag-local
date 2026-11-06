#!/usr/bin/env python3
"""Train and cross-check start/end fusion heads on exact reviewed rallies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def robust_rows(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    return (values - center) / np.maximum(scale, 1e-4)


def load_replays(path: Path, video: str) -> list[dict]:
    payload = json.loads(path.read_text())
    if "sources" in payload:
        matches = [row for row in payload["sources"].values() if row["video"] == video]
        if len(matches) != 1:
            raise ValueError("could not resolve replay source")
        return matches[0]["exclude_intervals"]
    if payload["video"] != video:
        raise ValueError("replay metadata belongs to another video")
    return payload.get("excluded_intervals", [])


def event_context(rows: list[dict], time: float, before: float, after: float,
                  score_key: str) -> tuple[float, float, float]:
    eligible = [row for row in rows if time - before <= float(row["time_s"]) <= time + after]
    if not eligible:
        return -5.0, before + after, 0.0
    best = max(eligible, key=lambda row: float(row.get(score_key, 1.0)))
    return float(best.get(score_key, 1.0)), float(best["time_s"]) - time, float(len(eligible))


def build_source(spec: dict) -> dict:
    queue = json.loads(Path(spec["queue"]).read_text())
    results = json.loads(Path(spec["results"]).read_text())
    motion = np.load(spec["motion_features"])
    item_ids = motion["item_ids"].astype(str)
    lookup = {row["id"]: row for row in queue["items"]}
    items = [lookup[item_id] for item_id in item_ids]
    times = np.asarray([float(row["time_s"]) for row in items])
    normalized_motion = robust_rows(motion["features"].astype(np.float64))
    frame_means = normalized_motion[:, [0, 4, 8, 12, 16]]
    court_bands = normalized_motion[:, 20:35]
    motion_summary = np.column_stack((
        frame_means.mean(axis=1), frame_means.std(axis=1),
        frame_means.min(axis=1), frame_means.max(axis=1),
        frame_means[:, -1] - frame_means[:, 0],
        court_bands.mean(axis=1), court_bands.std(axis=1),
        normalized_motion[:, -3:],
    ))
    previous = np.vstack((motion_summary[:2], motion_summary[:-2]))
    following = np.vstack((motion_summary[2:], motion_summary[-2:]))
    motion_context = np.concatenate((motion_summary, motion_summary - previous,
                                     following - motion_summary), axis=1)
    whistles = json.loads(Path(spec["whistles"]).read_text())["candidates"]
    if whistles:
        scores = np.asarray([float(row["score"]) for row in whistles])
        normalized = robust_rows(scores[:, None])[:, 0]
        whistles = [{**row, "normalized_score": float(score)}
                    for row, score in zip(whistles, normalized)]
    referee_payload = json.loads(Path(spec["referee_scores"]).read_text())
    referee = referee_payload["raw_candidates"]
    scoreboard = json.loads(Path(spec["scoreboard"]).read_text())["events"]
    replays = load_replays(Path(spec["replays"]), queue["video"])
    features = []
    for index, time in enumerate(times):
        whistle = event_context(whistles, time, 8, 1, "normalized_score")
        close_whistle = event_context(whistles, time, 2, 1, "normalized_score")
        referee_any = event_context(referee, time, 2.5, 2.5, "boundary_score")
        referee_start = event_context(
            [row for row in referee if row.get("predicted_label") == "serve_authorization"],
            time, 3, 3, "boundary_score")
        referee_end = event_context(
            [row for row in referee if row.get("predicted_label") == "rally_end_team_to_serve"],
            time, 3, 3, "boundary_score")
        score_future = event_context(scoreboard, time, 1, 15, "change_score")
        score_close = event_context(scoreboard, time, 2, 3, "change_score")
        inside_replay = float(any(row["start_s"] <= time <= row["end_s"] for row in replays))
        next_replay = min((row["start_s"] - time for row in replays
                           if 0 <= row["start_s"] - time <= 10), default=11.0)
        since_replay = min((time - row["end_s"] for row in replays
                            if 0 <= time - row["end_s"] <= 10), default=11.0)
        features.append(np.concatenate((motion_context[index], np.asarray([
            *whistle, *close_whistle, *referee_any, *referee_start, *referee_end,
            *score_future, *score_close, inside_replay, next_replay, since_replay,
        ], dtype=np.float64))))
    rallies = results["rallies"]
    labels = {}
    for kind, key in (("start", "start_s"), ("end", "end_s")):
        values = np.zeros(len(times), dtype=bool)
        retained = []
        for rally in rallies:
            index = int(np.argmin(np.abs(times - float(rally[key]))))
            if abs(times[index] - float(rally[key])) <= 1.0:
                values[index] = True
                retained.append(float(rally[key]))
        labels[kind] = values
        labels[f"{kind}_times"] = retained
    start_anchors = ([{"time_s": float(row["time_s"]), "kind": "whistle"}
                      for row in whistles if row["normalized_score"] >= 0]
                     + [{"time_s": float(row["time_s"]), "kind": "referee"}
                        for row in referee
                        if row.get("predicted_label") == "serve_authorization"])
    end_anchors = ([{"time_s": float(row["time_s"]), "kind": "scoreboard"}
                    for row in scoreboard]
                   + [{"time_s": float(row["start_s"]), "kind": "replay"}
                      for row in replays]
                   + [{"time_s": float(row["time_s"]), "kind": "referee"}
                      for row in referee
                      if row.get("predicted_label") == "rally_end_team_to_serve"])
    return {"name": spec["name"], "times": times, "features": np.stack(features),
            "labels": labels, "rallies": rallies,
            "duration_minutes": (times[-1] - times[0]) / 60,
            "start_anchors": start_anchors, "end_anchors": end_anchors}


def select_peaks(times: np.ndarray, scores: np.ndarray, threshold: float,
                 separation_s: float = 4.0) -> list[dict]:
    indexes = [index for index, score in enumerate(scores) if score >= threshold and
               score >= scores[max(0, index - 2):min(len(scores), index + 3)].max()]
    retained = []
    for index in sorted(indexes, key=lambda value: scores[value], reverse=True):
        if all(abs(times[index] - times[row]) >= separation_s for row in retained):
            retained.append(index)
    return [{"time_s": float(times[index]), "score": float(scores[index])}
            for index in sorted(retained)]


def match_events(truth: list[float], candidates: list[dict], tolerance_s: float = 2.0) -> list[dict]:
    pairs = sorted((abs(time - row["time_s"]), truth_index, candidate_index)
                   for truth_index, time in enumerate(truth)
                   for candidate_index, row in enumerate(candidates)
                   if abs(time - row["time_s"]) <= tolerance_s)
    used_truth, used_candidates, matches = set(), set(), []
    for error, truth_index, candidate_index in pairs:
        if truth_index in used_truth or candidate_index in used_candidates:
            continue
        used_truth.add(truth_index); used_candidates.add(candidate_index)
        matches.append({"truth_s": truth[truth_index],
                        "candidate_s": candidates[candidate_index]["time_s"],
                        "error_s": error})
    return matches


def choose_threshold(times: np.ndarray, scores: np.ndarray, truth: list[float],
                     duration_minutes: float) -> tuple[float, dict]:
    best = None
    for threshold in np.unique(np.quantile(scores, np.linspace(0, 1, 101))):
        candidates = select_peaks(times, scores, float(threshold))
        matches = match_events(truth, candidates)
        recall = len(matches) / max(len(truth), 1)
        precision = len(matches) / max(len(candidates), 1)
        density = len(candidates) / max(duration_minutes, 1e-8)
        eligible = density <= 8.0
        key = (eligible, recall, precision, -density, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), {"recall": recall, "precision": precision,
                                            "candidates_per_minute": density})
    assert best is not None
    return best[1], best[2]


def guided_candidates(times: np.ndarray, scores: np.ndarray, anchors: list[dict],
                      before_s: float, after_s: float,
                      separation_s: float = 4.0) -> list[dict]:
    proposals = []
    for anchor in anchors:
        eligible = np.where((times >= anchor["time_s"] - before_s)
                            & (times <= anchor["time_s"] + after_s))[0]
        if not len(eligible):
            continue
        index = int(eligible[np.argmax(scores[eligible])])
        proposals.append({"time_s": float(times[index]), "score": float(scores[index]),
                          "anchor_time_s": anchor["time_s"], "anchor_kind": anchor["kind"]})
    retained = []
    for row in sorted(proposals, key=lambda value: value["score"], reverse=True):
        if all(abs(row["time_s"] - old["time_s"]) >= separation_s for old in retained):
            retained.append(row)
    return sorted(retained, key=lambda row: row["time_s"])


def choose_guided_threshold(candidates: list[dict], truth: list[float],
                            duration_minutes: float) -> tuple[float, dict]:
    best = None
    values = np.asarray([row["score"] for row in candidates])
    for threshold in np.unique(np.r_[0.0, values]):
        retained = [row for row in candidates if row["score"] >= threshold]
        matches = match_events(truth, retained)
        recall = len(matches) / max(len(truth), 1)
        precision = len(matches) / max(len(retained), 1)
        density = len(retained) / max(duration_minutes, 1e-8)
        key = (density <= 8.0, recall, precision, -density, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), {"recall": recall, "precision": precision,
                                            "candidates_per_minute": density})
    assert best is not None
    return best[1], best[2]


def pair_intervals(starts: list[dict], ends: list[dict]) -> list[dict]:
    pending, result = [], []
    events = sorted([(row["time_s"], "start", row) for row in starts]
                    + [(row["time_s"], "end", row) for row in ends])
    for time, kind, row in events:
        if kind == "start":
            pending.append(row)
            pending = [value for value in pending if time - value["time_s"] <= 45]
        else:
            eligible = [value for value in pending if 1 <= time - value["time_s"] <= 45]
            if not eligible:
                continue
            start = max(eligible, key=lambda value: value["time_s"])
            result.append({"start_s": start["time_s"], "end_s": time,
                           "start_score": start["score"], "end_score": row["score"]})
            pending = []
    return result


def interval_matches(truth: list[dict], predicted: list[dict]) -> list[dict]:
    pairs = sorted((abs(row["start_s"] - gt["start_s"]) + abs(row["end_s"] - gt["end_s"]),
                    gt_index, pred_index)
                   for gt_index, gt in enumerate(truth)
                   for pred_index, row in enumerate(predicted)
                   if abs(row["start_s"] - gt["start_s"]) <= 2 and
                   abs(row["end_s"] - gt["end_s"]) <= 2)
    used_gt, used_pred, matches = set(), set(), []
    for _, gt_index, pred_index in pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index); used_pred.add(pred_index)
        matches.append({"gt_id": truth[gt_index]["id"], "prediction_index": pred_index})
    return matches


def fold(training: dict, evaluation: dict, c: float, proposal_mode: str) -> dict:
    outputs = {}
    candidates = {}
    for kind in ("start", "end"):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, class_weight="balanced", max_iter=5000, random_state=51))
        model.fit(training["features"], training["labels"][kind])
        training_scores = model.predict_proba(training["features"])[:, 1]
        if proposal_mode == "global":
            threshold, calibration = choose_threshold(
                training["times"], training_scores,
                training["labels"][f"{kind}_times"], training["duration_minutes"])
        else:
            if kind == "start":
                training_proposals = guided_candidates(
                    training["times"], training_scores, training["start_anchors"], 1, 8)
            else:
                training_proposals = guided_candidates(
                    training["times"], training_scores, training["end_anchors"], 15, 2)
            threshold, calibration = choose_guided_threshold(
                training_proposals, training["labels"][f"{kind}_times"],
                training["duration_minutes"])
        scores = model.predict_proba(evaluation["features"])[:, 1]
        if proposal_mode == "global":
            rows = select_peaks(evaluation["times"], scores, threshold)
        else:
            if kind == "start":
                proposals = guided_candidates(
                    evaluation["times"], scores, evaluation["start_anchors"], 1, 8)
            else:
                proposals = guided_candidates(
                    evaluation["times"], scores, evaluation["end_anchors"], 15, 2)
            rows = [row for row in proposals if row["score"] >= threshold]
        matches = match_events(evaluation["labels"][f"{kind}_times"], rows)
        candidates[kind] = rows
        outputs[kind] = {
            "threshold": threshold, "training_calibration": calibration,
            "anchors": len(evaluation[f"{kind}_anchors"]),
            "truth": len(evaluation["labels"][f"{kind}_times"]),
            "candidates": len(rows), "matches": len(matches),
            "recall": len(matches) / max(len(evaluation["labels"][f"{kind}_times"]), 1),
            "precision": len(matches) / max(len(rows), 1), "details": matches,
        }
    intervals = pair_intervals(candidates["start"], candidates["end"])
    matches = interval_matches(evaluation["rallies"], intervals)
    outputs["intervals"] = {
        "truth": len(evaluation["rallies"]), "candidates": len(intervals),
        "matches": len(matches), "recall": len(matches) / len(evaluation["rallies"]),
        "precision": len(matches) / max(len(intervals), 1),
        "predictions": intervals, "details": matches,
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sources = [build_source(row) for row in config["sources"]]
    c = float(config.get("logistic_c", .01))
    proposal_mode = config.get("proposal_mode", "global")
    if proposal_mode not in {"global", "guided"}:
        raise ValueError("proposal_mode must be global or guided")
    folds = []
    for training, evaluation in ((sources[0], sources[1]), (sources[1], sources[0])):
        folds.append({"training": training["name"], "evaluation": evaluation["name"],
                      **fold(training, evaluation, c, proposal_mode)})
    report = {"schema_version": 1, "kind": "volleytag.rally-boundary-fusion-loso",
              "frozen_tests_used": False,
              "protocol": "bidirectional whole-source transfer; thresholds selected on training source",
              "logistic_c": c, "proposal_mode": proposal_mode,
              "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.model_output:
        features = np.concatenate([row["features"] for row in sources])
        package = {"schema_version": 1, "kind": "volleytag.rally-boundary-fusion",
                   "frozen_tests_used": False, "proposal_mode": proposal_mode,
                   "training_sources": [row["name"] for row in sources], "models": {},
                   "thresholds": {}}
        for kind in ("start", "end"):
            labels = np.concatenate([row["labels"][kind] for row in sources])
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=51))
            model.fit(features, labels)
            package["models"][kind] = model
            package["thresholds"][kind] = float(np.median(
                [row[kind]["threshold"] for row in folds]))
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(package, args.model_output)
        report["model_output"] = str(args.model_output)
        report["final_thresholds"] = package["thresholds"]
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
