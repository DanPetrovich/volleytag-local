#!/usr/bin/env python3
"""Fuse scoreboard boundaries with unmatched referee-motion fallback candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fuse_events(score_events: list[dict], motion_items: list[dict],
                before_s: float, after_s: float) -> tuple[list[dict], set[int]]:
    """Fuse evidence while keeping score changes as the safe end timestamp.

    Referee motion can be a serve authorization that happens before the score
    overlay updates.  It corroborates a score event but must never move the
    rally end earlier and cut live play from the interval.
    """
    used_motion: set[int] = set()
    events = []
    for score_event in score_events:
        score_time = float(score_event["time_s"])
        eligible = [
            (abs(float(row["time_s"]) - score_time), motion_index, row)
            for motion_index, row in enumerate(motion_items)
            if motion_index not in used_motion and
            score_time - before_s <= float(row["time_s"]) <= score_time + after_s
        ]
        if eligible:
            _, motion_index, row = min(eligible)
            used_motion.add(motion_index)
            events.append({"time_s": score_time, "reason": "scoreboard+motion",
                           "scoreboard_time_s": score_time,
                           "matched_motion_time_s": float(row["time_s"]),
                           "motion_score": row["motion_score"]})
        else:
            events.append({"time_s": score_time, "reason": "scoreboard_only",
                           "scoreboard_time_s": score_time})
    events.extend({"time_s": row["time_s"], "reason": "motion_fallback",
                   "motion_score": row["motion_score"]}
                  for index, row in enumerate(motion_items) if index not in used_motion)
    events.sort(key=lambda row: row["time_s"])
    return events, used_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scoreboard", type=Path)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--roi", type=float, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--before-score-change-s", type=float, default=10)
    parser.add_argument("--after-score-change-s", type=float, default=6)
    parser.add_argument("--expected-score-changes", type=int)
    parser.add_argument("--post-frozen-timestamp-fix", action="store_true",
                        help="mark a route rebuilt after frozen timestamp inspection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    args = parser.parse_args()
    scoreboard = json.loads(args.scoreboard.read_text())
    motion = json.loads(args.motion.read_text())
    if scoreboard["video"] != motion["video"]:
        raise ValueError("scoreboard and motion candidates belong to different videos")
    events, used_motion = fuse_events(scoreboard["events"], motion["items"],
                                      args.before_score_change_s,
                                      args.after_score_change_s)
    fallback = [row for index, row in enumerate(motion["items"]) if index not in used_motion]
    duration_minutes = (float(scoreboard["end_s"]) - float(scoreboard["start_s"])) / 60
    frozen = bool(scoreboard["frozen_tests_used"] or motion["frozen_tests_used"])
    route = {
        "schema_version": 1,
        "kind": "volleytag.fused-rally-boundary-candidates",
        "video": scoreboard["video"],
        "start_s": scoreboard["start_s"],
        "end_s": scoreboard["end_s"],
        "frozen_tests_used": frozen,
        "post_frozen_timestamp_fix": bool(args.post_frozen_timestamp_fix),
        "parameters": {
            "before_score_change_s": args.before_score_change_s,
            "after_score_change_s": args.after_score_change_s,
        },
        "scoreboard_events": len(scoreboard["events"]),
        "expected_score_changes": args.expected_score_changes,
        "scoreboard_count_matches_expected": (
            None if args.expected_score_changes is None
            else len(scoreboard["events"]) == args.expected_score_changes),
        "matched_scoreboard_motion": len(used_motion),
        "motion_fallback": len(fallback),
        "candidate_density_per_minute": len(events) / max(duration_minutes, 1e-8),
        "events": events,
    }
    review_items = [{
        "id": f"frozen-motion-fallback-s{args.source}-t{float(row['time_s']):.3f}",
        "source": args.source,
        "video": scoreboard["video"],
        "time_s": float(row["time_s"]),
        "window_start_s": max(float(scoreboard["start_s"]), float(row["time_s"]) - 3),
        "window_end_s": min(float(scoreboard["end_s"]), float(row["time_s"]) + 3),
        "roi_box_norm": args.roi,
        "selection_reason": "frozen_motion_fallback",
        "motion_score": row["motion_score"],
    } for row in fallback]
    review = {
        "schema_version": 1,
        "kind": "volleytag.referee-gesture-review-queue",
        "frozen_tests_used": frozen,
        "labels": ["serve_authorization", "rally_end_team_to_serve", "other_signal",
                   "no_signal", "uncertain"],
        "counts": {"frozen_motion_fallback": len(review_items)},
        "items": review_items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n")
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    args.review_output.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "scoreboard_events": len(scoreboard["events"]),
        "scoreboard_count_matches_expected": route["scoreboard_count_matches_expected"],
        "matched_scoreboard_motion": len(used_motion),
        "motion_fallback": len(fallback),
        "total_candidates": len(events),
        "candidate_density_per_minute": route["candidate_density_per_minute"],
    }, indent=2))


if __name__ == "__main__":
    main()
