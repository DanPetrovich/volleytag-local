#!/usr/bin/env python3
"""Replace delayed legacy Cloud action marks without touching their archive."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--frozen-output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    cloud = json.loads(args.cloud.read_text(encoding="utf-8"))
    old_cloud = [event for event in base.get("events", []) if event.get("source") in (0, 1)]
    retained = [event for event in base.get("events", []) if event.get("source") not in (0, 1)]
    new_training = [event for event in cloud.get("events", []) if event.get("split") == "train"]
    new_frozen = [event for event in cloud.get("events", []) if event.get("split") == "frozen_eval"]

    training = {**base}
    training["events"] = sorted(retained + new_training,
                                key=lambda event: (int(event.get("source", 999)), float(event["time_s"])))
    training["review_progress"] = {
        **base.get("review_progress", {}),
        **{key: value for key, value in cloud.get("review_progress", {}).items()
           if key in {"training-set1", "training-set2"}},
    }
    training["cloud_replacement"] = {
        "source": str(args.cloud),
        "removed_delayed_legacy_events": len(old_cloud),
        "added_new_training_events": len(new_training),
        "old_events_preserved_in": str(args.base),
        "frozen_events_kept_separate": str(args.frozen_output),
    }

    frozen = {
        "schema_version": 1,
        "kind": "volleytag.action-events.frozen-eval",
        "source": str(args.cloud),
        "training_use_forbidden": True,
        "events": sorted(new_frozen, key=lambda event: float(event["time_s"])),
        "review_progress": {
            key: value for key, value in cloud.get("review_progress", {}).items()
            if key == "set3-unseen"
        },
    }
    write_json(args.training_output, training)
    write_json(args.frozen_output, frozen)

    report = {
        "training_events": len(training["events"]),
        "training_by_source": dict(Counter(str(event["source"]) for event in training["events"])),
        "training_by_action": dict(Counter(event["action"] for event in training["events"])),
        "frozen_events": len(new_frozen),
        "frozen_by_action": dict(Counter(event["action"] for event in new_frozen)),
        "removed_old_cloud": len(old_cloud),
        "added_new_cloud_train": len(new_training),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
