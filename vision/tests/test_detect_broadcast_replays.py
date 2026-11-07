import importlib.util
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional vision dependency unavailable: {error.name}")


MODULE_PATH = Path(__file__).parents[1] / "detect_broadcast_replays.py"
SPEC = importlib.util.spec_from_file_location("detect_broadcast_replays", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BroadcastReplayIntervalsTest(unittest.TestCase):
    def test_classifies_fixed_bumper_layouts_without_ocr(self):
        replay = np.full((90, 160, 3), (80, 0, 120), dtype=np.uint8)
        replay[35:55, 45:115] = 255
        league = np.full((90, 160, 3), (80, 0, 120), dtype=np.uint8)
        league[15:35, 45:130] = 255
        kinds = [row["bumper_kind"]
                 for row in MODULE.classify_bumpers([replay, league])]
        self.assertEqual(kinds, ["replay_start", "replay_end"])

    def test_pairs_typed_bumpers_even_for_long_highlight_package(self):
        runs = [
            {"start_s": 10, "end_s": 12, "bumper_kind": "replay_start"},
            {"start_s": 52, "end_s": 54, "bumper_kind": "replay_end"},
            {"start_s": 80, "end_s": 82, "bumper_kind": "replay_start"},
            {"start_s": 86, "end_s": 88, "bumper_kind": "replay_end"},
        ]
        pairs = MODULE.replay_pairs(runs, max_duration_s=60)
        self.assertEqual([(row["start_s"], row["end_s"]) for row in pairs],
                         [(10, 54), (80, 88)])

    def test_uses_ambiguous_bumper_before_next_start_as_end(self):
        runs = [
            {"start_s": 10, "end_s": 12, "bumper_kind": "replay_start"},
            {"start_s": 20, "end_s": 22, "bumper_kind": "unknown"},
            {"start_s": 40, "end_s": 42, "bumper_kind": "replay_start"},
            {"start_s": 46, "end_s": 48, "bumper_kind": "replay_end"},
        ]
        pairs = MODULE.replay_pairs(runs, max_duration_s=60)
        self.assertEqual([(row["start_s"], row["end_s"]) for row in pairs],
                         [(10, 22), (40, 48)])

    def test_ignores_orphan_end_and_long_bumper(self):
        runs = [
            {"start_s": 0, "end_s": 90, "bumper_kind": "replay_start"},
            {"start_s": 95, "end_s": 97, "bumper_kind": "replay_end"},
            {"start_s": 110, "end_s": 112, "bumper_kind": "replay_end"},
        ]
        self.assertEqual(MODULE.replay_pairs(runs, max_duration_s=60), [])


if __name__ == "__main__":
    unittest.main()
