import importlib.util
import unittest
from pathlib import Path

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
    def test_pairs_short_bumpers_greedily(self):
        runs = [
            {"start_s": 10, "end_s": 12},
            {"start_s": 20, "end_s": 22},
            {"start_s": 30, "end_s": 32},
            {"start_s": 60, "end_s": 62},
        ]
        pairs = MODULE.replay_pairs(runs, max_duration_s=30)
        self.assertEqual([(row["start_s"], row["end_s"]) for row in pairs],
                         [(10, 22), (30, 62)])

    def test_does_not_pair_long_intro(self):
        runs = [{"start_s": 0, "end_s": 90}, {"start_s": 95, "end_s": 97}]
        self.assertEqual(MODULE.replay_pairs(runs, max_duration_s=30), [])


if __name__ == "__main__":
    unittest.main()
