import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_rally_intervals.py"
SPEC = importlib.util.spec_from_file_location("build_rally_intervals", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyIntervalsTest(unittest.TestCase):
    def test_uses_previous_end_as_recall_safe_start(self):
        route = {"start_s": 0}
        endings = [{"time_s": 10, "reason": "scoreboard_only"},
                   {"time_s": 25, "reason": "motion_fallback"}]
        intervals = MODULE.build_intervals(route, endings, [16])
        self.assertEqual(intervals[0]["safe_start_s"], 0)
        self.assertEqual(intervals[1]["safe_start_s"], 10)
        self.assertEqual(intervals[1]["estimated_start_s"], 16)
        self.assertEqual(intervals[1]["end_s"], 25)

    def test_rejects_non_positive_interval(self):
        with self.assertRaises(ValueError):
            MODULE.build_intervals({"start_s": 10},
                                   [{"time_s": 10, "reason": "scoreboard_only"}], [])


if __name__ == "__main__":
    unittest.main()
