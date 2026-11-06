import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "evaluate_whistle_candidate_grouping.py"
SPEC = importlib.util.spec_from_file_location("evaluate_whistle_candidate_grouping", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class WhistleCandidateGroupingTest(unittest.TestCase):
    def test_keeps_latest_candidate_in_connected_group(self):
        rows = [{"time_s": 1.0}, {"time_s": 2.5}, {"time_s": 4.4},
                {"time_s": 8.0}]
        grouped = MODULE.group_latest(rows, max_gap_s=2.0)
        self.assertEqual([row["time_s"] for row in grouped], [4.4, 8.0])

    def test_anchor_hit_uses_asymmetric_serve_window(self):
        serves = [10.0, 30.0]
        rows = [{"time_s": 3.0}, {"time_s": 31.5}]
        self.assertEqual(MODULE.anchor_hits(serves, rows, 8.0, 1.0), 1)


if __name__ == "__main__":
    unittest.main()
