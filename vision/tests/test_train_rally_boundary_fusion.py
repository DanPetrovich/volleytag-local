import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import sklearn  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional fusion dependency unavailable: {error.name}")


MODULE_PATH = Path(__file__).parents[1] / "train_rally_boundary_fusion.py"
SPEC = importlib.util.spec_from_file_location("train_rally_boundary_fusion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyBoundaryFusionTest(unittest.TestCase):
    def test_event_matching_is_one_to_one(self):
        matches = MODULE.match_events(
            [10.0, 12.0], [{"time_s": 11.0}, {"time_s": 14.0}], tolerance_s=2)
        self.assertEqual(len(matches), 2)

    def test_guided_candidate_searches_relative_to_anchor(self):
        rows = MODULE.guided_candidates(
            np.asarray([8.0, 9.0, 10.0, 11.0]),
            np.asarray([.1, .2, .9, .3]),
            [{"time_s": 9.0, "kind": "whistle"}], before_s=1, after_s=2)
        self.assertEqual(rows[0]["time_s"], 10.0)

    def test_interval_pair_uses_latest_start_before_end(self):
        intervals = MODULE.pair_intervals(
            [{"time_s": 10.0, "score": .7}, {"time_s": 12.0, "score": .8}],
            [{"time_s": 20.0, "score": .9}])
        self.assertEqual(intervals[0]["start_s"], 12.0)


if __name__ == "__main__":
    unittest.main()
